import math

import torch
import torch.nn.functional as F

from examples.speech_recognition.models.context_model import FairseqContextModel
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel


class ContextAwareSequenceGenerator(SequenceGenerator):
    """
    Generates translations of a given source sentence with a context.

    Args:
        tgt_dict (~fairseq.data.Dictionary): target dictionary
        beam_size (int, optional): beam width (default: 1)
        max_len_a/b (int, optional): generate sequences of maximum length
            ax + b, where x is the source length
        min_len (int, optional): the minimum length of the generated output
            (not including end-of-sentence)
        normalize_scores (bool, optional): normalize scores by the length
            of the output (default: True)
        len_penalty (float, optional): length penalty, where <1.0 favors
            shorter, >1.0 favors longer sentences (default: 1.0)
        unk_penalty (float, optional): unknown word penalty, where <0
            produces more unks, >0 produces fewer (default: 0.0)
        retain_dropout (bool, optional): use dropout when generating
            (default: False)
        temperature (float, optional): temperature, where values
            >1.0 produce more uniform samples and values <1.0 produce
            sharper samples (default: 1.0)
        match_source_len (bool, optional): outputs should match the source
            length (default: False)
    """
    def __init__(
            self,
            tgt_dict,
            beam_size=1,
            max_len_a=0,
            max_len_b=200,
            min_len=1,
            normalize_scores=True,
            len_penalty=1.,
            unk_penalty=0.,
            retain_dropout=False,
            temperature=1.,
            match_source_len=False,
            no_repeat_ngram_size=0,
            search_strategy=None,
    ):
        super().__init__(
            tgt_dict,
            beam_size=beam_size,
            max_len_a=max_len_a,
            max_len_b=max_len_b,
            min_len=min_len,
            normalize_scores=normalize_scores,
            len_penalty=len_penalty,
            unk_penalty=unk_penalty,
            retain_dropout=retain_dropout,
            temperature=temperature,
            match_source_len=match_source_len,
            no_repeat_ngram_size=no_repeat_ngram_size,
            search_strategy=search_strategy,
        )
        self.buffer = self.default_buffer
        self.context = None

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = ContextAwareEnsembleModel(models)
        src_tokens = sample['net_input']['src_tokens']
        self.context = self.next_batch_context(src_tokens, self.buffer)
        assert src_tokens.size(0) == self.context.size(0),\
            "Batch size ({}) is different from context batch size ({}).".format(
                src_tokens.size(0), self.context.size(0))
        self.context = self.context.to(src_tokens.device)
        context_lengths = torch.LongTensor([self.context.size(1)]).to(src_tokens.device)
        model.forward_context(self.context, context_lengths)
        hypos = self._generate(model, sample, **kwargs)
        self.buffer = self.new_prev_buffer(src_tokens, hypos)
        return hypos

    @property
    def default_buffer(self):
        raise NotImplementedError

    def next_batch_context(self, src_tokens, buffer):
        raise NotImplementedError

    def new_prev_buffer(self, src_tokens, hypos):
        raise NotImplementedError


class TargetContextAwareSequenceGenerator(ContextAwareSequenceGenerator):
    """
    Generates translations of a given source sentence.
    This class provides the output generated for the previous sentence as
    context for the next one.
    """

    @property
    def default_buffer(self):
        return torch.LongTensor([[self.eos]])

    def next_batch_context(self, src_tokens, buffer):
        return buffer

    def new_prev_buffer(self, src_tokens, hypos):
        # We take the most likely sentence from previous generation as context for the next
        return hypos[0][0]['tokens'].unsqueeze(0)


class AudioContextAwareSequenceGenerator(ContextAwareSequenceGenerator):
    """
    Generates translations of a given source sentence.
    This class provides the source audio of the previous sentence as
    context for the next one.
    """

    @property
    def default_buffer(self):
        return None

    def next_batch_context(self, src_tokens, buffer):
        if buffer is not None:
            prev_spectrograms = buffer
        else:
            prev_spectrograms = torch.zeros(
                (src_tokens.size(1), src_tokens.size(2))).to(src_tokens.device)

        if src_tokens.shape[0] == 1:
            return prev_spectrograms.unsqueeze(0)

        context = src_tokens[:-1]
        if context.shape[1] > prev_spectrograms.shape[0]:
            prev_spectrograms = F.pad(
                prev_spectrograms, (0, 0, 0, context.shape[1] - prev_spectrograms.shape[0]))
        elif context.shape[1] < prev_spectrograms.shape[0]:
            context = F.pad(
                context, (0, 0, 0, prev_spectrograms.shape[0] - context.shape[1]))

        return torch.cat([prev_spectrograms.unsqueeze(0), context], dim=0)

    def new_prev_buffer(self, src_tokens, hypos):
        # We take the most likely sentence from previous generation as context for the next
        return src_tokens[-1]


class ContextAwareEnsembleModel(EnsembleModel):
    def __init__(self, models):
        for m in models:
            assert isinstance(m, FairseqContextModel)
        super().__init__(models)
        self.context_outs = None

    def forward_context(self, context, context_lengths):
        self.context_outs = [
            model.context_encoder(context, src_lengths=context_lengths) for model in self.models]

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        encoder_outs = []
        for model, context_out in zip(self.models, self.context_outs):
            encoder_outs.append(model.encoder(**encoder_input, context_out=context_out))
        return encoder_outs

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.context_outs[0],
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out, context_out in zip(self.models, encoder_outs, self.context_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                context_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, context_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens,
                encoder_out=encoder_out,
                context_out=context_out,
                incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, context_out=context_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1]
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        new_encoder_out = super().reorder_encoder_out(encoder_outs, new_order)
        self.context_outs = [
            model.context_encoder.reorder_encoder_out(context_out, new_order)
            for model, context_out in zip(self.models, self.context_outs)
        ]
        return new_encoder_out
