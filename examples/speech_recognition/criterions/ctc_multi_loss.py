import math

import torch
from torch import nn
import torch.nn.functional as F


from examples.speech_recognition.criterions.CTC_loss import CTCCriterion
from fairseq import utils, metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models import BaseFairseqModel


class CTCEncoderWrapperModel(BaseFairseqModel):
    def __init__(self, args, ctc_dictionary):
        super().__init__()
        self.fc_out = nn.Linear(args.encoder_embed_dim, len(ctc_dictionary))
        self.ctc_encoder_layer = args.ctc_encoder_layer

    def forward(self, model, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        if 'return_all_hiddens' in kwargs:
            del kwargs['return_all_hiddens']
        encoder_out = model.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=True, **kwargs)
        decoder_out = model.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        if isinstance(encoder_out, dict):
            encoder_states = encoder_out["encoder_states"]
            encoder_padding_mask = encoder_out["encoder_padding_mask"]
        elif hasattr(encoder_out, "encoder_states"):
            encoder_states = encoder_out.encoder_states
            encoder_padding_mask = encoder_out.encoder_padding_mask
        else:
            raise NotImplementedError("Encoder output not supported by CTC multi loss")
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # B x T => T x B
        ctc_features = encoder_states[self.ctc_encoder_layer - 1]
        return decoder_out, {
            "encoder_out": self.fc_out(ctc_features),
            "encoder_padding_mask": encoder_padding_mask
        }


class FakeEncoderModel(nn.Module):
    def __init__(self, encoder, net_out, target):
        super().__init__()
        self.net_out = net_out
        self.target = target
        if hasattr(encoder, "output_batch_first"):
            self.output_batch_first = encoder.output_batch_first

    def forward(self, **unused):
        return self.net_out

    def get_targets(self, *unused):
        return self.target

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output["encoder_out"]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                probs = F.log_softmax(logits, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            if hasattr(self, "output_batch_first"):
                probs.batch_first = self.output_batch_first
            return probs
        raise NotImplementedError


class FakeDecoderModel(nn.Module):
    def __init__(self, model, net_out, target):
        super().__init__()
        self.model = model
        self.net_out = net_out
        self.target = target

    def forward(self, **unused):
        return self.net_out

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        return self.model.get_normalized_probs(net_output, log_probs, sample=sample)

    def get_targets(self, *unused):
        return self.target

    @property
    def decoder(self):
        return self.model.decoder


class BaseCTCLoss(CTCCriterion):
    def __init__(self, args, task):
        super(CTCCriterion, self).__init__(args, task)
        self.args = args
        self.blank_idx = task.source_dictionary.index("<ctc_blank>")
        self.pad_idx = task.source_dictionary.pad()


@register_criterion("ctc_multi_loss")
class CTCMultiLoss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        assert task.source_dictionary is not None
        self.ctc_aware_model = CTCEncoderWrapperModel(args, task.source_dictionary)
        self.ctc_criterion = BaseCTCLoss(args, task)
        self.real_criterion = CTCMultiLoss.build_real_criterion(args, task)
        self.ctc_weight = args.ctc_weight

    @staticmethod
    def build_real_criterion(args, task):
        saved_criterion = args.criterion
        args.criterion = args.underlying_criterion
        assert saved_criterion != args.underlying_criterion
        underlying_criterion = task.build_criterion(args)
        args.criterion = saved_criterion
        return underlying_criterion

    @staticmethod
    def add_args(parser):
        CTCCriterion.add_args(parser)
        parser.add_argument('--ctc-encoder-layer', default=6, type=int, metavar='LAYER_NUM',
                            help='The encoder layer whose feature are used to compute the CTC loss')
        parser.add_argument('--ctc-weight', default=1.0, type=float, metavar='W',
                            help='The relative weight to assign to the CTC loss')
        parser.add_argument('--underlying-criterion', type=str, metavar='VAL', required=True,
                            help='underlying criterion to use for the model output loss')

    def forward(self, model, sample, reduce=True, log_probs=True):
        decoder_out, encoder_out = self.ctc_aware_model(model, **sample["net_input"])
        encoder_fake_model = FakeEncoderModel(model.encoder, encoder_out, sample["encoder_target"])
        decoder_fake_model = FakeDecoderModel(model, decoder_out, sample["target"])
        encoder_sample = {
            "net_input": sample["net_input"],
            "target": sample["encoder_target"],
            "target_lengths": sample["encoder_target_lengths"],
            "ntokens": sum(sample["encoder_target_lengths"]).item()
        }
        ctc_loss, ctc_sample_size, ctc_logging_output = self.ctc_criterion(
            encoder_fake_model, encoder_sample, reduce=reduce, log_probs=log_probs)
        real_loss, _, real_logging_output = self.real_criterion(
            decoder_fake_model, sample, reduce=reduce)
        loss = self.ctc_weight * ctc_loss + real_loss

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ctc_loss": ctc_logging_output['loss'],
            "ntokens": real_logging_output['ntokens'],
            "nsentences": real_logging_output['nsentences'],
            "sample_size": real_logging_output['sample_size'],
            "ctc_errors": ctc_logging_output['errors'],
            "ctc_total": ctc_logging_output['total'],
            "nframes": ctc_logging_output['nframes'],
        }
        if 'nll_loss' in real_logging_output:
            logging_output['nll_loss'] = real_logging_output['nll_loss']
        return loss, ctc_sample_size, logging_output

    @staticmethod
    def logging_outputs_can_be_summed():
        return True

    @staticmethod
    def reduce_metrics(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        ctc_loss_sum = utils.item(sum(log.get('ctc_loss', 0) for log in logging_outputs))
        if logging_outputs and 'nll_loss' in logging_outputs[0]:
            nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        else:
            nll_loss_sum = loss_sum - ctc_loss_sum  # NLL computed on the real loss, not on the auxiliary CTC
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        ctc_errors = sum(log.get("ctc_errors", 0) for log in logging_outputs)
        ctc_total = sum(log.get("ctc_total", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        metrics.log_scalar('ctc_loss', ctc_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('ctc_acc', 100.0 - min(ctc_errors * 100.0 / ctc_total, 100.0))
        metrics.log_scalar('nframes', nframes)
