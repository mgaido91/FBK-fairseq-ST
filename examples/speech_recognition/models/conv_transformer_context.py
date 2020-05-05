import logging
import math
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.models import register_model, FairseqEncoder, register_model_architecture
from fairseq.models.transformer import EncoderOut, TransformerDecoder
from fairseq.modules import TransformerEncoderLayer, PositionalEmbedding
from . import conv_transformer

from .context_model import FairseqContextModel
from .conv_transformer import ConvolutionalTransformerModel, ConvolutionalTransformerEncoder
from ..modules.conv_transformer_context_layer import ConvTransformerContextAwareEncoderLayer, \
    TransformerContextAwareDecoderLayer

logger = logging.getLogger(__name__)


@register_model('conv_transformer_context')
class ConvolutionalTransformerContextAwareModel(FairseqContextModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        ConvolutionalTransformerModel.add_args(parser)
        parser.add_argument('--context-encoder-layers', type=int, metavar='N',
                            help='num context encoder layers', default=1)
        parser.add_argument('--context-position', type=str, default="both",
                            choices=['both', 'encoder', "decoder"], )
        parser.add_argument('--context-decoder-attention-type', type=str, default="sequential",
                            choices=['parallel', "sequential"])
        parser.add_argument('--pretrained-model', type=str, default=None,
                            help='path to a pretrained context-unaware model')
        parser.add_argument('--freeze-pretrained', type=str, default="encoder",
                            choices=['all', 'encoder', "none"],
                            help='by default ("encoder") only encoder pretrained weights are freezed; '
                                 'if set to "none", no parameter is freezed (be careful to OOM);'
                                 'if set to "all", all params loaded from the pretrained model are freezed.')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 100000
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 100000

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        decoder_embed_tokens = build_embedding(
            tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)

        if args.context_type == 'src':
            context_encoder = PreviousAudioContextEncoder(args, task)
        else:
            context_encoder = PreviousTargetContextEncoder(args, tgt_dict, decoder_embed_tokens)
        encoder = ConvolutionalTransformerContextAwareEncoder(
            args, tgt_dict, audio_features=args.input_feat_per_channel)
        decoder = TransformerContextAwareDecoder(args, tgt_dict, decoder_embed_tokens)
        model = ConvolutionalTransformerContextAwareModel(encoder, decoder, context_encoder)
        if args.pretrained_model is not None:
            pretrained_model_state = checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_model)
            incompatible_keys = model.load_state_dict(pretrained_model_state["model"], strict=False)
            if len(incompatible_keys.unexpected_keys) != 0:
                logger.error("Cannot load the following keys from checkpoint: {}".format(
                    incompatible_keys.unexpected_keys))
                raise ValueError("Cannot load from checkpoint: {}".format(args.pretrained_model))

            for missing_key in incompatible_keys.missing_keys:
                if 'context' not in missing_key:
                    logger.error("Loaded checkpoint misses the parameter: {}.".format(missing_key))
            if args.freeze_pretrained != "none":
                for p_name, p_val in model.named_parameters():
                    if p_name in pretrained_model_state["model"] and \
                            (args.freeze_pretrained == "all" or "decoder" not in p_name):
                        p_val.requires_grad = False

        return model


class PreviousAudioContextEncoder(FairseqEncoder):
    def __init__(self, args, task):
        super().__init__(None)
        assert args.pretrained_model is not None
        pretrained_models, _ = checkpoint_utils.load_model_ensemble([args.pretrained_model], task=task)
        self.audio_encoder = pretrained_models[0].encoder
        if args.freeze_pretrained != "none":
            for p_name, p_val in self.audio_encoder.named_parameters():
                p_val.requires_grad = False
        self.n_layers = args.context_encoder_layers
        self.dropout = args.dropout
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(self.n_layers)
        ])

    def forward(self, src_tokens, src_lengths=None):
        prev_encoder_out = self.audio_encoder(src_tokens, src_lengths=src_lengths)
        x = prev_encoder_out.encoder_out
        padding_mask = prev_encoder_out.encoder_padding_mask
        x = F.dropout(x, p=self.dropout, training=self.training)

        # encoder layers
        for layer in self.layers:
            x = layer(x, padding_mask)

        # shapes are: T x B x C, B x T
        return {
            'context_out': x,
            'context_padding_mask': padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['context_out'] is not None:
            encoder_out['context_out'] = \
                encoder_out['context_out'].index_select(1, new_order)
        if encoder_out['context_padding_mask'] is not None:
            encoder_out['context_padding_mask'] = \
                encoder_out['context_padding_mask'].index_select(0, new_order)
        return encoder_out


class PreviousTargetContextEncoder(FairseqEncoder):
    def __init__(self, args, tgt_dict, embed_tokens):
        super().__init__(tgt_dict)
        self.n_layers = args.context_encoder_layers
        self.embed_dim = args.decoder_embed_dim
        self.embed_scale = math.sqrt(self.embed_dim)  # todo: try with input_embed_dim
        self.embed_tokens = embed_tokens
        self.dropout = args.dropout
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, self.embed_dim, self.embed_tokens.padding_idx,
            learned=args.decoder_learned_pos,
        )
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(args) for _ in range(self.n_layers)
        ])
        input_embed_dim = embed_tokens.embedding_dim
        self.project_in_dim = Linear(input_embed_dim, self.embed_dim, bias=False) \
            if self.embed_dim != input_embed_dim else None

    @staticmethod
    def create_mask(lengths):
        max_len = max(lengths)
        mask = lengths.new_zeros(len(lengths), max_len).bool()
        for i, l in enumerate(lengths):
            mask[i, l:] = 1
        if not mask.any():
            mask = None
        return mask

    def forward(self, src_tokens, src_lengths=None):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.project_in_dim is not None:
            x = self.project_in_dim(x)
        x = x + self.embed_positions(src_tokens, incremental_state=None, timestep=src_lengths)
        x = F.dropout(x, p=self.dropout, training=self.training)
        padding_mask = PreviousTargetContextEncoder.create_mask(src_lengths)
        # self attention requires T x B x C
        x = x.transpose(1, 0)

        # encoder layers
        for layer in self.layers:
            x = layer(x, padding_mask)

        # shapes are: T x B x C, B x T
        return {
            'context_out': x,
            'context_padding_mask': padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['context_out'] is not None:
            encoder_out['context_out'] = \
                encoder_out['context_out'].index_select(1, new_order)
        if encoder_out['context_padding_mask'] is not None:
            encoder_out['context_padding_mask'] = \
                encoder_out['context_padding_mask'].index_select(0, new_order)
        return encoder_out


class ConvolutionalTransformerContextAwareEncoder(ConvolutionalTransformerEncoder):
    """Transformer encoder."""

    def __init__(self, args, dictionary, audio_features=40):
        self.n_encoder_layers = args.encoder_layers
        args.encoder_layers = 0  # To avoid generating useless layers overridden later
        super().__init__(args, dictionary, audio_features)
        args.encoder_layers = self.n_encoder_layers
        self.layers = nn.ModuleList([
            ConvTransformerContextAwareEncoderLayer(args) for _ in range(self.n_encoder_layers)
        ])

    def forward(
            self,
            src_tokens,
            src_lengths,
            context_out,
            cls_input: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
    ):
        x = src_tokens.unsqueeze(1)
        for i, conv in enumerate(self.convolutions):
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = self.bn[i](self.activation_fn(x))
            src_lengths = torch.ceil(src_lengths.float() / 2).long()
            x = F.dropout(x, p=max(self.dropout, .1), training=self.training)

        if hasattr(self, 'attn_2d'):
            residual = x
            x, _ = self.attn_2d[0](query=x, key=x, value=x)
            x = x + residual
            residual = x
            x, _ = self.attn_2d[1](query=x, key=x, value=x)
            x = x + residual

        # B x Cout x T x F -> T x B x C
        bsz, out_channels, time, feats = x.size()
        x = x.transpose(1, 2).contiguous().view(bsz, time, -1).contiguous().transpose(0, 1)
        x = self.activation_fn(self.fc3(x))

        x = x + self.embed_positions(x.transpose(0, 1), src_lengths).transpose(0, 1)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        encoder_padding_mask = self.create_mask(src_lengths)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(
                    x, encoder_padding_mask,
                    context=context_out['context_out'],
                    context_padding_mask=context_out['context_padding_mask'])
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=None,
            encoder_states=encoder_states,  # List[T x B x C]
        )


class TransformerContextAwareDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerContextAwareDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.n_decoder_layers = args.decoder_layers
        args.decoder_layers = 0  # To avoid generating useless layers overridden later
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        args.decoder_layers = self.n_decoder_layers
        self.layers = nn.ModuleList([
            TransformerContextAwareDecoderLayer(args) for _ in range(self.n_decoder_layers)
        ])

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        context_out: Optional[Dict[str, torch.Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            context_out (optional): output from the context encoder, used
                for context attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            context_out=context_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        context_out: Optional[Dict[str, torch.Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[torch.Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[torch.Tensor] = None
        inner_states: List[Optional[torch.Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state: Optional[torch.Tensor] = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_states = encoder_out.encoder_states
                    assert encoder_states is not None
                    encoder_state = encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn, _ = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask
                    if encoder_out is not None
                    else None,
                    incremental_state,
                    context=context_out['context_out'],
                    context_padding_mask=context_out['context_padding_mask'],
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('conv_transformer_context', 'conv_transformer_context')
def base_architecture(args):
    conv_transformer.base_architecture(args)


@register_model_architecture('conv_transformer_context', 'conv_transformer_context_big')
def speechtransformer_big(args):
    conv_transformer.speechtransformer_big(args)


@register_model_architecture('conv_transformer_context', 'conv_transformer_context_big2')
def speechtransformer_big2(args):
    conv_transformer.speechtransformer_big2(args)


@register_model_architecture('conv_transformer_context', 'conv_transformer_context_giant')
def speechtransformer_giant(args):
    conv_transformer.speechtransformer_giant(args)
