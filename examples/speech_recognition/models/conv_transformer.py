# Copyright (c) 2017-present, FBK.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
import logging
import math
from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, NamedTuple

from examples.speech_recognition.modules.conv_attention_2d import ConvAttention2D
from examples.speech_recognition.modules.conv_transformer_layer import ConvTransformerEncoderLayer
from examples.speech_recognition.modules.positional_embedding_audio import PositionalEmbeddingAudio
from fairseq import utils
from fairseq.models import register_model, FairseqEncoderDecoderModel, fconv, register_model_architecture, FairseqEncoder
from fairseq.models.transformer import TransformerDecoder, EncoderOut, TransformerModel

logger = logging.getLogger(__name__)


CTCAwareEncoderOut = NamedTuple(
    'CTCAwareEncoderOut',
    list(EncoderOut._field_types.items()) + [
        ("ctc_out", torch.Tensor),
        ("ctc_padding_mask", torch.Tensor)],)


@register_model('conv_transformer')
class ConvolutionalTransformerModel(FairseqEncoderDecoderModel):
    """
    A transformer model whose encoder has convolutions and convolutional 2D attention layers
    before the self-attention layers.
    Please see Di Gangi et al., Enhancing Transformer for End-to-end Speech-to-Text Translation
    (https://www.aclweb.org/anthology/W19-6603.pdf).
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        TransformerModel.add_args(parser)
        parser.add_argument('--encoder-convolutions', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--normalization-constant', type=float, default=1.0)
        parser.add_argument('--no-attn-2d', action='store_true', default=False,
                            help="Whether to use 2d attention")
        parser.add_argument('--distance-penalty', type=str, default=False,
                            choices=['log', 'gauss'],
                            help='Add distance penalty to the encoder')
        parser.add_argument('--init-variance', type=float, default=1.0,
                            help='Initialization value for variance')
        parser.add_argument('--ctc-compress-out',  action='store_true', default=False,
                            help="If set, compress the CTC output based on predictions")
        parser.add_argument('--freeze-pretrained', action='store_true',
                            help='if set, all params loaded from the pretrained model are freezed')

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
        encoder = ConvolutionalTransformerEncoder(
            args, src_dict if src_dict is not None else tgt_dict, audio_features=args.input_feat_per_channel)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return ConvolutionalTransformerModel(encoder, decoder)

    def raw_state_dict_upgrade(self, state_dict):
        if self.encoder.ctc_compress_out and "encoder.ctc_fc.weight" not in state_dict["model"]:
            if "ctc_aware_model.fc_out.weight" in state_dict["criterion"]:
                state_dict["model"]["encoder.ctc_fc.weight"] = \
                    state_dict["criterion"]["ctc_aware_model.fc_out.weight"]
                state_dict["model"]["encoder.ctc_fc.bias"] = \
                    state_dict["criterion"]["ctc_aware_model.fc_out.bias"]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, args=None):
        loaded_model = super().load_state_dict(state_dict, strict)
        if getattr(args, "freeze_pretrained", False):
            logger.info("Freezing pretrained weights")
            for p_name, p_val in loaded_model.named_parameters():
                if p_name in state_dict["model"]:
                    p_val.requires_grad = False
        return loaded_model


class ConvolutionalTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of specified convolution layers, 2D attention layers and
    *args.encoder_layers* transformer layers.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """
    def __init__(self, args, dictionary, audio_features=40):
        super().__init__(dictionary)
        convolutions = eval(args.encoder_convolutions) if args.encoder_convolutions is not None else ((512, 3),) * 2
        stride = 2
        self.dropout = args.dropout

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )

        convolutions = fconv.extend_conv_spec(convolutions)
        self.convolutions = nn.ModuleList()
        in_channels = 1
        for i, (out_channels, kernel_size, kernel_width) in enumerate(convolutions):
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(Conv2D(
                in_channels, out_channels, kernel_size, dropout=self.dropout, padding=padding, stride=stride))
            in_channels = out_channels
        if args.attn_2d:
            self.attn_2d = nn.ModuleList(
                [ConvAttention2D(out_channels, 4, dropout=self.dropout) for _ in range(2)])
        self.bn = nn.ModuleList([BatchNorm(out_channels) for _ in range(len(convolutions))])

        if args.distance_penalty == True:
            args.distance_penalty = 'log'

        flat_dim = audio_features
        for _ in range(len(self.convolutions)):
            flat_dim = math.ceil(flat_dim / stride)
        flat_dim *= out_channels
        self.fc3 = Linear(flat_dim, args.encoder_embed_dim)
        self.embed_positions = PositionalEmbeddingAudio(
            args.max_source_positions, args.encoder_embed_dim, 0, learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None
        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)
        self.encoder_layerdrop = args.encoder_layerdrop

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [ConvTransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(args.encoder_embed_dim)
        else:
            self.layernorm_embedding = None
        self.ctc_compress_out = args.ctc_compress_out
        if self.ctc_compress_out:
            self.ctc_fc = nn.Linear(args.encoder_embed_dim, len(dictionary))
            assert args.criterion == "ctc_multi_loss"
            self.ctc_layer = args.ctc_encoder_layer

    def forward(
        self,
        src_tokens,
        src_lengths,
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
        for l_idx, layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if self.ctc_compress_out and self.ctc_layer == l_idx + 1:
                    ctc_padding_mask = encoder_padding_mask
                    x_ctc, x, src_lengths = self.average_same_ctc_features(x, src_lengths)
                    encoder_padding_mask = self.create_mask(src_lengths)

                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x
        if self.ctc_compress_out:
            return CTCAwareEncoderOut(
                encoder_out=x,  # T x B x C
                encoder_padding_mask=encoder_padding_mask,  # B x T
                encoder_embedding=None,
                encoder_states=encoder_states,  # List[T x B x C]
                ctc_out=x_ctc,  # T x B x D
                ctc_padding_mask=ctc_padding_mask
            )
        else:
            return EncoderOut(
                encoder_out=x,  # T x B x C
                encoder_padding_mask=encoder_padding_mask,  # B x T
                encoder_embedding=None,
                encoder_states=encoder_states,  # List[T x B x C]
            )

    def average_same_ctc_features(self, x, src_lengths):
        x_ctc = self.ctc_fc(x)
        with torch.no_grad():
            batch_predicted = []
            prob_ctc = F.softmax(x_ctc, dim=-1).transpose(0, 1)  # from T x B x D to B x T x D
            for b in range(prob_ctc.shape[0]):
                predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()
                batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])

            new_lengths = [len(p) for p in batch_predicted]
            new_maxlen = max(new_lengths)
            weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=x.dtype)
            processed_inputs_cnt = 0
            for b_idx, pred in enumerate(batch_predicted):
                for t_idx, same in enumerate(pred):
                    new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                    weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = 1.0 / same[1]
                    processed_inputs_cnt = new_processed_inputs_cnt
            weights_matrix = weights_matrix.to(x.device)
        # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
        compressed_output = x.permute(1, 2, 0).bmm(weights_matrix)  # B x C x T'
        return x_ctc, compressed_output.permute(2, 0, 1), src_lengths.new(new_lengths)

    def create_mask(self, lengths):
        max_len = max(lengths)
        mask = lengths.new_zeros(len(lengths), max_len).bool()
        for i, l in enumerate(lengths):
            mask[i, l:] = 1
        if not mask.any():
            mask = None
        return mask

    @property
    def output_batch_first(self):
        return False

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(
                    0, new_order
                )
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out


def Conv2D(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv2d layer"""
    m = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def BatchNorm(embedding_dim):
    m = nn.BatchNorm2d(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m


@register_model_architecture('conv_transformer', 'conv_transformer')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.attn_2d = not getattr(args, 'no_attn_2d', False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(64, 3, 3)] * 2')
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 768)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.distance_penalty = getattr(args, 'distance_penalty', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', 256)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 768)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)


@register_model_architecture('conv_transformer', 'conv_transformer_big')
def speechtransformer_fbk(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.attn_2d = not getattr(args, 'no_attn_2d', False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(64, 3, 3)] * 2')
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.distance_penalty = getattr(args, 'distance_penalty', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)


@register_model_architecture('conv_transformer', 'conv_transformer_big2')
def speechtransformer_fbk(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.attn_2d = not getattr(args, 'no_attn_2d', False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(64, 3, 3)] * 2')
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.distance_penalty = getattr(args, 'distance_penalty', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)


@register_model_architecture('conv_transformer', 'conv_transformer_giant')
def speechtransformer_fbk(args):
    args.dropout = getattr(args, 'dropout', 0.3)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.attn_2d = not getattr(args, 'no_attn_2d', False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(128, 3, 3)] * 2')
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.distance_penalty = getattr(args, 'distance_penalty', False)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 1024)
    args.decoder_output_dim = getattr(args, 'decoder_output_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
