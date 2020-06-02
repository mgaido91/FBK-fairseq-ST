# Copyright (c) 2017-present, FBK.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


from collections import OrderedDict
import logging

import torch
from typing import Optional

from torch import nn

from examples.speech_recognition.data.data_utils import lengths_to_encoder_padding_mask
from examples.speech_recognition.models.conv_transformer import ConvolutionalTransformerModel, \
    ConvolutionalTransformerEncoder, base_architecture, speechtransformer_big2, speechtransformer_big
from fairseq import utils, checkpoint_utils
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.models import register_model, register_model_architecture, FairseqMultiModel
from fairseq.models.transformer import TransformerDecoder
from fairseq.tasks.multilingual_translation import _lang_token_index


logger = logging.getLogger(__name__)


@register_model('multilingual_conv_transformer')
class MultilingualConvolutionalTransformerModel(FairseqMultiModel):
    """
    A multilingual transformer model whose encoder has convolutions and convolutional 2D attention layers
    before the self-attention layers.
    Please see Di Gangi et al., Enhancing Transformer for End-to-end Speech-to-Text Translation
    (https://www.aclweb.org/anthology/W19-6603.pdf).
    """
    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 100000
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 100000

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_decoder_embed_tokens = None
        if args.share_decoder_embeddings:
            shared_decoder_embed_tokens = (
                FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embed_dim=args.decoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.decoder_embed_path,
                )
            )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(src_lang, tgt_lang):
            if src_lang not in lang_encoders:
                lang_encoders[src_lang] = TokenWiseConvolutionalTransformerEncoder(
                    args, task.dicts[tgt_lang], audio_features=args.input_feat_per_channel, langs=task.langs)
                if args.pretrained_encoder is not None:
                    checkpoint_utils.load_pretrained_component_from_model(
                        lang_encoders[src_lang], args.pretrained_encoder, args.allow_partial_restore)
            return lang_encoders[src_lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang], args.decoder_embed_dim, args.decoder_embed_path
                    )
                lang_decoders[lang] = TokenWiseTransformerDecoder(
                    args, task.dicts[lang], decoder_embed_tokens, task.args.target_lang)
                if args.pretrained_decoder is not None:
                    decoder_loaded_state = load_checkpoint_to_cpu(args.pretrained_decoder)
                    if args.encoder_langtok is not None or args.decoder_langtok:

                        def resize_model_to_new_dict(weights_tensor):
                            old_shape = weights_tensor.shape
                            new_tensor = weights_tensor.new_empty((old_shape[0] + len(task.langs), old_shape[1]))
                            nn.init.xavier_uniform_(new_tensor, gain=nn.init.calculate_gain('relu'))
                            new_tensor[:old_shape[0], :] = weights_tensor
                            return new_tensor

                        decoder_loaded_state["model"]["decoder.embed_tokens.weight"] = resize_model_to_new_dict(
                            decoder_loaded_state["model"]["decoder.embed_tokens.weight"]
                        )
                        decoder_loaded_state["model"]["decoder.output_projection.weight"] = resize_model_to_new_dict(
                            decoder_loaded_state["model"]["decoder.output_projection.weight"]
                        )
                    new_component_state_dict = OrderedDict()
                    for key in decoder_loaded_state["model"].keys():
                        if key.startswith("decoder"):
                            # decoder.input_layers.0.0.weight --> input_layers.0.0.weight
                            component_subkey = key[8:]
                            new_component_state_dict[component_subkey] = decoder_loaded_state["model"][key]
                    incompatible_keys = lang_decoders[lang].load_state_dict(
                        new_component_state_dict, strict=(not args.allow_partial_restore))
                    if len(incompatible_keys.unexpected_keys) != 0:
                        logger.error("Cannot load the following keys from checkpoint: {}".format(
                            incompatible_keys.unexpected_keys))
                        raise ValueError("Cannot load from checkpoint: {}".format(args.pretrained_decoder))
                    if len(incompatible_keys.missing_keys) > 0:
                        logger.info("Loaded checkpoint misses the parameters: {}".format(
                            incompatible_keys.missing_keys))
            return lang_decoders[lang]

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0], tgt_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = shared_encoder if shared_encoder is not None else get_encoder(src, tgt)
            decoders[lang_pair] = shared_decoder if shared_decoder is not None else get_decoder(tgt)

        return MultilingualConvolutionalTransformerModel(encoders, decoders)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        ConvolutionalTransformerModel.add_args(parser)
        parser.add_argument('--pretrained-encoder', type=str, default=None,
                            help='path to a pretrained encoder')
        parser.add_argument('--pretrained-decoder', type=str, default=None,
                            help='path to a pretrained decoder')
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')

    def load_state_dict(self, state_dict, strict=True, args=None):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith('models.')
            lang_pair = k.split('.')[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        return super().load_state_dict(state_dict_subset, strict=strict, args=args)


class TokenWiseConvolutionalTransformerEncoder(ConvolutionalTransformerEncoder):
    """
    Transformer encoder consisting of specified convolution layers, 2D attention layers and
    *args.encoder_layers* transformer layers.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        audio_features (int): number of features for the input spectrogram
        langs (list[str]): list of the languages
    """
    def __init__(self, args, dictionary, audio_features=40, langs=None):
        super().__init__(args, dictionary, audio_features)
        assert langs is not None
        self.langs = langs
        self.lang_embeddings = Embedding(len(langs), audio_features, None)
        self.langtok_merge_strategy = args.langtok_merge_strategy

    def forward(
        self,
        src_tokens,
        src_lengths,
        langtok,
        cls_input: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        if langtok is not None:
            device = src_tokens.device
            embed = self.lang_embeddings(torch.tensor(self.langs.index(langtok)).to(device))
            if self.langtok_merge_strategy == 'sum':
                padding_mask = lengths_to_encoder_padding_mask(src_lengths, batch_first=True)[0]
                embeddings = padding_mask.logical_not().unsqueeze(-1).float() * embed
                src_tokens = src_tokens + embeddings
            else:
                embeddings = embed.repeat(src_tokens.shape[0], 1).unsqueeze(1)
                src_tokens = torch.cat([embeddings, src_tokens], dim=1)
                src_lengths = src_lengths + 1
        return super().forward(src_tokens, src_lengths, cls_input, return_all_hiddens)


class TokenWiseTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary,  embed_tokens, target_lang, no_encoder_attn=False):
        if args.langtok_merge_strategy == 'sum' and args.decoder_langtok:
            if target_lang is not None:
                target_lang_idx = _lang_token_index(dictionary, target_lang)
            else:
                target_lang_idx = None
            embed_tokens = EmbeddingsWithTokenSum(embed_tokens, dictionary.eos(), lang_token_index=target_lang_idx)
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out = None,
        incremental_state = None,
        features_only = False,
        alignment_layer = None,
        alignment_heads = None,
        src_lengths = None,
        return_all_hiddens = False,
        langtok = None,
    ):
        return super().forward(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class EmbeddingsWithTokenSum(nn.Module):
    def __init__(self, base_embeddings, bos_idx, lang_token_index=None):
        super().__init__()
        self.base_embeddings = base_embeddings
        self.bos_idx = bos_idx
        self.embedding_dim = base_embeddings.embedding_dim
        self.padding_idx = base_embeddings.padding_idx
        self.lang_token_index = lang_token_index

    def forward(self, tokens):
        device = tokens.device
        if self.lang_token_index is None:
            # Training phase
            embeddings = self.base_embeddings(tokens)
            # The first token is the lang token
            lang_embed = embeddings[0][0]
            embeddings[:, 0] = self.base_embeddings(torch.tensor(self.bos_idx).to(device))
            return embeddings + lang_embed
        else:
            # Generation phase
            bos = tokens.eq(self.lang_token_index)
            tokens[bos] = self.bos_idx
            embeddings = self.base_embeddings(tokens)
            return embeddings + self.base_embeddings(torch.tensor(self.lang_token_index).to(device))



@register_model_architecture('multilingual_conv_transformer', 'multilingual_conv_transformer')
def base_multilingual_architecture(args):
    base_architecture(args)


@register_model_architecture('multilingual_conv_transformer', 'multilingual_conv_transformer_big')
def speechtransformer_multilingual_big(args):
    speechtransformer_big(args)


@register_model_architecture('multilingual_conv_transformer', 'multilingual_conv_transformer_big2')
def speechtransformer_multilingual_big2(args):
    speechtransformer_big2(args)
