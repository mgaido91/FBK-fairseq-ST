# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import logging
import os

from examples.speech_recognition.data.langtok_fbank_dataset import LangtokFilterBanksToTextDataset
from examples.speech_recognition.data.transcription_dataset import TranscriptionWrapperDataset
from examples.speech_recognition.modules.specaugment import SpecAugment
from examples.speech_recognition.modules.time_stretch import TimeStretch
from examples.speech_recognition.tasks.speech_recognition import get_datasets_from_indexed_filterbanks
from fairseq.data import (
    RoundRobinZipDatasets,
    data_utils, ConcatDataset)
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask


logger = logging.getLogger(__name__)


@register_task('multilingual_speech_translation_with_transcr')
class MultilingualSpeechTranslationWithTranscriptionTask(MultilingualTranslationTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, which indicates the inference langauge direction.
    `--lang-pairs`, `--encoder-langtok`, `--decoder-langtok` have to be set to
    the same value as training.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        MultilingualTranslationTask.add_args(parser)
        # Speech related args:
        parser.add_argument(
            "--dataset-from-json", default=False,
            help="whether to read the data from a JSON file or from indexed data containing "
                 "the precomputed filterbanks")
        parser.add_argument('--skip-normalization', action='store_true', default=False,
                            help='if set, the input filterbanks are not normalized')
        parser.add_argument('--legacy-audio-fix-lua-indexing', action='store_true', default=False,
                            help='if set, the input filterbanks are subtracted 1 to remove +1 for lua indexing')
        parser.add_argument('--specaugment', action='store_true', default=False)
        parser.add_argument('--frequency-masking-pars', type=int, default=13,
                            help="Maximum number of frequencies that can be masked")
        parser.add_argument('--time-masking-pars', type=int, default=13,
                            help="Maximum number of time steps that can be masked")
        parser.add_argument('--frequency-masking-num', type=int, default=2,
                            help="Number of masks to apply along the frequency dimension")
        parser.add_argument('--time-masking-num', type=int, default=2,
                            help="Number of masks to apply along the time dimension")
        parser.add_argument('--specaugment-rate', type=float, default=1.0,
                            help="Probability to apply specaugment to a spectrogram")
        parser.add_argument('--time-stretch', action='store_true',
                            help="If set, activates time stretch on spectrograms")
        parser.add_argument('--time-stretch-rate', type=float, default=1.0,
                            help='Probability to apply time stretch to a spectrogram')
        parser.add_argument('--time-stretch-w', type=int, default=1, help='Window size for time stretch')
        parser.add_argument('--time-stretch-low', type=float, default=0.8, help='Low side of the stretch range')
        parser.add_argument('--time-stretch-high', type=float, default=1.25, help='High side of the stretch range')
        # End of speech args
        parser.add_argument('--langtok-merge-strategy', default='concat', type=str, choices=['concat', 'sum'],
                            metavar='MRG', help='strategy to use when merging the language token with the input, '
                                                'it can be \'sum\' or \'concat\'')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        if self.args.langtok_merge_strategy == 'sum' and self.args.decoder_langtok:
            raise ValueError('Merge strategy \'sum\' is not valid for decoder language token.')
        self.paths = args.data.split(os.pathsep)
        specaugment = getattr(args, 'specaugment', False)
        if specaugment:
            self.specaugment = SpecAugment(frequency_masking_pars=args.frequency_masking_pars,
                                           time_masking_pars=args.time_masking_pars,
                                           frequency_masking_num=args.frequency_masking_num,
                                           time_masking_num=args.time_masking_num,
                                           rate=args.specaugment_rate)
        else:
            self.specaugment = None
        time_stretch = getattr(args, 'time_stretch', False)
        if time_stretch:
            self.time_stretch = TimeStretch(
                args.time_stretch_rate, args.time_stretch_w, args.time_stretch_low, args.time_stretch_high)
        else:
            self.time_stretch = None

    def alter_dataset_langtok(
            self, ds, src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return ds

        encoder_lang_for_token = None
        if self.args.encoder_langtok is not None:
            if self.args.encoder_langtok == 'src':
                encoder_lang_for_token = src_lang
            else:
                encoder_lang_for_token = tgt_lang

        tgt_langtok = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            tgt_langtok = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return LangtokFilterBanksToTextDataset(
            ds,
            lang_for_token=encoder_lang_for_token,
            tgt_bos=tgt_eos,
            tgt_langtok=tgt_langtok)

    def __load_dataset(self, split, lang_pair):
        src, tgt = lang_pair.split('-')
        datasets = []
        transcr_datasets = []
        for path in self.paths:
            try:
                ds = get_datasets_from_indexed_filterbanks(
                    path,
                    tgt,
                    self.dicts[tgt],
                    split,
                    self.args.dataset_impl,
                    self.args.skip_normalization,
                    self.args.legacy_audio_fix_lua_indexing)
                transcr_ds = data_utils.load_indexed_dataset(
                    os.path.join(path, split) + "." + src,
                    self.dicts[src],
                    self.args.dataset_impl)
                assert transcr_ds is not None, "Transcription dataset not found in {}".format(
                    os.path.join(path, split))
                transcr_datasets.append(transcr_ds)
                datasets.append(ds)
            except Exception:
                logger.warning("Split {} not found in {}. Skipping...".format(split, path))
        assert len(datasets) > 0
        assert len(datasets) == len(transcr_datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
            transcr_dataset = ConcatDataset(transcr_datasets)
        else:
            dataset = datasets[0]
            transcr_dataset = transcr_datasets[0]
        return self.alter_dataset_langtok(
            TranscriptionWrapperDataset(dataset, transcr_dataset, self.dicts[src]),
            src_eos=self.dicts[src].eos(),
            src_lang=src,
            tgt_eos=self.dicts[tgt].eos(),
            tgt_lang=tgt)

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a dataset split."""
        if self.args.dataset_from_json:
            raise NotImplementedError
        else:
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([
                    (lang_pair, self.__load_dataset(split, lang_pair))
                    for lang_pair in self.lang_pairs
                ]),
                eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        raise NotImplementedError

    def build_model(self, args):
        assert self.args.langtok_merge_strategy == args.langtok_merge_strategy, \
            '--langtok-merge-strategy should be {}.'.format(args.langtok_merge_strategy)
        return super().build_model(args)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        from collections import defaultdict
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., defaultdict(float)
        for lang_pair in self.model_lang_pairs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue
            current_sample = sample[lang_pair]
            if self.time_stretch is not None:
                current_sample = self.time_stretch(current_sample)
            if self.specaugment is not None:
                current_sample = self.specaugment(current_sample)

            loss, sample_size, logging_output = criterion(model.models[lang_pair], current_sample)
            if ignore_grad:
                loss *= 0
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output
