# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections import OrderedDict

import numpy as np

from examples.speech_recognition.data.transcription_dataset import TranscriptionWrapperDataset
from examples.speech_recognition.tasks.multilingual_speech_translation import \
    MultilingualSpeechTranslationWithTranscriptionTask
from examples.speech_recognition.tasks.speech_recognition import get_datasets_from_indexed_filterbanks
from fairseq.data import data_utils, ConcatDataset, IndexedDataset, RoundRobinZipDatasets
from fairseq.data.knowledge_distillation import TeacherOutputDataset, DatasetWithTeacherOutput
from fairseq.tasks import register_task


logger = logging.getLogger(__name__)


@register_task('multilingual_speech_translation_with_transcr_kd')
class MultilingualSpeechTranslationWithTranscriptionKDTask(MultilingualSpeechTranslationWithTranscriptionTask):
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
        MultilingualSpeechTranslationWithTranscriptionTask.add_args(parser)
        parser.add_argument('--distill-topk', default=None, type=int, required=True, metavar='K',
                            help='source language')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)

    def __load_dataset(self, split, lang_pair):
        src, tgt = lang_pair.split('-')
        datasets = []
        transcr_datasets = []
        teacher_probs_datasets = []
        teacher_idxs_datasets = []
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
                tgt_prefix = os.path.join(path, split) + "." + tgt
                teacher_idxs_fname = tgt_prefix + '.top{}_idx'.format(self.args.distill_topk)
                teacher_probs_fname = tgt_prefix + '.top{}_out'.format(self.args.distill_topk)
                assert IndexedDataset.exists(teacher_idxs_fname) and IndexedDataset.exists(teacher_probs_fname), \
                    "Teacher datasets not found in {}".format(tgt_prefix)
                teacher_probs_datasets.append(TeacherOutputDataset(teacher_probs_fname, np.float32))
                teacher_idxs_datasets.append(TeacherOutputDataset(teacher_idxs_fname, np.int32))
                assert transcr_ds is not None, "Transcription dataset not found in {}".format(
                    os.path.join(path, split))
                transcr_datasets.append(transcr_ds)
                datasets.append(ds)
            except Exception:
                logger.warning("Split {} not found in {}. Skipping...".format(split, path))
        assert len(datasets) > 0
        assert len(datasets) == len(transcr_datasets)
        assert len(datasets) == len(teacher_probs_datasets)
        assert len(datasets) == len(teacher_idxs_datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
            transcr_dataset = ConcatDataset(transcr_datasets)
            teacher_idxs_dataset = ConcatDataset(teacher_idxs_datasets)
            teacher_probs_dataset = ConcatDataset(teacher_probs_datasets)
        else:
            dataset = datasets[0]
            transcr_dataset = transcr_datasets[0]
            teacher_idxs_dataset = teacher_idxs_datasets[0]
            teacher_probs_dataset = teacher_probs_datasets[0]
        dataset_with_transcr = TranscriptionWrapperDataset(dataset, transcr_dataset, self.dicts[src])
        dataset_with_kd = DatasetWithTeacherOutput(
            dataset_with_transcr, teacher_probs_dataset, teacher_idxs_dataset, self.dicts[tgt], self.args.distill_topk)
        return self.alter_dataset_langtok(
            dataset_with_kd,
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
