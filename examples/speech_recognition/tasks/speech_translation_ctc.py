# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from examples.speech_recognition.data.transcription_dataset import TranscriptionWrapperDataset
from examples.speech_recognition.tasks.speech_recognition import SpeechRecognitionTask
from fairseq.data import Dictionary, data_utils
from fairseq.tasks import register_task


@register_task("speech_translation_with_transcription")
class SpeechTranslationCTCTask(SpeechRecognitionTask):
    """
    Task for training speech recognition model.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechRecognitionTask.add_args(parser)
        parser.add_argument('-s', '--source-lang', default=None, metavar='TARGET',
                            help='source language')

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        # It is set in the setup task after the constructor
        self.src_dict = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        task = super(SpeechTranslationCTCTask, cls).setup_task(args)
        source_dict_path = os.path.join(args.data, "dict.{}.txt".format(args.source_lang))
        if not os.path.isfile(source_dict_path):
            raise FileNotFoundError("Dict not found: {}".format(source_dict_path))
        src_dict = Dictionary.load(source_dict_path)
        if args.criterion == "ctc_multi_loss":
            src_dict.add_symbol("<ctc_blank>")
        print("| CTC dictionary: {} types".format(len(src_dict)))
        task.src_dict = src_dict
        return task

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.args.dataset_from_json:
            # TODO: not implemented yet
            raise NotImplementedError
        else:
            super().load_dataset(split, combine=combine, **kwargs)
            transcr_dataset = data_utils.load_indexed_dataset(
                os.path.join(self.args.data, split) + "." + self.args.source_lang,
                self.src_dict,
                self.args.dataset_impl)
        self.datasets[split] = TranscriptionWrapperDataset(self.datasets[split], transcr_dataset, self.src_dict)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` for the transcription."""
        return self.src_dict
