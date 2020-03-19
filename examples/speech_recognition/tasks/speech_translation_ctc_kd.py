import numpy as np
import os

from examples.speech_recognition.tasks.speech_translation_ctc import SpeechTranslationCTCTask
from fairseq.data import IndexedDataset, ConcatDataset
from fairseq.data.knowledge_distillation import TeacherOutputDataset, DatasetWithTeacherOutput
from fairseq.tasks import register_task


@register_task("speech_translation_with_transcription_kd")
class SpeechTranslationCTCWithKDTask(SpeechTranslationCTCTask):
    """
    Task for training end-to-end speech translation model. It adds the transcription,
    which can be used in the training and the data for applying knowledge distillation
    from a teacher model, whose outputs are written using the generate_topk.py script.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechTranslationCTCTask.add_args(parser)
        parser.add_argument('--distill-topk', default=None, type=int, required=True, metavar='K',
                            help='source language')

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

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
            teacher_probs_datasets = []
            teacher_idxs_datasets = []
            for path in self.paths:
                prefix = os.path.join(path, split) + "." + self.args.target_lang
                teacher_idxs_fname = prefix + '.top{}_idx'.format(self.args.distill_topk)
                teacher_probs_fname = prefix + '.top{}_out'.format(self.args.distill_topk)
                if IndexedDataset.exists(teacher_idxs_fname) and IndexedDataset.exists(teacher_probs_fname):
                    teacher_probs_datasets.append(TeacherOutputDataset(teacher_probs_fname, np.float32))
                    teacher_idxs_datasets.append(TeacherOutputDataset(teacher_idxs_fname, np.int32))

            assert len(teacher_idxs_datasets) > 0
            assert len(teacher_probs_datasets) > 0
            if len(teacher_idxs_datasets) > 1:
                teacher_idxs_dataset = ConcatDataset(teacher_idxs_datasets)
                teacher_probs_dataset = ConcatDataset(teacher_probs_datasets)
            else:
                teacher_idxs_dataset = teacher_idxs_datasets[0]
                teacher_probs_dataset = teacher_probs_datasets[0]
        assert len(self.datasets[split]) == len(teacher_idxs_dataset)
        assert len(teacher_probs_dataset) == len(teacher_idxs_dataset)
        self.datasets[split] = DatasetWithTeacherOutput(
            self.datasets[split], teacher_probs_dataset, teacher_idxs_dataset, self.tgt_dict, self.args.distill_topk)
