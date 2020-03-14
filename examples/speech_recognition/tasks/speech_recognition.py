# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re

from examples.speech_recognition.data.fbank_dataset import FilterBanksDataset, FilterBankToTextDataset
from fairseq.data import Dictionary, data_utils, ConcatDataset
from fairseq.tasks import FairseqTask, register_task
from examples.speech_recognition.data import AsrDataset
from examples.speech_recognition.data.replabels import replabel_symbol
from examples.speech_recognition.modules.specaugment import SpecAugment
from examples.speech_recognition.modules.time_stretch import TimeStretch


logger = logging.getLogger(__name__)


def get_asr_dataset_from_json(data_json_path, tgt_dict, skip_norm):
    """
    Parse data json and create dataset.
    See scripts/asr_prep_json.py which pack json from raw files

    Json example:
    {
    "utts": {
        "4771-29403-0025": {
            "input": {
                "length_ms": 170,
                "path": "/tmp/file1.flac"
            },
            "output": {
                "text": "HELLO \n",
                "token": "HE LLO",
                "tokenid": "4815, 861"
            }
        },
        "1564-142299-0096": {
            ...
        }
    }
    """
    if not os.path.isfile(data_json_path):
        raise FileNotFoundError("Dataset not found: {}".format(data_json_path))
    with open(data_json_path, "rb") as f:
        data_samples = json.load(f)["utts"]
        assert len(data_samples) != 0
        sorted_samples = sorted(
            data_samples.items(),
            key=lambda sample: int(sample[1]["input"]["length_ms"]),
            reverse=True,
        )
        aud_paths = [s[1]["input"]["path"] for s in sorted_samples]
        ids = [s[0] for s in sorted_samples]
        speakers = []
        for s in sorted_samples:
            m = re.search("(.+?)-(.+?)-(.+?)", s[0])
            speakers.append(m.group(1) + "_" + m.group(2))
        frame_sizes = [s[1]["input"]["length_ms"] for s in sorted_samples]
        tgt = [
            [int(i) for i in s[1]["output"]["tokenid"].split(", ")]
            for s in sorted_samples
        ]
        # append eos
        tgt = [[*t, tgt_dict.eos()] for t in tgt]
        return AsrDataset(aud_paths, frame_sizes, tgt, tgt_dict, ids, speakers, skip_normalization=skip_norm)


def get_datasets_from_indexed_filterbanks(
        data_path, tgt_lang, tgt_dict, split, dataset_impl, skip_norm, legacy_audio_fix_lua_indexing):
    """
    Creates a dataset reading precomputed filterbanks adn the corresponding target saved as indexed datasets.
    """
    assert tgt_lang is not None
    prefix = os.path.join(data_path, split)

    src_dataset = FilterBanksDataset(prefix + ".npz", dataset_impl == "cached", legacy_audio_fix_lua_indexing)
    tgt_dataset = data_utils.load_indexed_dataset(prefix + "." + tgt_lang, tgt_dict, dataset_impl)
    return FilterBankToTextDataset(src_dataset, tgt_dataset, tgt_dict, skip_normalization=skip_norm)


@register_task("speech_recognition")
class SpeechRecognitionTask(FairseqTask):
    """
    Task for training speech recognition model.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory. For multiple directories, "
                                         "use column to concatenate them.")
        parser.add_argument(
            "--silence-token", default="\u2581", help="token for silence (used by w2l)"
        )
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument(
            "--dataset-from-json", default=False,
            help="whether to read the data from a JSON file or from indexed data containing "
                 "the precomputed filterbanks")
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
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
        parser.add_argument('--time-stretch-w', type=int, default=1, help='Window size for time stretch')
        parser.add_argument('--time-stretch-low', type=float, default=0.8, help='Low side of the stretch range')
        parser.add_argument('--time-stretch-high', type=float, default=1.25, help='High side of the stretch range')

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
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
            self.time_stretch = TimeStretch(args.time_stretch_w, args.time_stretch_low, args.time_stretch_high)
        else:
            self.time_stretch = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        if args.target_lang is None:
            dict_basename = "dict.txt"
        else:
            dict_basename = "dict.{}.txt".format(args.target_lang)
        dict_path = os.path.join(args.data.split(os.pathsep)[0], dict_basename)
        if not os.path.isfile(dict_path):
            raise FileNotFoundError("Dict not found: {}".format(dict_path))
        tgt_dict = Dictionary.load(dict_path)

        if args.criterion == "ctc_loss":
            tgt_dict.add_symbol("<ctc_blank>")
        elif args.criterion == "asg_loss":
            for i in range(1, args.max_replabel + 1):
                tgt_dict.add_symbol(replabel_symbol(i))

        print("| dictionary: {} types".format(len(tgt_dict)))
        return cls(args, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        datasets = []
        for path in self.paths:
            try:
                if self.args.dataset_from_json:
                    data_json_path = os.path.join(path, "{}.json".format(split))
                    ds = get_asr_dataset_from_json(data_json_path, self.tgt_dict, self.args.skip_normalization)
                else:
                    ds = get_datasets_from_indexed_filterbanks(
                        path,
                        self.args.target_lang,
                        self.tgt_dict,
                        split,
                        self.args.dataset_impl,
                        self.args.skip_normalization,
                        self.args.legacy_audio_fix_lua_indexing)
                datasets.append(ds)
            except Exception:
                logger.warning("Split {} not found in {}. Skipping...".format(split, path))
        assert len(datasets) > 0
        if len(datasets) > 1:
            self.datasets[split] = ConcatDataset(datasets)
        else:
            self.datasets[split] = datasets[0]

    def build_generator(self, args):
        w2l_decoder = getattr(args, "w2l_decoder", None)
        if w2l_decoder == "viterbi":
            from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder

            return W2lViterbiDecoder(args, self.target_dictionary)
        elif w2l_decoder == "kenlm":
            from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder

            return W2lKenLMDecoder(args, self.target_dictionary)
        else:
            return super().build_generator(args)

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.tgt_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None

    def max_positions(self):
        """Return the max speech and sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        if self.time_stretch is not None:
            sample = self.time_stretch(sample)
        if self.specaugment is not None:
            sample = self.specaugment(sample)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output
