# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re

from examples.speech_recognition.data.fbank_dataset import FilterBanksDataset, FilterBankToTextDataset
from fairseq.data import Dictionary, data_utils
from fairseq.tasks import FairseqTask, register_task
from examples.speech_recognition.data import AsrDataset
from examples.speech_recognition.data.replabels import replabel_symbol


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


def get_datasets_from_indexed_filterbanks(data_path, tgt_lang, tgt_dict, split, dataset_impl, skip_norm):
    """
    Creates a dataset reading precomputed filterbanks adn the corresponding target saved as indexed datasets.
    """
    assert tgt_lang is not None
    prefix = os.path.join(data_path, split)

    src_dataset = FilterBanksDataset(prefix + ".npz", dataset_impl == "cached")
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
        parser.add_argument("data", help="path to data directory")
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

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        if args.target_lang is None:
            dict_basename = "dict.txt"
        else:
            dict_basename = "dict.{}.txt".format(args.target_lang)
        dict_path = os.path.join(args.data, dict_basename)
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
        if self.args.dataset_from_json:
            data_json_path = os.path.join(self.args.data, "{}.json".format(split))
            ds = get_asr_dataset_from_json(data_json_path, self.tgt_dict, self.args.skip_normalization)
        else:
            ds = get_datasets_from_indexed_filterbanks(
                self.args.data,
                self.args.target_lang,
                self.tgt_dict,
                split,
                self.args.dataset_impl,
                self.args.skip_normalization)
        self.datasets[split] = ds

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
