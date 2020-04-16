import logging
import os

import numpy as np

from examples.speech_recognition.data.context_dataset import ContextAwareDataset
from examples.speech_recognition.data.fbank_dataset import FilterBanksDataset
from examples.speech_recognition.sequence_generator_with_context import TargetContextAwareSequenceGenerator, \
    AudioContextAwareSequenceGenerator
from examples.speech_recognition.tasks.speech_recognition import SpeechRecognitionTask, \
    get_datasets_from_indexed_filterbanks
from fairseq import search
from fairseq.data import ConcatDataset, data_utils, iterators, FairseqDataset
from fairseq.tasks import register_task

logger = logging.getLogger(__name__)


@register_task('speech_recognition_context')
class ContextAwareSpeechRecognitionTask(SpeechRecognitionTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        SpeechRecognitionTask.add_args(parser)
        parser.add_argument('--context-type', default='tgt', choices=['tgt', 'src'],
                            help='if src, the context is considered to be the previous audio, otherwise'
                                 'it is assumed to be the previous text')

    @classmethod
    def setup_task(cls, args, **kwargs):
        task = super(ContextAwareSpeechRecognitionTask, cls).setup_task(args)
        if hasattr(args, 'beam'):  # TODO: find a better way to determine whether it is training or not...
            task.training = False
        return task

    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.training = True  # It may be set to False in setup_task

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if self.args.dataset_from_json:
            raise NotImplementedError
        datasets = []
        for path in self.paths:
            try:
                ds = get_datasets_from_indexed_filterbanks(
                    path,
                    self.args.target_lang,
                    self.tgt_dict,
                    split,
                    self.args.dataset_impl,
                    self.args.skip_normalization,
                    self.args.legacy_audio_fix_lua_indexing)
                if self.training:
                    if self.args.context_type == 'src':
                        context_ds = FilterBanksDataset(
                            os.path.join(path, split) + ".context.npz",
                            self.args.dataset_impl == "cached",
                            self.args.legacy_audio_fix_lua_indexing)
                    else:
                        context_ds = data_utils.load_indexed_dataset(
                            os.path.join(path, split) + ".context." + self.args.target_lang,
                            self.tgt_dict,
                            self.args.dataset_impl)
                    datasets.append(ContextAwareDataset(
                        ds, context_ds, self.tgt_dict, self.args.context_type == 'src'))
                else:
                    datasets.append(ds)
            except Exception:
                logger.warning("Split {} not found in {}. Skipping...".format(split, path))
        assert len(datasets) > 0
        if len(datasets) > 1:
            self.datasets[split] = ConcatDataset(datasets)
        else:
            self.datasets[split] = datasets[0]

    def build_generator(self, args):
        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, 'sampling', False)
        sampling_topk = getattr(args, 'sampling_topk', -1)
        sampling_topp = getattr(args, 'sampling_topp', -1.0)
        diverse_beam_groups = getattr(args, 'diverse_beam_groups', -1)
        diverse_beam_strength = getattr(args, 'diverse_beam_strength', 0.5),
        match_source_len = getattr(args, 'match_source_len', False)
        diversity_rate = getattr(args, 'diversity_rate', -1)
        if (
                sum(
                    int(cond)
                    for cond in [
                        sampling,
                        diverse_beam_groups > 0,
                        match_source_len,
                        diversity_rate > 0,
                    ]
                )
                > 1
        ):
            raise ValueError('Provided Search parameters are mutually exclusive.')
        assert sampling_topk < 0 or sampling, '--sampling-topk requires --sampling'
        assert sampling_topp < 0 or sampling, '--sampling-topp requires --sampling'

        if sampling:
            search_strategy = search.Sampling(self.target_dictionary, sampling_topk, sampling_topp)
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength)
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary, min_len_a=1, min_len_b=0, max_len_a=1, max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(self.target_dictionary, diversity_rate)
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)
        if args.context_type == 'src':
            seq_cls = AudioContextAwareSequenceGenerator
        else:
            seq_cls = TargetContextAwareSequenceGenerator
        return seq_cls(
            self.target_dictionary,
            beam_size=getattr(args, 'beam', 5),
            max_len_a=getattr(args, 'max_len_a', 0),
            max_len_b=getattr(args, 'max_len_b', 200),
            min_len=getattr(args, 'min_len', 1),
            normalize_scores=(not getattr(args, 'unnormalized', False)),
            len_penalty=getattr(args, 'lenpen', 1),
            unk_penalty=getattr(args, 'unkpen', 0),
            temperature=getattr(args, 'temperature', 1.),
            match_source_len=getattr(args, 'match_source_len', False),
            no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            search_strategy=search_strategy,
        )

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        This method differs from the superclass as it ensures that the
        samples are provided in the original order during the generation.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        if self.training:
            with data_utils.numpy_seed(seed):
                indices = dataset.ordered_indices()
        else:
            # Ensure the correct order is kept while generating
            indices = np.arange(len(dataset))

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices, dataset, max_positions, raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter

    def max_positions(self):
        """Return the max speech and sentence length allowed by the task."""
        if self.args.context_type == 'src':
            context_max_positions = self.args.max_source_positions
        else:
            context_max_positions = self.args.max_target_positions
        return (self.args.max_source_positions, self.args.max_target_positions, context_max_positions)
