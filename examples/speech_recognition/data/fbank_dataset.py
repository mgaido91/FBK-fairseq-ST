# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from functools import lru_cache

import numpy as np
import torch

from fairseq.data import FairseqDataset, IndexedCachedDataset

from . import data_utils
from .collaters import Seq2SeqCollater


class FilterBankToTextDataset(FairseqDataset):
    """
    A dataset representing speech (represented as precomputed filterbanks) and corresponding text.

    Args:
        src_dataset: (~fairseq.data.FairseqDataset): A dataset of FloatTensors containing the precomputed filterbanks
            of the source audio
        tgt_dataset (~fairseq.data.FairseqDataset): A dataset of LongTensors containing the indices
            of target text.
        tgt_dict (~fairseq.data.Dictionary): target vocabulary.
        skip_normalization (bool): whether to normalize (if False) the audio filterbanks for each item or not
    """
    def __init__(self, src_dataset, tgt_dataset, tgt_dict, skip_normalization=False):
        assert len(src_dataset) == len(tgt_dataset)
        self.tgt_dict = tgt_dict
        self.src_dataset = src_dataset
        self.tgt_dataset = tgt_dataset
        self.skip_normalization = skip_normalization

        self.s2s_collater = Seq2SeqCollater(
            0, 1, pad_index=self.tgt_dict.pad(),
            eos_index=self.tgt_dict.eos(), move_eos_to_beginning=True
        )

    def __getitem__(self, index):
        tgt_item = self.tgt_dataset[index] if self.tgt_dataset is not None else None
        src_item = self.src_dataset[index]
        if not self.skip_normalization:
            src_item = data_utils.apply_mv_norm(src_item)

        return {"id": index, "data": [src_item, tgt_item]}

    def __len__(self):
        return len(self.src_dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        return self.s2s_collater.collate(samples)

    def num_tokens(self, index):
        return self.src_dataset.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_dataset.size(index),
            self.tgt_dataset.size(index) if self.tgt_dataset is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return np.arange(len(self))

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return (hasattr(self.src_dataset, 'supports_prefetch') and self.src_dataset.supports_prefetch) or \
               (hasattr(self.tgt_dataset, 'supports_prefetch') and self.tgt_dataset.supports_prefetch)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        if self.src_dataset.supports_prefetch:
            self.src_dataset.prefetch(indices)
        if self.tgt_dataset.supports_prefetch:
            self.tgt_dataset.prefetch(indices)


class FilterBanksDataset(IndexedCachedDataset):
    """Loader for TorchNet dataset containing precomputed filterbanks"""
    _HDR_MAGIC = b'TNTIDX\x00\x00'

    def __init__(self, path, cached=True):
        super().__init__(path)
        self.path = path
        self.cached = cached
        assert self.dtype == np.float32
        assert len(self.sizes) == len(self) * 2

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        if self.cached:
            return self.get_cached(i)
        else:
            return self.get_from_disk(i)

    def get_from_disk(self, i):
        if not self.data_file:
            self.read_data(self.path)
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        self.data_file.seek(self.data_offsets[i] * self.element_size)
        self.data_file.readinto(a)
        return torch.from_numpy(a)

    def get_cached(self, i):
        self.check_index(i)
        tensor_size = self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]]
        a = np.empty(tensor_size, dtype=self.dtype)
        ptx = self.cache_index[i]
        np.copyto(a, self.cache[ptx: ptx + a.size])
        return torch.from_numpy(a)

    def __len__(self):
        return self._len

    def num_tokens(self, index):
        # The filterbanks are array of 2 dimensions, the first being the number of filterbanks
        # collected and the second being the number of features
        return self.sizes[index * 2]

    def size(self, index):
        # The filterbanks are array of 2 dimensions, the first being the number of filterbanks
        # collected and the second being the number of features
        return self.sizes[index * 2]

    @property
    def supports_prefetch(self):
        return self.cached
