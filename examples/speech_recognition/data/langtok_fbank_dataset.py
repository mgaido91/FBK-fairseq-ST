from fairseq.data import FairseqDataset


class LangtokFilterBanksToTextDataset(FairseqDataset):
    def __init__(self, wrapped_ds, lang_for_token=None, tgt_bos=None, tgt_langtok=None):
        super().__init__()
        self.wrapped_ds = wrapped_ds
        self.lang_for_token = lang_for_token
        self.tgt_bos = tgt_bos
        self.tgt_langtok = tgt_langtok

    def __getitem__(self, index):
        return self.wrapped_ds[index]

    def __len__(self):
        return len(self.wrapped_ds)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        batch = self.tgt_dataset.collater(samples)
        # In case of an empty batch, return an empty dict
        if len(batch) == 0:
            return {}
        batch['net_input']['langtok'] = self.lang_for_token

        if self.tgt_langtok is not None and 'prev_output_tokens' in samples['net_input']:
            assert (samples['net_input']['prev_output_tokens'][:, 0] != self.tgt_bos).sum() == 0
            samples['net_input']['prev_output_tokens'][:, 0] = self.tgt_langtok
        return batch

    def num_tokens(self, index):
        return self.wrapped_ds.num_tokens(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.wrapped_ds.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return self.wrapped_ds.ordered_indices()

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return getattr(self.wrapped_ds, 'supports_prefetch', False)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        self.wrapped_ds.prefetch(indices)
