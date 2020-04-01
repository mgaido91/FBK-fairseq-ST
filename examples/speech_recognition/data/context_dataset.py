import torch

from fairseq.data import FairseqDataset
from fairseq.data import data_utils as fairseq_data_utils


class ContextAwareDataset(FairseqDataset):
    """
    A dataset which wraps a provided :class:`~fairseq.data.FairseqDataset` adding the context
    which is another :class:`~fairseq.data.FairseqDataset`.

    Args:
        dataset (FairseqDataset): the base dataset to wrap
        context_dataset (FairseqDataset): a dataset which contains the context
        context_dict (Dictionary): the dictionary for the context
    """
    def __init__(self, dataset, context_dataset, context_dict):
        super().__init__()
        self.dataset = dataset
        self.context_dataset = context_dataset
        self.context_dict = context_dict

    def __getitem__(self, index):
        item = self.dataset[index]
        item['context'] = self.context_dataset[index]
        return item

    def __len__(self):
        return len(self.tgt_dataset)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[int]): sample indices to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        batch = self.dataset.collater(samples)
        # In case of an empty batch, return an empty dict
        if len(batch) == 0:
            return {}
        context_map = {}
        context_lens = []
        for i, s in enumerate(samples):
            context_map[s['id']] = i
            context_lens.append(s['context'].shape[0])
        sort_order = []
        for s_id in batch['id'].tolist():
            sort_order.append(context_map[s_id])
        sort_order = torch.tensor(sort_order)
        context = fairseq_data_utils.collate_tokens(
            [s["context"] for s in samples],
            self.context_dict.pad(),
            self.context_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        context = context.index_select(0, sort_order)
        context_lengths = torch.LongTensor(context_lens).index_select(0, sort_order)
        batch['net_input']['context_tokens'] = context
        batch['net_input']['context_lengths'] = context_lengths
        return batch

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.dataset.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return (hasattr(self.context_dataset, 'supports_prefetch') and
                self.context_dataset.supports_prefetch) or \
               (hasattr(self.dataset, 'supports_prefetch') and self.dataset.supports_prefetch)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        if self.context_dataset.supports_prefetch:
            self.context_dataset.prefetch(indices)
        if self.dataset.supports_prefetch:
            self.dataset.prefetch(indices)
