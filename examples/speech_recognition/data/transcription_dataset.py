import torch

from fairseq.data import FairseqDataset
from fairseq.data import data_utils as fairseq_data_utils


class TranscriptionWrapperDataset(FairseqDataset):
    def __init__(self, tgt_dataset, transcription_dataset, transcription_dict):
        super().__init__()
        self.tgt_dataset = tgt_dataset
        self.transcription_dataset = transcription_dataset
        self.transcription_dict = transcription_dict

    def __getitem__(self, index):
        item = self.tgt_dataset[index]
        item['encoder_target'] = self.transcription_dataset[index]
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
        transcriptions_map = {}
        transcr_lens = []
        for i, s in enumerate(samples):
            transcriptions_map[s['id']] = i
            transcr_lens.append(s['encoder_target'].shape[0])
        batch = self.tgt_dataset.collater(samples)
        sort_order = []
        for s_id in batch['id'].tolist():
            sort_order.append(transcriptions_map[s_id])
        sort_order = torch.tensor(sort_order)
        encoder_target = fairseq_data_utils.collate_tokens(
            [s["encoder_target"] for s in samples],
            self.transcription_dict.pad(),
            self.transcription_dict.eos(),
            left_pad=False,
            move_eos_to_beginning=False,
        )
        encoder_target = encoder_target.index_select(0, sort_order)
        encoder_target_lengths = torch.LongTensor(transcr_lens).index_select(0, sort_order)
        batch['encoder_target'] = encoder_target
        batch['encoder_target_lengths'] = encoder_target_lengths
        return batch

    def num_tokens(self, index):
        return self.tgt_dataset.num_tokens(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self.tgt_dataset.size(index)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        return self.tgt_dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return (hasattr(self.transcription_dataset, 'supports_prefetch') and
                self.transcription_dataset.supports_prefetch) or \
               (hasattr(self.tgt_dataset, 'supports_prefetch') and self.tgt_dataset.supports_prefetch)

    def prefetch(self, indices):
        """Prefetch the data required for this epoch."""
        if self.transcription_dataset.supports_prefetch:
            self.transcription_dataset.prefetch(indices)
        if self.tgt_dataset.supports_prefetch:
            self.tgt_dataset.prefetch(indices)
