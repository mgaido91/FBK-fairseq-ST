from torch import nn

from fairseq.modules import LearnedPositionalEmbedding, SinusoidalPositionalEmbedding


class PositionalEmbeddingAudio(nn.Module):
    """This module learns audio positional embeddings up to a fixed maximum size.

    Padding symbols are ignored, but it is necessary to specify whether padding
    is added on the left side (left_pad=True) or right side (left_pad=False).
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx, learned=True):
        super().__init__()
        if learned:
            self.embeddings = LearnedPositionalEmbedding(num_embeddings, embedding_dim, padding_idx)
        else:
            self.embeddings = SinusoidalPositionalEmbedding(int(embedding_dim), padding_idx)
        self.padding_idx = padding_idx

    def forward(self, input, lengths, incremental_state=None):
        """Input is expected to be of size [bsz x seqlen x feature_dim]."""
        max_length = max(lengths)
        pos_tensor = lengths.new(input.size(0), max_length).fill_(self.padding_idx)
        for i, l in enumerate(lengths):
            pos_tensor[i, :l] = self.padding_idx + 1
        return self.embeddings(pos_tensor)

    def max_positions(self):
        """Maximum number of supported positions."""
        return self.embeddings.max_positions()


    @property
    def weight(self):
        return self.embeddings.weight
