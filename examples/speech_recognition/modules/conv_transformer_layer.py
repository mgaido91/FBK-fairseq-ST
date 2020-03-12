import torch
from torch import nn
from torch.nn import Parameter

from examples.speech_recognition.modules.local_attention import LocalAttention
from fairseq.modules import TransformerEncoderLayer


class ConvTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        if args.distance_penalty:
            self.self_attn = LocalAttention(
                self.embed_dim,
                args.encoder_attention_heads,
                dropout=args.attention_dropout,
                self_attention=True,
                penalty=PENALTIES[args.distance_penalty](args)
            )


class LogPenalty(nn.Module):
    def __init__(self, *input):
        super().__init__()

    def forward(self, position_distances, batch_size):
        return torch.max(torch.zeros_like(position_distances), torch.log(position_distances))


class GaussPenalty(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.variance = Parameter(torch.Tensor(self.num_heads).fill_(args.init_variance))

    def forward(self, position_distances, batch_size):
        num = (position_distances * position_distances).unsqueeze(0)
        denom = (2 * self.variance * self.variance).unsqueeze(1).unsqueeze(2).repeat(batch_size, 1, 1)
        return num / denom


PENALTIES = {
    "log": LogPenalty,
    "gauss": GaussPenalty
}
