import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from fairseq import utils


class ConvAttention2D(nn.Module):
    """
    A multihead attention module which uses convolutions instead of linear transformations.
    Please see Di Gangi et al., Enhancing Transformer for End-to-end Speech-to-Text Translation
    (https://www.aclweb.org/anthology/W19-6603.pdf).
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim  # // num_heads
        # assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self._mask = None

        self.in_proj_weight = Parameter(torch.Tensor(3 * num_heads, embed_dim, 3, 3))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * num_heads))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Conv2d(2 * self.num_heads, embed_dim, 3, padding=1, bias=bias)
        self.bn_q = BatchNorm(self.num_heads)
        self.bn_k = BatchNorm(self.num_heads)
        self.bn_v = BatchNorm(self.num_heads)
        self.bn_out = BatchNorm(embed_dim)
        self.relu = F.relu

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, mask_future_timesteps=False,
                key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Future timesteps can be masked with the
        `mask_future_timesteps` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        bsz, channels, tgt_len, embed_dim = query.size()
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                # this will allow us to concat it with previous value and get
                # just get the previous value
                k = v = q.new(0)
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        src_len = k.size(2)
        freq_len = k.size(3)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        q = self.relu(self.bn_q(q.contiguous()).view(bsz * self.num_heads, tgt_len, freq_len))
        k = self.relu(self.bn_k(k.contiguous()).view(bsz * self.num_heads, src_len, freq_len))
        v = self.relu(self.bn_v(v.contiguous()).view(bsz * self.num_heads, src_len, freq_len))

        attn_weights_t = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights_t.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # only apply masking at training time (when incremental state is None)
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights_t = attn_weights_t.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights_t = attn_weights_t.float().masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            ).type_as(attn_weights_t)  # FP16 support: cast to float and back
            attn_weights_t = attn_weights_t.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights_t = F.softmax(attn_weights_t.float(), dim=-1).type_as(attn_weights_t)
        attn_weights_t = F.dropout(attn_weights_t, p=self.dropout, training=self.training)

        attn_t = torch.bmm(attn_weights_t, v)
        assert list(attn_t.size()) == [bsz * self.num_heads, tgt_len, freq_len]
        q_f = q.transpose(2, 1)
        v_f = v.transpose(2, 1)

        attn_weights_f = torch.bmm(q_f, k)
        assert list(attn_weights_f.size()) == [bsz * self.num_heads, freq_len, freq_len]

        # only apply masking at training time (when incremental state is None)
        attn_weights_f = F.softmax(attn_weights_f.float(), dim=-1).type_as(attn_weights_f)
        attn_weights_f = F.dropout(attn_weights_f, p=self.dropout, training=self.training)

        attn_f = torch.bmm(attn_weights_f, v_f)
        assert list(attn_f.size()) == [bsz * self.num_heads, freq_len, tgt_len]
        attn_t = attn_t.view(bsz, self.num_heads, tgt_len, freq_len).contiguous()
        attn_f = attn_f.transpose(1, 2).view(bsz, self.num_heads, tgt_len, freq_len).contiguous()
        attn = torch.cat([attn_t, attn_f], dim=1).contiguous()
        attn = self.relu(self.bn_out(self.out_proj(attn)))

        if need_weights:
            # average attention weights over heads
            attn_weights_t = attn_weights_t.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights_t = attn_weights_t.sum(dim=1) / self.num_heads
        else:
            attn_weights_t = None

        return attn, attn_weights_t

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=1).chunk(2, dim=1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=1)

    def in_proj_k(self, key):
        return self._in_proj(key, start=1, end=2)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2)

    def _in_proj(self, input, start=None, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        if end is not None:
            weight = weight[:end, :, :, :]
            if bias is not None:
                bias = bias[:end]
        if start is not None:
            weight = weight[start:, :, :, :]
            if bias is not None:
                bias = bias[start:]
        return F.conv2d(input, weight, bias=bias, padding=1)

    def buffered_mask(self, tensor):
        dim = tensor.size(-1)
        if self._mask is None:
            self._mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._mask.size(0) < dim:
            self._mask = torch.triu(utils.fill_with_neg_inf(self._mask.resize_(dim, dim)), 1)
        return self._mask[:dim, :dim]

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(1, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )


def BatchNorm(embedding_dim):
    m = nn.BatchNorm2d(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m
