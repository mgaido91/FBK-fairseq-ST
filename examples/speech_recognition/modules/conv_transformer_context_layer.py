from typing import Optional, Dict, List

import torch
from torch import nn, Tensor
from torch.nn import Linear, functional as F

from examples.speech_recognition.modules.conv_transformer_layer import ConvTransformerEncoderLayer
from fairseq.modules import MultiheadAttention, TransformerDecoderLayer


class ConvTransformerContextAwareEncoderLayer(ConvTransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        self.add_context = args.context_position in ["both", "encoder"]
        if self.add_context:
            self.context_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout,
            )
            self.context_gating_wi = Linear(self.embed_dim, self.embed_dim)
            self.context_gating_ws = Linear(self.embed_dim, self.embed_dim)
            if self.context_attention_type == "sequential":
                self.context_layer_norm = LayerNorm(self.embed_dim)

    def forward(
            self,
            x,
            encoder_padding_mask,
            context,
            context_padding_mask,
            attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            context (Tensor): external context to the layer of shape `(ctx_len, batch, embed_dim)`
            context_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, ctx_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Context attention
        if self.add_context:
            residual = x
            if self.normalize_before:
                x = self.context_layer_norm(x)
            x, _ = self.context_attn(query=x, key=context, value=context, key_padding_mask=context_padding_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            lambda_gating = torch.sigmoid(self.context_gating_wi(residual) + self.context_gating_ws(x))
            x = lambda_gating * residual + (1 - lambda_gating) * x
            if not self.normalize_before:
                x = self.context_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class TransformerContextAwareDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            args, no_encoder_attn=no_encoder_attn, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        self.add_context = args.context_position in ["both", "decoder"]
        self.context_attention_type = args.context_decoder_attention_type
        if self.add_context:
            self.context_attn = MultiheadAttention(
                self.embed_dim, args.decoder_attention_heads,
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.context_gating_wi = Linear(self.embed_dim, self.embed_dim)
            self.context_gating_ws = Linear(self.embed_dim, self.embed_dim)
            self.context_attn_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        context: Optional[torch.Tensor] = None,
        context_padding_mask: Optional[torch.Tensor] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            context (Tensor): external context to the layer of shape `(ctx_len, batch, embed_dim)`
            context_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, ctx_len)` where padding elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            # If parallel context attention is enabled, we need the normalized
            # input for the context cross-attention
            if self.add_context and self.context_attention_type == "parallel":
                query_ctx = x
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        # Context attention
        ctx_gate = None
        if self.add_context:
            if self.context_attention_type == "sequential":
                residual = x
                if self.normalize_before:
                    x = self.context_attn_layer_norm(x)
                x, _ = self.context_attn(
                    query=x, key=context, value=context, key_padding_mask=context_padding_mask,
                    incremental_state=incremental_state, static_kv=True)
                x = F.dropout(x, p=self.dropout, training=self.training)
                lambda_gating = torch.sigmoid(self.context_gating_wi(residual) + self.context_gating_ws(x))
                x = lambda_gating * residual + (1 - lambda_gating) * x
                if not self.normalize_before:
                    x = self.context_attn_layer_norm(x)
            elif self.context_attention_type == "parallel":
                c_x, _ = self.context_attn(
                    query=query_ctx, key=context, value=context, key_padding_mask=context_padding_mask,
                    incremental_state=incremental_state, static_kv=True)
                c_x = F.dropout(c_x, p=self.dropout, training=self.training)
                lambda_gating = torch.sigmoid(self.context_gating_wi(x) + self.context_gating_ws(c_x))
                x = lambda_gating * x + (1 - lambda_gating) * c_x
                ctx_gate = (1 - lambda_gating)
            else:
                raise RuntimeError("Invalid decoder context attention type {}".format(self.context_attention_type))

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state, ctx_gate
        return x, attn, None, ctx_gate


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    nn.init.constant_(m.weight, 1)
    nn.init.constant_(m.bias, 0)
    return m
