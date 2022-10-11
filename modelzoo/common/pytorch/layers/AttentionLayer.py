# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn

from modelzoo.common.pytorch.model_utils.create_initializer import (
    create_initializer,
)


class MultiheadAttention(nn.Module):
    """Multi-head attention layer. Adapted from:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention

    Args:
        embed_dim (int): Number of input units in each projection output
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate for key-query weights. Defaults to 0.0.
        batch_first (bool): If True, then the input and output tensors are 
            provided as (batch, seq, feature), otherwise the format will be
            (seq, batch, feature). Default: True (batch, seq, feature).
        add_bias_kv (bool): If specified, adds bias to the key and value sequences at dim=0. Default: False.
        add_zero_attn (bool): If specified, adds a new batch of zeros to the key and value 
            sequences at dim=1. Default: False
        kdim (int):  Number of output units in key projection
        vdim (int):  Number of output units in  projection
        use_projection_bias (bool): Whether to use bias in the key, query, and
            value projections.
        use_ffn_bias (bool): Whether to use bias in the output projection.
        attention_initializer (str): Projection kernel initializer. Defaults to
            ``xavier_uniform``.
        output_layer_initializer (str or initializer): If not None, use this
            initializer for the output transform layer. Defaults to None.
        attention_type (str): The attention variant to execute. Currently
            accepts ``dot_product`` and ``scaled_dot_product``. Defaults to
            ``scaled_dot_product``.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        batch_first=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        use_projection_bias=None,
        use_ffn_bias=False,
        attention_initializer="xavier_uniform",
        output_layer_initializer=None,
        attention_type="scaled_dot_product",
        device=None,
    ):
        _SUPPORTED_ATTENTION_TYPES = ["dot_product", "scaled_dot_product"]
        assert (
            attention_type in _SUPPORTED_ATTENTION_TYPES
        ), f"Attention type {attention_type} is not supported."
        assert (
            embed_dim % num_heads == 0
        ), "embed_dim must be divisible by num_heads."

        assert batch_first, "Currently, only batch_first=True is supported"
        assert not add_bias_kv, "add_bias_kv=True is not supported."
        assert not add_zero_attn, "add_zero_attn=True is not supported."
        assert kdim is None, "kdim should be set to None."
        assert vdim is None, "vdim should be set to None."
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.scale_dot_product = attention_type == "scaled_dot_product"

        self.proj_q_dense_layer = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_k_dense_layer = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=use_projection_bias,
            device=device,
        )
        self.proj_v_dense_layer = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            bias=use_projection_bias,
            device=device,
        )

        self.dropout_layer = nn.Dropout(dropout)

        self.proj_output_dense_layer = nn.Linear(
            self.embed_dim, self.embed_dim, bias=use_ffn_bias, device=device,
        )

        # handle initialization
        output_initializer = attention_initializer
        if output_layer_initializer is not None:
            output_initializer = output_layer_initializer

        self.initializer = create_initializer(attention_initializer)
        self.output_initializer = create_initializer(output_initializer)

        self._reset_parameters()

    def _reset_parameters(self):
        # q, k, v projections
        weight_initializer = self.initializer
        weight_initializer(self.proj_q_dense_layer.weight.data)
        weight_initializer(self.proj_k_dense_layer.weight.data)
        weight_initializer(self.proj_v_dense_layer.weight.data)

        # output projections
        weight_initializer = self.output_initializer
        weight_initializer(self.proj_output_dense_layer.weight.data)

    def forward(
        self,
        q,
        k,
        v,
        attn_mask=None,
        key_padding_mask=None,
        need_weights=False,
        average_attn_weights=True,
        past_kv=None,
        cache_present_kv=False,
        position_bias=None,
        rotary_position_embedding_helper=None,
    ):
        """Applies the attention mechanism to queries ``q``, keys ``k`` and values ``v``.

        Args:
            q (Tensor): Queries, shape ``[batch_size, seq_length, embed_dim]``.
            k (Tensor): Keys, shape ``[batch_size, seq_length, embed_dim]``.
            v (Tensor): Values, shape ``[batch_size, seq_length, embed_dim]``.
            attn_mask (Tensor): Attention mask. Can be 2D of shape
                ``[batch_size, seq_length]``, or 3D of shape
                ``[batch, query_length, seq_length]``.
            key_padding_mask (Tensor): If specified, a mask of shape (N, S) indicating 
                which elements within key to ignore for the purpose of attention
                (i.e. treat as “padding”). Defaults to None.
            need_weights (bool): If specified, returns attn_output_weights in addition
                to attn_outputs. Default: False.
            average_attn_weights (bool): If true, indicates that the returned attn_weights
                should be averaged across heads. Otherwise, attn_weights are provided
                separately per head. Note that this flag only has an effect when
                need_weights=True. Default: True (i.e. average weights across heads)
            past_kv (Tensor): Past keys and values. Has shape
                ``[2, batch_size, num_heads, seq_length, embed_dim / num_heads]``.
                The tensors in ``[0,:,:,:,:]`` and ``[1,:,:,:,:]`` contain the
                past keys and values, respectively. Defaults to ``None``.
            cache_present_kv (bool): Specifies if the present keys and values
                must be cached and returned. Needed to speed up the
                computations when the decoder is called within an
                autoregressive loop. Defaults to ``False``.
            training (bool): Training the model if ``True``. Needed to call the
                ``dropout`` (after softmax) in the appropriate mode.
            position_bias (Tensor): Tensor containing position bias to apply in attention.

        Returns:
            If ``cache_present_kv`` is ``False``, no entry for present keys and values
            is provided.
        """

        assert (
            key_padding_mask is None
        ), "Key-padding mask is not implemented yet."
        assert not (
            rotary_position_embedding_helper and position_bias
        ), "Cannot specify both rotary and relative position embeddings, pick one!"

        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = q.shape[:2]
        real_seq_length = seq_length

        # linearly project the query (q), key (k) and value (v) using different
        # learned projections
        q = self.proj_q_dense_layer(q)
        k = self.proj_k_dense_layer(k)
        v = self.proj_v_dense_layer(v)

        # split q, k, v into heads
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        present_kv = None
        if cache_present_kv:
            present_kv = torch.stack([k, v])

        offset_length = 0
        if past_kv is not None:
            offset_length = past_kv[0].shape[-2]
            real_seq_length += offset_length

        if rotary_position_embedding_helper:
            k = rotary_position_embedding_helper.rotate_tensor(
                k, real_seq_length, offset=offset_length
            )
            q = rotary_position_embedding_helper.rotate_tensor(
                q, real_seq_length, offset=offset_length
            )

        if past_kv is not None:
            k_past, v_past = past_kv[0], past_kv[1]
            k = torch.cat([k_past, k], dim=-2)
            v = torch.cat([v_past, v], dim=-2)

        key_length = real_seq_length if present_kv is None else seq_length
        if self.scale_dot_product:
            depth = self.embed_dim // self.num_heads
            q = q * torch.tensor(1 / float(depth) ** 0.5, dtype=torch.float16,)

        # calculate dot product attention
        logits = torch.matmul(q, k.transpose(-1, -2))

        # apply attention mask
        if attn_mask is not None:
            if (
                attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.float32
            ):
                logits = logits + attn_mask
            else:
                neg_inf = -1e4
                assert len(mask.shape) in [
                    2,
                    3,
                ], "Only 2D/3D masks are supported"

                if len(mask.shape) == 2:
                    if past_kv is not None:
                        past_mask = torch.zeros(
                            (q.shape[0], past_kv.shape[-2]), dtype=mask.dtype
                        )
                        mask = torch.cat([past_mask, mask], axis=-1)

                    batch_size, seq_length = mask.shape[:2]
                    query_length = 1
                else:
                    if past_kv is not None:
                        past_mask = torch.zeros(
                            (q.shape[0], q.shape[-2], past_kv.shape[-2]),
                            dtype=mask.dtype,
                        )
                        mask = torch.cat([past_mask, mask], axis=-1)

                    batch_size, query_length, seq_length = mask.shape[:3]

                # compute the attention_bias based on the mask.
                # shape: (batch_size, 1, 1, seq_length)
                attention_bias = (
                    mask.view(batch_size, 1, query_length, seq_length) * neg_inf
                )
                logits += attention_bias

        # Add relative position bias, if any
        if position_bias is not None:
            logits += position_bias

        weights = nn.functional.softmax(logits.float(), dim=-1).type_as(logits)
        weights = self.dropout_layer(weights)

        # Shape: (batch_size, num_heads, query_length, embed_dim / num_heads)
        attention_output = torch.matmul(weights, v)

        # Recombine heads --> [batch_size, seq_length, embed_dim].
        attention_output = self._combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.proj_output_dense_layer(attention_output)

        if cache_present_kv:
            return attention_output, present_kv

        if not need_weights:
            return attention_output
        else:
            if average_attn_weights:
                weights = torch.mean(weights, dim=1).squeeze()
            return (
                attention_output,
                weights,
            )

    def _split_heads(self, x):
        """Split x into different heads, and transpose the resulting value. The
        tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.

        Args:
            x: A tensor with shape ``[batch_size, seq_length, embed_dim]``.

        Returns:
            A tensor with shape
            ``[batch_size, num_heads, seq_length, embed_dim/num_heads]``.
        """
        batch_size, seq_length = x.shape[:2]
        depth = self.embed_dim // self.num_heads
        return x.view(batch_size, seq_length, self.num_heads, depth).transpose(
            1, 2
        )

    def _combine_heads(self, x):
        """Combine tensor that has been split.

        Args:
            x: A tensor with shape
            ``[batch_size, num_heads, seq_length, embed_dim/num_heads]``.

        Returns:
            A tensor with shape ``[batch_size, seq_length, embed_dim]``.
        """
        batch_size, seq_length = x.shape[0], x.shape[2]
        return x.transpose(1, 2).reshape(batch_size, seq_length, self.embed_dim)
