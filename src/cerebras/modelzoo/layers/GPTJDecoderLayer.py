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

from typing import Dict, Optional, Union

from torch import Tensor

from cerebras.modelzoo.layers.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)
from cerebras.modelzoo.layers.TransformerDecoderLayer import (
    SelfAndCrossAttnKV,
    SelfAttnKV,
    TransformerDecoderLayer,
)


class GPTJDecoderLayer(TransformerDecoderLayer):
    """
    GPTJDecoderLayer is inherited from `TransformerDecoderLayer`, it has 2 modifications:

    1. It uses parallel decoder architecture instead of the sequential one

    2. It supports both gptj and gpt-neox which uses untied_layer_norm

    Reference: https://www.cerebras.net/blog/how-to-harness-the-predictive-power-of-gpt-j

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multihead-attention models (required).
        use_untied_layer_norm (bool): whether to use untied layer_norm. Should be False for GPTJ and True for Neox
        kwargs: the rest of the arguments the same as `TransformerDecoderLayer`
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        use_untied_layer_norm: bool = False,
        **kwargs,
    ):
        super(GPTJDecoderLayer, self).__init__(d_model, nhead, **kwargs)
        self.use_untied_layer_norm = use_untied_layer_norm
        if not self.use_untied_layer_norm:
            self.norm3 = None

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        rotary_position_embedding_helper: Optional[
            RotaryPositionEmbeddingHelper
        ] = None,
        past_kv: Optional[Union[SelfAttnKV, SelfAndCrossAttnKV]] = None,
        cache_present_kv: bool = False,
        self_attn_position_bias: Optional[Tensor] = None,
        cross_attn_position_bias: Optional[Tensor] = None,
        layer_idx: Optional[int] = None,
        expert_hash_idx: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        special_token_meta: Dict[str, Tensor] = None,
    ) -> Tensor:
        """GPTJ layer with rotary position embeddings and parallel decoder architecture

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            rotary_position_embedding_helper (Optional[RotaryPositionEmbeddingHelper]):
                A helper class to apply rotary embedding on the input tensor.
            past_kv: Past keys and values for self attention and (if applicable) cross
                attention modules. Key/value tensors have shape
                ``[batch_size, num_heads, seq_length, embed_dim / num_heads]``. (optional).
            cache_present_kv: Specifies if the present keys and values
                must be cached and returned. Needed to speed up the
                computations when the decoder is called within an
                autoregressive loop. (optional).
            self_attn_position_bias: the tensor containing position bias to apply in self-attention,
                can be obtained from relative or alibi position embeddings.
            expert_hash_idx: tensor containing mixture-of-experts expert
                selection indices for each token in the batch. Only used with
                MoE with hash-based routing enabled (optional).

        Shape:
            Output tensor with shape

        """

        x = tgt
        residual = x
        hidden_normed = self.norm1(x)
        attn_output = self._sa_block(
            hidden_normed,
            tgt_mask,
            tgt_key_padding_mask,
            rotary_position_embedding_helper,
            past_kv=past_kv[:2] if past_kv is not None else None,
            cache_present_kv=cache_present_kv,
            self_attn_position_bias=self_attn_position_bias,
            layer_idx=layer_idx,
            position_ids=position_ids,
        )

        # Apply untied layernorm in neox
        if self.norm3 is not None:
            hidden_normed = self.norm3(x)

        if self.moe_enabled:
            ffn_output, routing_weights, expert_mask = self.ffn(
                hidden_normed,
                expert_hash_idx=expert_hash_idx,
            )
        else:
            ffn_output = self.ffn(hidden_normed)

        outputs = residual + ffn_output + attn_output[0]
        if self.moe_enabled:
            return outputs, routing_weights, expert_mask
        else:
            return outputs
