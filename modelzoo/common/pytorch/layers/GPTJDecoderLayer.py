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

from typing import Optional, Union

from torch import Tensor

from modelzoo.common.pytorch.layers.TransformerDecoderLayer import (
    SelfAndCrossAttnKV,
    SelfAttnKV,
    TransformerDecoderLayer,
)
from modelzoo.common.pytorch.model_utils.RotaryPositionEmbeddingHelper import (
    RotaryPositionEmbeddingHelper,
)


class GPTJDecoderLayer(TransformerDecoderLayer):
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
        rotary_position_embedding_helper: Optional[
            RotaryPositionEmbeddingHelper
        ] = None,
        past_kv: Optional[Union[SelfAttnKV, SelfAndCrossAttnKV]] = None,
        cache_present_kv: bool = False,
        self_attn_position_bias: Optional[Tensor] = None,
        cross_attn_position_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """GPTJ layer with rotary position embeddings and parallel decoder architecture
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
        )

        # Apply untied layernorm in neox
        if self.norm3 is not None:
            hidden_normed = self.norm3(x)

        ffn_output = self.ffn(hidden_normed)
        outputs = residual + ffn_output + attn_output[0]
        return outputs
