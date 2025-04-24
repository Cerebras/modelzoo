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

from typing import Callable, Optional, Union

import torch.nn as nn

from cerebras.modelzoo.config import BaseConfig
from cerebras.modelzoo.layers.AttentionLayer import MultiheadAttention
from cerebras.modelzoo.layers.create_initializer import create_initializer
from cerebras.modelzoo.layers.MultiQueryAttentionLayer import (
    MultiQueryAttention,
)

_SUPPORTED_ATTENTION_MODULE_TYPES = (
    MultiheadAttention,
    MultiQueryAttention,
)

AttentionModuleType = Union[_SUPPORTED_ATTENTION_MODULE_TYPES]


def create_projection_func_override(
    ordinal_proj_module_name: str, memory_proj_module_name: str
) -> Callable:
    def projection_func_override(self, inp, special_token_meta=None):
        """Apply regular projection weights to regular tokens
        and memory token projection weights to memory tokens,
        locations of which are given by special_token_meta
        """
        memtok_mask = special_token_meta["memory_token_mask"].unsqueeze(-1)

        memory_proj = getattr(self, memory_proj_module_name)(inp)
        ordinal_proj = getattr(self, ordinal_proj_module_name)(inp)
        out = memory_proj * memtok_mask + ordinal_proj * (~memtok_mask)

        return out

    return projection_func_override


def create_reset_parameters_override(
    reset_parameters_method: Callable,
) -> Callable:
    def reset_parameters_method_override(self):
        reset_parameters_method(self)
        # bias initialization for memory token weights
        bias_initializer = create_initializer(self.bias_initializer)
        if self.use_projection_bias:
            bias_initializer(self.proj_q_memory_tokens_layer.bias.data)
            bias_initializer(self.proj_k_memory_tokens_layer.bias.data)
            bias_initializer(self.proj_v_memory_tokens_layer.bias.data)
        if self.use_ffn_bias:
            bias_initializer(self.proj_output_memory_tokens_layer.bias.data)

        # q projection init for memory token weights
        weight_initializer = create_initializer(self.query_initializer)
        weight_initializer(self.proj_q_memory_tokens_layer.weight.data)

        # k, v projections init for memory token weights
        weight_initializer = create_initializer(self.initializer)
        weight_initializer(self.proj_k_memory_tokens_layer.weight.data)
        weight_initializer(self.proj_v_memory_tokens_layer.weight.data)

        # output projections init for memory token weights
        weight_initializer = create_initializer(self.output_initializer)
        weight_initializer(self.proj_output_memory_tokens_layer.weight.data)

    return reset_parameters_method_override


class AddMemoryTokenWeightsMeta(type):
    """Wrapper for attention modules that overrides
    'get_projection', 'reset_parameters' and __init__ methods
    to enable a separate set of projection weights for memory tokens
    """

    def __new__(cls, *args, **kwargs):
        new_class = super().__new__(cls, *args, **kwargs)

        projection_methods_override = {
            "get_query_projection": create_projection_func_override(
                "proj_q_dense_layer", "proj_q_memory_tokens_layer"
            ),
            "get_key_projection": create_projection_func_override(
                "proj_k_dense_layer", "proj_k_memory_tokens_layer"
            ),
            "get_value_projection": create_projection_func_override(
                "proj_v_dense_layer", "proj_v_memory_tokens_layer"
            ),
            "get_attention_output_projection": create_projection_func_override(
                "proj_output_dense_layer", "proj_output_memory_tokens_layer"
            ),
        }

        for method_name, method in projection_methods_override.items():
            setattr(new_class, method_name, method)

        setattr(
            new_class,
            "reset_parameters",
            create_reset_parameters_override(
                getattr(new_class, "reset_parameters")
            ),
        )

        return new_class

    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):

            original_init(self, *args, **kwargs)

            kv_inner_dim = self.inner_dim
            if hasattr(self, "num_kv_groups"):
                # GQA / MQA case
                kv_inner_dim = self.num_kv_groups * self.head_dim

            self.proj_q_memory_tokens_layer = nn.Linear(
                self.embed_dim,
                self.inner_dim,
                bias=self.use_projection_bias,
                device=kwargs["device"],
            )
            self.proj_k_memory_tokens_layer = nn.Linear(
                self.kdim,
                kv_inner_dim,
                bias=self.use_projection_bias,
                device=kwargs["device"],
            )
            self.proj_v_memory_tokens_layer = nn.Linear(
                self.vdim,
                kv_inner_dim,
                bias=self.use_projection_bias,
                device=kwargs["device"],
            )
            self.proj_output_memory_tokens_layer = nn.Linear(
                self.inner_dim,
                self.embed_dim,
                bias=self.use_ffn_bias,
                device=kwargs["device"],
            )

        cls.__init__ = new_init


def create_mem_token_attn_module(attn_module_cls: AttentionModuleType):
    """Wrap an attention module class to override 'get_projection' layers
    to make memory tokens go through their own set of projection weights
    """
    assert (
        attn_module_cls in _SUPPORTED_ATTENTION_MODULE_TYPES
    ), f"Adding memory token parameters not implemented for {attn_module_cls} module type"

    class AttnModuleWithMemTokens(
        attn_module_cls, metaclass=AddMemoryTokenWeightsMeta
    ):
        pass

    return AttnModuleWithMemTokens


class MemoryTokensConfig(BaseConfig):
    attn_memory_chunk_size: Optional[int] = None
    "Length of regular token segment that is inverleaved with memory tokens"

    num_memory_tokens_per_chunk: Optional[int] = None
    "Number of memory tokens to insert after every input segment"

    add_qkv_memory_weights: bool = False
    "Whether to add extra parallel memory token weights for QKV projections in attention layers"

    add_extra_embedding: bool = False
    "Whether to add extra learnable embedding for memory tokens"

    add_chunked_attn_mask: bool = False
    "Whether to use a chunked attention mask with global attention for memory tokens"

    memory_token_id: int = 0
    "Memory token ID (in the case when extra embedding is not added)"

    @property
    def memory_tokens_enabled(self):
        return (
            self.num_memory_tokens_per_chunk is not None
            and self.attn_memory_chunk_size is not None
        )

    def update_model_config_params(
        self,
        max_position_embeddings,
        position_embedding_type,
        moe_enabled,
        fixed_sparse_attention,
        attention_sliding_window_length,
        attention_vertical_column_spacing,
        attention_vertical_column_width,
        attention_chunk_size,
    ):
        if moe_enabled:
            raise ValueError("Memory tokens are not supported for MoE models")

        if position_embedding_type != "rotary":
            raise ValueError(
                f"Memory tokens are only supported with rotary position embedding, "
                f"but got type {position_embedding_type}"
            )

        total_chunk_size = (
            self.attn_memory_chunk_size + self.num_memory_tokens_per_chunk
        )
        updated_params = {}

        if self.add_chunked_attn_mask:
            # Verify parameters for triangular chunked mask with vertical columns
            if (
                fixed_sparse_attention is not None
                or attention_sliding_window_length is not None
            ):
                raise ValueError(
                    "Memory token option is not compatible with custom attention masks"
                )

            if attention_chunk_size is None:
                updated_params["attention_chunk_size"] = total_chunk_size
            elif attention_chunk_size != total_chunk_size:
                raise ValueError(
                    f"Expected attention_chunk_size={total_chunk_size}, "
                    f"got {attention_chunk_size}"
                )
            else:
                updated_params["attention_chunk_size"] = attention_chunk_size

            if attention_vertical_column_spacing is None:
                updated_params["attention_vertical_column_spacing"] = (
                    total_chunk_size
                )
            elif attention_vertical_column_spacing != total_chunk_size:
                raise ValueError(
                    f"Expected attention_vertical_column_spacing={total_chunk_size}, "
                    f"got {attention_vertical_column_spacing}"
                )
            else:
                updated_params["attention_vertical_column_spacing"] = (
                    attention_vertical_column_spacing
                )

            if attention_vertical_column_width is None:
                updated_params["attention_vertical_column_width"] = (
                    self.num_memory_tokens_per_chunk
                )
            elif (
                attention_vertical_column_width
                != self.num_memory_tokens_per_chunk
            ):
                raise ValueError(
                    f"Expected attention_vertical_column_width={self.num_memory_tokens_per_chunk}, "
                    f"got {attention_vertical_column_width}"
                )
            else:
                updated_params["attention_vertical_column_width"] = (
                    attention_vertical_column_width
                )

        # Update max_position_embeddings to accomodate sequence length
        # after memory token insertion
        updated_params["max_position_embeddings"] = (
            max_position_embeddings
            + self.num_memory_tokens_per_chunk
            * (self.attn_memory_chunk_size + max_position_embeddings)
            // self.attn_memory_chunk_size
        )
        return updated_params
