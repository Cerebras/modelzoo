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

import torch
import torch.nn as nn

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


def bcast2h(t: torch.Tensor, H: int):
    return t[..., None].broadcast_to(-1, -1, H)


def create_projection_func_override(
    ordinal_proj_module_name: str, memory_proj_module_name: str
) -> Callable:
    def projection_func_override(self, inp, special_token_indices=None):
        """Apply regular projection weights to regular tokens
        and memory token projection weights to memory tokens,
        locations of which are given by special_token_indices
        """
        embedding_size = inp.shape[-1]
        memory_indices = special_token_indices["memory_tokens"]
        regular_indices = special_token_indices["regular_tokens"]

        memory_proj = getattr(self, memory_proj_module_name)(
            torch.gather(inp, 1, bcast2h(memory_indices, embedding_size))
        )
        ordinal_proj = getattr(self, ordinal_proj_module_name)(
            torch.gather(inp, 1, bcast2h(regular_indices, embedding_size))
        )

        out = torch.zeros(
            (inp.shape[0], inp.shape[1], ordinal_proj.shape[-1]),
            dtype=ordinal_proj.dtype,
            device=ordinal_proj.device,
        )
        out = torch.scatter(
            out, 1, bcast2h(memory_indices, ordinal_proj.shape[-1]), memory_proj
        )
        out = torch.scatter(
            out,
            1,
            bcast2h(regular_indices, ordinal_proj.shape[-1]),
            ordinal_proj,
        )

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


def memory_tokens_validate_param_consistency(
    position_embedding_type: str,
    moe_enabled: bool,
    fixed_sparse_attention: Optional[dict] = None,
    attention_sliding_window_length: Optional[int] = None,
    attention_chunk_size: Optional[int] = None,
    attention_vertical_column_spacing: Optional[int] = None,
    attention_vertical_column_width: Optional[int] = None,
) -> dict:
    """Checks for options that are incompatible with memory tokens
    and returns a dict of parameters to be set in order to be
    aligned with memory token usage algorithm
    """
    if (
        attention_sliding_window_length is not None
        or attention_chunk_size is not None
        or fixed_sparse_attention is not None
    ):
        raise ValueError(
            "Memory token option is not compatible with custom attention masks"
        )
    if position_embedding_type != "rotary":
        raise ValueError(
            f"Memory tokens are only supported with rotary position embedding, "
            f"but got type {position_embedding_type}"
        )
    if moe_enabled:
        raise ValueError("Memory tokens are not supported for MoE models")
    if (
        attention_vertical_column_spacing is not None
        or attention_vertical_column_width is not None
    ):
        raise ValueError(
            "Vertical column spacing/width parameters for attention masks "
            "cannot be set together with memory tokens"
        )


def memory_tokens_update_params(
    num_memory_tokens_per_chunk, attn_memory_chunk_size, max_position_embeddings
):
    # Setting sliding window length and vertical column spacing/width
    # in attention parameters to be aligned with memory token locations.
    # max_position_embeddings needs to be updated as effective sequence
    # length is higher with memory tokens
    updated_params = {}

    total_chunk_size = attn_memory_chunk_size + num_memory_tokens_per_chunk
    updated_params["attention_chunk_size"] = total_chunk_size
    updated_params["attention_vertical_column_spacing"] = total_chunk_size
    updated_params["attention_vertical_column_width"] = (
        num_memory_tokens_per_chunk
    )

    updated_params["max_position_embeddings"] = (
        max_position_embeddings
        + num_memory_tokens_per_chunk
        * (attn_memory_chunk_size + max_position_embeddings)
        // attn_memory_chunk_size
    )
    return updated_params
