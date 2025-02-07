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

from typing import Literal, Union

import torch.nn as nn

from .AttentionLayer import MultiheadAttention
from .MultiQueryAttentionLayer import MultiQueryAttention

ATTENTION_TYPE_DICT = {
    "aiayn_attention": MultiheadAttention,
    "multiquery_attention": MultiQueryAttention,
}

AttentionType = Literal[tuple(ATTENTION_TYPE_DICT)]


def get_attention_module(attn_module: Union[str, nn.Module], extra_params):
    """
    This function retrieves the attention module according to
    `attn_module`. If the input is a string, the function lookups
    the corresponding attention class and checks if provided extra_params is
    correctly related to the attention module. If the input is a nn.Module,
    it returns the input (and ensures that the extra_params are correct if
    applicable)
    """
    # attn_module is a nn.Module:
    if isinstance(attn_module, nn.Module):
        if hasattr(attn_module, "check_extra_params"):
            attn_module.check_extra_params(extra_params)
        return attn_module

    # attn_module is a string:
    attn_module = attn_module.lower()
    assert (
        attn_module in ATTENTION_TYPE_DICT
    ), f"Attention {attn_module} not supported"
    attn_class = ATTENTION_TYPE_DICT[attn_module]
    attn_class.check_extra_params(extra_params)
    return attn_class
