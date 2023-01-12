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

from .AttentionLayer import MultiheadAttention

ATTENTION_TYPE_DICT = {
    "aiayn_attention": MultiheadAttention,
}


def get_attention_module(attn_type: str, extra_params):
    """
        This function retrieves the attention module according to `attn_type` and 
        checks if provided extra_params is correctly related to the attention module
    """
    assert (
        attn_type in ATTENTION_TYPE_DICT
    ), f"Attention {attn_type} not supported"
    attn_module = ATTENTION_TYPE_DICT[attn_type.lower()]
    attn_module.check_extra_params(extra_params)
    return attn_module
