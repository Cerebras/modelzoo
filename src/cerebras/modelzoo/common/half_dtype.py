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

"""Module which provides utilities for selecting 16-bit floating point representation."""

from typing import Any, Dict

import torch

import cerebras.pytorch as cstorch


def set_half_dtype_from_params(params: Dict[str, Any]) -> torch.dtype:
    """Sets the half dtype in cstorch from the given model params.

    Note that after this call, reading the half dtype from params is discouraged. Instead, use
    `cstorch.amp.get_half_dtype()` to get the dtype to use in the model.

    Args:
        params: Model params where to find "fp16_type" key.
    Returns:
        The proxy dtype to use in the model.
    """
    if "use_bfloat16" in params:
        raise KeyError(
            f"The flag \"use_bfloat16\" is deprecated. Use the flag \"fp16_type\" instead. "
            f"This flag accepts one of \"float16\", \"bfloat16\", or \"cbfloat16\"."
        )

    return cstorch.amp.set_half_dtype(params.get("fp16_type", "float16"))
