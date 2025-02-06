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

import torch

import cerebras.pytorch as cstorch


def _maybe_to_half_dtype_impl(tensor: torch.Tensor) -> torch.Tensor:
    """Return tensor cast to half dtype if on CSX or autocast CPU/GPU ctx."""
    if (
        cstorch.use_cs()
        or torch.is_autocast_enabled()
        or torch.is_autocast_cpu_enabled()
    ):
        return tensor.to(cstorch.amp.get_half_dtype())

    return tensor


def maybe_to_half_dtype(
    tree: torch.utils._pytree.PyTree,
) -> torch.utils._pytree.PyTree:
    """Return tree with tensors cast to half dtype if on CSX or autocast CPU/GPU ctx."""
    return torch.utils._pytree.tree_map(_maybe_to_half_dtype_impl, tree)


def cb16_to_fp32(tensor: torch.Tensor) -> torch.Tensor:
    """Return tensor cast to float32 if it is cbfloat16."""
    if cstorch.amp.is_cbfloat16_tensor(tensor):
        return tensor.float()
    return tensor
