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

"""Attention mask ranges annotation support."""

from dataclasses import dataclass, field
from typing import List, Tuple

import torch

from cerebras.pytorch.backend import use_cs
from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation


@dataclass(frozen=True)
class AttentionMaskRangesInfo:
    """
    This attention mask sparsity descriptor is given to our kernels during model
    compilation and they will use it to configure themselves to save FLOPs.
    """

    starts: List[List[Tuple[int]]]
    ends: List[List[Tuple[int]]]
    S: int  # target sequence length
    dims: Tuple[int] = (2, 3)  # which logical dims are Sout and Sin

    # Produces a tensor that fills each region from the annotation with
    # its index (i.e., the first range will be filled with 1s, the second
    # with 2s and so on).
    def mask_from_ranges(self, device='cpu'):
        x, i = torch.zeros([self.S, self.S], device=device), 1
        for _s, _e in zip(self.starts, self.ends):
            for s, e in zip(_s, _e):
                x[s[0] : e[0], s[1] : e[1]] = i
                i += 1
        return x.T

    # Applies this mask range to an input tensor. Every non-zero entry
    # in the mask generated from the annotation means that the respective
    # element in the input tensor can never be pruned. All other entries
    # in the input tensor will be set to zero and could potentially be
    # pruned in the kernel implementation.
    def apply_to_mask(self, input_tensor: torch.Tensor):
        return torch.where(self.mask_from_ranges().bool(), input_tensor, 0)


@dataclass
class AnnotationConfig(AnnotationMode.Config):
    value: List
    enable_fwd: bool = field(default=True, init=False)
    enable_bwd: bool = field(default=True, init=False)


def get_attribute(config, is_backward: bool):
    """Returns mask_range attribute"""
    return AnnotationMode.Attribute('mask_range', config.value)


def mask_range(value: AttentionMaskRangesInfo):
    """
    Annotation decorator for attention mask ranges to save FLOPs based on sparse
    attention mask structures.
    """
    if not use_cs():
        return lambda fn: fn

    if value is None or not isinstance(value, AttentionMaskRangesInfo):
        return lambda fn: fn

    return create_annotation(
        AnnotationConfig,
        get_attribute,
        value=[
            value.ends,
            value.starts,
        ],  # stack and kernels expect this in reverse
    )
