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

"""
Pytorch-level API for recompute

This API aims to give user ability to control the recompute algorithm in compiler, by providing
two python decorators to annotate two things:
1. all operations that in this user-control recompute region.
2. recompute checkpoints, whose result should be saved in memory.

After annotating the recompute region and checkpoints, the compiler will determine which operation
should be recomputed from user-specified checkpoints.
"""

from dataclasses import dataclass
from typing import Optional

from cerebras.pytorch.core.annotation import AnnotationMode, create_annotation


@dataclass
class CkptNamedConfig(AnnotationMode.Config):
    name: Optional[str]


@dataclass
class RecomputeRegionNamedConfig(AnnotationMode.Config):
    name: Optional[str]


def get_recompute_ckpt_attribute(config, is_backward: bool):
    """Returns recompute_ckpt attribute"""
    postfix = "bwd" if is_backward else "fwd"
    name = f"{config.name}.{postfix}"
    return AnnotationMode.Attribute('recompute_ckpt', name)


def get_recompute_region_attribute(config, is_backward: bool):
    """Returns recompute_region attribute"""
    postfix = "bwd" if is_backward else "fwd"
    name = f"{config.name}.{postfix}"
    return AnnotationMode.Attribute('recompute_region', name)


def get_autgrad_func_attribute(config, is_backward: bool):
    """Returns recompute_region attribute"""
    postfix = "custom_bwd"
    name = f"{config.name}.{postfix}"
    return AnnotationMode.Attribute('recompute_region', name)


def recompute_ckpt(
    name: Optional[str] = None,
    enable_fwd: bool = True,
    enable_bwd: bool = False,
):
    """
    Python decorator for annotating checkpoint operation for the recompute algorithm.
    The annotated operation will have the "recompute_ckpt" attribute in "cs.internal"
    DictionaryAttr.
    """
    return create_annotation(
        CkptNamedConfig,
        get_recompute_ckpt_attribute,
        name=name,
        enable_fwd=enable_fwd,
        enable_bwd=enable_bwd,
    )


def recompute_region(
    name: Optional[str] = None,
    enable_fwd: bool = True,
    enable_bwd: bool = True,
    is_autograd_func: bool = False,
):
    """
    Python decorator for annotating operation in the recompute region.
    The annotated operation will have the "recompute_region" attribute in "cs.internal"
    DictionaryAttr.

    Note: if user uses 'torch.autograd.Function' to custom the backward function, we need to
    annotate this customized backward function separately with 'is_autograd_func=True'.
    """
    if is_autograd_func:
        return create_annotation(
            RecomputeRegionNamedConfig,
            get_autgrad_func_attribute,
            name=name,
            enable_fwd=enable_fwd,
            enable_bwd=enable_bwd,
        )
    else:
        return create_annotation(
            RecomputeRegionNamedConfig,
            get_recompute_region_attribute,
            name=name,
            enable_fwd=enable_fwd,
            enable_bwd=enable_bwd,
        )
