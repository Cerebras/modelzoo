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

from functools import partial
from typing import Literal

import torchvision.ops as ops
from torch import nn

from cerebras.modelzoo.layers import (
    AdaLayerNorm,
    BatchChannelNorm2D,
    GroupInstanceNorm,
    RMSNorm,
)


def get_norm(norm_string):
    # NORM2CLASS mapping must stay inside the scope of this function,
    # so that the class being returned is resolved each time that
    # `get_norm` gets called. If this were outside the scope of
    # `get_norm`, then the class pointer is bound at the time of the
    # first load of this source file.
    global NORM2CLASS
    NORM2CLASS = {
        "adalayer": AdaLayerNorm,
        "batchchannel2d": BatchChannelNorm2D,
        "batchnorm1d": nn.BatchNorm1d,
        "batchnorm2d": nn.BatchNorm2d,
        "batchnorm3d": nn.BatchNorm3d,
        "biasless-layernorm": partial(nn.LayerNorm, bias=False),
        "nonparametric-layernorm": partial(
            nn.LayerNorm, elementwise_affine=False, bias=False
        ),
        "frozenbatchnorm2d": ops.FrozenBatchNorm2d,
        "group": nn.GroupNorm,
        "group_instance": GroupInstanceNorm,  # used to emulate instance norm with group norm
        "instance1d": nn.InstanceNorm1d,
        "instance2d": nn.InstanceNorm2d,
        "instance3d": nn.InstanceNorm3d,
        "layernorm": nn.LayerNorm,
        "rmsnorm": RMSNorm,
        None: nn.Identity,
    }

    if norm_string is not None:
        norm_string = norm_string.lower()
    if norm_string in NORM2CLASS:
        return NORM2CLASS[norm_string]
    else:
        raise KeyError(
            f"class {norm_string} not found in NORM2CLASS mapping {list(NORM2CLASS.keys())}"
        )


# Dummy call to get_norm to initialize NORM2CLASS
get_norm("layernorm")

NormType = Literal[tuple(NORM2CLASS)]
