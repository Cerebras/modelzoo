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

import torch.nn as nn
import torchvision.ops as ops

from modelzoo.vision.pytorch.layers.BatchChannelNorm import BatchChannelNorm2D

NORM2CLS = {
    "batchnorm1d": nn.BatchNorm1d,
    "batchnorm2d": nn.BatchNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "instance1d": nn.InstanceNorm1d,
    "instance2d": nn.InstanceNorm2d,
    "instance3d": nn.InstanceNorm3d,
    "group": nn.GroupNorm,
    "layer": nn.LayerNorm,
    "batchchannel2d": BatchChannelNorm2D,
    "frozenbatchnorm2d": ops.FrozenBatchNorm2d,
    None: nn.Identity,
}


def get_normalization(normalization_str):
    if normalization_str is not None:
        normalization_str = normalization_str.lower()
    if normalization_str in NORM2CLS:
        return NORM2CLS[normalization_str]
    else:
        raise KeyError(
            f"function {normalization_str} not found in NORM2CLS mapping "
            f"{list(NORM2CLS.keys())}"
        )
