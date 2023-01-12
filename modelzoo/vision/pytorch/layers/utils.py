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

import math
from typing import Optional

import torch.nn as nn


def adjust_channels(
    channels: int,
    width_multiplier: float,
    divisor: Optional[int] = 8,
    min_value: Optional[int] = None,
    round_limit: Optional[int] = 0.9,
) -> int:
    return _make_divisible(
        channels * width_multiplier, divisor, min_value, round_limit
    )


def adjust_depth(num_layers: int, depth_multiplier: float):
    return int(math.ceil(num_layers * depth_multiplier))


def _make_divisible(
    v: float,
    divisor: int,
    min_value: Optional[int] = None,
    round_limit: Optional[int] = 0.9,
) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class ModuleWrapperClass(nn.Module):
    def __init__(self, fcn, name=None, kwargs=None):
        self.fcn = fcn
        self.name = name
        self.kwargs = kwargs
        super(ModuleWrapperClass, self).__init__()

    def extra_repr(self) -> str:
        repr_str = 'fcn={}'.format(
            self.name if self.name is not None else self.fcn.__name__
        )
        if self.kwargs is not None:
            for k, val in self.kwargs.items():
                repr_str += f", {k}={val}"

        return repr_str

    def forward(self, input):
        return self.fcn(input)
