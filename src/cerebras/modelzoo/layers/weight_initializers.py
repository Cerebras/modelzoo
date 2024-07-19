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

from torch.nn.init import _calculate_fan_in_and_fan_out, trunc_normal_


def variance_scaling_(
    tensor, scale=1.0, mode="fan_in", distribution="truncated_normal"
):
    r"""Adapted from TensorFlow's initializations
    https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling

    Fills the input Tensor with values given scale, mode and distribution.

    Args:
        tensor (torch.Tensor): an n-dimensional `torch.Tensor`
        scale (float): scaling factor (positive float)
        mode (str): mode of weight initialization. Defaults to `fan_in`
        distribution (str): distributino to initialize tensors with. Defaults to
            `truncated_normal`

    Examples:
        >>> w = torch.empty(3, 3)
        >>> variance_scaling_(w)

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = max(1.0, fan_in)
    elif mode == 'fan_out':
        denom = max(1.0, fan_out)
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
        denom = max(1.0, denom)

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    r"""Adapted from TensorFlow's initializations
    https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal

    Args:
        tensor (torch.Tensor): an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 3)
        >>> lecun_normal_(w)
    """
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def lecun_uniform_(tensor):
    r"""Adapted from TensorFlow's initializations
    https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunUniform

    Args:
        tensor (torch.Tensor): an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 3)
        >>> lecun_uniform_(w)
    """
    variance_scaling_(tensor, mode="fan_in", distribution="uniform")
