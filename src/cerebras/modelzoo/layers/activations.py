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

# This code is adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/activations.py
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from typing import Literal

import torch
from torch import nn

# TODO: Figure logging
# from .utils import logging
# logger = logging.get_logger(__name__)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x))
        )
    )


def gelu_fast(x):
    return (
        0.5
        * x
        * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    )


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


def squared_gelu(x):
    g = nn.functional.gelu(x)
    return g * g


def squared_relu(x):
    """
    Taken from https://arxiv.org/abs/2109.08668

    Args:
        x (torch.Tensor): Input tensor for the activation

    Returns:
        torch.Tensor with activation applied
    """
    r = nn.functional.relu(x)
    return r * r


def linear_act(x):
    return x


# GLU bivariate Activations implementation
def glu_bivariate_base_fn(x1, x2, activation_fn):
    assert (
        x1.shape == x2.shape
    ), "GLU activation inputs must have the same shape"
    return x1 * activation_fn(x2)


def liglu(x1, x2):
    identity = lambda x: x
    return glu_bivariate_base_fn(x1, x2, identity)


def geglu(x1, x2):
    return glu_bivariate_base_fn(x1, x2, nn.functional.gelu)


def reglu(x1, x2):
    return glu_bivariate_base_fn(x1, x2, nn.functional.relu)


def swiglu(x1, x2):
    return glu_bivariate_base_fn(x1, x2, nn.functional.silu)


GLU_ACTIVATIONS = {
    "liglu",
    "geglu",
    "reglu",
    "swiglu",
}

ACT2FN = {
    "relu": nn.functional.relu,
    "leaky_relu": nn.functional.leaky_relu,
    "silu": nn.functional.silu,
    "swish": nn.functional.silu,
    "gelu": nn.functional.gelu,
    "tanh": torch.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "squared_gelu": squared_gelu,
    "squared_relu": squared_relu,
    "mish": nn.functional.mish,
    "linear": linear_act,
    "sigmoid": torch.sigmoid,
    "relu6": nn.functional.relu6,
    "liglu": liglu,
    "geglu": geglu,
    "reglu": reglu,
    "swiglu": swiglu,
    None: linear_act,
}

ActivationType = Literal[tuple(ACT2FN)]


def get_activation(activation):
    if callable(activation):
        return activation
    if activation is not None:
        activation = activation.lower()
    if activation in ACT2FN:
        return ACT2FN[activation]
    else:
        raise KeyError(
            f"function {activation} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )


def is_glu_activation(activation):
    if hasattr(activation, "is_glu_activation"):
        return getattr(activation, "is_glu_activation")
    if isinstance(activation, str):
        activation = activation.lower()
    return activation in GLU_ACTIVATIONS
