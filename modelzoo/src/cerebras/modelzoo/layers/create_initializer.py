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

import logging

import torch.nn as nn

from cerebras.modelzoo.layers.weight_initializers import (
    lecun_normal_,
    lecun_uniform_,
    variance_scaling_,
)

INIT2FN = {
    "constant": nn.init.constant_,
    "ones": nn.init.ones_,
    "zeros": nn.init.zeros_,
    "eye": nn.init.eye_,
    "uniform": nn.init.uniform_,
    "normal": nn.init.normal_,
    "xavier_normal": nn.init.xavier_normal_,
    "glorot_normal": nn.init.xavier_normal_,  # alias for `xavier_normal`
    "xavier_uniform": nn.init.xavier_uniform_,
    "glorot_uniform": nn.init.xavier_uniform_,  # alias for `xavier_uniform`
    "truncated_normal": nn.init.trunc_normal_,
    "variance_scaling": variance_scaling_,
    "lecun_normal": lecun_normal_,
    "lecun_uniform": lecun_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
}


def create_initializer(spec):
    """
    Creates the specified initializer.

    :param dict/str spec: either a string indicating the name of the initializer
        or a dict that includes the name + other params if relevant.
    :param int seed: random seed for the initializer or None to run unseeded.
    :returns: initializer that can be passed to layers
    """
    from cerebras.modelzoo.layers.init import Initializer

    if isinstance(spec, Initializer):
        return spec

    if type(spec) == str:
        spec = {"name": spec}
    if "name" not in spec:
        raise ValueError("Initializer name must be provided")
    name = spec["name"].lower()

    if name == "constant":
        return lambda tensor: INIT2FN[name](
            tensor, val=_get_spec_value(spec, "val", 0)
        )
    elif name in ["ones", "zeros", "eye", "lecun_normal", "lecun_uniform"]:
        return lambda tensor: INIT2FN[name](tensor)
    elif name == "uniform":
        return lambda tensor: INIT2FN[name](
            tensor,
            a=_get_spec_value(spec, "a", -0.05),
            b=_get_spec_value(spec, "b", 0.05),
        )
    elif name == "normal":
        return lambda tensor: INIT2FN[name](
            tensor,
            mean=_get_spec_value(spec, "mean", 0.0),
            std=_get_spec_value(spec, "std", 0.05),
        )
    elif name in [
        "xavier_normal",
        "xavier_uniform",
        "glorot_normal",
        "glorot_uniform",
    ]:
        return lambda tensor: INIT2FN[name](
            tensor, gain=_get_spec_value(spec, "gain", 1.0)
        )
    elif name == "kaiming_normal":
        return lambda tensor: INIT2FN[name](
            tensor,
            a=_get_spec_value(spec, "a", 0.0),
            mode=_get_spec_value(spec, "mode", "fan_in"),
            nonlinearity=_get_spec_value(
                spec, "nonlinearity", "leaky_relu", override_gain_calc=True
            ),
        )
    elif name == "kaiming_uniform":
        return lambda tensor: INIT2FN[name](
            tensor,
            a=_get_spec_value(spec, "a", 0.0),
            mode=_get_spec_value(spec, "mode", "fan_in"),
            nonlinearity=_get_spec_value(
                spec, "nonlinearity", "leaky_relu", override_gain_calc=True
            ),
        )
    elif name == "truncated_normal":
        std = _get_spec_value(spec, "std", 0.05)
        return lambda tensor: INIT2FN[name](
            tensor,
            mean=_get_spec_value(spec, "mean", 0.0),
            std=std,
            a=_get_spec_value(spec, "a", -2 * std),
            b=_get_spec_value(spec, "b", 2 * std),
        )
    elif name == "variance_scaling":
        return lambda tensor: INIT2FN[name](
            tensor,
            scale=_get_spec_value(spec, "scale", 1.0),
            mode=_get_spec_value(spec, "mode", "fan_in"),
            distribution=_get_spec_value(
                spec, "distribution", "truncated_normal"
            ),
        )
    else:
        raise ValueError(f"Invalid or unsupported initializer, '{name}'. ")


def _get_spec_value(spec, key, default_value, override_gain_calc=False):
    """
    Returns value of spec[key].
    If key is not present, gives a warning and returns default_value.
    """

    def is_nonlinearity(value):
        return value in [
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
            "sigmoid",
            "tanh",
            "relu",
            "leaky_relu",
        ]

    name = spec["name"]
    value = spec.get(key)
    if value is None:
        logging.debug(
            f"{name} initializer's {key} parameter not specified. "
            f"Using {default_value}."
        )
        value = default_value
    elif override_gain_calc:
        pass
    elif is_nonlinearity(value):
        value = nn.init.calculate_gain(value)
    return value
