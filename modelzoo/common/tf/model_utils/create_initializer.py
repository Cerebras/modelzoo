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

import tensorflow as tf


def create_initializer(spec, seed=None):
    """
    Creates the specified initializer.

    :param dict/str spec: either a string indicating the name of the initializer
        or a dict that includes the name + other params if relevant.
    :param int seed: random seed for the initializer or None to run unseeded. 
    Note: Currently seed is removed until TF upgade. 
    passed seed is going to be ignored here if someone passes seed in a call to create_initializer
    :returns: initializer that can be passed to layers
    """
    if type(spec) == str:
        spec = {"name": spec}
    if "name" not in spec:
        raise ValueError("Initializer name must be provided")
    name = spec["name"]
    if name == "constant":
        return tf.keras.initializers.Constant(
            value=_get_spec_value(spec, "value", 0)
        )
    elif name == "uniform":
        return tf.keras.initializers.RandomUniform(
            minval=_get_spec_value(spec, "minval", -0.05),
            maxval=_get_spec_value(spec, "maxval", 0.05),
        )
    elif name == "glorot_uniform":
        return tf.keras.initializers.GlorotUniform()
    elif name == "normal":
        return tf.keras.initializers.RandomNormal(
            mean=_get_spec_value(spec, "mean", 0.0),
            stddev=_get_spec_value(spec, "stddev", 0.05),
        )
    elif name == "glorot_normal":
        return tf.keras.initializers.GlorotNormal()
    elif name == "truncated_normal":
        return tf.keras.initializers.TruncatedNormal(
            mean=_get_spec_value(spec, "mean", 0.0),
            stddev=_get_spec_value(spec, "stddev", 0.05),
        )
    elif name == "variance_scaling":
        return tf.keras.initializers.VarianceScaling(
            scale=_get_spec_value(spec, "scale", 1.0),
            mode=_get_spec_value(spec, "mode", "fan_in"),
            distribution=_get_spec_value(
                spec, "distribution", "truncated_normal"
            ),
        )
    else:
        raise ValueError(
            f"Invalid initializer, '{name}'. "
            "Supported initializers include: "
            "constant, uniform, glorot_uniform, normal, glorot_normal, truncated_normal, "
            "variance_scaling."
        )


def _get_spec_value(spec, key, default_value):
    """
    Returns value of spec[key].
    If key is not present, gives a warning and returns default_value.
    """
    name = spec["name"]
    value = spec.get(key)
    if value is None:
        tf.compat.v1.logging.warn(
            f"{name} initializer's {key} parameter not specified. "
            f"Using {default_value}."
        )
        value = default_value
    return value
