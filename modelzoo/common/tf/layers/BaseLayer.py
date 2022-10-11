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

_SUPPORTED_MP_POLICIES = ["float16", "mixed_float16", "float32"]
# note: pure "float16" is experimental, not fully tested


class BaseLayer(tf.keras.layers.Layer):
    """
    Base layer for the reference models.

    Args:
        boundary_casting (bool): If ``True``, outputs the values in half
            precision and casts the input values up to full precision.
        tf_summary (bool): If ``True``, saves the activations with
            ``summary_layer``.
    """

    def __init__(self, boundary_casting=False, tf_summary=False, **kwargs):
        super(BaseLayer, self).__init__(**kwargs)

        assert self._dtype_policy, "Failed to setup an MP policy."
        assert (
            self._dtype_policy.name in _SUPPORTED_MP_POLICIES
        ), f"Unsupported MP policy type: {self._dtype_policy.name}"

        self.dtype_policy = self._dtype_policy
        self.variable_dtype = self._dtype_policy.variable_dtype
        self.compute_dtype = self._dtype_policy.compute_dtype

        self.boundary_casting = boundary_casting
        self.tf_summary = tf_summary

        if self.boundary_casting and self.compute_dtype != "float32":
            raise ValueError(
                f"Boundary casting can only be enabled for \
                compute_type = 'float32'. Received {self.compute_dtype}."
            )

    def call(self):
        raise NotImplementedError
