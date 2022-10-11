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

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class RMSNormalizationLayer(BaseLayer):
    """Construct the RMSNorm module taken from the paper
    `Root Mean Square Layer Normalization <https://arxiv.org/abs/1910.07467>`_.

    Args:
        axis (int): The axis to apply normalization along. Defaults to ``-1``.
        epsilon (float): The epsilon value to ensure non-zero division.
            Defaults to ``1.0e-6``.
        boundary_casting (bool): See the documentation for ``BaseLayer``.
        tf_summary (bool): See the documentation for ``BaseLayer``.
        **kwargs: Additional keyword arguments for ``BaseLayer``.
    """

    def __init__(
        self,
        axis=-1,
        epsilon=1e-6,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(RMSNormalizationLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.axis = axis
        self.epsilon = epsilon
        self.boundary_casting = boundary_casting
        self.tf_summary = tf_summary

    def build(self, input_shape):
        self.weight = self.add_weight(
            "weight",
            shape=(input_shape[self.axis],),
            initializer="ones",
            dtype=self.variable_dtype,
            experimental_autocast=False,
        )
        super().build(input_shape)

    def call(self, inputs):
        """Apply the normalization.

        Args:
            inputs (Tensor): Arbitrary tensor.

        Returns:
            Tensor: A normalized tensor of the same shape as input.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        inputs = tf.cast(inputs, dtype=tf.float32)
        variance = tf.math.reduce_mean(
            tf.math.square(inputs), axis=self.axis, keepdims=True,
        )
        inputs = inputs * tf.math.rsqrt(variance + self.epsilon)
        output = tf.cast(self.weight, dtype=self.compute_dtype) * tf.cast(
            inputs, dtype=self.compute_dtype
        )

        if self.tf_summary:
            output = summary_layer(output)
        return output
