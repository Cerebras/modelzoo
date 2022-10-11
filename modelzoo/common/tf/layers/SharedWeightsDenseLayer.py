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
from tensorflow.python.keras import (
    activations,
    constraints,
    initializers,
    regularizers,
)

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class SharedWeightsDenseLayer(BaseLayer):
    """Dense layer that takes in a kernel as a shared weight. Can also
    optionally add a bias.

    """

    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        bias_initializer='zeros',
        bias_regularizer=None,
        bias_constraint=None,
        boundary_casting=False,
        tf_summary=False,
        *args,
        **kwargs,
    ):
        super(SharedWeightsDenseLayer, self).__init__(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            *args,
            **kwargs,
        )
        self.units = units
        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=[units],
                dtype=self.variable_dtype,
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )

    def call(self, inputs, kernel, transpose_kernel=True, **kwargs):
        """Apply the densely-connected layer.

        Args:
            inputs (Tensor): An N-D tensor with the shape:
                ``(batch_size, ..., input_dim)``.
            kernel (Tensor): A 2-D tensor with the shape:
                ``(units, input_dim)``. The dense kernel.
            transpose_kernel (bool): Whether to transpose the kernel when
                performing ``tf.matmul(inputs, kernel)``.

        Returns:
            Tensor: An N-D tensor with shape: ``(batch_size, ..., units)``.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)
        kernel_shape = [self.units, inputs.shape[-1]]
        if transpose_kernel:
            kernel_shape.reverse()
        assert inputs.shape[-1] == kernel_shape[0], (
            "Input kernel has shape"
            f"{inputs.shape[-1]} when it should be {kernel_shape[0]}"
        )
        kernel = (tf.cast(kernel, inputs.dtype),)
        output = tf.matmul(inputs, kernel, transpose_b=transpose_kernel)
        if self.use_bias:
            output += tf.cast(self.bias, output.dtype)
        output = self.activation(output)
        if self.tf_summary:
            output = summary_layer(output)
        return output
