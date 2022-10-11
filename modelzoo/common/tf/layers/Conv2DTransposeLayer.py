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


class Conv2DTransposeLayer(BaseLayer):
    """Wrapper around the Keras 2D transposed convolution layer.
    """

    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        output_padding=None,
        data_format=None,
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(Conv2DTransposeLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.layer = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides,
            padding,
            output_padding,
            data_format,
            dilation_rate,
            activation,
            use_bias,
            kernel_initializer,
            bias_initializer,
            kernel_regularizer,
            bias_regularizer,
            activity_regularizer,
            kernel_constraint,
            bias_constraint,
            name=self.name,
            dtype=self.dtype_policy,
        )

    def call(self, inputs, **kwargs):
        """Apply the 2D transposed convolution layer.

        Args:
            inputs: A 4D tensor with shape: ``(samples, channels, rows, cols)``
                if ``data_format='channels_first'`` or 4D tensor with shape:
                ``(samples, rows, cols, channels)`` if
                ``data_format='channels_last'``.
        Returns:
            Tensor: A 4D tensor with shape:
            ``(samples, filters, new_rows, new_cols)`` if
            ``data_format='channels_first'`` or a 4D tensor with shape:
            ``(samples, new_rows, new_cols, filters)`` if
            ``data_format='channels_last'``. Note that ``rows`` and
            ``cols`` values might have changed due to padding.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)

        if self.tf_summary:
            inputs = summary_layer(inputs)

        output = self.layer(inputs)
        if self.tf_summary:
            output = summary_layer(output)
        return output
