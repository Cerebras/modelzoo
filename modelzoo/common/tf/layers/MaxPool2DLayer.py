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


class MaxPool2DLayer(BaseLayer):
    """Wrapper around the Keras 2D max pooling layer.
    """

    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding='valid',
        data_format=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(MaxPool2DLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.layer = tf.keras.layers.MaxPool2D(
            pool_size,
            strides,
            padding,
            data_format,
            name=self.name,
            dtype=self.dtype_policy,
        )

    def call(self, inputs, **kwargs):
        """Applies the 2D max pooling layer.

        Args:
            inputs (Tensor): A 4D tensor with the shape:
                ``(samples, channels, rows, cols)`` if
                ``data_format='channels_first'`` or a 4D tensor with the
                shape ``(samples, rows, cols, channels)`` if
                ``data_format='channels_last'``.
        Returns:
            Tensor: A 4D tensor with the shape:
            ``(batch_size, channels, pooled_rows, pooled_cols)`` if
            ``data_format='channels_first'`` or a 4D tensor with
            shape: ``(batch_size, pooled_rows, pooled_cols, channels)`` if
            ``data_format='channels_last'``.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)
        output = self.layer(inputs)
        if self.tf_summary:
            output = summary_layer(output)
        return output
