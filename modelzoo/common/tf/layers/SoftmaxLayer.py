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
from tensorflow.keras.layers import Softmax

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class SoftmaxLayer(BaseLayer):
    """Wrapper around the Keras softmax layer.
    """

    def __init__(
        self, axis=-1, boundary_casting=False, tf_summary=False, **kwargs
    ):
        super(SoftmaxLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.layer = Softmax(axis=axis, name=self.name, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        """Performs the softmax.

        Args:
            inputs: Arbitrary tensor.

        Returns:
            Tensor: A tensor of the same shape as input.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)
        inputs = tf.cast(inputs, tf.float32)
        output = self.layer(inputs)
        output = tf.cast(output, self.compute_dtype)
        if self.tf_summary:
            output = summary_layer(output)
        return output
