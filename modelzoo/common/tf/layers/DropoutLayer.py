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

from tensorflow.keras.layers import Dropout

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class DropoutLayer(BaseLayer):
    """Wrapper around the Keras dropout layer.
    """

    def __init__(
        self,
        rate,
        noise_shape=None,
        seed=None,
        boundary_casting=False,
        tf_summary=False,
        **kwargs,
    ):
        super(DropoutLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.layer = Dropout(
            rate, noise_shape, seed, name=self.name, dtype=self.dtype_policy
        )

    def call(self, inputs, training=True, **kwargs):
        """Performs the dropout.

        Args:
            inputs (Tensor): Arbitrary tensor.
            training (bool): Training mode if set to ``True``.

        Returns:
            Tensor: A tensor of same shape as input.
        """

        if self.boundary_casting:
            inputs = boundary_cast(inputs)
        output = self.layer(inputs, training=training)
        if self.tf_summary:
            output = summary_layer(output)
        return output
