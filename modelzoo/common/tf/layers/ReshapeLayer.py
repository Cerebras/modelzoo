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

from tensorflow.keras.layers import Reshape

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class ReshapeLayer(BaseLayer):
    """Wrapper around the Keras layer that reshapes the input.
    """

    def __init__(
        self, target_shape, boundary_casting=False, tf_summary=False, **kwargs
    ):
        super(ReshapeLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.layer = Reshape(
            target_shape, dtype=self.dtype_policy, name=self.name
        )

    def call(self, input, **kwargs):
        """Apply the reshape layer to an input.

        Args:
            inputs (Tensor): A tensor.

        Returns:
            Tensor: The tensor after reshape.
        """

        if self.boundary_casting:
            input = boundary_cast(input)
        output = self.layer(input)
        if self.tf_summary:
            output = summary_layer(output)
        return output
