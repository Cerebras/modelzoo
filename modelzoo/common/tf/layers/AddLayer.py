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

from tensorflow.keras.layers import Add

from modelzoo.common.tf.layers.BaseLayer import BaseLayer
from modelzoo.common.tf.layers.utils import boundary_cast, summary_layer


class AddLayer(BaseLayer):
    """Wrapper around the Keras layer. Adds a list of inputs.
    """

    def __init__(self, boundary_casting=False, tf_summary=False, **kwargs):
        super(AddLayer, self).__init__(boundary_casting, tf_summary, **kwargs)
        self.layer = Add(name=self.name, dtype=self.dtype_policy)

    def call(self, inputs, **kwargs):
        """Apply the ``AddLayer`` to sum up a list of inputs.

        Args:
            inputs: List of input tensors (at least 2).
        Returns:
            Tensor: A tensor containing the sum of inputs.
        """
        if self.boundary_casting:
            inputs = [boundary_cast(i) for i in inputs]
        output = self.layer(inputs)
        if self.tf_summary:
            output = summary_layer(output)
        return output
