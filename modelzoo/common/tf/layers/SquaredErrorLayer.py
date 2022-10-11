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


class SquaredErrorLayer(BaseLayer):
    """Squared error between prediction and labels.

    Args:
        boundary_casting (bool): If ``True``, outputs the values in half
            precision and casts the input values up to full precision.
        tf_summary (bool): If ``True``, saves the activations with
            ``summary_layer``.
    """

    def __init__(self, boundary_casting=False, tf_summary=False, **kwargs):
        super(SquaredErrorLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

    def call(self, labels, pred):
        """Calculates the squared error between prediction and labels.

        Args:
            labels (Tensor): Labels.
            pred (Tensor): Predictions (same shape as labels).

        Returns:
            Tensor: Loss tensor of the same shape and type as ``pred``.
        """

        if self.boundary_casting:
            pred = boundary_cast(pred)
        loss = tf.cast(
            tf.square(labels - tf.cast(pred, tf.float32)), self.compute_dtype,
        )
        if self.tf_summary:
            loss = summary_layer(loss)
        return loss
