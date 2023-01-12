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


class CrossEntropyFromLogitsLayer(BaseLayer):
    """Cross entropy loss, given logits. Compares logits against labels.

    Args:
        boundary_casting (bool):
        tf_summary (bool):
    """

    def __init__(self, boundary_casting=False, tf_summary=False, **kwargs):
        super(CrossEntropyFromLogitsLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

    def call(self, labels, logits):
        """Calculating cross entropy over logits.

        Args:
            labels (Tensor): Label indices.
            logits (Tensor): Logits (non-normalized).

        Returns:
            Tensor: A tensor of the same shape as labels and of the same type
            as logits with the softmax cross entropy loss.
        """

        if self.boundary_casting:
            logits = boundary_cast(logits)
        if logits.shape[-1] == 1:
            raise ValueError(
                f"Last dimension of `logits` in `CrossEntropyFromLogitsLayer` "
                f"should be > 1. It was {logits.shape[-1]} instead."
            )
        crossent = tf.cast(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=tf.cast(logits, tf.float32)
            ),
            self.compute_dtype,
        )
        if self.tf_summary:
            crossent = summary_layer(crossent)
        return crossent
