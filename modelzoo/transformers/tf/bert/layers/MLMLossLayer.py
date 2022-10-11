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
from modelzoo.common.tf.layers.CrossEntropyFromLogitsLayer import (
    CrossEntropyFromLogitsLayer,
)
from modelzoo.common.tf.run_utils import ExecutionMode, get_execution_mode


class MLMLossLayer(BaseLayer):
    """
    MLM loss layer

    :param bool boundary_casting: See documentation for ``BaseLayer``
    :param bool tf_summary: See documentation for ``BaseLayer``
    """

    def __init__(
        self, boundary_casting=False, tf_summary=False, **kwargs,
    ):
        super(MLMLossLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )
        self.softmax_ce_layer = CrossEntropyFromLogitsLayer(
            dtype=self.dtype_policy
        )

    def call(self, masked_lm_ids, masked_lm_weights, logits, batch_size=None):
        """
        MLM loss. Based on
        https://github.com/google-research/bert/blob/master/run_pretraining.py

        :param Tensor masked_lm_ids: The target tokens for the masked
            positions of shape [batch_size, max_predictions_per_seq].
            Might be zero-padded (if the sequence is too short to
            have the maximum number of predictions).
        :param Tensor masked_lm_weights: The `label_weights` tensor
            of shape [batch_size, max_predictions_per_seq].
            Has a value of 0.0 for the padding predictions and arbitrary
            non-zero values for real predictions.
        :param Tensor logits: The logits tensor of shape
            [batch_size, max_predictions_per_seq, vocab_size].
        :param int batch_size: for scaling the loss
        :returns: The MLM loss scalar tensor.
        """
        per_example_loss = self.softmax_ce_layer(masked_lm_ids, logits=logits)
        if get_execution_mode() == ExecutionMode.WeightStreaming:
            label_weights = tf.cast(masked_lm_weights, tf.float32)
            per_example_loss = tf.cast(per_example_loss, tf.float32)
        else:
            label_weights = tf.cast(masked_lm_weights, logits.dtype)
        masked_per_ex_loss = tf.cast(
            label_weights * per_example_loss, tf.float32
        )
        loss = tf.reduce_sum(input_tensor=masked_per_ex_loss)
        return tf.cast(loss / batch_size, logits.dtype)
