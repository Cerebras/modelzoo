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
from modelzoo.transformers.tf.bert.layers.MLMLossLayer import MLMLossLayer


class MultipleCLSLossLayer(BaseLayer):
    """
    Multiple CLS loss layer.
    Binary cross entropy over all CLS tokens is used for
    loss function.
    :param bool boundary_casting: See documentation for ``BaseLayer``.
    :param bool tf_summary: See documentation for ``BaseLayer``.
    """

    def __init__(
        self, boundary_casting=False, tf_summary=False, **kwargs,
    ):
        super(MultipleCLSLossLayer, self).__init__(
            boundary_casting, tf_summary, **kwargs
        )

        self.mlm_loss_layer = MLMLossLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.dtype_policy,
        )

    def call(
        self, cls_tokens_labels, cls_token_weights, logits, batch_size=None
    ):
        """
        Binary Multiple CLS loss.

        :param Tensor cls_tokens_labels: The target tokens for the CLS
            positions of shape [batch_size, max_cls_tokens].
            Might be zero-padded (if the sequence is too short to
            have the maximum number of predictions).
        :param Tensor cls_token_weights: The `cls_token_weights` tensor
            of shape [batch_size, max_cls_tokens].
            Has a value of 1.0 for every real prediction
            and 0.0 for the padding predictions.
        :param Tensor logits: The logits tensor of shape
            [batch_size, max_cls_tokens, vocab_size].
        :param int batch_size: for scaling the loss
        :returns: The Multiple CLS loss scalar tensor.
        """
        # We need to obtain extra column with zeros in order to pass it
        # to softmax layer.
        logits = tf.concat(
            (tf.zeros(logits.shape, dtype=logits.dtype), logits), -1,
        )
        return self.mlm_loss_layer(
            masked_lm_ids=cls_tokens_labels,
            masked_lm_weights=cls_token_weights,
            logits=logits,
            batch_size=batch_size,
        )
