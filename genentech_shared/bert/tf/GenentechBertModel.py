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

from modelzoo.common.tf.metrics.ece_loss_metric import ece_loss_metric
from modelzoo.transformers.tf.bert.BertModel import BertModel


class GenentechBertModel(BertModel):
    """
    The Genentech BERT model https://arxiv.org/abs/1906.08230.
    Differs from Bert model only in eval metrics computations.

    :param dict params: Model configuration parameters.
    """

    def __init__(self, params, encode_only=False):

        super(GenentechBertModel, self).__init__(
            params=params, encode_only=encode_only
        )
        self.encode_only = encode_only

    def build_eval_metric_ops(self, eval_metric_inputs, labels, features):
        """
        Compute Genentech BERT eval metrics which compared to BERT
        add ECE loss calculation (exponential cross-entropy):
            -- formula: tf.exp(loss).

        :param eval_metric_inputs: tuple containing:
                -- `nsp_output`: NSP branch output returned by call method
                    if `disable_nsp` is True, otherwise set to None;
                -- `mlm_pred`: MLM prediction tensor;
                -- `mlm_xentr` MLM cross entropy tensor.
        :param labels: Tensor of shape (batch_size,). Contains
                       next sentence labels used for bert pretraining.
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        assert (
            not self.encode_only
        ), "To evaluate GenentechBertModel, encode_only must be False"

        metrics_dict = super().build_eval_metric_ops(
            eval_metric_inputs, labels, features
        )

        nsp_output, mlm_pred, mlm_xentr = eval_metric_inputs

        # We avoid using mlm_loss_layer here in order to sum MLM Loss in FP32,
        # which helps prevent overflow issues.

        weights = tf.where(
            tf.cast(features["masked_lm_weights"], tf.bool), 1, 0
        )
        unscaled_mlm_loss = tf.reduce_sum(
            input_tensor=tf.cast(
                mlm_xentr * tf.cast(weights, mlm_xentr.dtype), tf.float32,
            )
        )
        num_masked = tf.reduce_sum(weights)
        metrics_dict["eval/mlm_ece"] = ece_loss_metric(
            unscaled_mlm_loss, num_masked
        )
        return metrics_dict
