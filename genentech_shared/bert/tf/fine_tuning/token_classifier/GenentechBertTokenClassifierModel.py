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

from modelzoo.transformers.tf.bert.fine_tuning.token_classifier.BertTokenClassifierModel import (
    BertTokenClassifierModel,
)


class GenentechBertTokenClassifierModel(BertTokenClassifierModel):
    """
    The Genentech BERT Token Classifier model https://arxiv.org/abs/1906.08230.
    Differs from Bert Token Classifier model only in eval metrics computations.

    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):

        super(GenentechBertTokenClassifierModel, self).__init__(params=params)

    def build_eval_metric_ops(self, model_outputs, labels, features):
        """
        Compute BERT Token Classifier eval metrics which compared to BERT
        calculate token level accuracy only.

        :param  model_outputs: cls logits
        :param labels: Tensor of shape (batch_size,).
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        predictions = tf.argmax(model_outputs, axis=-1, output_type=tf.int32)
        metrics_dict = {}
        metrics_dict[
            "eval/token_level_accuracy"
        ] = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=predictions
        )
        return metrics_dict
