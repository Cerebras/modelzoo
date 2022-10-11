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

from modelzoo.common.tf.layers.CrossEntropyFromLogitsLayer import (
    CrossEntropyFromLogitsLayer,
)
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.bert.BertModel import BertModel


class BertQuestionAnsweringModel(TFBaseModel):
    """
    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):

        super(BertQuestionAnsweringModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # CS util params for layers.
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]

        # Weight initialization params.
        initializer_spec = params["model"]["initializer"]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]

        # Set up initializer.
        initializer = create_initializer(
            initializer_spec, weight_initialization_seed
        )

        # Set up layers.
        self.bert = BertModel(params, encode_only=True)

        self.qa_dense_layer = DenseLayer(
            units=2,
            kernel_initializer=initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="qa_logits",
        )

        self.loss_layer = CrossEntropyFromLogitsLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

    def build_model(self, features, mode):
        """
        Build the model (up to loss).

        :param dict features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: qa logits (start_logits, end_logits)
        """

        encoder_outputs = self.bert(features, mode)[2]
        logits = self.qa_dense_layer(encoder_outputs)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        return start_logits, end_logits

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.

        :param  model_outputs: qa logits (start_logits, end_logits)
        :param features: Dictionary of input features.
        :param labels: Tensor of shape (batch_size,).
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: Total loss tensor.
        """

        start_logits, end_logits = model_outputs
        start_labels = labels[:, 0]
        end_labels = labels[:, 1]
        start_loss = self._loss(start_labels, start_logits, "start_loss")
        end_loss = self._loss(end_labels, end_logits, "end_loss")
        qa_loss = start_loss + end_loss

        total_loss = tf.reduce_mean(qa_loss)

        self._write_summaries(total_loss)

        return total_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.bert.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels, features):
        """
        Compute eval metrics (of which there are none for now).

        :param  model_outputs: qa logits (start_logits, end_logits)
        :param labels: Tensor of shape (batch_size,).
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        return dict()

    def _loss(self, labels, logits, name):
        """
        Cross Entropy loss.
        """
        return tf.reduce_mean(self.loss_layer(labels, logits=logits), name=name)

    def _write_summaries(self, total_loss):
        """
        Write train metrics summaries

        :param total_loss: total loss tensor
        """

        # Use GradAccumSummarySaverHook to add
        # loss summaries when trained with
        # gradient accumulation
        if self.bert.trainer.is_grad_accum():
            return

        total_loss = tf.cast(total_loss, tf.float32)
        tf.compat.v1.summary.scalar('train/cost_qa', total_loss)

    @property
    def trainer(self):
        return self.bert.trainer
