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
from genentech_shared.bert.tf.fine_tuning.regression.metrics.pearson_correlation import (
    pearson_correlation_metric,
)

from modelzoo.common.tf.layers.SquaredErrorLayer import SquaredErrorLayer
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.bert.BertModel import BertModel
from modelzoo.transformers.tf.bert.layers.CLSLayer import CLSLayer


class BertRegressionModel(TFBaseModel):
    """
    The BERT model https://arxiv.org/pdf/1810.04805.pdf
    Here, `CLSLayer` refers to the layer built on the CLS token.

    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):

        super(BertRegressionModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # cls layer params
        pooler_type = params["model"]["pooler_type"]
        hidden_size = params["model"]["hidden_size"]
        use_cls_bias = params["model"]["use_cls_bias"]
        cls_nonlinearity = params["model"]["cls_nonlinearity"]
        cls_dropout_rate = params["model"]["cls_dropout_rate"]
        dropout_seed = params["model"]["dropout_seed"]
        num_regression_values = params["model"]["num_regression_values"]

        # CS util params for layers
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]

        # Weight initialization params
        initializer_spec = params["model"]["initializer"]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]

        # Set up initializer
        initializer = create_initializer(
            initializer_spec, weight_initialization_seed
        )

        # Set up layers
        self.bert = BertModel(params, encode_only=True)

        self.cls_layer = CLSLayer(
            hidden_size,
            num_regression_values,
            pooler_type=pooler_type,
            nonlinearity=cls_nonlinearity,
            use_bias=use_cls_bias,
            dropout_rate=cls_dropout_rate,
            dropout_seed=dropout_seed,
            kernel_initializer=initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="cls_layer",
        )

        # Loss layer: Squared Error
        self.squared_error_loss_layer = SquaredErrorLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=tf.float32,
        )

    def build_model(self, features, mode):
        """
        Build the model (up to loss).

        :param dict features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL, PREDICT).
        :returns: preds (outputs of CLS layer)
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"The model supports estimator TRAIN, EVAL, and PREDICT modes."

        encoder_outputs = self.bert(features, mode)[2]
        training = mode == tf.estimator.ModeKeys.TRAIN
        preds = self.cls_layer(encoder_outputs, training=training)
        return preds

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.

        :param  model_outputs: preds
        :param features: Dictionary of input features.
        :param labels: Tensor of shape (batch_size,).
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: Total loss tensor.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
        ], f"The model supports only TRAIN and EVAL modes."

        squared_error_loss = self._squared_error_loss(labels, model_outputs)
        # Reduction is needed so that estimator can know what to do when the
        # non-scalar loss arrives as an array.
        total_loss = tf.reduce_sum(input_tensor=squared_error_loss)

        self._write_summaries(total_loss)

        return total_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.bert.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels, features):
        """
        Compute regression eval metrics.

        :param  model_outputs: predictions
        :param labels: Tensor of shape (batch_size,).
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        metrics_dict = {}

        metrics_dict["eval/pearson_correlation"] = pearson_correlation_metric(
            labels, model_outputs
        )

        return metrics_dict

    def _squared_error_loss(self, labels, preds):
        """
        squared error loss.
        """
        # Casting labels to fp16 is required by a supporting kernel
        labels_fp16 = tf.cast(labels, tf.float16)
        loss = tf.reduce_mean(
            tf.cast(
                self.squared_error_loss_layer(labels_fp16, preds), tf.float32
            ),
            name="squared_error_loss",
        )
        return tf.cast(loss, preds.dtype)

    def _write_summaries(self, total_loss):
        """
        Write train metrics summaries

        :param total_loss: total loss tensor
        """

        # Use `GradAccumSummarySaverHook` to add loss summaries when training
        # with gradient accumulation.
        if self.bert.trainer.is_grad_accum():
            return

        total_loss = tf.cast(total_loss, tf.float32)
        tf.compat.v1.summary.scalar("train/cost_squared_error", total_loss)

    @property
    def trainer(self):
        return self.bert.trainer
