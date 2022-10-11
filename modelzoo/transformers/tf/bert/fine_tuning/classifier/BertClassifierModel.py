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
from modelzoo.common.tf.metrics.f1_score import f1_score_metric
from modelzoo.common.tf.metrics.mcc import mcc_metric
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.bert.BertModel import BertModel
from modelzoo.transformers.tf.bert.layers.CLSLayer import CLSLayer


class BertClassifierModel(TFBaseModel):
    """
    The BERT model https://arxiv.org/pdf/1810.04805.pdf

    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):

        super(BertClassifierModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # cls layer params
        hidden_size = params["model"]["hidden_size"]
        cls_dropout_rate = params["model"]["cls_dropout_rate"]
        num_cls_classes = params["model"]["num_cls_classes"]

        # CS util params for layers
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]

        # Set up initializer
        initializer_spec = params["model"]["initializer"]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]
        initializer = create_initializer(
            initializer_spec, weight_initialization_seed
        )

        # Set up layers
        self.backbone = self._build_backbone(params)

        self.cls_layer = CLSLayer(
            hidden_size,
            num_cls_classes,
            nonlinearity="tanh",
            use_bias=True,
            dropout_rate=cls_dropout_rate,
            kernel_initializer=initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="cls_layer",
        )

        # Loss layer: CLS loss
        self.cls_loss_layer = CrossEntropyFromLogitsLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

        self.num_cls_classes = num_cls_classes

    def _build_backbone(self, params):
        """
        Builds pretraining model through the encoder.
        Can be overwritten to fine tune BERT variants.
        """
        return BertModel(params, encode_only=True)

    def build_model(self, features, mode):
        """
        Build the model (up to loss).

        :param dict features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL, PREDICT).
        :returns: cls logits
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"The model supports estimator TRAIN, EVAL, and PREDICT modes."

        encoder_outputs = self.backbone(features, mode)[2]
        training = mode == tf.estimator.ModeKeys.TRAIN
        cls_outputs = self.cls_layer(encoder_outputs, training=training)
        return cls_outputs

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.

        :param  model_outputs: cls logits
        :param features: Dictionary of input features.
        :param labels: Tensor of shape (batch_size,).
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: Total loss tensor.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
        ], f"The model supports only TRAIN and EVAL modes."

        cls_loss = self._cls_loss(labels, model_outputs)
        # reduction is needed so that estimator can know what to do when the
        # non-scalar loss arrives as an array.
        total_loss = tf.reduce_sum(input_tensor=cls_loss)

        self._write_summaries(total_loss)

        return total_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.backbone.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels, features):
        """
        Compute classification eval metrics.

        :param  model_outputs: cls logits
        :param labels: Tensor of shape (batch_size,).
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        metrics_dict = dict()
        predictions = tf.argmax(model_outputs, axis=-1, output_type=tf.int32)

        metrics_dict["eval/accuracy_cls"] = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=predictions
        )
        # add in matched/mismatched accuracies for MNLI
        if "is_matched" in features:
            metrics_dict[
                "eval/accuracy_matched"
            ] = tf.compat.v1.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                weights=features["is_matched"],
            )
            metrics_dict[
                "eval/accuracy_mismatched"
            ] = tf.compat.v1.metrics.accuracy(
                labels=labels,
                predictions=predictions,
                weights=1 - features["is_matched"],
            )

        if self.num_cls_classes == 2:
            metrics_dict["eval/f1_score"] = f1_score_metric(
                labels, predictions,
            )
            metrics_dict["eval/mcc"] = mcc_metric(labels, predictions)

        return metrics_dict

    def _cls_loss(self, labels, logits):
        """
        CLS loss.
        """
        loss = tf.reduce_mean(
            tf.cast(self.cls_loss_layer(labels, logits=logits), tf.float32),
            name='cls_loss',
        )
        return tf.cast(loss, logits.dtype)

    def _write_summaries(self, total_loss):
        """
        Write train metrics summaries

        :param total_loss: total loss tensor
        """

        # Use GradAccumSummarySaverHook to add
        # loss summaries when trained with
        # gradient accumulation
        if self.backbone.trainer.is_grad_accum():
            return

        total_loss = tf.cast(total_loss, tf.float32)
        tf.compat.v1.summary.scalar('train/cost_cls', total_loss)

    @property
    def trainer(self):
        return self.backbone.trainer
