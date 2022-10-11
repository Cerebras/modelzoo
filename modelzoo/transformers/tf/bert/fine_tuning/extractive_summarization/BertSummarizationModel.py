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

from modelzoo.common.tf.metrics.rouge_score import rouge_score_metric
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.bert.BertModel import BertModel
from modelzoo.transformers.tf.bert.fine_tuning.extractive_summarization.layers.MultipleCLSLayer import (
    MultipleCLSLayer,
)
from modelzoo.transformers.tf.bert.fine_tuning.extractive_summarization.layers.MultipleCLSLossLayer import (
    MultipleCLSLossLayer,
)
from modelzoo.transformers.tf.bert.fine_tuning.extractive_summarization.utils import (
    extract_text_words_given_cls_indices,
)


class BertSummarizationModel(TFBaseModel):
    """
    The BERTSUM model https://arxiv.org/abs/1903.10318.
    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):
        super(BertSummarizationModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        loss_weight = params["model"]["loss_weight"]
        enable_gpu_optimizations = params["model"]["enable_gpu_optimizations"]
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

        # Set up backbone layer (same for each CLS token).
        self.backbone = self._build_backbone(params)

        # Pool multiple CLS tokens.
        self.multiple_cls_layer = MultipleCLSLayer(
            output_size=1,  # binary classification.
            nonlinearity=None,
            use_bias=True,
            kernel_initializer=initializer,
            enable_gpu_optimizations=enable_gpu_optimizations,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="multiple_cls_layer",
        )

        # Get binary multiple CLS loss.
        self.multiple_cls_loss_layer = MultipleCLSLossLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

        # Store params
        self.train_batch_size = params["train_input"]["batch_size"]
        self.eval_batch_size = params["eval_input"]["batch_size"]
        self.loss_weight = loss_weight
        self.vocab_file = params["train_input"]["vocab_file"]
        self.vocab_size = params["train_input"]["vocab_size"]

    def _build_backbone(self, params):
        """
        Builds pretraining model through the encoder.
        Can be overwritten to finetune BERT variants.
        """
        return BertModel(params, encode_only=True)

    def build_model(self, features, mode):
        """
        Build model (up to loss).
        :param Dictionary features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL, PREDICT).
        :returns: Dictionary of CLS logits per each CLS token.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"The model supports estimator TRAIN, EVAL and PREDICT mode."

        encoder_outputs = self.backbone(features, mode)[2]
        # Result of shape [batch_size, max_cls_tokens, 1].
        logits = self.multiple_cls_layer(
            encoder_outputs, features["cls_indices"]
        )
        return logits

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.
        :param model_outputs: cls logits.
        :param features: Dictionary of input features.
        :param labels: Tensor of shape (batch_size, max_cls_tokens).
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: Total loss tensor.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
        ], f"Can only build loss in TRAIN, EVAL modes."

        total_loss = self.multiple_cls_loss_layer(
            cls_tokens_labels=labels,
            cls_token_weights=features["cls_weights"],
            logits=model_outputs,
            batch_size=self.train_batch_size
            if mode == tf.estimator.ModeKeys.TRAIN
            else self.eval_batch_size,
        )
        total_loss *= self.loss_weight

        self._write_summaries(total_loss)

        return total_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.backbone.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels, features):
        """
        :param model_outputs: cls logits.
        :param labels: Tensor of shape (batch_size, max_cls_tokens).
        :param features: Dictionary of input features.
        :returns: Dictionary of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        metrics_dict = {}
        probabilities = tf.math.sigmoid(tf.cast(model_outputs, tf.float32))
        predictions = tf.reshape(
            tf.cast(probabilities >= 0.5, dtype=tf.int32), labels.shape
        )

        # String types are not supported for copying between GPU and CPU
        # in tensorflow: https://github.com/TensorSpeech/TensorFlowASR/issues/71.
        with tf.device("cpu:0"):
            hypothesis = extract_text_words_given_cls_indices(
                tf.cast(predictions, tf.int32),
                features["cls_indices"],
                features["cls_weights"],
                features["input_ids"],
                self.vocab_file,
            )

            references = extract_text_words_given_cls_indices(
                labels,
                features["cls_indices"],
                features["cls_weights"],
                features["input_ids"],
                self.vocab_file,
            )

            # Measures quality on uni-grams.
            scores_1, update_op_1 = rouge_score_metric(
                hypothesis, references, max_n=1
            )

            # Measures quality on bi-grams.
            scores_2, update_op_2 = rouge_score_metric(
                hypothesis, references, max_n=2
            )

        for key, value in scores_1.items():
            metrics_dict[f"eval/rouge1-{key}"] = value, update_op_1

        for key, value in scores_2.items():
            metrics_dict[f"eval/rouge2-{key}"] = value, update_op_2

        return metrics_dict

    def _write_summaries(self, total_loss):
        """
        Write train metrics summaries.
        :param total_loss: total loss tensor.
        """
        total_loss = tf.cast(total_loss, tf.float32)
        tf.compat.v1.summary.scalar("train/total_loss", total_loss)
