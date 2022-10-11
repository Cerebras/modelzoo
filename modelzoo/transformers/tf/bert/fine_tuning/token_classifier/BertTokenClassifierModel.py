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
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.metrics.fbeta_score import fbeta_score_metric
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.data_processing.utils import get_label_id_map
from modelzoo.transformers.tf.bert.BertModel import BertModel


class BertTokenClassifierModel(TFBaseModel):
    """
    The BERT model https://arxiv.org/pdf/1810.04805.pdf
    Classifies each encoder output to one of num_classes.
    
    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):

        super(BertTokenClassifierModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # Batch size params
        self.train_batch_size = params["train_input"]["batch_size"]
        self.eval_batch_size = params["eval_input"]["batch_size"]

        # Token classification layer params
        encoder_output_dropout_rate = params["model"][
            "encoder_output_dropout_rate"
        ]
        dropout_seed = params["model"]["dropout_seed"]
        self.num_classes = params["model"]["num_classes"]

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
        # Get label_vocab
        self.label_map_id = get_label_id_map(
            params["model"]["label_vocab_file"]
        )

        # Ignore token labels in eval which dont
        # refer to a token beginning or inside.
        # Labels such as
        # "O", [CLS], [SEP], [PAD], "O", "X"
        # are ignored during eval
        self.eval_ignore_labels = []
        if self.label_map_id is not None:
            for key, label_id in self.label_map_id.items():
                if not (key.startswith("B") or key.startswith("I")):
                    self.eval_ignore_labels.append(label_id)

        # Set up layers
        self.backbone = self._build_backbone(params)

        self.encoder_output_dropout_layer = DropoutLayer(
            encoder_output_dropout_rate,
            seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="encoder_output_dropout",
        )

        self.dense_layer = DenseLayer(
            self.num_classes,
            activation=None,
            use_bias=True,
            kernel_initializer=initializer,
            bias_initializer='zeros',
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="output_logits",
        )

        self.include_padding_in_loss = params["model"][
            "include_padding_in_loss"
        ]
        self.loss_weight = params["model"]["loss_weight"]

        self.loss_layer = CrossEntropyFromLogitsLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

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
        :returns: tokens logits
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"The model supports estimator TRAIN, EVAL, and PREDICT modes."

        # BertModel -> Dropout -> Dense(shape:[hidden_size, num_classes]) -> CrossEntropyLoss
        encoder_outputs = self.backbone(features, mode)[2]

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        dropout_outputs = self.encoder_output_dropout_layer(
            encoder_outputs, training=is_training
        )

        # logits shape = [bsz, max_seq_len, num_classes]
        logits = self.dense_layer(dropout_outputs)

        return logits

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
        ], f"Can build loss value only in TRAIN and EVAL modes."

        batch_size = (
            self.train_batch_size
            if mode == tf.estimator.ModeKeys.TRAIN
            else self.eval_batch_size
        )
        tokens_loss = self._tokens_loss(
            labels,
            model_outputs,
            batch_size,
            input_mask=None
            if self.include_padding_in_loss
            else features["input_mask"],
        )
        # reduction is needed so that estimator can know what to do when the
        # non-scalar loss arrives as an array.
        total_loss = tf.reduce_sum(input_tensor=tokens_loss)

        self._write_summaries(total_loss)

        return total_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.backbone.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels, features):
        """
        Compute BERT eval metrics

        :param  model_outputs: cls logits
        :param labels: Tensor of shape (batch_size,).
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        predictions = tf.argmax(model_outputs, axis=-1, output_type=tf.int32)
        metrics_dict = dict()

        # Macro F1 score - compute f1 scores of each class
        # interested and take mean of them
        metrics_dict["eval/token_level_macro_f1"] = fbeta_score_metric(
            labels,
            predictions,
            num_classes=self.num_classes,
            ignore_labels=self.eval_ignore_labels,
            beta=1,
            average_type="macro",
        )
        return metrics_dict

    def _tokens_loss(self, labels, model_outputs, batch_size, input_mask=None):
        """
        Tokens loss. 
        Mean of Cross Entropy Loss computed for each token in sequence.
        :params labels: Tensor of shape [bsz, max_seq_len] representing actual labels.
        :params model_outputs: Tensor of shape [bsz, max_seq_len, num_classes] representing logits 
            produced by the model.
        :params batch_size: integer number for batch size.
        :params input_mask: Tensor of shape [bsz, max_seq_len]. Contains
            `1` at tokens to be ignored during loss computation and
            `0` at tokens to be considered during loss computation.
        """

        # per_token_loss shape = [bsz, max_seq_len]
        per_token_loss = self.loss_layer(labels, model_outputs)

        if input_mask is not None:
            input_mask = 1.0 - tf.cast(input_mask, model_outputs.dtype)
            per_token_loss = per_token_loss * tf.cast(
                input_mask, per_token_loss.dtype
            )
            total_loss = tf.reduce_sum(per_token_loss, name="tokens_loss")
            total_loss *= self.loss_weight
            total_loss = total_loss / tf.cast(batch_size, model_outputs.dtype)
        else:
            total_loss = tf.reduce_mean(per_token_loss, name="tokens_loss")

        return total_loss

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
