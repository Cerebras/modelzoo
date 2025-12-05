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

from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.EmbeddingLayer import EmbeddingLayer
from modelzoo.common.tf.layers.Input import SetupInputTensor
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.layers.PositionEmbeddingLayer import (
    PositionEmbeddingLayer,
)
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.bert.layers.MLMLayer import MLMLayer
from modelzoo.transformers.tf.bert.layers.MLMLossLayer import MLMLossLayer
from modelzoo.transformers.tf.layers.Encoder import Encoder


class GenomicBertModel(TFBaseModel):
    """
    The genomic BERT model.

    :param dict params: Model configuration parameters.
    """

    def __init__(self, params, encode_only=False):

        super(GenomicBertModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # Position embedding params
        use_position_embedding = params["model"]["use_position_embedding"]
        position_embedding_type = params["model"]["position_embedding_type"]

        # Encoder params
        vocab_size_dna = params["train_input"]["vocab_size_dna"]
        vocab_size_ideas = params["train_input"]["vocab_size_ideas"]
        hidden_size = params["model"]["hidden_size"]
        num_heads = params["model"]["num_heads"]
        num_hidden_layers = params["model"]["num_hidden_layers"]
        filter_size = params["model"]["filter_size"]
        encoder_nonlinearity = params["model"]["encoder_nonlinearity"]
        dropout_rate = params["model"]["dropout_rate"]

        attention_type = params["model"]["attention_type"]
        attention_dropout_rate = params["model"]["attention_dropout_rate"]
        use_projection_bias_in_attention = params["model"][
            "use_projection_bias_in_attention"
        ]
        use_ffn_bias_in_attention = params["model"]["use_ffn_bias_in_attention"]
        use_ffn_bias = params["model"]["use_ffn_bias"]

        # Task-specific layer params. Assume that
        # both MLM layers have identical parametrization.
        use_ffn_bias_in_mlm = params["model"]["use_ffn_bias_in_mlm"]
        use_output_bias_in_mlm = params["model"]["use_output_bias_in_mlm"]
        mlm_nonlinearity = params["model"]["mlm_nonlinearity"]

        mlm_loss_weight = params["model"]["mlm_loss_weight"]
        batch_size = params["train_input"]["batch_size"]
        layer_norm_epsilon = params["model"]["layer_norm_epsilon"]
        disable_layer_norm = params["model"]["disable_layer_norm"]

        # CS util params for layers
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]

        # Setup layers
        initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        self.embedding_layer_dna = EmbeddingLayer(
            input_dim=vocab_size_dna,
            output_dim=hidden_size,
            embeddings_initializer=initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="input_embedding_dna",
        )

        self.embedding_layer_ideas = EmbeddingLayer(
            input_dim=vocab_size_ideas,
            output_dim=hidden_size,
            embeddings_initializer=initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="input_embedding_ideas",
        )

        self.position_embedding = PositionEmbeddingLayer(
            embedding_type=position_embedding_type,
            embeddings_initializer=initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="position_embedding",
        )

        self.layer_norm = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="embedding_layer_norm",
        )

        self.dropout_layer = DropoutLayer(
            dropout_rate,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="embedding_dropout",
        )

        self.encoder = Encoder(
            hidden_size,
            num_heads,
            num_hidden_layers,
            filter_size,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=initializer,
            ffn_initializer=initializer,
            weight_regularizer=None,
            attention_type=attention_type,
            attention_dropout_rate=attention_dropout_rate,
            nonlinearity=encoder_nonlinearity,
            dropout_rate=dropout_rate,
            disable_layer_norm=disable_layer_norm,
            layer_norm_epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="encoder",
        )

        self.mlm_layer_dna = MLMLayer(
            hidden_size,
            vocab_size_dna,
            nonlinearity=mlm_nonlinearity,
            use_ffn_bias=use_ffn_bias_in_mlm,
            use_output_bias=use_output_bias_in_mlm,
            kernel_initializer=initializer,
            disable_layer_norm=disable_layer_norm,
            layer_norm_epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="mlm_layer_dna",
        )

        self.mlm_layer_ideas = MLMLayer(
            hidden_size,
            vocab_size_ideas,
            nonlinearity=mlm_nonlinearity,
            use_ffn_bias=use_ffn_bias_in_mlm,
            use_output_bias=use_output_bias_in_mlm,
            kernel_initializer=initializer,
            disable_layer_norm=disable_layer_norm,
            layer_norm_epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="mlm_layer_ideas",
        )

        self.mlm_loss_layer_dna = MLMLossLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

        self.mlm_loss_layer_ideas = MLMLossLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

        # Model trainer
        self.trainer = Trainer(
            params=params["optimizer"],
            tf_summary=tf_summary,
            mixed_precision=params["model"]["mixed_precision"],
        )

        # Store params
        self.encode_only = encode_only
        self.use_position_embedding = use_position_embedding
        self.mlm_loss_weight = mlm_loss_weight
        self.batch_size = batch_size
        self.disable_layer_norm = disable_layer_norm
        self.input_pad_id = params["model"]["input_pad_id"]
        # remove after SW-18721 resolved
        self.batch_size = batch_size
        self.tf_summary = tf_summary

    def build_model(self, features, mode):
        """
        Build the model (up to loss).

        :param dict features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns list: List of model outputs, where the 0th
            entry is the mlm-dna output, and 1st entry
            is the mlm-ideas output.
        """
        inputs_dna = SetupInputTensor(
            features["input_ids_dna"], tf_summary=self.tf_summary
        )
        inputs_ideas = SetupInputTensor(
            features["input_ids_ideas"], tf_summary=self.tf_summary
        )

        # Combine DNA and ideas embeddings
        embedded_inputs_dna = self.embedding_layer_dna(
            inputs_dna, pad_id=self.input_pad_id
        )
        embedded_inputs_ideas = self.embedding_layer_ideas(
            inputs_ideas, pad_id=self.input_pad_id
        )
        encoder_inputs = embedded_inputs_dna + embedded_inputs_ideas
        if self.use_position_embedding:
            encoder_inputs = self.position_embedding(encoder_inputs)

        if not self.disable_layer_norm:
            encoder_inputs = self.layer_norm(encoder_inputs)

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        encoder_inputs = self.dropout_layer(
            encoder_inputs, training=is_training
        )

        # Run through encoder
        encoder_outputs = self.encoder(
            encoder_inputs,
            self_attention_mask=features["input_mask"],
            training=is_training,
        )

        # The DNA and ideas MLM heads
        mlm_outputs_dna = (
            self.mlm_layer_dna(
                encoder_outputs,
                masked_lm_positions=features["masked_lm_positions"],
                embedding_table=self.embedding_layer_dna.embedding_table(),
            )
            if not self.encode_only
            else None
        )

        mlm_outputs_ideas = (
            self.mlm_layer_ideas(
                encoder_outputs,
                masked_lm_positions=features["masked_lm_positions"],
                embedding_table=self.embedding_layer_ideas.embedding_table(),
            )
            if not self.encode_only
            else None
        )

        return [mlm_outputs_dna, mlm_outputs_ideas, encoder_outputs]

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.

        :param  model_outputs: list containing model outputs
        :param features: Dictionary of input features.
        :param labels: Tensor of shape (batch_size,).
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
            remove once SW-18721 is resolved
        :returns: Total loss tensor.
        """
        assert self.mlm_loss_weight >= 0.0 and self.mlm_loss_weight <= 1.0, (
            "The value of mlm_loss_weight should be in [0, 1]."
            + f"Received {self.mlm_loss_weight}."
        )

        assert (
            not self.encode_only
        ), "To train GenomicBertModel, encode_only must be False."

        mlm_loss_dna = (
            self.mlm_loss_layer_dna(
                features["masked_lm_ids_dna"],
                features["masked_lm_weights_dna"],
                model_outputs[0],
                self.batch_size,
            )
            * self.mlm_loss_weight
        )

        mlm_loss_ideas = self.mlm_loss_layer_ideas(
            features["masked_lm_ids_ideas"],
            features["masked_lm_weights_ideas"],
            model_outputs[1],
            self.batch_size,
        ) * (1.0 - self.mlm_loss_weight)

        total_loss = mlm_loss_dna + mlm_loss_ideas
        # reduction is needed so that estimator can know what to do when the
        # non-scalar loss arrives as an array.
        total_loss = tf.reduce_sum(input_tensor=total_loss)

        self._write_summaries(
            features, total_loss, mlm_loss_dna, mlm_loss_ideas,
        )

        return total_loss

    def _write_summaries(
        self, features, total_loss, mlm_loss_dna, mlm_loss_ideas
    ):
        """
        Write train metrics summaries

        :param features: Dictionary of input features.
        :param total_loss: total loss tensor
        :param mlm_loss_dna: mlm dna loss tensor
        :param mlm_loss_ideas: mlm ideas loss tensor

        __Note__: Currently tf.reduce_sum is the only reduction we support for
        summaries in this model. We are working on supporting other reductions
        in an upcoming release. Also, note that the values for the summaries
        here are scaled by the batch size.

        """
        tf.compat.v1.summary.scalar('train/total_loss', total_loss)
        tf.compat.v1.summary.scalar(
            'train/loss_mlm_dna',
            tf.reduce_sum(mlm_loss_dna, name="loss_mlm_dna",),
        )
        tf.compat.v1.summary.scalar(
            'train/loss_mlm_ideas',
            tf.reduce_sum(mlm_loss_ideas, name="loss_mlm_ideas",),
        )

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.trainer.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, features, labels):
        """
        MLM accuracies for DNA and IDEAS predictions.
        """
        assert (
            not self.encode_only
        ), "To evaluate GenomicBertModel, encode_only must be False."

        mlm_logits_dna, mlm_logits_ideas = model_outputs
        # We use a temporary hack where we transpose
        # logits before passing to the eval call.
        # Hence we need to transpose back here.
        mlm_logits_dna = tf.transpose(mlm_logits_dna, perm=[0, 2, 1])
        mlm_logits_ideas = tf.transpose(mlm_logits_ideas, perm=[0, 2, 1])

        metrics_dict = dict()

        mlm_pred_dna = tf.argmax(mlm_logits_dna, axis=-1)
        weights_dna = tf.where(
            tf.cast(features["masked_lm_weights_dna"], tf.bool), 1, 0
        )
        metrics_dict[
            "eval/accuracy_masked_lm_dna"
        ] = tf.compat.v1.metrics.accuracy(
            labels=features["masked_lm_ids_dna"],
            predictions=mlm_pred_dna,
            weights=weights_dna,
        )

        mlm_pred_ideas = tf.argmax(mlm_logits_ideas, axis=-1)
        weights_ideas = tf.where(
            tf.cast(features["masked_lm_weights_ideas"], tf.bool), 1, 0
        )
        metrics_dict[
            "eval/accuracy_masked_lm_ideas"
        ] = tf.compat.v1.metrics.accuracy(
            labels=features["masked_lm_ids_ideas"],
            predictions=mlm_pred_ideas,
            weights=weights_ideas,
        )

        return metrics_dict
