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

# Copyright 2021 Cerebras Systems.
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

import math

import tensorflow as tf
from tensorflow.compat.v1.layers import flatten

from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.EmbeddingLayer import EmbeddingLayer
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.layers.Decoder import Decoder
from modelzoo.transformers.tf.layers.Encoder import Encoder


class Spacetimeformer(TFBaseModel):
    """
    Spacetimeformer model based on
    https://arxiv.org/pdf/2109.12218
    """

    def __init__(self, params):
        """ Make a transformer object. """
        super(Spacetimeformer, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # Embeddings params.
        self._num_vars = params["train_input"]["num_vars"]
        self._target_length = params["train_input"]["target_length"]
        self._context_length = params["train_input"]["context_length"]

        # Encoder and Decoder params.
        hidden_size = params["model"]["hidden_size"]
        num_heads = params["model"]["num_heads"]  # number of attn heads.
        encoder_num_hidden_layers = params["model"][
            "encoder_num_hidden_layers"
        ]  # Number of identical layers(Attn-> Norm-> FFN) in encoder stack.
        decoder_num_hidden_layers = params["model"]["decoder_num_hidden_layers"]
        filter_size = params["model"][
            "filter_size"
        ]  # num output units for FFN-1 (referring to FFN-1->non-linearity->FFN-2).
        encoder_nonlinearity = params["model"][
            "encoder_nonlinearity"
        ]  # non-linearity after FFN-1.
        decoder_nonlinearity = params["model"]["decoder_nonlinearity"]
        dropout_rate = params["model"][
            "dropout_rate"
        ]  # Embedding_dropout + Encoder/Decoder dropout layers.
        dropout_seed = params["model"][
            "dropout_seed"
        ]  # Seed for all dropout layers.

        attention_dropout_rate = params["model"][
            "attention_dropout_rate"
        ]  # dropout for attention context matrix.
        layer_norm_epsilon = params["model"]["layer_norm_epsilon"]
        use_ffn_bias = params["model"]["use_ffn_bias"]
        use_pre_normalization = params["model"]["use_pre_normalization"]

        # CS util params for layers.
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]
        mixed_precision = params["model"]["mixed_precision"]

        # Weight initialization params.
        embedding_initializer_spec = params["model"]["embedding_initializer"]
        embedding_projection_initializer_spec = params["model"][
            "embedding_projection_initializer"
        ]
        attention_initializer_spec = params["model"]["attention_initializer"]
        feed_forward_initializer_spec = params["model"][
            "feed_forward_initializer"
        ]
        output_projection_initializer_spec = params["model"][
            "output_projection_initializer"
        ]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]

        # Set up initializers.
        embedding_initializer = create_initializer(
            embedding_initializer_spec, weight_initialization_seed
        )
        embedding_projection_initializer = create_initializer(
            embedding_projection_initializer_spec, weight_initialization_seed
        )
        attention_initializer = create_initializer(
            attention_initializer_spec, weight_initialization_seed
        )
        feed_forward_initializer = create_initializer(
            feed_forward_initializer_spec, weight_initialization_seed
        )
        output_projection_initializer = create_initializer(
            output_projection_initializer_spec, weight_initialization_seed
        )

        # Set up layers.
        # Embedding layers.
        (
            self.val_time_proj_enc,
            self.val_time_proj_dec,
            self.var_emb_enc,
            self.givens_emb_enc,
            self.var_emb_dec,
            self.givens_emb_dec,
        ) = self._create_embedding_layers(
            num_vars=self._num_vars,
            hidden_size=hidden_size,
            embedding_initializer=embedding_initializer,
            embedding_projection_initializer=embedding_projection_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

        self.pre_encoder_layer_norm = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="pre_encoder_layer_norm",
        )

        self.pre_decoder_layer_norm = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="pre_decoder_layer_norm",
        )

        self.pre_encoder_dropout_layer = DropoutLayer(
            dropout_rate,
            seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="pre_encoder_embedding_dropout",
        )

        self.pre_decoder_dropout_layer = DropoutLayer(
            dropout_rate,
            seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="pre_decoder_embedding_dropout",
        )

        # Encoder Stack.
        self.encoder = Encoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_hidden_layers=encoder_num_hidden_layers,
            filter_size=filter_size,
            use_projection_bias_in_attention=False,
            use_ffn_bias_in_attention=False,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            ffn_initializer=feed_forward_initializer,
            attention_dropout_rate=attention_dropout_rate,
            nonlinearity=encoder_nonlinearity,
            dropout_rate=dropout_rate,
            dropout_seed=dropout_seed,
            layer_norm_epsilon=layer_norm_epsilon,
            use_pre_normalization=use_pre_normalization,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="encoder",
        )

        # Decoder Stack.
        self.decoder = Decoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_hidden_layers=decoder_num_hidden_layers,
            filter_size=filter_size,
            use_ffn_bias_in_attention=False,
            use_projection_bias_in_attention=False,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=attention_initializer,
            ffn_initializer=feed_forward_initializer,
            attention_dropout_rate=attention_dropout_rate,
            nonlinearity=decoder_nonlinearity,
            dropout_rate=dropout_rate,
            dropout_seed=dropout_seed,
            layer_norm_epsilon=layer_norm_epsilon,
            use_pre_normalization=use_pre_normalization,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="decoder",
        )

        # Dense layer for the decoder output.
        self.output_dense_layer = DenseLayer(
            units=1,
            kernel_initializer=output_projection_initializer,
            dtype=self.policy,
            name="final_dense",  # 1 for single real value prediction
        )

        # Model trainer.
        self._trainer = Trainer(
            params=params["optimizer"],
            tf_summary=tf_summary,
            mixed_precision=mixed_precision,
        )

        # Store params.
        self.embedding_scale = math.sqrt(hidden_size)
        self.use_pre_normalization = use_pre_normalization

        # Batch size params.
        self.train_batch_size = params["train_input"]["batch_size"]
        self.eval_batch_size = params["eval_input"]["batch_size"]

    def _create_embedding_layers(
        self,
        num_vars,
        hidden_size,
        embedding_initializer,
        embedding_projection_initializer,
        boundary_casting=False,
        tf_summary=False,
        dtype=None,
    ):
        # Value&Time Projection
        val_time_proj_enc = DenseLayer(
            units=hidden_size,
            kernel_initializer=embedding_projection_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="ValueTimeProjectionLayerEncoder",
        )
        # Value&Time Projection
        val_time_proj_dec = DenseLayer(
            units=hidden_size,
            kernel_initializer=embedding_projection_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="ValueTimeProjectionLayerDecoder",
        )

        # Variable Embedding
        var_emb_enc = EmbeddingLayer(
            input_dim=num_vars,
            output_dim=hidden_size,
            embeddings_initializer=embedding_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="VariableEmbeddingLayer",
        )

        # Givens Embedding (context or target)
        givens_emb_enc = EmbeddingLayer(
            input_dim=2,
            output_dim=hidden_size,
            embeddings_initializer=embedding_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="GivensEmbeddingLayer",
        )

        var_emb_dec = EmbeddingLayer(
            input_dim=self._num_vars,
            output_dim=hidden_size,
            embeddings_initializer=embedding_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="DecoderVariableEmbeddingLayer",
        )

        givens_emb_dec = EmbeddingLayer(
            input_dim=2,
            output_dim=hidden_size,
            embeddings_initializer=embedding_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="DecoderGivensEmbeddingLayer",
        )

        return (
            val_time_proj_enc,
            val_time_proj_dec,
            var_emb_enc,
            givens_emb_enc,
            var_emb_dec,
            givens_emb_dec,
        )

    def build_model(self, features, mode):
        """ Forward graph.
        """
        # Embeddings -> Dropout -> Layernorm -> Encoder/Decoder.
        ## Embedding and positional encoding.
        encoder_input = self.val_time_proj_enc(features['enc_in'])
        encoder_input += self.var_emb_enc(features['spatial_ind_enc'])
        encoder_input += self.givens_emb_enc(features['givens_enc'])

        # Dropout.
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        encoder_input = self.pre_encoder_dropout_layer(
            encoder_input, training=is_training
        )

        # Layernorm.
        if not self.use_pre_normalization:
            encoder_input = self.pre_encoder_layer_norm(encoder_input)

        encoder_output = self.encoder(
            encoder_input,
            self_attention_mask=None,  # no padding tokens
            training=is_training,
        )

        # Decoder embedding and positional encoding.
        decoder_input = self.val_time_proj_dec(features['dec_in'])

        decoder_input += self.var_emb_dec(features['spatial_ind_dec'])
        decoder_input += self.givens_emb_dec(features['givens_dec'])

        # Dropout.
        decoder_input = self.pre_decoder_dropout_layer(
            decoder_input, training=is_training
        )

        # Layernorm.
        if not self.use_pre_normalization:
            decoder_input = self.pre_decoder_layer_norm(decoder_input)

        attn_autoregressive_mask = None

        decoder_output, present_keys_values = self.decoder(
            decoder_input,
            self_attention_mask=attn_autoregressive_mask,
            encoder_output=encoder_output,
            cross_attention_mask=None,  # no padding tokens
            training=is_training,
        )

        out = self.output_dense_layer(decoder_output)
        out = flatten(out[:, -(self._target_length * self._num_vars) :, :])

        return out

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.

        :param model_outputs: decoder outputs returned
            by build_model.
        :param dict features: Dictionary of input features.
        :param Tensor labels: Tensor of shape (batch_size,). Contains
            next sentence labels used for bert pretraining.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: Total loss tensor.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
        ], f"The model supports only TRAIN and EVAL modes."

        batch_size = (
            self.train_batch_size
            if mode == tf.estimator.ModeKeys.TRAIN
            else self.eval_batch_size
        )

        labels = flatten(labels)
        labels = tf.cast(labels, model_outputs.dtype)

        loss = tf.cast(
            # MSE does loss in FP32
            tf.compat.v1.losses.mean_squared_error(
                labels=labels, predictions=model_outputs,
            ),
            model_outputs.dtype,
        )
        loss = tf.reduce_sum(loss)
        self._write_summaries(features, loss)

        return loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self._trainer.build_train_ops(total_loss)

    def build_eval_metric_ops(self, flat_preds, flat_labels, features):
        """
        Compute Transformer eval metrics - BLEU score.

        :param  flat_preds: Tensor of shape (batch_size, tgt_length*num_vars)
                with logits.
        :param flat_labels: Tensor of shape (batch_size, tgt_length*num_vars)
                Contains expected reference translation labels.
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """

        flat_preds = tf.cast(flat_preds, tf.float32)
        metrics_dict = {}
        metrics_dict[
            "eval/normalized_mse"
        ] = tf.compat.v1.metrics.mean_squared_error(flat_labels, flat_preds,)
        metrics_dict[
            "eval/standard_mse"
        ] = tf.compat.v1.metrics.mean_squared_error(
            flat_labels * features['scales'] + features['means'],
            flat_preds * features['scales'] + features['means'],
        )

        mse = tf.compat.v1.losses.mean_squared_error(
            labels=flat_labels, predictions=flat_preds,
        )
        tf.compat.v1.summary.scalar("loss/eval_mse", mse)

        return metrics_dict

    def _write_summaries(
        self, features, total_loss,
    ):
        """
        Write train metrics summaries.

        :param features: Dictionary of input features.
        :param total_loss: total loss tensor.

        """
        # Use GradAccumSummarySaverHook to add
        # loss summaries when trained with
        # gradient accumulation.
        if self._trainer.is_grad_accum():
            return

        tf.compat.v1.summary.scalar("loss/total_loss", total_loss)

    @property
    def trainer(self):
        return self._trainer
