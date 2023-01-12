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

import math

import tensorflow as tf

from modelzoo.common.tf.layers.CrossEntropyFromLogitsLayer import (
    CrossEntropyFromLogitsLayer,
)
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.EmbeddingLayer import EmbeddingLayer
from modelzoo.common.tf.layers.Input import SetupInputTensor
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.layers.PositionEmbeddingLayer import (
    PositionEmbeddingLayer,
)
from modelzoo.common.tf.layers.SharedWeightsDenseLayer import (
    SharedWeightsDenseLayer,
)
from modelzoo.common.tf.metrics.perplexity import perplexity_metric
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.layers.Decoder import Decoder
from modelzoo.transformers.tf.layers.Encoder import Encoder
from modelzoo.transformers.tf.transformer.input.data_processing.utils import (
    get_special_tokens,
)
from modelzoo.transformers.tf.transformer_utils import (
    create_autoregressive_attention_mask,
)


class Transformer(TFBaseModel):
    """
    Transformer model from Attention Is All you Need.
    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, params):
        """Make a transformer object."""
        super(Transformer, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # save the params as they will be used in the host_call. Since the
        # params are just a constant dict (no tf resource variables used)
        # we can easily use this in both model and host call code.
        self.params = params

        # Embeddings params.
        src_vocab_size = params["train_input"]["src_vocab_size"]
        tgt_vocab_size = params["train_input"]["tgt_vocab_size"]
        position_embedding_type = params["model"]["position_embedding_type"]
        share_encoder_decoder_embedding = params["model"][
            "share_encoder_decoder_embedding"
        ]
        max_position_embeddings = params["model"]["max_position_embeddings"]

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
        input_pad_id = params["model"]["input_pad_id"]

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
        attention_initializer_spec = params["model"]["attention_initializer"]
        feed_forward_initializer_spec = params["model"][
            "feed_forward_initializer"
        ]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]

        # Set up initializers.
        embedding_initializer = create_initializer(
            embedding_initializer_spec, weight_initialization_seed
        )
        attention_initializer = create_initializer(
            attention_initializer_spec, weight_initialization_seed
        )
        feed_forward_initializer = create_initializer(
            feed_forward_initializer_spec, weight_initialization_seed
        )

        # Set up layers.
        ## Embedding layers.
        (
            self.encoder_position_embedding_layer,
            self.decoder_position_embedding_layer,
            self.encoder_token_embedding_layer,
            self.decoder_token_embedding_layer,
        ) = self._create_embedding_layers(
            share_encoder_decoder_embedding,
            src_vocab_size,
            tgt_vocab_size,
            embedding_size=hidden_size,  # Output dimension same as `hidden_size` i.e. `d_model`.
            embeddings_initializer=embedding_initializer,
            max_position_embeddings=max_position_embeddings,
            position_embedding_type=position_embedding_type,
            position_embeddings_initializer=embedding_initializer,
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
        self.output_dense_layer = SharedWeightsDenseLayer(
            tgt_vocab_size, dtype=self.policy, name="final_dense"
        )

        # Final loss layer.
        self.loss_layer = CrossEntropyFromLogitsLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
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
        self.input_pad_id = input_pad_id

        # Batch size params.
        self.train_batch_size = params["train_input"]["batch_size"]
        self.eval_batch_size = params["eval_input"]["batch_size"]

        # Params for eval.
        self.special_tokens = get_special_tokens()
        self.special_words = list(self.special_tokens.values())

    def _create_embedding_layers(
        self,
        share_encoder_decoder_embedding,
        src_vocab_size,
        tgt_vocab_size,
        embedding_size,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        max_position_embeddings=None,
        position_embedding_type=None,
        position_embeddings_initializer="uniform",
        position_embeddings_regularizer=None,
        boundary_casting=False,
        tf_summary=False,
        dtype=None,
    ):

        # Position Embedding layers.
        encoder_position_embedding = PositionEmbeddingLayer(
            max_position_embeddings=max_position_embeddings,
            embedding_type=position_embedding_type,
            embeddings_initializer=position_embeddings_initializer,
            embeddings_regularizer=position_embeddings_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="encoder_position_encoding",
        )
        decoder_position_embedding = PositionEmbeddingLayer(
            max_position_embeddings=max_position_embeddings,
            embedding_type=position_embedding_type,
            embeddings_initializer=position_embeddings_initializer,
            embeddings_regularizer=position_embeddings_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="decoder_position_encoding",
        )

        # Token embedding layers.
        encoder_token_embedding = EmbeddingLayer(
            input_dim=src_vocab_size,
            output_dim=embedding_size,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=dtype,
            name="encoder_token_embedding",
        )
        if share_encoder_decoder_embedding:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError(
                    f"Cannot share embeddings between encoder and decoder due to "
                    f"different vocab sizes. `src_vocab_size`={src_vocab_size} and "
                    f"`tgt_vocab_size`={tgt_vocab_size}."
                )

            decoder_token_embedding = encoder_token_embedding

        else:
            decoder_token_embedding = EmbeddingLayer(
                input_dim=tgt_vocab_size,
                output_dim=embedding_size,
                embeddings_initializer=embeddings_initializer,
                embeddings_regularizer=embeddings_regularizer,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=dtype,
                name="decoder_token_embedding",
            )

        return (
            encoder_position_embedding,
            decoder_position_embedding,
            encoder_token_embedding,
            decoder_token_embedding,
        )

    def build_model(self, features, mode):
        """Forward graph."""
        # Inputs.
        encoder_input_ids = SetupInputTensor(features["encoder_input_ids"])
        decoder_input_ids = SetupInputTensor(features["decoder_input_ids"])

        # Embeddings -> Dropout -> Layernorm -> Encoder/Decoder.
        ## Embedding and positional encoding.
        encoder_input = self.encoder_token_embedding_layer(
            encoder_input_ids,
            pad_id=self.input_pad_id,
            scale=self.embedding_scale,
        )
        encoder_input = self.encoder_position_embedding_layer(encoder_input)

        ## Dropout.
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        encoder_input = self.pre_encoder_dropout_layer(
            encoder_input, training=is_training
        )

        ## Layernorm.
        if not self.use_pre_normalization:
            encoder_input = self.pre_encoder_layer_norm(encoder_input)

        ## Encoder.
        encoder_output = self.encoder(
            encoder_input,
            self_attention_mask=features["encoder_mask"],
            training=is_training,
        )

        # Decoder embedding and positional encoding.
        decoder_input = self.decoder_token_embedding_layer(
            decoder_input_ids,
            pad_id=self.input_pad_id,
            scale=self.embedding_scale,
        )
        decoder_input = self.decoder_position_embedding_layer(decoder_input)

        ## Dropout.
        decoder_input = self.pre_decoder_dropout_layer(
            decoder_input, training=is_training
        )

        ## Layernorm.
        if not self.use_pre_normalization:
            decoder_input = self.pre_decoder_layer_norm(decoder_input)

        attn_autoregressive_mask = create_autoregressive_attention_mask(
            batch_size=tf.shape(decoder_input)[0],
            max_sequence_length=tf.shape(decoder_input)[1],
            dtype=decoder_input.dtype,
        )

        decoder_output, present_keys_values = self.decoder(
            decoder_input,
            self_attention_mask=attn_autoregressive_mask,
            encoder_output=encoder_output,
            cross_attention_mask=features["encoder_mask"],
            training=is_training,
        )

        decoder_output = self.output_dense_layer(
            decoder_output,
            kernel=self.decoder_token_embedding_layer.embedding_table(),
            transpose_kernel=True,
        )

        return decoder_output

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

        # loss shape: [bsz, tgt_max_seq_len].
        loss = self.loss_layer(labels, logits=model_outputs)

        decoder_mask = self._get_decoder_mask(
            features, model_outputs.dtype
        )  # shape: [bsz, tgt_max_seq_len].
        decoder_mask *= tf.cast(features["loss_scale"], decoder_mask.dtype)

        loss = loss * tf.cast(decoder_mask, loss.dtype)
        total_loss = tf.reduce_sum(input_tensor=loss, name="total_loss")
        total_loss = tf.cast(total_loss / batch_size, model_outputs.dtype)

        self._write_summaries(features, total_loss)

        return total_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self._trainer.build_train_ops(total_loss)

    def build_eval_metric_inputs(self, model_outputs, labels, features):
        """
        Build inputs for eval metrics computations.
        :param model_outputs: Model output tensor returned by call method.
        :param labels: (2D tensor) decoder target token sequence.
        :param features: Dictionary of input features.

        :return: `eval_metric_inputs`: tuple containing:
                -- `predictions`: predicted labels for each token;
                -- `cross_entropy_loss`: tensor with cross entropy loss for each decoder token
        """
        predictions = tf.argmax(model_outputs, axis=-1, output_type=tf.int32)

        cross_entropy_loss = CrossEntropyFromLogitsLayer(dtype=self.policy)(
            labels, logits=model_outputs
        )

        return (predictions, cross_entropy_loss)

    def build_eval_metric_ops(self, eval_metric_inputs, labels, features):
        """
        Compute Transformer eval metrics - BLEU score.
        :param `eval_metric_inputs`: tuple containing:
            -- `predictions`: predicted labels for each token;
            -- `decoder_xetr`: tensor with cross entropy loss for each decoder token
            Tensors of shape (batch_size, tgt_max_sequence_length).
        :param labels: Tensor of shape (batch_size, tgt_max_sequence_length).
                Contains expected reference translation labels.
        :param features: Dictionary of input features.
        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """

        predictions, cross_entropy_loss = eval_metric_inputs

        decoder_mask = self._get_decoder_mask(features, tf.float16)

        metrics_dict = dict()
        metrics_dict["eval/accuracy_lm"] = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=predictions, weights=decoder_mask,
        )

        unscaled_lm_loss = tf.reduce_sum(
            input_tensor=tf.cast(cross_entropy_loss * decoder_mask, tf.float32,)
        )
        num_masked = tf.reduce_sum(decoder_mask)
        metrics_dict["eval/perplexity_lm"] = perplexity_metric(
            unscaled_lm_loss, num_masked
        )

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

        total_loss = tf.cast(total_loss, tf.float32)

        decoder_mask = self._get_decoder_mask(
            features, total_loss.dtype
        )  # shape: [bsz, tgt_max_seq_len].
        num_non_padded_tokens = tf.math.reduce_sum(decoder_mask)

        # Log losses: total.
        tf.compat.v1.summary.scalar("loss/total_loss", total_loss)

    def _get_decoder_mask(self, features, output_type):

        # features["decoder_mask"] has
        # 1's in padded positions
        # 0's in non-padded positions.
        # Inverting the values inorder to have 1's in non-padded positions.

        # Shape: [batch_size, tgt_max_sequence_length].
        decoder_mask = 1.0 - tf.cast(features["decoder_mask"], output_type)
        return decoder_mask

    @property
    def trainer(self):
        return self._trainer
