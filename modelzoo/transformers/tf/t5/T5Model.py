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
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.EmbeddingLayer import EmbeddingLayer
from modelzoo.common.tf.layers.Input import SetupInputTensor
from modelzoo.common.tf.layers.SharedWeightsDenseLayer import (
    SharedWeightsDenseLayer,
)
from modelzoo.common.tf.metrics.perplexity import perplexity_metric
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.t5.layers.T5Decoder import T5Decoder
from modelzoo.transformers.tf.t5.layers.T5Encoder import T5Encoder
from modelzoo.transformers.tf.transformer_utils import (
    create_autoregressive_attention_mask,
)


class T5Model(TFBaseModel):
    """
    The T5 model https://arxiv.org/pdf/1910.10683.pdf.
    :param dict params: Model configuration parameters.
    """

    def __init__(self, params):
        super(T5Model, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )
        # Model-level and encoder-decoder-shared params.
        vocab_size = params["train_input"]["vocab_size"]

        # Batch size params.
        train_batch_size = params["train_input"]["batch_size"]
        eval_batch_size = params["eval_input"]["batch_size"]

        # Max sequence lengths size params.
        train_tgt_max_sequence_length = params["train_input"][
            "tgt_max_sequence_length"
        ]
        eval_tgt_max_sequence_length = params["eval_input"][
            "tgt_max_sequence_length"
        ]

        # Size of the key, query, value projections per attention head.
        d_kv = params["model"]["d_kv"]
        share_encoder_decoder_embedding = params["model"][
            "share_encoder_decoder_embedding"
        ]

        # Loss option
        mlm_loss_scaling = params["model"]["mlm_loss_scaling"]
        lm_loss_weight = params["model"]["lm_loss_weight"]

        num_heads = params["model"]["num_heads"]
        # Size of the intermediate feed forward layer in t5 blocks.
        d_ff = params["model"]["d_ff"]
        # Size of the encoder layers and the pooler layer.
        d_model = params["model"]["d_model"]
        # Dropout rate is used in embedding, attention, feed forward layers.
        dropout_rate = params["model"]["dropout_rate"]
        attention_type = params["model"]["attention_type"]
        use_ffn_bias = params["model"]["use_ffn_bias"]

        # Encoder params.
        encoder_num_hidden_layers = params["model"]["encoder_num_hidden_layers"]
        decoder_num_hidden_layers = params["model"]["decoder_num_hidden_layers"]
        encoder_nonlinearity = params["model"]["encoder_nonlinearity"]
        decoder_nonlinearity = params["model"]["decoder_nonlinearity"]

        # Compute params.
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]
        mixed_precision = params["model"]["mixed_precision"]

        # Other params.
        layer_norm_epsilon = params["model"]["layer_norm_epsilon"]
        use_relative_attention_bias = params["model"][
            "use_relative_attention_bias"
        ]
        num_relative_attention_buckets = params["model"][
            "num_relative_attention_buckets"
        ]
        input_pad_id = params["model"]["input_pad_id"]

        # Weight initialization params.
        embedding_initializer_spec = params["model"]["embedding_initializer"]
        query_layer_initializer_spec = params["model"][
            "query_layer_initializer"
        ]
        key_layer_initializer_spec = params["model"]["key_layer_initializer"]
        value_layer_initializer_spec = params["model"][
            "value_layer_initializer"
        ]
        relative_attention_bias_weight_initializer_spec = params["model"][
            "relative_attention_bias_weight_initializer"
        ]
        output_layer_initializer_spec = params["model"][
            "output_layer_initializer"
        ]
        feed_forward_input_layer_initializer_spec = params["model"][
            "feed_forward_input_layer_initializer"
        ]
        feed_forward_output_layer_initializer_spec = params["model"][
            "feed_forward_output_layer_initializer"
        ]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]

        # Set up initializers.
        embedding_initializer = create_initializer(
            embedding_initializer_spec, weight_initialization_seed
        )
        query_layer_initializer = create_initializer(
            query_layer_initializer_spec, weight_initialization_seed
        )
        key_layer_initializer = create_initializer(
            key_layer_initializer_spec, weight_initialization_seed
        )
        value_layer_initializer = create_initializer(
            value_layer_initializer_spec, weight_initialization_seed
        )
        relative_attention_bias_weight_initializer = create_initializer(
            relative_attention_bias_weight_initializer_spec,
            weight_initialization_seed,
        )
        output_layer_initializer = create_initializer(
            output_layer_initializer_spec, weight_initialization_seed
        )
        feed_forward_input_layer_initializer = create_initializer(
            feed_forward_input_layer_initializer_spec,
            weight_initialization_seed,
        )
        feed_forward_output_layer_initializer = create_initializer(
            feed_forward_output_layer_initializer_spec,
            weight_initialization_seed,
        )

        # Embedding layers for both encoder and decoder
        # Mesh TF embedding layer does not have scaling in the init, see:
        # https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/layers.py#L2096
        # But have scaling during training, see:
        # https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer.py#L2175
        self.encoder_embedding_layer = EmbeddingLayer(
            input_dim=vocab_size,
            output_dim=d_model,
            embeddings_initializer=embedding_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="encoder_embedding",
        )

        if share_encoder_decoder_embedding:
            self.decoder_embedding_layer = self.encoder_embedding_layer
        else:
            self.decoder_embedding_layer = EmbeddingLayer(
                input_dim=vocab_size,
                output_dim=d_model,
                embeddings_initializer=embedding_initializer,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
                name="decoder_embedding",
            )

        # Encoder and decoder can share the same embedding dropout.
        self.dropout_layer = DropoutLayer(
            dropout_rate,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="embedding_dropout",
        )

        # Encoder stack.
        self.encoder = T5Encoder(
            d_kv,  # Size of the key, query, value projections per attention head.
            d_model,  # Size of the encoder layers and the pooler layer.
            num_heads,  # d_kv * num_heads = hidden size.
            encoder_num_hidden_layers,
            d_ff,  # Size of the intermediate feed forward layer in t5 blocks.
            use_ffn_bias=use_ffn_bias,
            query_layer_initializer=query_layer_initializer,
            key_layer_initializer=key_layer_initializer,
            value_layer_initializer=value_layer_initializer,
            relative_attention_bias_weight_initializer=relative_attention_bias_weight_initializer,
            output_layer_initializer=output_layer_initializer,
            feed_forward_input_layer_initializer=feed_forward_input_layer_initializer,
            feed_forward_output_layer_initializer=feed_forward_output_layer_initializer,
            attention_type=attention_type,
            nonlinearity=encoder_nonlinearity,
            dropout_rate=dropout_rate,
            use_relative_attention_bias=use_relative_attention_bias,
            num_relative_attention_buckets=num_relative_attention_buckets,
            layer_norm_epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="encoder",
        )

        # Decoder stack.
        self.decoder = T5Decoder(
            d_kv,  # Size of the key, query, value projections per attention head.
            d_model,  # Size of the encoder layers and the pooler layer.
            num_heads,  # d_kv * num_heads = hidden size.
            decoder_num_hidden_layers,
            d_ff,  # Size of the intermediate feed forward layer in t5 blocks.
            use_ffn_bias=use_ffn_bias,
            query_layer_initializer=query_layer_initializer,
            key_layer_initializer=key_layer_initializer,
            value_layer_initializer=value_layer_initializer,
            relative_attention_bias_weight_initializer=relative_attention_bias_weight_initializer,
            output_layer_initializer=output_layer_initializer,
            feed_forward_input_layer_initializer=feed_forward_input_layer_initializer,
            feed_forward_output_layer_initializer=feed_forward_output_layer_initializer,
            attention_type=attention_type,
            nonlinearity=decoder_nonlinearity,
            dropout_rate=dropout_rate,
            use_relative_attention_bias=use_relative_attention_bias,
            num_relative_attention_buckets=num_relative_attention_buckets,
            layer_norm_epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="decoder",
        )

        # Dense layer for the decoder output.
        self.output_dense_layer = SharedWeightsDenseLayer(
            vocab_size,
            use_bias=False,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="final_dense",
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
        self.d_model = d_model
        self.num_heads = num_heads
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.mlm_loss_scaling = mlm_loss_scaling
        self.lm_loss_weight = lm_loss_weight
        self.mixed_precision = mixed_precision
        self.use_relative_attention_bias = use_relative_attention_bias
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_tgt_max_sequence_length = train_tgt_max_sequence_length
        self.eval_tgt_max_sequence_length = eval_tgt_max_sequence_length
        self.input_pad_id = input_pad_id
        self.tf_summary = tf_summary

    def build_model(self, features, mode):
        """
        Build the model (up to loss).
        :param dict features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :return: list: model outputs.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
        ], f"The model supports estimator TRAIN, and EVAL modes."

        # Encoder embedding
        encoder_input_ids = SetupInputTensor(
            features["encoder_input_ids"], tf_summary=self.tf_summary
        )
        encoder_input = self.encoder_embedding_layer(
            encoder_input_ids, pad_id=self.input_pad_id
        )
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        encoder_input = self.dropout_layer(encoder_input, training=is_training)

        # We cache `position_bias` only when `use_relative_attention_bias`
        # is set to `True`.
        encoder_output = self.encoder(
            encoder_input,
            self_attention_mask=features["encoder_mask"],
            cache_position_bias=self.use_relative_attention_bias,
            is_training=is_training,
        )

        # Decoder embedding
        decoder_input_ids = SetupInputTensor(
            features["decoder_input_ids"], tf_summary=self.tf_summary
        )
        decoder_input = self.decoder_embedding_layer(
            decoder_input_ids, pad_id=self.input_pad_id
        )
        decoder_input = self.dropout_layer(decoder_input, training=is_training)

        # We cache `position_bias` only when `use_relative_attention_bias
        # is set to `True`.
        decoder_output, _ = self.decoder(
            inputs=decoder_input,
            self_attention_mask=create_autoregressive_attention_mask(
                batch_size=decoder_input.shape[0],
                max_sequence_length=decoder_input.shape[1],
                dtype=decoder_input.dtype,
            ),
            encoder_output=encoder_output,
            cross_attention_mask=features["encoder_mask"],
            cache_position_bias=self.use_relative_attention_bias,
            is_training=is_training,
        )

        # Rescale output before projecting on vocab, see:
        # https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer.py#L2174
        decoder_output = decoder_output * (self.d_model ** -0.5)

        # Share embedding table with output dense layer
        output = self.output_dense_layer(
            decoder_output,
            kernel=self.decoder_embedding_layer.embedding_table(),
            transpose_kernel=True,
        )

        return output

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.
        :param list model_outputs: Model output tensor returned by call method.
        :param dict features: Dictionary of input features.
        :param Tensor labels: (2D tensor) decoder target token sequence.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :return: Total loss tensor.
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
        tgt_max_sequence_length = (
            self.train_tgt_max_sequence_length
            if mode == tf.estimator.ModeKeys.TRAIN
            else self.eval_tgt_max_sequence_length
        )
        # loss shape: [bsz, tgt_max_seq_len].
        loss = self.loss_layer(labels, logits=model_outputs)

        decoder_mask = self._get_decoder_mask(features, model_outputs.dtype)
        if self.mlm_loss_scaling == "precomputed_num_masked":
            decoder_mask *= tf.cast(features["loss_scale"], decoder_mask.dtype)
        loss = loss * tf.cast(decoder_mask, loss.dtype)
        total_loss = tf.reduce_sum(input_tensor=loss, name="total_loss")
        if self.mlm_loss_scaling == "precomputed_num_masked":
            total_loss = total_loss / batch_size
        elif self.mlm_loss_scaling == "batch_size":
            total_loss = total_loss / batch_size * self.lm_loss_weight
        elif self.mlm_loss_scaling == "num_masked":
            total_loss = total_loss / tf.reduce_sum(decoder_mask)
        else:
            raise ValueError(
                f"{self.mlm_loss_scaling} is not supported. Choose between `num_masked`, `precomputed_num_masked`, `batch_size`."
            )

        self._write_summaries(features, total_loss, mode)
        return tf.cast(total_loss, model_outputs.dtype)

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
                -- `cross_entropy_loss` containing cross entropy loss
                 computed over non-padded tokens based on the `decoder_mask`.
        """
        predictions = tf.argmax(model_outputs, axis=-1, output_type=tf.int32)
        cross_entropy_loss = CrossEntropyFromLogitsLayer(dtype=self.policy)(
            labels=labels, logits=model_outputs
        )
        eval_metric_inputs = (predictions, cross_entropy_loss)
        return eval_metric_inputs

    def build_eval_metric_ops(self, eval_metric_inputs, labels, features):
        """
        Compute T5 eval metrics.
        :param eval_metric_inputs: tuple containing:
            -- `predictions`: predicted labels for each token;
            -- `cross_entropy_loss` containing cross entropy loss
                computed over non-padded tokens based on the `decoder_mask`.
        :param labels: (2D tensor) decoder target token sequence.
        :param features: Dictionary of input features.

        :return: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        predictions, cross_entropy_loss = eval_metric_inputs

        metrics_dict = {}
        decoder_mask = self._get_decoder_mask(features, tf.float16)
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

    def _write_summaries(self, features, total_loss, mode):
        """
        Write train metrics summaries.

        :param features: Dictionary of input features.
        :param total_loss: total loss tensor.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        """

        # Use GradAccumSummarySaverHook to add loss summaries when trained
        # with gradient accumulation.
        if self._trainer.is_grad_accum():
            return

        total_loss = tf.cast(total_loss, tf.float32)
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
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.num_heads

    @property
    def num_hidden_layers(self):
        return self.encoder_num_hidden_layers

    @property
    def trainer(self):
        return self._trainer
