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
from modelzoo.common.tf.layers.EmbeddingLayer import EmbeddingLayer
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.layers.SharedWeightsDenseLayer import (
    SharedWeightsDenseLayer,
)
from modelzoo.common.tf.metrics.bits_per_x import (
    bits_per_x_metric,
    calculate_bits_per_x,
)
from modelzoo.common.tf.metrics.perplexity import (
    calculate_perplexity,
    perplexity_metric,
)
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.gptj.layers.GptJDecoder import GptJDecoder
from modelzoo.transformers.tf.transformer_utils import (
    create_autoregressive_attention_mask,
)


class GptJModel(TFBaseModel):
    """
    GPT model with parallel attention & feed-forward layers. Implements the
    following two variants:

        - GPT-J model :: `<https://github.com/kingoflolz/mesh-transformer-jax>`_.
        - GPT-NeoX model :: `<https://arxiv.org/abs/2204.06745>`_.

    Args:
        params (dict): Model configuration params
    """

    def __init__(self, params):
        super(GptJModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # embedding params
        vocab_size = params["train_input"]["vocab_size"]
        embedding_dropout_rate = params["model"]["embedding_dropout_rate"]
        share_embedding_weights = params["model"]["share_embedding_weights"]
        max_position_embeddings = params["model"]["max_position_embeddings"]

        # model params
        hidden_size = params["model"]["hidden_size"]
        num_heads = params["model"]["num_heads"]
        num_hidden_layers = params["model"]["num_hidden_layers"]
        use_projection_bias_in_attention = params["model"][
            "use_projection_bias_in_attention"
        ]
        use_ffn_bias_in_attention = params["model"]["use_ffn_bias_in_attention"]
        use_ffn_bias = params["model"]["use_ffn_bias"]
        filter_size = params["model"]["filter_size"]
        nonlinearity = params["model"]["nonlinearity"]
        rotary_dim = params["model"]["rotary_dim"]
        attention_dropout_rate = params["model"]["attention_dropout_rate"]
        residual_dropout_rate = params["model"]["residual_dropout_rate"]
        attention_softmax_fp32 = params["model"]["attention_softmax_fp32"]
        use_cache = params["model"]["use_cache"]
        use_bias_in_output = params["model"]["use_bias_in_output"]

        # layernorm parameters
        use_untied_layer_norm = params["model"]["use_untied_layer_norm"]
        layer_norm_epsilon = params["model"]["layer_norm_epsilon"]

        # Eval Metrics params
        bits_per_x_dataset = params["model"]["bits_per_x_dataset"]

        # Seeds
        dropout_seed = params["model"]["dropout_seed"]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]

        # Set up weight initializers with given parameters
        initializer_spec = params["model"]["initializer"]
        embedding_initializer_spec = params["model"]["embedding_initializer"]
        output_layer_initializer_spec = params["model"][
            "output_layer_initializer"
        ]
        initializer = create_initializer(
            initializer_spec, weight_initialization_seed
        )
        embedding_initializer = (
            initializer
            if embedding_initializer_spec is None
            else create_initializer(
                embedding_initializer_spec, weight_initialization_seed
            )
        )
        output_layer_initializer = (
            initializer
            if output_layer_initializer_spec is None
            else create_initializer(
                output_layer_initializer_spec, weight_initialization_seed
            )
        )

        # CS params for layers
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]

        self.wte = EmbeddingLayer(
            input_dim=vocab_size,
            output_dim=hidden_size,
            embeddings_initializer=embedding_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="input_embedding",
        )

        if embedding_dropout_rate > 0.0:
            self.embedding_dropout = DropoutLayer(
                rate=embedding_dropout_rate,
                seed=dropout_seed,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
                name="embedding_dropout",
            )
        else:
            self.embedding_dropout = None

        self.h = GptJDecoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            filter_size=filter_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            use_untied_layer_norm=use_untied_layer_norm,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            rotary_dim=rotary_dim,
            attention_initializer=initializer,
            ffn_initializer=initializer,
            output_initializer=output_layer_initializer,
            attention_dropout_rate=attention_dropout_rate,
            attention_residual_dropout_rate=residual_dropout_rate,
            ffn_residual_dropout_rate=residual_dropout_rate,
            dropout_seed=dropout_seed,
            nonlinearity=nonlinearity,
            layer_norm_epsilon=layer_norm_epsilon,
            attention_softmax_fp32=attention_softmax_fp32,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="gptj_decoder",
        )

        self.ln_f = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="post_decoder_layer_norm",
        )

        backbone_only = params["model"]["backbone_only"]
        if not backbone_only:

            if share_embedding_weights:
                self.output_dense_layer = SharedWeightsDenseLayer(
                    vocab_size,
                    use_bias=use_bias_in_output,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.policy,
                    name="lm_head",
                )
            else:
                self.output_dense_layer = DenseLayer(
                    vocab_size,
                    use_bias=use_bias_in_output,
                    kernel_initializer=initializer,
                    boundary_casting=boundary_casting,
                    tf_summary=tf_summary,
                    dtype=self.policy,
                    name="lm_head",
                )

            self.softmax_ce_loss = CrossEntropyFromLogitsLayer(
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
                name="softmax_ce_loss",
            )
        else:
            self.output_dense_layer = None
            self.softmax_ce_loss = None

        self._trainer = Trainer(
            params=params["optimizer"],
            tf_summary=tf_summary,
            mixed_precision=params["model"]["mixed_precision"],
        )

        self.num_hidden_layers = num_hidden_layers
        self.use_cache = use_cache
        self.backbone_only = backbone_only
        self.share_embedding_weights = share_embedding_weights
        self.bits_per_x_dataset = bits_per_x_dataset

    def build_model(self, features, mode):
        """
        Build the model (up to loss).

        Args:
            features (dict): Input features.
            mode (tf.estimator.ModeKeys): Mode (TRAIN, EVAL, PREDICT).

        Returns
            A list of model outputs, where the 0th entry is the logits tensor
            and the 1st entry is the decoder output.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"The model supports TRAIN, EVAL, and PREDICT modes."
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        # inference workflow only
        past_keys_values = features.get("past_keys_values", None)
        if mode is not tf.estimator.ModeKeys.PREDICT:
            assert (
                past_keys_values is None and self.use_cache is False
            ), "Currently support past_keys_values only in PREDICT mode."

        input_ids = features["input_ids"]
        hidden_states = self.wte(input_ids)
        if self.embedding_dropout is not None:
            hidden_states = self.embedding_dropout(
                hidden_states, training=is_training
            )

        # mask for attention
        attention_mask = features.get(
            "attention_mask",
            create_autoregressive_attention_mask(
                max_sequence_length=input_ids.shape[1],
                dtype=hidden_states.dtype,
            ),
        )
        hidden_states, present_key_values = self.h(
            hidden_states,
            attention_mask=attention_mask,
            past_keys_values=past_keys_values,
            use_cache=self.use_cache,
            training=is_training,
        )
        hidden_states = self.ln_f(hidden_states)

        if self.backbone_only:
            return hidden_states, present_key_values

        if self.share_embedding_weights:
            logits = self.output_dense_layer(
                hidden_states,
                self.wte.embedding_table(),
                transpose_kernel=True,
            )
        else:
            logits = self.output_dense_layer(hidden_states)

        return logits, hidden_states, present_key_values

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.

        Args:
            model_outputs (list): list containing logits and decoder outputs.
            features (dict): Dictionary of input features.
            labels (Tensor): Tensor of shape (batch_size,).
            mode (tf.estimator.ModeKeys): Mode (TRAIN, EVAL).

        Returns:
            Total loss tensor.
        """

        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
        ], f"The model supports only TRAIN and EVAL modes."

        loss = self._compute_loss(model_outputs, features, labels, mode)
        self._write_summaries(features, loss)

        return loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self._trainer.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels, features):
        metrics_dict = dict()

        # initial computations for metrics
        logits = model_outputs[0]
        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        softmax_ce_loss = self.softmax_ce_loss(labels, logits=logits)

        mask = tf.cast(1 - features["input_mask"], softmax_ce_loss.dtype)
        # accuracy
        metrics_dict["eval/accuracy"] = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=pred, weights=mask,
        )
        # perplexity
        unscaled_loss = tf.reduce_sum(
            input_tensor=tf.cast(softmax_ce_loss * mask, tf.float32,)
        )
        num_tokens = self._compute_num_tokens(mask)
        metrics_dict["eval/lm_perplexity"] = perplexity_metric(
            unscaled_loss, num_tokens
        )
        # bits_per_byte
        metrics_dict["eval/bits_per_byte"] = bits_per_x_metric(
            unscaled_loss,
            num_tokens,
            bits_per_type="per_byte",
            dataset=self.bits_per_x_dataset,
        )

        return metrics_dict

    def _compute_loss(self, model_outputs, features, labels, mode):
        logits = model_outputs[0]
        softmax_ce_loss = self.softmax_ce_loss(labels, logits=logits)
        mask = tf.cast(1 - features["input_mask"], softmax_ce_loss.dtype)
        cross_entropy = tf.reduce_sum(
            input_tensor=tf.cast(softmax_ce_loss * mask, tf.float32)
        )
        scale = self._compute_num_tokens(mask)
        return tf.cast(cross_entropy / scale, logits.dtype)

    def _write_summaries(
        self, features, loss,
    ):

        # Use GradAccumSummarySaverHook to add loss summaries when trained
        # with gradient accumulation
        if self._trainer.is_grad_accum():
            return

        # loss
        loss = tf.cast(loss, tf.float32)
        tf.compat.v1.summary.scalar("train/lm_cost", loss)

        # perplexity
        unscaled_loss = loss
        num_tokens = self._compute_num_tokens(1 - features["input_mask"])
        unscaled_loss *= num_tokens
        ppl = calculate_perplexity(unscaled_loss, num_tokens)
        tf.compat.v1.summary.scalar("train/lm_perplexity", ppl)

        # bits_per_byte
        bpb = calculate_bits_per_x(
            unscaled_loss,
            num_tokens,
            bits_per_type="per_byte",
            dataset=self.bits_per_x_dataset,
        )
        tf.compat.v1.summary.scalar("train/lm_bpb", bpb)

    def _compute_num_tokens(self, mask):
        return tf.maximum(
            tf.cast(tf.reduce_sum(input_tensor=mask), tf.float32), 1e-5,
        )

    @property
    def trainer(self):
        return self._trainer
