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

"""GPT-2 model.
"""
import tensorflow as tf

from modelzoo.common.tf.layers.CrossEntropyFromLogitsLayer import (
    CrossEntropyFromLogitsLayer,
)
from modelzoo.common.tf.layers.DenseLayer import DenseLayer
from modelzoo.common.tf.layers.DropoutLayer import DropoutLayer
from modelzoo.common.tf.layers.Input import SetupInputTensor
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
from modelzoo.common.tf.run_utils import ExecutionMode, get_execution_mode
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.layers.Decoder import Decoder
from modelzoo.transformers.tf.transformer_utils import (
    create_autoregressive_attention_mask,
    create_embedding_layers,
    create_fixed_sparse_attention_mask,
)


class Gpt2Model(TFBaseModel):
    """
    GPT model with sequential attention & feed-forward layers. Implements the
    following two variants:

        - GPT-2 model (Radford, et al 2018):
            `<https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_.
        - GPT-3 model (Brown, et al 2020): `<https://arxiv.org/abs/2005.14165>`_.

    Args:
        params (dict): Model configuration params
    """

    def __init__(self, params):
        """GPT-2 Model initialization.

        :param dict params:
            Dictionary with input, model, and training parameters.
        """
        super(Gpt2Model, self).__init__(
            mixed_precision=params['model']['mixed_precision']
        )

        # Embedding params
        vocab_size = params["train_input"]["vocab_size"]
        use_position_embedding = params["model"]["use_position_embedding"]
        position_embedding_type = params["model"]["position_embedding_type"]
        max_position_embeddings = params["model"]["max_position_embeddings"]
        embedding_size = params["model"]["embedding_size"]
        share_embedding_weights = params["model"]["share_embedding_weights"]

        # Decoder params
        hidden_size = params["model"]["hidden_size"]
        num_heads = params["model"]["num_heads"]
        num_hidden_layers = params["model"]["num_hidden_layers"]
        filter_size = params["model"]["filter_size"]
        nonlinearity = params["model"]["nonlinearity"]
        dropout_rate = params["model"]["dropout_rate"]
        dropout_seed = params["model"]["dropout_seed"]
        attention_softmax_fp32 = params["model"]["attention_softmax_fp32"]

        fixed_sparse_attention = params["model"].get(
            "fixed_sparse_attention", None
        )
        attention_dropout_rate = params["model"]["attention_dropout_rate"]
        use_projection_bias_in_attention = params["model"][
            "use_projection_bias_in_attention"
        ]
        use_ffn_bias_in_attention = params["model"]["use_ffn_bias_in_attention"]
        use_ffn_bias = params["model"]["use_ffn_bias"]
        use_pre_normalization = params["model"]["use_pre_normalization"]
        layer_norm_epsilon = params["model"]["layer_norm_epsilon"]

        # Loss params
        loss_weight = params["model"]["loss_weight"]
        loss_scaling = params["model"]["loss_scaling"]

        # Batch size params.
        train_batch_size = params["train_input"]["batch_size"]
        eval_batch_size = params["eval_input"]["batch_size"]

        # CS util params for layers
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]

        # Weight initialization params
        initializer_spec = params["model"]["initializer"]
        embedding_initializer_spec = params["model"]["embedding_initializer"]
        output_layer_initializer_spec = params["model"][
            "output_layer_initializer"
        ]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]

        # Eval Metrics params
        bits_per_x_dataset = params["model"]["bits_per_x_dataset"]

        # Set up initializers
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

        # Set up layers
        (
            self.token_embedding_layer,
            self.position_embedding_layer,
            _,
        ) = create_embedding_layers(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            embeddings_initializer=embedding_initializer,
            max_position_embeddings=max_position_embeddings,
            position_embeddings_type=position_embedding_type
            if use_position_embedding
            else None,
            position_embeddings_initializer=initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

        use_factorized_embedding = embedding_size != hidden_size

        if use_factorized_embedding:
            tf.compat.v1.logging.warning(
                'Using a factorized embedding, since embedding_size {} '
                'does not match hidden_size {}'.format(
                    embedding_size, hidden_size
                )
            )
            # Add projection for embedding to the
            # hidden dimension of the decoder.
            self.embedding_projection = DenseLayer(
                hidden_size,
                activation=None,
                use_bias=False,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
                name="input_embed_projection",
            )

        self.pre_decoder_layer_norm = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="pre_decoder_layer_norm",
        )

        self.dropout_layer = DropoutLayer(
            dropout_rate,
            seed=dropout_seed,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="embedding_dropout",
        )

        self.decoder = Decoder(
            hidden_size,
            num_heads,
            num_hidden_layers,
            filter_size,
            use_projection_bias_in_attention=use_projection_bias_in_attention,
            use_ffn_bias_in_attention=use_ffn_bias_in_attention,
            use_ffn_bias=use_ffn_bias,
            attention_initializer=initializer,
            ffn_initializer=initializer,
            output_layer_initializer=output_layer_initializer,
            attention_dropout_rate=attention_dropout_rate,
            nonlinearity=nonlinearity,
            dropout_rate=dropout_rate,
            dropout_seed=dropout_seed,
            layer_norm_epsilon=layer_norm_epsilon,
            use_pre_normalization=use_pre_normalization,
            attention_softmax_fp32=attention_softmax_fp32,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="decoder",
        )

        self.post_decoder_layer_norm = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="post_decoder_layer_norm",
        )

        if use_factorized_embedding:
            self.decoder_output_projection = DenseLayer(
                embedding_size,
                activation=None,
                use_bias=False,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
                name='output_embed_projection',
            )

        if share_embedding_weights:
            self.output_dense_layer = SharedWeightsDenseLayer(
                vocab_size,
                activation=None,
                use_bias=False,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
                name='lm_output_layer',
            )
        else:
            self.output_dense_layer = DenseLayer(
                vocab_size,
                activation=None,
                use_bias=False,
                kernel_initializer=initializer,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
                name='lm_output_layer',
            )

        self.softmax_ce_loss = CrossEntropyFromLogitsLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name='softmax_ce_loss',
        )

        # Model trainer
        self._trainer = Trainer(
            params=params["optimizer"],
            tf_summary=tf_summary,
            mixed_precision=params["model"]["mixed_precision"],
        )

        # Store params
        self.use_factorized_embedding = use_factorized_embedding
        self.share_embedding_weights = share_embedding_weights
        self.use_pre_normalization = use_pre_normalization
        self.loss_weight = loss_weight
        self.loss_scaling = loss_scaling
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tf_summary = tf_summary
        self.bits_per_x_dataset = bits_per_x_dataset
        self.num_heads = num_heads
        self.fixed_sparse_attention = fixed_sparse_attention

    def build_model(self, features, mode):
        """
        Build the model (up to loss).

        :param dict features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns list: List of model outputs, where the 0th
            entry is the logits tensor and the 1st entry is
            the decoder output.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"The model supports TRAIN, EVAL, and PREDICT modes."

        input_ids = SetupInputTensor(
            features["input_ids"], tf_summary=self.tf_summary
        )

        # Tensor of past keys and values. Needed for
        # autoregressive decoding in the predict mode.
        past_keys_values = features.get("past_keys_values", None)
        cache_present_keys_values = features.get(
            'cache_present_keys_values', False
        )
        if mode is not tf.estimator.ModeKeys.PREDICT:
            assert (
                past_keys_values is None and cache_present_keys_values is False
            ), "Currently support past_keys_values only in PREDICT mode."

        decoder_input = self.token_embedding_layer(input_ids)

        if self.use_factorized_embedding:
            decoder_input = self.embedding_projection(decoder_input)

        if self.position_embedding_layer:
            position_ids = None
            if past_keys_values is not None:
                position_ids = (
                    tf.range(tf.shape(input_ids)[1])
                    + tf.shape(past_keys_values)[-2]
                )
            decoder_input = self.position_embedding_layer(
                decoder_input, position_ids
            )

        if not self.use_pre_normalization:
            decoder_input = self.pre_decoder_layer_norm(decoder_input)

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        decoder_input = self.dropout_layer(decoder_input, training=is_training)

        auto_regressive_mask = create_autoregressive_attention_mask(
            max_sequence_length=input_ids.shape[1], dtype=decoder_input.dtype,
        )

        sparse_attention_mask = None
        if self.fixed_sparse_attention:
            sparse_attention_mask = create_fixed_sparse_attention_mask(
                max_sequence_length=input_ids.shape[1],
                n_heads=self.num_heads,
                dtype=decoder_input.dtype,
                **self.fixed_sparse_attention,
            )

        decoder_output, present_keys_values = self.decoder(
            decoder_input,
            self_attention_mask=auto_regressive_mask,
            sparse_attention_mask=sparse_attention_mask,
            past_keys_values=past_keys_values,
            cache_present_keys_values=cache_present_keys_values,
            training=is_training,
        )

        if self.use_pre_normalization:
            decoder_output = self.post_decoder_layer_norm(decoder_output)

        if self.use_factorized_embedding:
            decoder_output = self.decoder_output_projection(decoder_output)

        if self.share_embedding_weights:
            logits = self.output_dense_layer(
                decoder_output,
                self.token_embedding_layer.embedding_table(),
                transpose_kernel=True,
            )
        else:
            logits = self.output_dense_layer(decoder_output)

        return logits, decoder_output, present_keys_values

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.

        :param list model_outputs: list containing logits and decoder outputs.
        :param dict features: Dictionary of input features.
        :param Tensor labels: Tensor of shape (batch_size,).
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns: Total loss tensor.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
        ], f"The model supports only TRAIN and EVAL modes."

        loss = (
            self._compute_loss(model_outputs, features, labels, mode)
            * self.loss_weight
        )
        self._write_summaries(features, loss)

        return loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self._trainer.build_train_ops(total_loss)

    def build_eval_metric_inputs(self, model_outputs, labels, features):
        logits = model_outputs[0]

        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        softmax_ce_loss = self.softmax_ce_loss(labels, logits=logits)
        eval_metric_inputs = (pred, softmax_ce_loss)
        return eval_metric_inputs

    def build_eval_metric_ops(self, model_outputs, labels, features):
        if get_execution_mode() == ExecutionMode.Pipeline:
            pred = model_outputs[0]
            softmax_ce_loss = model_outputs[1]
        else:
            logits = model_outputs[0]
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            softmax_ce_loss = self.softmax_ce_loss(labels, logits=logits)

        metrics_dict = dict()
        mask = tf.cast(1 - features["input_mask"], softmax_ce_loss.dtype)
        # accuracy
        metrics_dict['eval/accuracy'] = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=pred, weights=mask
        )
        # perplexity
        unscaled_loss = tf.reduce_sum(
            input_tensor=tf.cast(softmax_ce_loss * mask, tf.float32,)
        )
        num_tokens = _compute_num_tokens(mask)
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
        mask = 1 - features['input_mask']
        cross_entropy = tf.reduce_sum(
            input_tensor=tf.cast(
                self.softmax_ce_loss(labels, logits=logits) * mask, tf.float32
            )
        )

        if self.loss_scaling == "num_tokens":
            scale = _compute_num_tokens(mask)
        elif self.loss_scaling == "batch_size":
            scale = self._get_batch_size(mode)
        else:
            raise ValueError(
                f"Loss scaling can't be set to {self.loss_scaling}. \
                Should be either 'num_tokens' or 'batch_size'"
            )

        return tf.cast(cross_entropy / scale, logits.dtype)

    def _write_summaries(self, features, loss):
        # Use GradAccumSummarySaverHook to add loss summaries when trained
        # with gradient accumulation
        if self._trainer.is_grad_accum():
            return

        # loss
        loss = tf.cast(loss, tf.float32)
        tf.compat.v1.summary.scalar('train/lm_cost', loss)

        # perplexity
        unscaled_loss = loss / self.loss_weight
        num_tokens = _compute_num_tokens(1 - features['input_mask'])

        if self.loss_scaling == "num_tokens":
            unscaled_loss *= num_tokens
        elif self.loss_scaling == "batch_size":
            unscaled_loss *= self._get_batch_size(
                mode=tf.estimator.ModeKeys.TRAIN
            )
        else:
            raise ValueError(
                f"Loss scaling can't be set to {self.loss_scaling}. \
                Should be either 'num_tokens' or 'batch_size'"
            )

        ppl = calculate_perplexity(unscaled_loss, num_tokens)
        tf.compat.v1.summary.scalar('train/lm_perplexity', ppl)
        # bits_per_byte
        bpb = calculate_bits_per_x(
            unscaled_loss,
            num_tokens,
            bits_per_type="per_byte",
            dataset=self.bits_per_x_dataset,
        )
        tf.compat.v1.summary.scalar("train/lm_bpb", bpb)

    def _get_batch_size(self, mode):
        # Set batch size based on the estimator mode.
        return (
            self.train_batch_size
            if mode == tf.estimator.ModeKeys.TRAIN
            else self.eval_batch_size
        )

    @property
    def trainer(self):
        return self._trainer


def _compute_num_tokens(mask):
    return tf.maximum(
        tf.cast(tf.reduce_sum(input_tensor=mask), tf.float32), 1e-5,
    )
