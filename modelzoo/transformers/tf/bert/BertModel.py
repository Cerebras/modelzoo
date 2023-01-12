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
from modelzoo.common.tf.layers.Input import SetupInputTensor
from modelzoo.common.tf.layers.LayerNormalizationLayer import (
    LayerNormalizationLayer,
)
from modelzoo.common.tf.metrics.perplexity import perplexity_metric
from modelzoo.common.tf.model_utils.create_initializer import create_initializer
from modelzoo.common.tf.optimizers.Trainer import Trainer
from modelzoo.common.tf.TFBaseModel import TFBaseModel
from modelzoo.transformers.tf.bert.layers.CLSLayer import CLSLayer
from modelzoo.transformers.tf.bert.layers.MLMLayer import MLMLayer
from modelzoo.transformers.tf.bert.layers.MLMLossLayer import MLMLossLayer
from modelzoo.transformers.tf.layers.Encoder import Encoder
from modelzoo.transformers.tf.transformer_utils import create_embedding_layers


class BertModel(TFBaseModel):
    """
    The BERT model https://arxiv.org/pdf/1810.04805.pdf

    :param dict params: Model configuration parameters.
    """

    def __init__(self, params, encode_only=False):

        super(BertModel, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        # Position and segment embedding params
        vocab_size = params["train_input"]["vocab_size"]
        use_position_embedding = params["model"]["use_position_embedding"]
        use_segment_embedding = params["model"]["use_segment_embedding"]
        position_embedding_type = params["model"]["position_embedding_type"]
        max_position_embeddings = params["model"]["max_position_embeddings"]
        embedding_size = params["model"]["embedding_size"]

        # Encoder params
        hidden_size = params["model"]["hidden_size"]
        num_heads = params["model"]["num_heads"]
        num_hidden_layers = params["model"]["num_hidden_layers"]
        filter_size = params["model"]["filter_size"]
        encoder_nonlinearity = params["model"]["encoder_nonlinearity"]
        dropout_rate = params["model"]["dropout_rate"]
        dropout_seed = params["model"]["dropout_seed"]
        attention_dropout_rate = params["model"]["attention_dropout_rate"]
        layer_norm_epsilon = params["model"]["layer_norm_epsilon"]
        use_ffn_bias = params["model"]["use_ffn_bias"]
        use_pre_normalization = params["model"]["use_pre_normalization"]
        attention_type = params["model"]["attention_type"]
        use_projection_bias_in_attention = params["model"][
            "use_projection_bias_in_attention"
        ]
        use_ffn_bias_in_attention = params["model"]["use_ffn_bias_in_attention"]

        # Task-specific layer params
        share_embedding_weights = params["model"]["share_embedding_weights"]
        num_cls_classes = params["model"]["num_cls_classes"]

        disable_nsp = params["model"]["disable_nsp"]
        if not encode_only and not disable_nsp:
            assert (
                num_cls_classes == 2
            ), "Number of classification classes must be 2 for NSP"

        # Loss params
        mlm_loss_weight = params["model"]["mlm_loss_weight"]

        # Whether to use model optimizations targeted at GPUs
        enable_gpu_optimizations = params["model"]["enable_gpu_optimizations"]

        # Batch size params
        train_batch_size = params["train_input"]["batch_size"]
        eval_batch_size = params["eval_input"]["batch_size"]

        # CS util params for layers
        boundary_casting = params["model"]["boundary_casting"]
        tf_summary = params["model"]["tf_summary"]

        # Weight initialization params
        initializer_spec = params["model"]["initializer"]
        embedding_initializer_spec = params["model"]["embedding_initializer"]
        weight_initialization_seed = params["model"][
            "weight_initialization_seed"
        ]

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

        # Set up layers
        (
            self.token_embedding_layer,
            self.position_embedding_layer,
            self.segment_embedding_layer,
        ) = create_embedding_layers(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            segment_embedding_size=hidden_size,
            embeddings_initializer=embedding_initializer,
            max_position_embeddings=max_position_embeddings,
            position_embeddings_type=position_embedding_type
            if use_position_embedding
            else None,
            position_embeddings_initializer=initializer,
            num_segments=2 if use_segment_embedding else None,
            segment_embeddings_initializer=initializer,
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
            # hidden dimension of the encoder.
            self.embedding_projection = DenseLayer(
                hidden_size,
                activation=None,
                use_bias=False,
                boundary_casting=boundary_casting,
                tf_summary=tf_summary,
                dtype=self.policy,
                name="input_embed_projection",
            )

        self.output_embedding_layer = EmbeddingLayer(
            vocab_size,
            embedding_size,
            embeddings_initializer=embedding_initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="output_embedding",
        )

        self.pre_encoder_layer_norm = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="pre_encoder_layer_norm",
        )

        self.dropout_layer = DropoutLayer(
            dropout_rate,
            seed=dropout_seed,
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

        self.post_encoder_layer_norm = LayerNormalizationLayer(
            epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="post_encoder_layer_norm",
        )

        # Task-specific layer: NSP
        # Used for next sentence prediction during BERT pretraining
        self.nsp_layer = CLSLayer(
            hidden_size,
            num_cls_classes,
            nonlinearity="tanh",
            use_bias=True,
            kernel_initializer=initializer,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            name="cls_layer",
        )

        # Task-specific layer: MLM
        self.mlm_layer = MLMLayer(
            hidden_size,
            vocab_size,
            embedding_size=embedding_size,
            use_ffn_bias=True,
            use_output_bias=True,
            kernel_initializer=initializer,
            layer_norm_epsilon=layer_norm_epsilon,
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
            enable_gpu_optimizations=enable_gpu_optimizations,
            name="mlm_layer",
        )

        # Loss layers: NSP loss
        self.nsp_loss_layer = CrossEntropyFromLogitsLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

        # Loss layers: MLM loss
        self.mlm_loss_layer = MLMLossLayer(
            boundary_casting=boundary_casting,
            tf_summary=tf_summary,
            dtype=self.policy,
        )

        # Model trainer
        self._trainer = Trainer(
            params=params["optimizer"],
            tf_summary=tf_summary,
            mixed_precision=params["model"]["mixed_precision"],
        )

        # Store params
        self.encode_only = encode_only
        self.disable_nsp = disable_nsp
        self.vocab_size = vocab_size
        self.use_factorized_embedding = use_factorized_embedding
        self.share_embedding_weights = share_embedding_weights
        self.mlm_loss_weight = mlm_loss_weight
        self.use_pre_normalization = use_pre_normalization
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.input_pad_id = params["model"]["input_pad_id"]
        self.segment_pad_id = params["model"]["segment_pad_id"]
        self.tf_summary = tf_summary

    def build_model(self, features, mode):
        """
        Build the model (up to loss).

        :param dict features: Input features.
        :param tf.estimator.ModeKeys mode: Mode (TRAIN, EVAL).
        :returns list: List of model outputs, where the 0th
            entry is the NSP branch output, the 1st entry
            is the MLM branch output, and the 2nd entry is
            the encoder outputs.
        """
        assert mode in [
            tf.estimator.ModeKeys.TRAIN,
            tf.estimator.ModeKeys.EVAL,
            tf.estimator.ModeKeys.PREDICT,
        ], f"The model supports estimator TRAIN, EVAL, and PREDICT modes."
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        input_ids = SetupInputTensor(
            features["input_ids"], tf_summary=self.tf_summary
        )

        encoder_input = self.token_embedding_layer(
            input_ids, pad_id=self.input_pad_id
        )

        if self.use_factorized_embedding:
            encoder_input = self.embedding_projection(encoder_input)

        if self.position_embedding_layer:
            encoder_input = self.position_embedding_layer(encoder_input)

        if self.segment_embedding_layer:
            seg_embeds = self.segment_embedding_layer(
                features["segment_ids"], pad_id=self.segment_pad_id
            )
            encoder_input += seg_embeds

        if not self.use_pre_normalization:
            encoder_input = self.pre_encoder_layer_norm(encoder_input)

        encoder_input = self.dropout_layer(encoder_input, training=is_training)

        encoder_output = self.encoder(
            encoder_input,
            self_attention_mask=features["input_mask"],
            training=is_training,
        )

        if self.use_pre_normalization:
            encoder_output = self.post_encoder_layer_norm(encoder_output)

        padding_mask = 1 - tf.cast(
            features["input_mask"], dtype=encoder_output.dtype
        )
        nsp_output = (
            self.nsp_layer(
                encoder_output, padding_mask=padding_mask, training=is_training,
            )
            if not self.encode_only and not self.disable_nsp
            else None
        )

        mlm_embedding = (
            self.token_embedding_layer
            if self.share_embedding_weights
            else self.output_embedding_layer
        )
        mlm_output = (
            self.mlm_layer(
                encoder_output,
                masked_lm_positions=features["masked_lm_positions"],
                embedding_table=mlm_embedding.embedding_table(),
            )
            if not self.encode_only
            else None
        )

        return nsp_output, mlm_output, encoder_output

    def build_total_loss(self, model_outputs, features, labels, mode):
        """
        Compute total loss. Also logs loss summaries.

        :param list model_outputs: nsp, mlm, encoder outputs returned
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

        assert (
            not self.encode_only
        ), "To train BertModel, encode_only must be False"

        nsp_loss = 0.0
        if not self.disable_nsp:
            nsp_loss = self._nsp_loss(labels, features, logits=model_outputs[0])

        batch_size = (
            self.train_batch_size
            if mode == tf.estimator.ModeKeys.TRAIN
            else self.eval_batch_size
        )
        mlm_loss = (
            self.mlm_loss_layer(
                features["masked_lm_ids"],
                features["masked_lm_weights"],
                model_outputs[1],
                batch_size,
            )
            * self.mlm_loss_weight
        )

        total_loss = nsp_loss + mlm_loss
        total_loss = tf.reduce_sum(input_tensor=total_loss)

        self._write_summaries(
            features, total_loss, nsp_loss, mlm_loss,
        )

        self._nsp_loss = nsp_loss
        self._mlm_loss = mlm_loss

        return total_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self._trainer.build_train_ops(total_loss)

    def build_eval_metric_inputs(self, model_outputs, labels, features):
        """
        Build inputs for eval metrics computations.

        :param model_outputs: Model output returned by call method.
        :param labels: Tensor of shape (batch_size,). Contains
                       next sentence labels used for bert pretraining.
        :param features: Dictionary of input features.

        :returns: `eval_metric_inputs`: tuple containing:
                -- `nsp_output`: NSP branch output returned by call method;
                -- `mlm_pred`: MLM prediction tensor;
                -- `mlm_xentr` MLM cross entropy tensor.
        """

        features["input_mask"] = tf.cast(features["input_mask"], tf.float16)

        nsp_output = model_outputs[0]
        mlm_output = model_outputs[1]

        mlm_pred = tf.argmax(mlm_output, axis=-1, output_type=tf.int32)

        mlm_xentr = CrossEntropyFromLogitsLayer(dtype=self.policy)(
            features["masked_lm_ids"], logits=mlm_output
        )

        eval_metric_inputs = (nsp_output, mlm_pred, mlm_xentr)
        return eval_metric_inputs

    def build_eval_metric_ops(self, eval_metric_inputs, labels, features):
        """
        Compute BERT eval metrics (currently NSP/MLM accuracy).

        :param eval_metric_inputs: tuple containing:
                -- `nsp_output`: NSP branch output returned by call method;
                -- `mlm_pred`: MLM prediction tensor;
                -- `mlm_xentr` MLM cross entropy tensor.
        :param labels: Tensor of shape (batch_size,). Contains
                       next sentence labels used for bert pretraining.
        :param features: Dictionary of input features.

        :returns: Dict of metric results keyed by name.
            The values of the dict can be one of the following:
            (1) instance of Metric class. (2) Results of calling a metric
            function, namely a (metric_tensor, update_op) tuple.
        """
        assert (
            not self.encode_only
        ), "To evaluate BertModel, encode_only must be False"

        nsp_output, mlm_pred, mlm_xentr = eval_metric_inputs

        metrics_dict = dict()
        if not self.disable_nsp:
            nsp_pred = tf.argmax(nsp_output, axis=-1, output_type=tf.int32)
            metrics_dict["eval/accuracy_cls"] = tf.compat.v1.metrics.accuracy(
                labels=labels, predictions=nsp_pred
            )

        weights = tf.where(
            tf.cast(features["masked_lm_weights"], tf.bool), 1, 0
        )
        metrics_dict["eval/accuracy_masked_lm"] = tf.compat.v1.metrics.accuracy(
            labels=features["masked_lm_ids"],
            predictions=mlm_pred,
            weights=weights,
        )

        # We avoid using mlm_loss_layer here in order to sum MLM Loss in FP32,
        # which helps prevent overflow issues.
        unscaled_mlm_loss = tf.reduce_sum(
            input_tensor=tf.cast(
                mlm_xentr * tf.cast(weights, mlm_xentr.dtype), tf.float32,
            )
        )
        num_masked = tf.reduce_sum(weights)
        metrics_dict["eval/mlm_perplexity"] = perplexity_metric(
            unscaled_mlm_loss, num_masked
        )

        return metrics_dict

    def _nsp_loss(self, labels, features, logits):
        """
        NSP loss.
        """
        assert (
            not self.encode_only and not self.disable_nsp
        ), f"NSP loss can't be used unless the NSP head is enabled."
        loss = tf.reduce_mean(
            tf.cast(self.nsp_loss_layer(labels, logits=logits), tf.float32),
            name='nsp_loss',
        )
        return tf.cast(loss, logits.dtype)

    def _write_summaries(
        self, features, total_loss, nsp_loss, mlm_loss,
    ):
        """
        Write train metrics summaries

        :param features: Dictionary of input features.
        :param total_loss: total loss tensor
        :param nsp_loss: NSP loss tensor
        :param mlm_loss: MLM loss tensor
        """

        # Use GradAccumSummarySaverHook to add loss summaries when trained
        # with gradient accumulation
        if self._trainer.is_grad_accum():
            return

        total_loss = tf.cast(total_loss, tf.float32)
        nsp_loss = tf.cast(nsp_loss, tf.float32)
        mlm_loss = tf.cast(mlm_loss, tf.float32)

        # Log losses: total, nsp, and mlm
        tf.compat.v1.summary.scalar('train/cost', total_loss)
        tf.compat.v1.summary.scalar('train/cost_cls', nsp_loss)
        tf.compat.v1.summary.scalar('train/cost_masked_lm', mlm_loss)

        # Log total loss per masked element
        batch_size = float(features["input_ids"].shape[0])
        total_loss_per_masked_element = batch_size * nsp_loss
        nmasked = tf.maximum(
            tf.reduce_sum(tf.cast(features["masked_lm_weights"], tf.float32),),
            1e-5,
        )

        total_loss_per_masked_element += batch_size * mlm_loss
        total_loss_per_masked_element /= nmasked

        tf.compat.v1.summary.scalar(
            "train/c/elem", total_loss_per_masked_element
        )

    @property
    def trainer(self):
        return self._trainer

    @property
    def nsp_loss(self):
        if not hasattr(self, "_nsp_loss"):
            raise AttributeError(
                "NSP loss not defined. "
                + "Make sure build_total_loss was called after build_model."
            )
        return self._nsp_loss

    @property
    def mlm_loss(self):
        if not hasattr(self, "_mlm_loss"):
            raise AttributeError(
                "MLM loss not defined. "
                + "Make sure build_total_loss was called after build_model."
            )
        return self._mlm_loss
