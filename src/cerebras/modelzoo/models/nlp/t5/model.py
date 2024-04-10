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

import logging

import torch.nn as nn

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.losses.T5ForConditionalGenerationLoss import (
    T5ForConditionalGenerationLoss,
)
from cerebras.modelzoo.models.nlp.t5.t5_model import T5ForConditionalGeneration
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


@registry.register_model(
    "t5", datasetprocessor=["T5HDF5DataProcessor", "T5DynamicDataProcessor"]
)
@registry.register_model(
    "transformer", datasetprocessor=["TransformerDynamicDataProcessor"]
)
class T5ForConditionalGenerationModel(nn.Module):
    """
    T5 models
    """

    def __init__(self, params):
        super().__init__()

        model_params = params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = T5ForConditionalGenerationLoss(
            params["model"].get("lm_loss_weight", 1.0),
            mlm_loss_scaling=params["model"].get(
                "mlm_loss_scaling", "batch_size"
            ),
            label_smoothing=params["model"].get("label_smoothing", 0.0),
        )

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )
        if self.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy_lm")
            self.perplexity_metric = PerplexityMetric(name="eval/perplexity_lm")

    def _post_device_transfer(self):
        self.model.tie_weights()

    def model_class(self):
        return T5ForConditionalGeneration

    def build_model(self, model_params):
        model = None
        kwargs = {
            "src_vocab_size": model_params.pop("src_vocab_size"),
            "tgt_vocab_size": model_params.pop("tgt_vocab_size", None),
            "mlm_loss_scaling": model_params.pop(
                "mlm_loss_scaling", "batch_size"
            ),
            "label_smoothing": model_params.pop("label_smoothing", 0.0),
            "extra_ids": model_params.pop("extra_ids", 0),
            "d_model": model_params.pop("d_model"),
            "d_kv": model_params.pop("d_kv"),
            "d_ff": model_params.pop("d_ff"),
            "encoder_num_hidden_layers": model_params.pop(
                "encoder_num_hidden_layers"
            ),
            "decoder_num_hidden_layers": model_params.pop(
                "decoder_num_hidden_layers", None
            ),
            "num_heads": model_params.pop("num_heads"),
            "use_projection_bias_in_attention": model_params.pop(
                "use_projection_bias_in_attention", False
            ),
            "relative_attention_num_buckets": model_params.pop(
                "relative_attention_num_buckets", 32
            ),
            # This param ties weights between lm_head and
            # decoder.embed_tokens layers.
            "share_embedding_weights": model_params.pop(
                "share_embedding_weights",
                True,
            ),
            "norm_type": model_params.pop("norm_type", "rmsnorm"),
            "dropout_rate": model_params.pop("dropout_rate"),
            "layer_norm_epsilon": float(
                model_params.pop("layer_norm_epsilon", 1.0e-5),
            ),
            "encoder_nonlinearity": model_params.pop("encoder_nonlinearity"),
            "decoder_nonlinearity": model_params.pop("decoder_nonlinearity"),
            "position_embedding_type": model_params.pop(
                "position_embedding_type", "relative"
            ),
            "src_max_position_embeddings": model_params.pop(
                "src_max_position_embeddings"
            ),
            "tgt_max_position_embeddings": model_params.pop(
                "tgt_max_position_embeddings"
            ),
            "use_dropout_outside_residual_path": model_params.pop(
                "use_dropout_outside_residual_path", True
            ),
            # This param ties weights between encoder.embed_tokens and
            # decoder.embed_tokens layers.
            "share_encoder_decoder_embedding": model_params.pop(
                "share_encoder_decoder_embedding", True
            ),
            "relu_dropout_rate": model_params.pop("relu_dropout_rate", None),
            "use_pre_encoder_decoder_dropout": model_params.pop(
                "use_pre_encoder_decoder_dropout", False
            ),
            "use_pre_encoder_decoder_layer_norm": model_params.pop(
                "use_pre_encoder_decoder_layer_norm", True
            ),
            "use_ffn_bias": model_params.pop("use_ffn_bias", False),
            "lm_loss_weight": model_params.pop("lm_loss_weight", 1.0),
            "use_transformer_initialization": model_params.pop(
                "use_transformer_initialization", False
            ),
            "attention_softmax_fp32": model_params.pop(
                "attention_softmax_fp32", True
            ),
            "attention_kernel": model_params.pop("attention_kernel", None),
        }

        # Updating input and model params to account extra ids
        # for T5 Language Modeling task.
        extra_ids = kwargs.pop("extra_ids", 0)
        kwargs["src_vocab_size"] += extra_ids

        # T5 model has the same vocabulary size for source and target
        # sequences.
        if kwargs["tgt_vocab_size"] is None:
            kwargs["tgt_vocab_size"] = kwargs["src_vocab_size"]
        else:
            kwargs["tgt_vocab_size"] += extra_ids

        # T5 model does not distinguish dropout rate for
        # after relu computations, and utilizes the common dropout rate
        # across the whole model. Transformer, however, is using `0`
        # dropout rate there.
        if kwargs["relu_dropout_rate"] is None:
            kwargs["relu_dropout_rate"] = kwargs["dropout_rate"]

        model_params.pop("mixed_precision", None)

        cls = self.model_class()
        model = cls(**kwargs)

        unused_params = [
            key for key in model_params.keys() if key not in ["fp16_type"]
        ]
        if unused_params:
            logging.warning(
                "The following model params are unused: "
                + ", ".join(unused_params)
            )
        return model

    def _xentropy_loss(self, labels, logits, weights=None):
        """
        Calculates MLM Cross-Entropy (to be used for Perplexity calculation)

        Args:
            labels: Tensor of shape (batch, sequence) and type int32.
            logits: Tensor of shape (batch, sequence, vocab) and type float.
            weights: Optional float Tensor of shape (batch, sequence).
        Returns:
            The loss tensor

        """
        labels = labels.detach()
        logits = logits.detach()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        vocab_size = logits.shape[2]
        loss = loss_fct(
            logits.view(-1, vocab_size),
            labels.view(-1).long(),
        )
        if weights is not None:
            weights = weights.detach()
            loss = loss * weights.view(-1)
        return loss.sum()

    def forward(self, data):
        kwargs = {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
            "decoder_input_ids": data["decoder_input_ids"],
            "decoder_attention_mask": data["decoder_attention_mask"],
            "labels": data["labels"],
        }
        logits = self.model(**kwargs)
        loss = None
        if data["labels"] is not None:
            loss = self.loss_fn(
                logits,
                data["labels"],
                data["decoder_attention_mask"],
                data.get("loss_weight", None),
            ).to(logits.dtype)

        # Calculate eval metrics if not training
        if not self.model.training and self.compute_eval_metrics:
            labels = data["labels"].clone()
            decoder_mask = (
                data["decoder_attention_mask"].clone().to(logits.dtype)
            )
            predictions = logits.argmax(-1).int()

            self.accuracy_metric(
                labels=labels,
                predictions=predictions,
                weights=decoder_mask,
                dtype=logits.dtype,
            )

            # eval/perplexity_lm
            cross_entropy_loss = self._xentropy_loss(
                labels, logits, decoder_mask
            )
            self.perplexity_metric(
                labels=labels,
                loss=cross_entropy_loss,
                weights=decoder_mask,
                dtype=logits.dtype,
            )
        return loss
