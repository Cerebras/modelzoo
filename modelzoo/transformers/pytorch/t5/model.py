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

from modelzoo.common.pytorch.metrics import AccuracyMetric, PerplexityMetric
from modelzoo.common.pytorch.model_utils.T5ForConditionalGenerationLoss import (
    T5ForConditionalGenerationLoss,
)
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.pytorch.t5.t5_model import T5ForConditionalGeneration
from modelzoo.transformers.pytorch.t5.utils import set_custom_stack_params


class T5ForConditionalGenerationModel(PyTorchBaseModel):
    """
    T5 models
    """

    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = T5ForConditionalGenerationLoss(
            self.params["model"].get("lm_loss_weight", 1.0),
            mlm_loss_scaling=self.params["model"].get(
                "mlm_loss_scaling", "batch_size"
            ),
            label_smoothing=self.params["model"].get("label_smoothing", 0.0),
        )

        super(T5ForConditionalGenerationModel, self).__init__(
            params=params, model=self.model, device=device
        )

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )
        if self.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy_lm")
            self.perplexity_metric = PerplexityMetric(name="eval/perplexity_lm")

        # Add custom Cerebras stack flags
        set_custom_stack_params(params)

    def _post_device_transfer(self):
        self.model.tie_weights()

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
                "decoder_num_hidden_layers"
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
            "tie_word_embeddings": model_params.pop(
                "share_embedding_weights", True,
            ),
            "use_t5_layer_norm": model_params.pop("use_t5_layer_norm", True),
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
                "use_pre_encoder_decoder_layer_norm", False
            ),
            "use_ffn_bias": model_params.pop("use_ffn_bias", False),
            "lm_loss_weight": model_params.pop("lm_loss_weight", 1.0),
            "use_transformer_initialization": model_params.pop(
                "use_transformer_initialization", False
            ),
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

        model_params.pop("to_float16", None)
        model_params.pop("mixed_precision", None)

        model = T5ForConditionalGeneration(**kwargs)

        self.enable_vts = model_params.pop("enable_vts", False)
        if self.enable_vts:
            from modelzoo.common.pytorch import cbtorch

            self.vts = cbtorch.nn.StripPadding()
        else:
            self.vts = None

        if len(model_params) > 0:
            logging.warning(
                "The following model params are unused: "
                + ", ".join(model_params.keys())
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
        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1).long(),)
        if weights is not None:
            weights = weights.detach()
            loss = loss * weights.view(-1)
        return loss.sum()

    def __call__(self, data):
        if self.enable_vts and not self.model.training:
            self.enable_vts = False
            logging.info(
                "VTS is only supported in train mode. Disabling for the "
                "current run."
            )
        if self.enable_vts:
            data["input_ids"] = self.vts(
                data["input_ids"], data["attention_mask"]
            )
            data["decoder_input_ids"] = self.vts(
                data["decoder_input_ids"], data["decoder_attention_mask"]
            )
            data["labels"] = self.vts(
                data["labels"], data["decoder_attention_mask"]
            )
        kwargs = {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
            "decoder_input_ids": data["decoder_input_ids"],
            "decoder_attention_mask": data["decoder_attention_mask"],
            "labels": data["labels"],
            "loss_weight": data.get("loss_weight", None),
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
