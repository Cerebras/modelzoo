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

import torch

from cerebras_pytorch.metrics import AccuracyMetric, PerplexityMetric
from modelzoo.common.pytorch.model_utils.GPTLMHeadModelLoss import (
    GPTLMHeadModelLoss,
)
from modelzoo.transformers.pytorch.gpt2.gpt2_model import GPT2LMHeadModel


class Gpt2Model(torch.nn.Module):
    """
    GPT-2 models
    """

    def __init__(self, params):
        super().__init__()

        model_params = params["model"].copy()
        self.model = self.build_model(model_params)

        self.loss_fn = GPTLMHeadModelLoss(
            params["model"]["vocab_size"], self.loss_scaling, self.loss_weight,
        )
        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )
        if self.compute_eval_metrics:
            self.perplexity_metric = PerplexityMetric(name="eval/lm_perplexity")
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, model_params):

        position_embedding_type = (
            model_params.pop("position_embedding_type", "learned")
            if model_params.pop("use_position_embedding", True)
            else None
        )

        self.loss_weight = model_params.pop("loss_weight", 1.0)
        self.loss_scaling = model_params.pop(
            "loss_scaling", "num_tokens"
        ).lower()
        if self.loss_weight != 1.0 and self.loss_scaling == "num_tokens":
            logging.warning(
                f"loss_weight cannot be {self.loss_weight} for num_tokens "
                f"loss_scaling. Setting loss_weight to 1.0."
            )
            self.loss_weight = 1.0
        self.output_logits_scale = model_params.pop("output_logits_scale", None)
        self.scale_qk_dot_by_d = model_params.pop("scale_qk_dot_by_d", None)
        self.embeddings_scale = model_params.pop("embeddings_scale", 1.0)
        default_dropout_rate = model_params.pop("dropout_rate")
        embedding_dropout_rate = model_params.pop(
            "embedding_dropout_rate", default_dropout_rate
        )
        attention_dropout_rate = model_params.pop(
            "attention_dropout_rate", default_dropout_rate
        )

        model_params.pop("mixed_precision", None)

        model = GPT2LMHeadModel(
            # Embedding
            vocab_size=model_params.pop("vocab_size"),
            max_position_embeddings=model_params.pop(
                "max_position_embeddings", 1024
            ),
            embd_pdrop=embedding_dropout_rate,
            position_embedding_type=position_embedding_type,
            position_embedding_offset=model_params.pop(
                "position_embedding_offset", 0
            ),
            hidden_size=model_params.pop("hidden_size"),
            share_embedding_weights=model_params.pop(
                "share_embedding_weights", True
            ),
            embedding_layer_norm=model_params.pop(
                "embedding_layer_norm", False
            ),
            num_relative_attention_buckets=model_params.pop(
                "num_relative_attention_buckets", 32
            ),
            rotary_dim=model_params.pop("rotary_dim", None),
            # Encoder
            num_hidden_layers=model_params.pop("num_hidden_layers"),
            dropout_rate=default_dropout_rate,
            norm_type=model_params.pop("norm_type", "layernorm"),
            layer_norm_epsilon=float(
                model_params.pop("layer_norm_epsilon", 1.0e-5),
            ),
            # Encoder - Attention
            num_heads=model_params.pop("num_heads"),
            attention_type=model_params.pop("attention_type"),
            attention_module=model_params.pop(
                "attention_module", "aiayn_attention"
            ),
            extra_attention_params=model_params.pop(
                "extra_attention_params", {}
            ),
            use_projection_bias_in_attention=model_params.pop(
                "use_projection_bias_in_attention", True
            ),
            use_ffn_bias_in_attention=model_params.pop(
                "use_ffn_bias_in_attention", True
            ),
            attention_dropout_rate=attention_dropout_rate,
            attention_softmax_fp32=model_params.pop(
                "attention_softmax_fp32", True
            ),
            attention_kernel=model_params.pop("attention_kernel", None),
            # Encoder - ffn
            filter_size=model_params.pop("filter_size"),
            nonlinearity=model_params.pop("nonlinearity", "gelu"),
            use_ffn_bias=model_params.pop("use_ffn_bias", True),
            # Task-specific
            use_bias_in_output=model_params.pop("use_bias_in_output", False),
            fixed_sparse_attention=model_params.pop(
                "fixed_sparse_attention", None
            ),
            # Initializers
            embedding_initializer=model_params.pop(
                "embedding_initializer", None
            ),
            initializer=model_params.pop("initializer", None),
            output_layer_initializer=model_params.pop(
                "output_layer_initializer", None
            ),
            initializer_range=model_params.pop("initializer_range", 0.02),
            # muP (maximal update parameterization)  parameters
            output_logits_scale=self.output_logits_scale,
            embeddings_scale=self.embeddings_scale,
            scale_qk_dot_by_d=self.scale_qk_dot_by_d,
            alibi_trainable_slopes=model_params.pop(
                "alibi_trainable_slopes", False
            ),
            scale_qk_dot_by_layer_idx=model_params.pop(
                "scale_qk_dot_by_layer_idx", False
            ),
        )

        # `use_bfloat16` and `precision_opt_level` are accessed later,
        # so we remove these from the list of unused params
        unused_params = [
            key
            for key in model_params.keys()
            if key != "use_bfloat16" and key != "precision_opt_level"
        ]
        if unused_params:
            logging.warning(
                "The following model params are unused: "
                + ", ".join(unused_params)
            )

        return model

    def forward(self, data):
        # Note: attention_mask is a misnomer in this model and actually acts as
        # a loss mask. In the model computation its contents are ignored and
        # only its shape is used.
        assert (
            "input_ids" in data
            and "attention_mask" in data
            and "labels" in data
        ), "GPT-2 model expects these data fields: input_ids, attention_mask, labels"
        assert (
            data["input_ids"].dtype == torch.int32
            and data["attention_mask"].dtype == torch.int32
            and data["labels"].dtype == torch.int32
        ), "The dtype for all inputs should be torch.int32"

        lm_logits = self.model(
            input_ids=data["input_ids"],
            attention_mask=data[
                "attention_mask"
            ],  # doesn't actually mask anything
        )
        loss = self.loss_fn(
            lm_logits,
            labels=data["labels"],
            attention_mask=data["attention_mask"],  # acts as a loss mask
        )

        # Calculate eval metrics if not training
        if not self.model.training and self.compute_eval_metrics:
            lm_labels = data["labels"].clone()
            lm_weights = data["attention_mask"].to(lm_logits.dtype).clone()
            lm_preds = lm_logits.argmax(-1).int()

            self.accuracy_metric(
                labels=lm_labels, predictions=lm_preds, weights=lm_weights,
            )

            if self.loss_scaling == "num_tokens":
                unscaled_loss = loss * torch.sum(
                    lm_weights, dtype=torch.float32
                )
            elif self.loss_scaling == "batch_size":
                unscaled_loss = loss * torch.tensor(
                    lm_labels.shape[0] / self.loss_weight, dtype=torch.float32
                )
            else:
                raise ValueError(
                    f"Loss scaling can't be set to {self.loss_scaling}. \
                    Should be either 'num_tokens' or 'batch_size'"
                )

            self.perplexity_metric(
                labels=lm_labels, loss=unscaled_loss, weights=lm_weights,
            )

        return loss
