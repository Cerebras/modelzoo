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

from modelzoo.common.pytorch.metrics import AccuracyMetric, PerplexityMetric
from modelzoo.common.pytorch.model_utils.GPTLMHeadModelLoss import (
    GPTLMHeadModelLoss,
)
from modelzoo.transformers.pytorch.gpt2.gpt2_model import GPT2LMHeadModel
from modelzoo.transformers.pytorch.gpt2.utils import set_custom_stack_params


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

        # Add custom Cerebras stack flags
        set_custom_stack_params()

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, model_params):
        if model_params.pop("position_embedding_type") != "learned":
            raise NotImplementedError(
                "Only learned position embeddings are supported."
            )

        self.loss_weight = model_params.pop("loss_weight", 1.0)
        self.loss_scaling = model_params.pop(
            "loss_scaling", "num_tokens"
        ).lower()
        if self.loss_weight != 1.0 and self.loss_scaling == "num_tokens":
            self.loss_scaling = "batch_size"
            logging.warning(
                "loss_weight and loss_scaling don't match, fall back to using 'batch_size' for loss_scaling"
            )
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
            position_embedding_type=model_params.pop(
                "position_embedding_type", "learned"
            )
            if model_params.pop("use_position_embedding", True)
            else None,
            hidden_size=model_params.pop("hidden_size"),
            share_embedding_weights=model_params.pop(
                "share_embedding_weights", True
            ),
            embedding_layer_norm=model_params.pop(
                "embedding_layer_norm", False
            ),
            # Encoder
            num_hidden_layers=model_params.pop("num_hidden_layers"),
            dropout_rate=default_dropout_rate,
            layer_norm_epsilon=float(
                model_params.pop("layer_norm_epsilon", 1.0e-5),
            ),
            # Encoder - Attention
            num_heads=model_params.pop("num_heads"),
            attention_type=model_params.pop("attention_type"),
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
            loss_weight=self.loss_weight,
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
            loss_scaling=self.loss_scaling,
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

    def __call__(self, data):
        lm_logits = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"],
        )
        loss = self.loss_fn(
            lm_logits,
            labels=data["labels"],
            attention_mask=data["attention_mask"],
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
                    data["attention_mask"].clone(), dtype=torch.float32
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
