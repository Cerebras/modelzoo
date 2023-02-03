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
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.pytorch.gptj.gptj_model import GPTJModel


class GptjModel(PyTorchBaseModel):
    """
    GPT-2 models
    """

    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        self.model = self.build_model(model_params)
        self.loss_fn = GPTLMHeadModelLoss(
            params["model"]["vocab_size"], self.loss_weight,
        )

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )
        if self.compute_eval_metrics:
            self.perplexity_metric = PerplexityMetric(name="eval/lm_perplexity")
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")

        super(GptjModel, self).__init__(
            params=params, model=self.model, device=device
        )

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, model_params):
        attention_type = model_params.pop("attention_type")
        if attention_type not in ["scaled_dot_product", "dot_product"]:
            raise ValueError(
                "attention_type should be 'scaled_dot_product' or 'dot_product'."
            )

        position_embedding_type = model_params.pop(
            "position_embedding_type", "rotary"
        ).lower()

        self.loss_weight = model_params.pop("loss_weight", 1.0)

        model = GPTJModel(
            hidden_size=model_params.pop("hidden_size"),
            # Embedding params
            vocab_size=model_params.pop("vocab_size"),
            max_position_embeddings=model_params.pop(
                "max_position_embeddings", 1024
            ),
            embd_pdrop=model_params.pop("embedding_dropout_rate", 0.1),
            share_embedding_weights=model_params.pop(
                "share_embedding_weights", True
            ),
            position_embedding_type=position_embedding_type,
            rotary_dim=model_params.pop("rotary_dim", None),
            # Decoder params
            num_hidden_layers=model_params.pop("num_hidden_layers"),
            filter_size=model_params.pop("filter_size"),
            dropout_rate=model_params.pop("residual_dropout_rate", 0.1),
            nonlinearity=model_params.pop("nonlinearity", "gelu"),
            layer_norm_epsilon=float(
                model_params.pop("layer_norm_epsilon", 1.0e-5)
            ),
            use_ffn_bias=model_params.pop("use_ffn_bias", False),
            use_untied_layer_norm=model_params.pop(
                "use_untied_layer_norm", False
            ),
            # Attention params
            num_heads=model_params.pop("num_heads"),
            attention_type=attention_type,
            attention_dropout_rate=model_params.pop(
                "attention_dropout_rate", 0.1
            ),
            attention_softmax_fp32=model_params.pop(
                "attention_softmax_fp32", True
            ),
            use_projection_bias_in_attention=model_params.pop(
                "use_projection_bias_in_attention", True
            ),
            use_ffn_bias_in_attention=model_params.pop(
                "use_ffn_bias_in_attention", True
            ),
            # Task-specific
            initializer_range=model_params.pop("initializer_range", 0.02),
            use_bias_in_output=model_params.pop("use_bias_in_output", False),
            norm_first=model_params.pop("norm_first", True),
            embedding_initializer=model_params.pop(
                "embedding_initializer", None
            ),
            attention_initializer=model_params.pop("initializer", None),
            output_layer_initializer=model_params.pop(
                "output_layer_initializer", None
            ),
        )

        model_params.pop("mixed_precision", None)
        # `use_bfloat16` is accessed later, so we remove it from the list of unused params
        unused_params = [
            key for key in model_params.keys() if key != "use_bfloat16"
        ]
        if unused_params:
            logging.warning(
                "The following model params are unused: "
                + ", ".join(unused_params)
            )
        logging.root.setLevel(logging.INFO)
        return model

    def __call__(self, data):
        lm_logits = self.model(**data)
        loss = self.loss_fn(lm_logits, data["labels"], data["attention_mask"],)

        # Calculate eval metrics if not training
        if not self.model.training and self.compute_eval_metrics:
            lm_labels = data["labels"].clone()
            lm_weights = data["attention_mask"].clone()
            lm_preds = lm_logits.argmax(-1).int()

            self.accuracy_metric(
                labels=lm_labels, predictions=lm_preds, weights=lm_weights,
            )

            unscaled_loss = loss * torch.sum(
                data["attention_mask"].clone(), dtype=torch.float32
            )

            self.perplexity_metric(
                labels=lm_labels, loss=unscaled_loss, weights=lm_weights,
            )

        return loss
