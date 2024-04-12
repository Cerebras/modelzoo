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

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss
from cerebras.modelzoo.models.nlp.gptj.gptj_model import GPTJModel
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


@registry.register_model(
    ["gptj", "falcon"],
    datasetprocessor=[
        "HuggingFaceDataProcessorEli5",
        "HuggingFaceIterableDataProcessorEli5",
        "DummyDataProcessor",
        "DummyIterableDataProcessor",
        "GptHDF5DataProcessor",
        "GptHDF5MapDataProcessor",
    ],
)
class GptjModel(torch.nn.Module):
    """
    GPT-2 models
    """

    def __init__(self, params):
        super().__init__()

        model_params = params["model"].copy()

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )
        if self.compute_eval_metrics:
            self.perplexity_metric = PerplexityMetric(name="eval/lm_perplexity")
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")

        self.model = self.build_model(model_params)

        self.loss_fn = GPTLMHeadModelLoss(
            params["model"]["vocab_size"],
            self.loss_scaling,
            self.loss_weight,
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

        assert (
            position_embedding_type != "alibi"
        ), "alibi position embedding is not yet supported by gptj"

        rope_theta = model_params.pop("rope_theta", 10000)
        pad_rope = model_params.pop("pad_rope", False)
        rotary_dim = None
        num_relative_attention_buckets = None
        if position_embedding_type == "rotary":
            rotary_dim = model_params.pop(
                "rotary_dim",
                int(
                    model_params["hidden_size"]
                    // model_params["num_heads"]
                    * 0.25
                ),
            )
            # https://github.com/huggingface/transformers/blob/f0577df6de36e7e7f28e90fa76da0657de038a39/src/transformers/models/gpt_neox/modeling_gpt_neox.py#L84-L85
            # https://arxiv.org/pdf/2104.09864.pdf Section 3.3
            assert (
                rotary_dim
                <= model_params["hidden_size"] / model_params["num_heads"]
            ), "Rotary dimensions should be <= hidden size divided by number of attention heads."
            assert (
                rotary_dim % 2 == 0
            ), "Rotary dimension must be an even number."
        else:
            # relative PE
            num_relative_attention_buckets = model_params.pop(
                "num_relative_attention_buckets", 32
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
            rotary_dim=rotary_dim,
            rope_theta=rope_theta,
            pad_rope=pad_rope,
            num_relative_attention_buckets=num_relative_attention_buckets,
            # Decoder params
            num_hidden_layers=model_params.pop("num_hidden_layers"),
            filter_size=model_params.pop("filter_size"),
            dropout_rate=model_params.pop("residual_dropout_rate", 0.1),
            nonlinearity=model_params.pop("nonlinearity", "gelu"),
            norm_type=model_params.pop("norm_type", "layernorm"),
            layer_norm_epsilon=float(
                model_params.pop("layer_norm_epsilon", 1.0e-5)
            ),
            use_ffn_bias=model_params.pop("use_ffn_bias", False),
            use_untied_layer_norm=model_params.pop(
                "use_untied_layer_norm", False
            ),
            # Attention params
            num_heads=model_params.pop("num_heads"),
            attention_module=model_params.pop(
                "attention_module", "aiayn_attention"
            ),
            attention_sliding_window_length=model_params.pop(
                "attention_sliding_window_length", None
            ),
            extra_attention_params=model_params.pop(
                "extra_attention_params", {}
            ),
            attention_type=attention_type,
            attention_dropout_rate=model_params.pop(
                "attention_dropout_rate", 0.1
            ),
            attention_softmax_fp32=model_params.pop(
                "attention_softmax_fp32", True
            ),
            attention_kernel=model_params.pop("attention_kernel", None),
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
            alibi_trainable_slopes=model_params.pop(
                "alibi_trainable_slopes", False
            ),
            pos_scaling_factor=float(
                model_params.pop("pos_scaling_factor", 1.0)
            ),
        )

        model_params.pop("mixed_precision", None)
        # `fp16_type` is accessed later,
        # so we remove these from the list of unused params
        unused_params = [
            key for key in model_params.keys() if key != "fp16_type"
        ]
        if unused_params:
            logging.warning(
                "The following model params are unused: "
                + ", ".join(unused_params)
            )
        return model

    def forward(
        self,
        data,
        reduce_batch=True,
        average_logps=False,
        output_logits=False,
    ):
        assert (
            "input_ids" in data
            and "attention_mask" in data
            and "labels" in data
        ), "GPT-J model expects these data fields: input_ids, attention_mask, labels"
        assert (
            data["input_ids"].dtype == torch.int32
            and data["attention_mask"].dtype == torch.int32
            and data["labels"].dtype == torch.int32
        ), "The dtype for all inputs should be torch.int32"

        lm_logits = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            attention_span=data.get("attention_span"),  # VSL-only input
            position_ids=data.get("position_ids"),  # VSL-only input
        )
        loss = self.loss_fn(
            lm_logits,
            data["labels"],
            data["attention_mask"],
            reduce_batch=reduce_batch,
            average_logps=average_logps,
        )

        # Calculate eval metrics if not training
        if not self.model.training and self.compute_eval_metrics:
            lm_labels = data["labels"].clone()
            lm_weights = data["attention_mask"].clone()
            lm_preds = lm_logits.argmax(-1).int()

            self.accuracy_metric(
                labels=lm_labels,
                predictions=lm_preds,
                weights=lm_weights,
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
                labels=lm_labels,
                loss=unscaled_loss,
                weights=lm_weights,
            )

        if output_logits:
            return loss, lm_logits
        else:
            return loss
