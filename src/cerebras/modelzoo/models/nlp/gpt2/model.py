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
from copy import deepcopy
from typing import Optional

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss
from cerebras.modelzoo.models.nlp.gpt2.gpt2_model import GPT2LMHeadModel
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


@registry.register_model(
    [
        "gpt2",
        "btlm",
        "bloom",
        "llama",
        "mistral",
        "mpt",
        "santacoder",
        "starcoder",
        "jais",
    ],
    datasetprocessor=[
        "GptHDF5DataProcessor",
        "HuggingFaceDataProcessorEli5",
        "HuggingFaceIterableDataProcessorEli5",
        "DummyDataProcessor",
        "DummyIterableDataProcessor",
    ],
)
class Gpt2Model(torch.nn.Module):
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
        position_embedding_type = model_params.pop(
            "position_embedding_type", "learned"
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
        self.embeddings_scale = model_params.pop("embeddings_scale", None)
        if self.embeddings_scale is None:
            self.embeddings_scale = 1.0
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
            rope_theta=model_params.pop("rope_theta", 10000),
            pad_rope=model_params.pop("pad_rope", False),
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
            attention_sliding_window_length=model_params.pop(
                "attention_sliding_window_length", None
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
            pos_scaling_factor=float(
                model_params.pop("pos_scaling_factor", 1.0)
            ),
            scale_qk_dot_by_layer_idx=model_params.pop(
                "scale_qk_dot_by_layer_idx", False
            ),
        )

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
        """The forward pass on the input data. This method
        returns the loss tensor if `output_logits` is False.
        If `output_logits` is True, the model call will also
        return the output logits tensor in addition to the
        loss as a (loss, lm_logits) tuple.

        This may be useful for performing post processing on
        the model's output logits.
        """
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
            attention_span=data.get("attention_span"),  # VSL-only input
            position_ids=data.get("position_ids"),  # VSL-only input
        )
        loss = self.loss_fn(
            lm_logits,
            labels=data["labels"],
            attention_mask=data["attention_mask"],  # acts as a loss mask
            reduce_batch=reduce_batch,
            average_logps=average_logps,
        )

        # Calculate eval metrics if not training
        if not self.model.training and self.compute_eval_metrics:
            lm_labels = data["labels"].clone()
            lm_weights = data["attention_mask"].to(lm_logits.dtype).clone()
            lm_preds = lm_logits.argmax(-1).int()

            self.accuracy_metric(
                labels=lm_labels,
                predictions=lm_preds,
                weights=lm_weights,
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
                labels=lm_labels,
                loss=unscaled_loss,
                weights=lm_weights,
            )

        if output_logits:
            return loss, lm_logits
        else:
            return loss


class GptInferenceModel(Gpt2Model):
    def __init__(self, params):
        params = deepcopy(params)

        if "start_token" not in params["model"]:
            raise KeyError(
                "Inference requires a start token. "
                "Please provide `start_token` in the model params."
            )
        if "stop_sequences" not in params["model"]:
            raise KeyError(
                "Inference requires a stop token sequence. "
                "Please provide `stop_sequences` in the model params."
            )

        self.loop_dim = params["model"].pop("loop_dim", 1)
        self.start_token = params["model"].pop("start_token")
        self.stop_sequences = params["model"].pop("stop_sequences")
        self.max_tokens = params["model"].pop("max_tokens", None)

        super().__init__(params)

    def forward(self, data, stop_sequences: Optional[int] = None):
        """The forward pass on the input data. This method
        returns the predictions of the network as tokens.
        """
        if "input_ids" not in data:
            raise KeyError("GPT-2 model expects these data fields: input_ids")
        elif data["input_ids"].dtype != torch.int32:
            raise TypeError("The dtype for all inputs should be torch.int32")

        input_ids = data["input_ids"]
        # Note: attention_mask is a misnomer in this model; its contents are
        # ignored and only its shape is used.
        lm_logits = self.model(
            input_ids=input_ids,
            attention_mask=input_ids,  # doesn't actually mask anything
        )

        predictions = torch.argmax(lm_logits, dim=-1).int()

        cstorch.experimental.run_implicit_autoregressive_loop(
            input_tensor=input_ids,
            output_tensor=predictions,
            loop_dim=self.loop_dim,
            start_token=self.start_token,
            stop_sequences=self.stop_sequences
            if stop_sequences is None
            else stop_sequences,
            max_tokens=self.max_tokens,
        )
        return predictions
