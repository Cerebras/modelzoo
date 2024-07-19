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
from dataclasses import asdict
from typing import Optional

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.common.utils.model.generation_utils import sample_tokens
from cerebras.modelzoo.common.utils.model.mup_utils import (
    LRAdjustmentGroup,
    is_mup,
)
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    get_embedding_dtype,
)
from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss
from cerebras.modelzoo.losses.LoadBalancingLoss import LoadBalancingLoss
from cerebras.modelzoo.models.nlp.gpt2.gpt2_model import GPT2LMHeadModel
from cerebras.modelzoo.trainer import summarize_scalar
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


@registry.register_model(
    [
        "gpt2",
        "gpt3",
        "btlm",
        "bloom",
        "llama",
        "mistral",
        "mixtral",
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
    GPT-2 models.
    """

    def __init__(self, params):
        super().__init__()

        self.lr_adjustment_groups = self.create_default_lr_adjustment_groups()
        if isinstance(params, dict):
            config_class = registry.get_config_class_from_model(self.__class__)
            params = config_class(**params)

        pol = cstorch.backends.csx.precision.optimization_level
        if pol == 2 or (pol == 1 and params.model.fp16_type == "cbfloat16"):
            params.model.attention_softmax_fp32 = False

        # Create a deep copy of the model parameters object to ensure immutability
        model_params = deepcopy(params.model)

        # Directly access compute_eval_metrics
        self.compute_eval_metrics = model_params.compute_eval_metrics

        # Initialize evaluation metrics if compute_eval_metrics is enabled
        if self.compute_eval_metrics:
            self.perplexity_metric = PerplexityMetric(name="eval/lm_perplexity")
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")

        # MoE (Mixture of Experts) settings
        if not model_params.moe:
            self.moe_enabled = False
            model_params.moe = dict(num_experts=1)
        elif "num_experts" not in model_params.moe:
            model_params.moe["num_experts"] = 1
        else:
            self.moe_enabled = model_params.moe["num_experts"] > 1

        # Load balancing loss if MoE is enabled
        if self.moe_enabled:
            self.load_balancing_loss_coef = model_params.moe.get(
                "load_balancing_loss_coef", 0.01
            )
            routing_algorithm = model_params.moe.get(
                "routing_algorithm", "learned"
            )
            if (
                routing_algorithm == "hash"
                and self.load_balancing_loss_coef > 0.0
            ):
                logging.warning(
                    "Disabling load balancing loss when using hash routing"
                )
                model_params.moe["load_balancing_loss_coef"] = 0.0
                self.load_balancing_loss_coef = 0.0
            self.load_balancing_loss_fn = LoadBalancingLoss(
                model_params.moe["num_experts"], model_params.moe["top_k"]
            )

        self.vocab_size = params.model.vocab_size

        # Directly assign attributes to instance variables
        self.start_token = params.model.start_token
        self.loop_dim = params.model.loop_dim
        self.stop_sequences = params.model.stop_sequences
        self.max_tokens = params.model.max_tokens

        # Sampling configuration
        self.temperature = params.model.temperature
        self.top_k = params.model.top_k
        self.top_p = params.model.top_p

        if self.start_token is not None:
            # Disable eval metrics
            params.model.compute_eval_metrics = False

        # Build the model using the model parameters object
        self.model = self.build_model(model_params)

        # Initialize the loss function directly using the model attributes
        self.loss_fn = GPTLMHeadModelLoss(
            params.model.vocab_size,
            self.loss_scaling,
            self.loss_weight,
        )

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, model_params):
        mup_config = is_mup(model_params)

        # Access other parameters directly
        self.loss_weight = model_params.loss_weight
        self.loss_scaling = model_params.loss_scaling.lower()

        # Validate the loss weight
        if self.loss_weight != 1.0 and self.loss_scaling == "num_tokens":
            logging.warning(
                f"loss_weight cannot be {self.loss_weight} for num_tokens "
                f"loss_scaling. Setting loss_weight to 1.0."
            )
            self.loss_weight = 1.0

        if model_params.scale_qk_dot_by_d is None:
            if mup_config:
                model_params.scale_qk_dot_by_d = True
                logging.warning(
                    "Found muP params but no scale_qk_dot_by_d was provided, "
                    "so it will be automatically set to 'True' as the muP "
                    "default."
                )
            else:
                model_params.scale_qk_dot_by_d = False

        model = GPT2LMHeadModel(
            # Embedding
            vocab_size=model_params.vocab_size,
            max_position_embeddings=model_params.max_position_embeddings,
            embd_pdrop=model_params.embedding_dropout_rate,
            position_embedding_type=model_params.position_embedding_type,
            position_embedding_offset=model_params.position_embedding_offset,
            hidden_size=model_params.hidden_size,
            share_embedding_weights=model_params.share_embedding_weights,
            embedding_layer_norm=model_params.embedding_layer_norm,
            num_relative_attention_buckets=model_params.num_relative_attention_buckets,
            rotary_dim=model_params.rotary_dim,
            rope_theta=model_params.rope_theta,
            pad_rope=model_params.pad_rope,
            # Encoder
            num_hidden_layers=model_params.num_hidden_layers,
            dropout_rate=model_params.dropout_rate,
            norm_type=model_params.norm_type,
            layer_norm_epsilon=float(model_params.layer_norm_epsilon),
            # Encoder - Attention
            num_heads=model_params.num_heads,
            attention_type=model_params.attention_type,
            attention_module=model_params.attention_module,
            attention_sliding_window_length=model_params.attention_sliding_window_length,
            extra_attention_params=model_params.extra_attention_params,
            extra_ffn_params=model_params.extra_ffn_params,
            use_projection_bias_in_attention=model_params.use_projection_bias_in_attention,
            use_ffn_bias_in_attention=model_params.use_ffn_bias_in_attention,
            attention_dropout_rate=(
                model_params.attention_dropout_rate or model_params.dropout_rate
            ),
            attention_softmax_fp32=model_params.attention_softmax_fp32,
            attention_kernel=model_params.attention_kernel,
            # Encoder - FFN
            filter_size=model_params.filter_size,
            nonlinearity=model_params.nonlinearity,
            use_ffn_bias=model_params.use_ffn_bias,
            # Task-specific
            use_bias_in_output=model_params.use_bias_in_output,
            fixed_sparse_attention=model_params.fixed_sparse_attention,
            # Initializers
            embedding_initializer=(
                asdict(model_params.embedding_initializer)
                if model_params.embedding_initializer
                else None
            ),  # InitializerConfig needs to be casted to dict
            initializer=(
                asdict(model_params.initializer)
                if model_params.initializer
                else None
            ),
            output_layer_initializer=(
                asdict(model_params.output_layer_initializer)
                if model_params.output_layer_initializer
                else None
            ),
            initializer_range=model_params.initializer_range,
            pos_scaling_factor=model_params.pos_scaling_factor,
            scale_qk_dot_by_layer_idx=model_params.scale_qk_dot_by_layer_idx,
            ffn_initializer=(
                asdict(model_params.ffn_initializer)
                if model_params.ffn_initializer
                else None
            ),
            ffn_output_layer_initializer=(
                asdict(model_params.ffn_output_layer_initializer)
                if model_params.ffn_output_layer_initializer
                else None
            ),
            # muP (maximal update parameterization)  parameters
            lr_adjustment_groups=self.lr_adjustment_groups,
            embeddings_scale=model_params.embeddings_scale,
            scale_qk_dot_by_d=model_params.scale_qk_dot_by_d,
            attention_logits_alpha=model_params.attention_logits_alpha,
            scale_output_logits_by_d=model_params.scale_output_logits_by_d,
            mup_base_hidden_size=model_params.mup_base_hidden_size,
            mup_base_filter_size=model_params.mup_base_filter_size,
            output_logits_alpha=model_params.output_logits_alpha,
            # muP backwards compatibility
            output_logits_scale=model_params.output_logits_scale,
            alibi_trainable_slopes=model_params.alibi_trainable_slopes,
            # MoE:
            moe_params=model_params.moe,
            dtype=get_embedding_dtype(
                model_params.mixed_precision,
                model_params.fp16_type,
            ),
        )

        return model

    def forward(self, *args, autoregressive=False, **kwargs):
        if autoregressive:
            return self.inference_step(*args, **kwargs)
        return self.training_step(*args, **kwargs)

    def training_step(
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

        model_outputs = self.model(
            input_ids=data["input_ids"],
            attention_mask=data[
                "attention_mask"
            ],  # doesn't actually mask anything
            attention_span=data.get("attention_span"),  # VSL-only input
            position_ids=data.get("position_ids"),  # VSL-only input
        )

        if self.moe_enabled:
            lm_logits, routing_weights, expert_masks = model_outputs
        else:
            lm_logits = model_outputs

        loss = self.loss_fn(
            lm_logits,
            labels=data["labels"],
            attention_mask=data["attention_mask"],  # acts as a loss mask
            reduce_batch=reduce_batch,
            average_logps=average_logps,
        )

        if (
            self.moe_enabled
            and self.load_balancing_loss_coef > 0.0
            and self.training
        ):
            load_balance_loss = (
                self.load_balancing_loss_coef
                * self.load_balancing_loss_fn(
                    routing_weights,
                    expert_masks,
                    attention_mask=data["attention_mask"],
                )
            )
            summarize_scalar(
                "expert_stats/load_balance_loss", load_balance_loss
            )
            loss = loss + load_balance_loss

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

    def create_default_lr_adjustment_groups(self):
        return {
            "embedding": LRAdjustmentGroup("*embedding*weight"),
            # decoder_kernel for muP backward compatibility
            "decoder_kernel": LRAdjustmentGroup(
                [
                    "*decoder*dense*weight",
                    "*decoder*linear*weight",
                    "*decoder*linear*expert_weights",  # moe
                ]
            ),
            "decoder_attention": LRAdjustmentGroup(
                "*decoder*attn*dense*weight"
            ),
            "decoder_input_ffn": LRAdjustmentGroup(
                [
                    "*decoder*ffn.ffn.[!1]*weight",
                    "*decoder*ffn.ffn.[!1]*expert_weights",  # moe
                ]
            ),
            "decoder_output_ffn": LRAdjustmentGroup(
                [
                    "*decoder*ffn.ffn.[1]*weight",
                    "*decoder*ffn.ffn.[1]*expert_weights",  # moe
                ]
            ),
        }

    def inference_step(self, data, stop_sequences: Optional[int] = None):
        """The forward pass on the input data. This method
        returns the predictions of the network as tokens.
        """
        if self.start_token is None:
            raise KeyError(
                "Inference requires a start token. "
                "Please provide `start_token` in the model params."
            )

        if self.stop_sequences is None:
            raise KeyError(
                "Inference requires a stop token sequence. "
                "Please provide `stop_sequences` in the model params."
            )

        if self.temperature and self.temperature <= 0:
            raise ValueError(
                "If sampling is needed, `temperature` must be a positive float "
                f"number, got {self.temperature}"
            )
        if self.top_k and self.top_k <= 0:
            raise ValueError(
                "If top-k sampling is needed, `top_k` must be a positive integer, "
                f"got {self.top_k}"
            )
        if self.top_p and (self.top_p < 0 or self.top_p > 1):
            raise ValueError(
                "If top-p sampling is needed, `top_p` must be a float between "
                f"(0.0, 1.0), got {self.top_p}"
            )

        if "input_ids" not in data:
            raise KeyError("GPT-2 model expects these data fields: input_ids")
        elif data["input_ids"].dtype != torch.int32:
            raise TypeError("The dtype for all inputs should be torch.int32")

        input_ids = data["input_ids"]
        loop_index = cstorch.experimental.start_implicit_loop(
            input_ids, loop_dim=1
        )

        # Note: attention_mask is a misnomer in this model; its contents are
        # ignored and only its shape is used.
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=input_ids,  # doesn't actually mask anything
            # By passing this, we get only the current token's logits instead
            # of all logits from the whole sequence
            inference_loop_index=loop_index,
        )

        if self.moe_enabled:
            token_logits, _, _ = model_outputs
        else:
            token_logits = model_outputs

        # run sampling, if needed
        token_pred = sample_tokens(
            token_logits, self.temperature, self.top_k, self.top_p
        )

        sequence_preds = cstorch.experimental.update_implicit_loop(
            input_tensor=input_ids,
            index_tensor=loop_index,
            update_tensor=token_pred,
            start_token=self.start_token,
            stop_sequences=(
                self.stop_sequences
                if stop_sequences is None
                else stop_sequences
            ),
            max_tokens=self.max_tokens,
        )
        return sequence_preds
