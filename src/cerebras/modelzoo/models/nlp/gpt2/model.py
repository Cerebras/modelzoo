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
from typing import List, Literal, Optional, Union

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.utils.model.generation_utils import sample_tokens
from cerebras.modelzoo.common.utils.model.mup_utils import is_mup
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    get_embedding_dtype,
)
from cerebras.modelzoo.losses.GPTLMHeadModelLoss import GPTLMHeadModelLoss
from cerebras.modelzoo.losses.LoadBalancingLoss import LoadBalancingLoss
from cerebras.modelzoo.models.nlp.gpt2.gpt2_model import (
    GPT2LMHeadModel,
    GPT2LMHeadModelConfig,
)
from cerebras.modelzoo.trainer import summarize_scalar
from cerebras.pytorch.metrics import AccuracyMetric, PerplexityMetric


class GPT2ModelConfig(GPT2LMHeadModelConfig):
    name: Literal["gpt2"]

    boundary_casting: Optional[bool] = False

    # Loss:
    loss_scaling: Literal["batch_size", "num_tokens"] = "num_tokens"
    """The scaling type used to calculate the loss. Accepts - `batch_size`, `num_tokens`.
    See [more](https://docs.cerebras.net/en/latest/wsc/general/num-tokens-loss-scaling.html).
    **Note:** It is recommended to set this to `num_tokens` for convenience."""

    loss_weight: float = 1.0
    """The weight for the loss scaling when `loss_scaling = 'batch_size'`, generally set to
    '1/max_sequence_length`.
    """

    # Optional inference parameters:
    start_token: Optional[Union[int, List[int]]] = None
    loop_dim: int = 1
    stop_sequences: Optional[Union[int, List[List[int]]]] = None
    max_tokens: Optional[int] = None

    temperature: Optional[float] = None
    "If set, use some form of sampling instead of greedy decoding"
    top_k: Optional[int] = None
    "Enable top-k sampling method, limiting the number of vocab positions"
    top_p: Optional[float] = None
    "Enable top-p sampling method, handling variable uncertainty better"

    # Misc:
    compute_eval_metrics: bool = True
    "Computes perplexity & accuracy metrics in addition to loss"


class Gpt2Model(torch.nn.Module):
    """
    GPT-2 models.
    """

    def __init__(self, config: GPT2ModelConfig):
        if isinstance(config, dict):
            if "model" in config:
                config = config["model"]
            config = GPT2ModelConfig(**config)

        super().__init__()

        pol = cstorch.backends.csx.precision.optimization_level
        if pol == 2 or (
            pol == 1 and cstorch.amp.get_half_dtype_str() == "cbfloat16"
        ):
            self.attention_softmax_fp32 = False
        else:
            self.attention_softmax_fp32 = config.attention_softmax_fp32

        # Directly access compute_eval_metrics
        self.compute_eval_metrics = config.compute_eval_metrics

        # Initialize evaluation metrics if compute_eval_metrics is enabled
        if self.compute_eval_metrics:
            self.perplexity_metric = PerplexityMetric(name="eval/lm_perplexity")
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")

        # MoE (Mixture of Experts) settings
        if not config.moe_params:
            self.moe_enabled = False
        else:
            total_experts = config.moe_params.num_experts
            if config.moe_params.num_shared_experts:
                total_experts += config.moe_params.num_shared_experts
            self.moe_enabled = total_experts > 1

        # Load balancing loss if MoE is enabled
        if self.moe_enabled:
            self.moe_params = config.moe_params
            self.load_balancing_loss_fn = LoadBalancingLoss(
                config.moe_params.num_experts, config.moe_params.top_k
            )

        self.vocab_size = config.vocab_size

        # Directly assign attributes to instance variables
        self.start_token = config.start_token
        self.loop_dim = config.loop_dim
        self.max_tokens = config.max_tokens

        # Sampling configuration
        self.temperature = config.temperature
        self.top_k = config.top_k
        self.top_p = config.top_p

        # if self.start_token is not None:
        #     # Disable eval metrics
        #     self.compute_eval_metrics = False

        # Build the model using the model parameters object
        self.model = self.build_model(config)

        # Initialize the loss function directly using the model attributes
        self.loss_fn = GPTLMHeadModelLoss(
            config.vocab_size,
            self.loss_scaling,
            self.loss_weight,
        )

    def _post_device_transfer(self):
        self.model.tie_weights()

    def build_model(self, config):
        mup_config = is_mup(config)

        # Access other parameters directly
        self.loss_weight = config.loss_weight
        self.loss_scaling = config.loss_scaling.lower()

        # Validate the loss weight
        if self.loss_weight != 1.0 and self.loss_scaling == "num_tokens":
            logging.warning(
                f"loss_weight cannot be {self.loss_weight} for num_tokens "
                f"loss_scaling. Setting loss_weight to 1.0."
            )
            self.loss_weight = 1.0

        scale_qk_dot_by_d = config.scale_qk_dot_by_d
        if scale_qk_dot_by_d is None:
            if mup_config:
                scale_qk_dot_by_d = True
                logging.warning(
                    "Found muP params but no scale_qk_dot_by_d was provided, "
                    "so it will be automatically set to 'True' as the muP "
                    "default."
                )
            else:
                scale_qk_dot_by_d = False

        model = GPT2LMHeadModel(
            config.copy(
                update=dict(
                    attention_softmax_fp32=self.attention_softmax_fp32,
                    scale_qk_dot_by_d=scale_qk_dot_by_d,
                    dtype=get_embedding_dtype(
                        dtype=cstorch.amp.get_floating_point_dtype_str(),
                    ),
                )
            )
        )

        # lr_adjustment_groups must be stored in self in order to be accessible
        # by configure_param_groups (optimizer initialization phase of muP runs)
        if model.lr_adjustment_groups is not None:
            self.lr_adjustment_groups = model.lr_adjustment_groups

        return model

    def forward(self, *args, autoregressive=False, **kwargs):
        if autoregressive:
            return self.inference_step(*args, **kwargs)
        return self.training_step(*args, **kwargs)

    def training_step(
        self,
        data,
        cross_attention_states=None,
        cross_attention_mask=None,
        full_text_row_masked_out_mask=None,
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
            cross_attention_states=cross_attention_states,
            cross_attention_mask=cross_attention_mask,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            attention_span=data.get("attention_span"),  # VSL-only input
            position_ids=data.get("position_ids"),
            special_token_meta=data.get("special_token_meta"),
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
            and self.moe_params.load_balancing_loss_coef > 0.0
            and self.training
        ):
            load_balance_loss = (
                self.moe_params.load_balancing_loss_coef
                * self.load_balancing_loss_fn(
                    routing_weights,
                    expert_masks,
                    attention_mask=data["attention_mask"],
                )
            )
            summarize_scalar(
                "expert_stats/load_balance_loss", load_balance_loss
            )
            summarize_scalar("loss/cross_entropy_loss", loss)
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

    def inference_step(self, data):
        """The forward pass on the input data. This method
        returns the predictions of the network as tokens.
        """
        if self.start_token is None:
            raise KeyError(
                "Inference requires a start token. "
                "Please provide `start_token` in the model params."
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
            token_logits,
            data["rand_uniform"],
            self.temperature,
            self.top_k,
            self.top_p,
        )

        sequence_preds = cstorch.experimental.update_implicit_loop(
            input_tensor=input_ids,
            index_tensor=loop_index,
            update_tensor=token_pred,
            stop_sequences_tensor=data["stop_sequences"],
            start_token=self.start_token,
            max_tokens=self.max_tokens,
        )
        return sequence_preds
