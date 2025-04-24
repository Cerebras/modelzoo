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

from copy import deepcopy
from typing import Literal, Union
from warnings import warn

import torch
from pydantic import Field, model_validator
from typing_extensions import Annotated

from cerebras.modelzoo.config import BaseConfig, ModelConfig
from cerebras.modelzoo.losses.DPOLoss import DPOLoss, DPOLossType
from cerebras.modelzoo.models.nlp.bloom.model import BloomModelConfig
from cerebras.modelzoo.models.nlp.falcon.model import FalconModelConfig
from cerebras.modelzoo.models.nlp.gpt2.model import GPT2ModelConfig
from cerebras.modelzoo.models.nlp.gpt3.model import GPT3ModelConfig
from cerebras.modelzoo.models.nlp.gptj.model import (
    GPTJModelConfig,
    GPTNeoXModelConfig,
)
from cerebras.modelzoo.models.nlp.llama.model import LlamaModelConfig
from cerebras.modelzoo.models.nlp.mistral.model import MistralModelConfig
from cerebras.modelzoo.models.nlp.mixtral.model import MixtralModelConfig
from cerebras.modelzoo.models.nlp.santacoder.model import SantaCoderModelConfig
from cerebras.modelzoo.models.nlp.starcoder.model import StarCoderModelConfig
from cerebras.modelzoo.trainer import summarize_scalar
from cerebras.pytorch.metrics import MeanMetric


class DPOParameters(BaseConfig):
    beta: float = 0.1
    reference_free: bool = False
    loss_type: DPOLossType = "sigmoid"
    disable_dropout: bool = True
    dpop_penalty_weight: float = 5.0
    """Corresponds to lambda in the DPOP paper(https://arxiv.org/pdf/2402.13228)
    This parameter only takes effect if loss_type=\"dpop\" """


class DPOModelConfig(ModelConfig):
    name: Literal["dpo"]

    policy_model: Annotated[
        Union[
            BloomModelConfig,
            FalconModelConfig,
            GPT2ModelConfig,
            GPT3ModelConfig,
            GPTJModelConfig,
            GPTNeoXModelConfig,
            LlamaModelConfig,
            MistralModelConfig,
            MixtralModelConfig,
            SantaCoderModelConfig,
            StarCoderModelConfig,
        ],
        Field(discriminator="name"),
    ] = ...

    dpo: DPOParameters = ...
    "Parameters for DPO configuration"

    compute_eval_metrics: bool = True

    @model_validator(mode="before")
    @classmethod
    def make_submodel(cls, data):
        if "policy_model" not in data:
            warn(
                f"Detected that DPO model parameters were not organized "
                f"under the `policy_model` key. This behaviour is deprecated "
                f"and support for it will be removed in the future"
            )
            model = {
                k: data.pop(k)
                for k in list(data)
                if k not in ("name", "dpo", "compute_eval_metrics")
            }
            if "model_name" in model:
                model["name"] = model.pop("model_name")

            data["policy_model"] = model

        return data


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


class DPOModel(torch.nn.Module):
    """
    Differential Privacy Optimization (DPO) enhanced GPT-2 models.
    """

    def __init__(self, config: DPOModelConfig):
        if isinstance(config, dict):
            config = DPOModelConfig(**config)

        super().__init__()

        self.compute_eval_metrics = config.compute_eval_metrics

        # Directly access the 'dpo' object attributes
        dpo_params = config.dpo
        self.beta = dpo_params.beta
        self.reference_free = dpo_params.reference_free
        disable_dropout = dpo_params.disable_dropout
        self.loss_type = dpo_params.loss_type
        self.dpop_penalty_weight = dpo_params.dpop_penalty_weight

        if self.compute_eval_metrics:
            if not self.reference_free:
                self.rewards_chosen_metric = MeanMetric(
                    name="eval/rewards_chosen"
                )
                self.rewards_rejected_metric = MeanMetric(
                    name="eval/rewards_rejected"
                )
                self.rewards_accuracies_metric = MeanMetric(
                    name="eval/rewards_accuracies"
                )
                self.rewards_margins_metric = MeanMetric(
                    name="eval/rewards_margins"
                )

            self.logps_rejected_metric = MeanMetric(name="eval/logps_rejected")
            self.logps_chosen_metric = MeanMetric(name="eval/logps_chosen")

        self.policy_model = config.policy_model()

        # Turn off eval metrics directly in the policy_model object
        self.policy_model.compute_eval_metrics = False

        if disable_dropout:
            disable_dropout_in_model(self.policy_model)

        if not self.reference_free:
            self.ref_model = deepcopy(self.policy_model)
            self.ref_model.eval()
            if disable_dropout:
                disable_dropout_in_model(self.ref_model)

        self.loss_fn = DPOLoss(
            beta=self.beta,
            loss_type=self.loss_type,
            reference_free=self.reference_free,
            dpop_penalty_weight=self.dpop_penalty_weight,
        )

    def forward(self, data):
        batch_size = data["chosen_input_ids"].shape[0]
        seq_length = data["chosen_input_ids"].shape[1]

        # stack 'chosen' and 'rejected' data, and reshape it back to original num_dimension.
        # This is a workaround for using torch.concat on batch dim directly.
        concatenated_shape = [batch_size * 2, seq_length]
        policy_input = torch.stack(
            [data["chosen_input_ids"], data["rejected_input_ids"]], dim=1
        ).view(concatenated_shape)
        policy_mask = torch.stack(
            [data["chosen_attention_mask"], data["rejected_attention_mask"]],
            dim=1,
        ).view(concatenated_shape)
        policy_label = torch.stack(
            [data["chosen_labels"], data["rejected_labels"]], dim=1
        ).view(concatenated_shape)

        all_logps, all_logits = self.policy_model(
            data={
                "input_ids": policy_input,
                "attention_mask": policy_mask,
                "labels": policy_label,
            },
            reduce_batch=False,
            average_logps=self.loss_type == "ipo",
            output_logits=True,
        )

        reshaped_logps = -all_logps.view([batch_size, 2])
        policy_chosen_logps = reshaped_logps[:, 0:1]
        policy_rejected_logps = reshaped_logps[:, 1:2]

        if not self.reference_free:
            with torch.no_grad():
                all_logps = -self.ref_model(
                    data={
                        "input_ids": policy_input,
                        "attention_mask": policy_mask,
                        "labels": policy_label,
                    },
                    reduce_batch=False,
                    average_logps=self.loss_type == "ipo",
                )
                # Reshape logps's first dimension to [batch, 2], and split into two tensors along
                # the second dimension.
                reshaped_logps = all_logps.view([batch_size, 2])

                reference_chosen_logps = reshaped_logps[:, 0:1]
                reference_rejected_logps = reshaped_logps[:, 1:2]

            chosen_rewards = self.beta * (
                policy_chosen_logps - reference_chosen_logps
            )
            rejected_rewards = self.beta * (
                policy_rejected_logps - reference_rejected_logps
            )
            margins = chosen_rewards - rejected_rewards
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

        else:
            reference_chosen_logps = None
            reference_rejected_logps = None

        loss = self.loss_fn(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # Collect summaries
        if not self.reference_free:
            summarize_scalar("chosen_rewards", chosen_rewards.mean())
            summarize_scalar("rejected_rewards", rejected_rewards.mean())
            summarize_scalar("reward_accuracies", reward_accuracies.mean())
            summarize_scalar("reward_margins", margins.mean())

        summarize_scalar("policy_rejected_logps", policy_rejected_logps.mean())
        summarize_scalar("policy_chosen_logps", policy_chosen_logps.mean())

        # Calculate eval metrics if not training
        if not self.policy_model.model.training and self.compute_eval_metrics:
            if not self.reference_free:
                self.rewards_chosen_metric(chosen_rewards.clone().mean())
                self.rewards_rejected_metric(rejected_rewards.clone().mean())
                self.rewards_accuracies_metric(reward_accuracies.clone().mean())
                self.rewards_margins_metric(margins.clone().mean())

            self.logps_rejected_metric(policy_rejected_logps.clone().mean())
            self.logps_chosen_metric(policy_chosen_logps.clone().mean())
        return loss
