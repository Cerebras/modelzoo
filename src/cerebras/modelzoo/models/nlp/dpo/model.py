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

import copy

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.losses.DPOLoss import DPOLoss
from cerebras.modelzoo.models.nlp.gpt2.model import Gpt2Model
from cerebras.modelzoo.models.nlp.gpt2.utils import (
    set_defaults as gpt2_set_defaults,
)
from cerebras.modelzoo.models.nlp.gptj.model import GptjModel
from cerebras.modelzoo.models.nlp.gptj.utils import (
    set_defaults as gptj_set_defaults,
)
from cerebras.pytorch.metrics import MeanMetric

MODEL_MAPPING = {
    "bloom": "gpt2",
    "falcon": "gptj",
    "gpt2": "gpt2",
    "gpt3": "gpt2",
    "gptj": "gptj",
    "gpt-neox": "gptj",
    "lambda": "gpt2",
    "llama": "gpt2",
    "mistral": "gpt2",
    "mpt": "gpt2",
    "opt": "gpt2",
    "palm": "gptj",
    "santacoder": "gpt2",
    "starcoder": "gpt2",
}


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


@registry.register_model("dpo", datasetprocessor=["DpoHDF5DataProcessor"])
class DPOModel(torch.nn.Module):
    """
    GPT-2 models
    """

    def __init__(self, params):
        super().__init__()

        model_params = params["model"]
        model_name = model_params.pop("model_name")

        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", True
        )

        dpo_params = model_params.pop("dpo")
        self.beta = dpo_params.get("beta", 0.1)
        self.reference_free = dpo_params.get("reference_free", False)
        disable_dropout = dpo_params.get("disable_dropout", True)
        self.loss_type = dpo_params.get("loss_type", "sigmoid")

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

        # Turn off GPT-2/J eval metrics
        params["model"]["compute_eval_metrics"] = False

        self.policy_model = self.build_model(MODEL_MAPPING[model_name], params)
        if disable_dropout:
            disable_dropout_in_model(self.policy_model)

        if not self.reference_free:
            self.ref_model = copy.deepcopy(self.policy_model)
            self.ref_model.eval()
            if disable_dropout:
                disable_dropout_in_model(self.ref_model)

        self.loss_fn = DPOLoss(
            beta=self.beta,
            loss_type=self.loss_type,
            reference_free=self.reference_free,
        )

    def build_model(self, model_backbone, params):
        if model_backbone == "gpt2":
            gpt2_set_defaults(params)
            model = Gpt2Model(params)
        elif model_backbone == "gptj":
            gptj_set_defaults(params)
            model = GptjModel(params)
        return model

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
            cstorch.summarize_scalar("chosen_rewards", chosen_rewards.mean())
            cstorch.summarize_scalar(
                "rejected_rewards", rejected_rewards.mean()
            )
            cstorch.summarize_scalar(
                "reward_accuracies", reward_accuracies.mean()
            )
            cstorch.summarize_scalar("reward_margins", margins.mean())

        cstorch.summarize_scalar(
            "policy_rejected_logps", policy_rejected_logps.mean()
        )
        cstorch.summarize_scalar(
            "policy_chosen_logps", policy_chosen_logps.mean()
        )

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
