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

# Based on https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from cerebras.modelzoo.common.registry import registry


@registry.register_loss("DPOLoss")
class DPOLoss(nn.Module):
    """
    DPO Loss
    :param beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        We ignore the reference model as beta -> 0.
    :param reference_free: If True, we ignore the _provided_ reference model and implicitly use a
        reference model that assigns equal probability to all responses.
    """

    def __init__(
        self,
        beta=0.1,
        loss_type="sigmoid",
        reference_free=False,
    ):
        super(DPOLoss, self).__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.reference_free = reference_free

    def forward(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    ):
        pi_logratios = policy_chosen_logps - policy_rejected_logps

        if self.reference_free:
            ref_logratios = 0
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = torch.pow(logits - 1 / (2 * self.beta), 2.0)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo']"
            )

        return losses.mean()
