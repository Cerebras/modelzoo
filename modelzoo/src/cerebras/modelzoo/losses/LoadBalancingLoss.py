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

import torch
import torch.nn as nn

from cerebras.modelzoo.trainer import summarize_scalar


class LoadBalancingLoss(nn.Module):
    def __init__(
        self,
        num_experts,
        top_k,
    ):
        super(LoadBalancingLoss, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        router_weights_list,
        expert_mask_list,
        attention_mask=None,
    ):
        """
        router_weights: Num hidden layers * [[batch_size, seq_len, experts]]
        expert_mask: Num hidden layers * [[batch_size, seq_len, experts]].
        """
        tokens_per_expert = torch.zeros_like(router_weights_list[0][:, 0, :])
        router_prob_per_expert = torch.zeros_like(
            router_weights_list[0][:, 0, :]
        )
        for router_weights, expert_mask in zip(
            router_weights_list, expert_mask_list
        ):
            if attention_mask is not None:
                extended_attention_mask = (
                    attention_mask[:, :, None]
                    .broadcast_to(expert_mask.shape)
                    .to(router_weights.dtype)
                )

                # Compute the percentage of tokens routed to each experts
                tokens_per_expert += torch.sum(
                    expert_mask * extended_attention_mask, dim=1
                ) / torch.sum(extended_attention_mask, dim=1)

                router_prob_per_expert += torch.sum(
                    router_weights * extended_attention_mask, dim=1
                ) / torch.sum(extended_attention_mask, dim=1)
            else:
                # Compute the percentage of tokens routed to each experts
                tokens_per_expert += torch.mean(expert_mask, dim=1)

                # Compute the average probability of routing to these experts
                router_prob_per_expert += torch.mean(router_weights, dim=1)

        tokens_per_expert /= len(router_weights_list)
        router_prob_per_expert /= len(router_weights_list)

        for expert_idx in range(self.num_experts):
            summarize_scalar(
                f"expert_stats/tokens_per_expert/expert_{expert_idx}",
                torch.mean(tokens_per_expert[:, expert_idx]),
            )
            summarize_scalar(
                f"expert_stats/router_prob_per_expert/expert_{expert_idx}",
                torch.mean(router_prob_per_expert[:, expert_idx]),
            )

        return (self.num_experts**2) * torch.mean(
            router_prob_per_expert * tokens_per_expert
        )
