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

# TODO: Move to a separate folder when more head models are added like MLM and NSP on bert, etc.

import torch
import torch.nn as nn


class GPTLMHeadModelLoss(nn.Module):
    def __init__(
        self,
        vocab_size,
        loss_scaling,
        loss_weight,
    ):
        super(GPTLMHeadModelLoss, self).__init__()
        self.vocab_size = vocab_size
        self.loss_weight = loss_weight
        self.loss_scaling = loss_scaling

        assert (
            self.loss_scaling == "num_tokens"
            or self.loss_scaling == "batch_size"
        ), f"Loss scaling can't be set to {self.loss_scaling}. \
            Should be either 'num_tokens' or 'batch_size'"

        if self.loss_scaling == "num_tokens":
            assert (
                self.loss_weight == 1.0
            ), f"Loss scaling with 'num_tokens' requires loss_weight == 1.0"

    def forward(
        self,
        lm_logits,
        labels,
        attention_mask,
        reduce_batch=True,
        average_logps=False,
    ):
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(
            lm_logits.view(-1, self.vocab_size),
            labels.view(-1).long(),
        )
        if reduce_batch:
            assert (
                not average_logps
            ), "average_logps can only be set to True, when reduce_batch=False"

            lm_loss = lm_loss * attention_mask.to(dtype=lm_logits.dtype).view(
                -1
            )

            if self.loss_scaling == "num_tokens":
                lm_loss = torch.sum(lm_loss) / torch.sum(
                    attention_mask.to(dtype=lm_logits.dtype)
                )
            else:
                lm_loss = (
                    torch.sum(lm_loss) / labels.shape[0]
                ) * self.loss_weight

            loss = lm_loss

        else:
            loss = lm_loss.view(attention_mask.shape) * attention_mask.to(
                dtype=lm_logits.dtype
            )
            loss = torch.sum(loss, dim=-1)
            if average_logps:
                loss = loss / torch.sum(
                    attention_mask.to(dtype=lm_logits.dtype), dim=-1
                )
        return loss
