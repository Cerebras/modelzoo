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
        self, vocab_size, loss_weight,
    ):
        super(GPTLMHeadModelLoss, self).__init__()
        self.vocab_size = vocab_size
        self.loss_weight = loss_weight

    def forward(
        self, lm_logits, labels, attention_mask,
    ):
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        lm_loss = loss_fct(
            lm_logits.view(-1, self.vocab_size), labels.view(-1).long(),
        )

        lm_loss *= attention_mask.to(dtype=lm_logits.dtype).view(-1)

        lm_loss = (torch.sum(lm_loss) / labels.shape[0]) * self.loss_weight

        loss = lm_loss.to(lm_logits.dtype)
        return loss
