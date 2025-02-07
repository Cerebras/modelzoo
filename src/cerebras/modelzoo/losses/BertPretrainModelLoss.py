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

# coding=utf-8
#
# This code is adapted from
# https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
#
# Copyright 2022 Cerebras Systems.
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from cerebras.modelzoo.common.utils.model.transformer_utils import smooth_loss


class BertPretrainModelLoss(nn.Module):
    def __init__(
        self,
        disable_nsp=False,
        mlm_loss_weight=1.0,
        label_smoothing=0.0,
    ):
        super(BertPretrainModelLoss, self).__init__()
        self.disable_nsp = disable_nsp
        self.mlm_loss_weight = mlm_loss_weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        mlm_logits,
        vocab_size,
        mlm_labels,
        nsp_logits,
        nsp_labels,
        mlm_weights,
        mlm_loss_scale=None,
    ):
        mlm_loss_fn = nn.CrossEntropyLoss(reduction="none")

        if mlm_loss_scale is not None:
            mlm_weights = mlm_weights * mlm_loss_scale

        mlm_loss = mlm_loss_fn(
            mlm_logits.view(-1, vocab_size),
            mlm_labels.view(-1).long(),
        )

        if self.label_smoothing > 0.0 and self.training:
            # Calculate loss correction for label smoothing
            mlm_loss = smooth_loss(
                mlm_logits, mlm_loss, self.label_smoothing, vocab_size
            )

        mlm_loss *= mlm_weights.view(-1)
        mlm_loss = torch.sum(mlm_loss) / mlm_labels.shape[0]

        if mlm_loss_scale is None:
            mlm_loss *= self.mlm_loss_weight

        total_loss = mlm_loss

        if not self.disable_nsp:
            nsp_loss_fn = nn.CrossEntropyLoss()
            nsp_loss = nsp_loss_fn(
                nsp_logits.view(-1, 2),
                nsp_labels.view(-1).long(),
            )
            total_loss += nsp_loss

        return total_loss
