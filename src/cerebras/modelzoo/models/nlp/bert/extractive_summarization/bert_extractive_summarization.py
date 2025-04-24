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

import os
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from annotated_types import Ge, Le
from pydantic import Field, field_validator
from typing_extensions import Annotated

from cerebras.modelzoo.config.types import AliasedPath
from cerebras.modelzoo.layers.activations import ActivationType
from cerebras.modelzoo.models.nlp.bert.bert_model import (
    BertModel,
    BertModelConfig,
)


class BertSummarizationModelConfig(BertModelConfig):
    name: Literal["bert/extractive_summarization"]

    num_labels: Literal[1, 2] = 2

    loss_weight: Optional[float] = 1.0
    use_cls_bias: bool = True
    vocab_file: Optional[AliasedPath] = None

    vocab_size: Annotated[int, Ge(1), Le(512000)] = 30522
    "The size of the vocabulary used in the model. Max supported value - `512000`."

    attention_dropout_rate: float = 0.1
    "Dropout rate for attention layer. Default - same as `dropout`"

    max_position_embeddings: int = 1024
    "The maximum sequence length that the model can handle."

    encoder_nonlinearity: Literal["gelu", "relu", "silu", "gelu_new"] = "gelu"
    pooler_nonlinearity: Optional[ActivationType] = None
    compute_eval_metrics: bool = False

    num_segments: Optional[int] = 2

    add_pooling_layer: Literal[False] = False

    # The following fields are deprecated and unused.
    # They will be removed in the future once all configs have been fixed
    # These are required because the checkpoint converter doesn't distinguish between bert model types
    mlm_loss_weight: Optional[Any] = Field(default=None, deprecated=True)
    mlm_nonlinearity: Optional[Any] = Field(default=None, deprecated=True)
    share_embedding_weights: Optional[Any] = Field(
        default=None, deprecated=True
    )

    @field_validator("loss_weight", mode="after")
    @classmethod
    def validate_loss_weight(cls, value):
        return value or 1.0

    @field_validator("vocab_file", mode="after")
    @classmethod
    def get_vocab_file(cls, vocab_file):
        if vocab_file is None:
            return vocab_file
        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file does not exist: {vocab_file}")
        return os.path.abspath(vocab_file)

    def post_init(self, context):
        super().post_init(context)

        if self.embedding_dropout_rate is None:
            self.embedding_dropout_rate = self.dropout_rate


class BertForSummarization(nn.Module):
    def __init__(self, config: BertSummarizationModelConfig):
        super().__init__()

        self.bert = BertModel(config)
        self.classifier = nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=config.use_cls_bias,
        )

        self.__reset_parameters()

    def reset_parameters(self):
        self.bert.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        cls_tokens_positions=None,
    ):
        encoded_outputs, _ = self.bert(
            input_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        hidden_size = list(encoded_outputs.size())[-1]

        batch_size, max_pred = list(cls_tokens_positions.size())
        index = torch.broadcast_to(
            cls_tokens_positions.unsqueeze(2),
            (batch_size, max_pred, hidden_size),
        ).long()
        masked_output = torch.gather(encoded_outputs, dim=1, index=index)
        encoded_outputs = masked_output

        logits = self.classifier(encoded_outputs)

        return logits


class BertForSummarizationLoss(nn.Module):
    def __init__(
        self,
        num_labels,
        loss_weight=1.0,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_labels = num_labels

    def forward(self, logits, labels, label_weights):
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1).long()
            )
            loss = loss * label_weights.view(-1)
            loss = loss.sum() / labels.shape[0] * self.loss_weight
        return loss
