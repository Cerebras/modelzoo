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

from typing import Any, Literal, Optional

import torch
import torch.nn as nn
from annotated_types import Ge, Le
from pydantic import Field
from typing_extensions import Annotated

from cerebras.modelzoo.models.nlp.bert.bert_model import (
    BertModel,
    BertModelConfig,
)


class BertForSequenceClassificationModelConfig(BertModelConfig):
    name: Literal["bert/classifier"]

    num_labels: int = ...
    problem_type: str = ...
    task_dropout: float = ...

    embedding_dropout_rate: Optional[Annotated[float, Ge(0.0), Le(1.0)]] = None

    vocab_size: Annotated[int, Ge(1), Le(512000)] = 30522
    "The size of the vocabulary used in the model. Max supported value - `512000`."

    attention_dropout_rate: Optional[float] = 0.1
    "Dropout rate for attention layer. Default - same as `dropout`"

    max_position_embeddings: int = 1024
    "The maximum sequence length that the model can handle."

    num_segments: Optional[int] = 2

    encoder_nonlinearity: Literal["gelu", "relu", "silu", "gelu_new"] = "gelu"
    compute_eval_metrics: bool = False

    # The following fields are deprecated and unused.
    # They will be removed in the future once all configs have been fixed
    # These are required because the checkpoint converter doesn't distinguish between bert model types
    mlm_loss_weight: Optional[Any] = Field(default=None, deprecated=True)
    mlm_nonlinearity: Optional[Any] = Field(default=None, deprecated=True)
    share_embedding_weights: Optional[Any] = Field(
        default=None, deprecated=True
    )

    def post_init(self, context):
        super().post_init(context)

        if self.embedding_dropout_rate is None:
            self.embedding_dropout_rate = self.dropout_rate

        self.nonlinearity = self.encoder_nonlinearity


class BertForSequenceClassification(nn.Module):
    def __init__(self, config: BertForSequenceClassificationModelConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.task_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
        token_type_ids=None,
        attention_mask=None,
    ):

        _, pooled_outputs = self.bert(
            input_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        return logits


class BertForSequenceClassificationLoss(nn.Module):
    def __init__(self, num_labels, problem_type):
        super().__init__()
        self.num_labels = num_labels
        self.problem_type = problem_type

    def forward(self, labels, logits):
        loss = None
        if labels is not None:
            if self.problem_type is None:
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze().float())
                else:
                    loss = loss_fct(logits, labels.float())
            elif self.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels),
                    labels.view(-1).long(),
                )
            elif self.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.reshape(-1))
        return loss
