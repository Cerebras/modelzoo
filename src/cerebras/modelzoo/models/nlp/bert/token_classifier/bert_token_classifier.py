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
from cerebras.modelzoo.models.nlp.bert.bert_model import (
    BertModel,
    BertModelConfig,
)


class BertForTokenClassificationModelConfig(BertModelConfig):
    name: Literal["bert/token_classifier"]

    num_classes: int = ...

    loss_weight: Optional[float] = 1.0
    include_padding_in_loss: Optional[bool] = False
    encoder_output_dropout_rate: float = ...
    dropout_rate: Optional[float] = 0.0
    embedding_dropout_rate: Optional[Annotated[float, Ge(0.0), Le(1.0)]] = None

    vocab_size: Annotated[int, Ge(1), Le(512000)] = 30522
    "The size of the vocabulary used in the model. Max supported value - `512000`."

    attention_dropout_rate: Optional[float] = 0.0
    "Dropout rate for attention layer. Default - same as `dropout`"

    max_position_embeddings: int = 1024
    "The maximum sequence length that the model can handle."

    label_vocab_file: Optional[AliasedPath] = None

    encoder_nonlinearity: Literal["gelu", "relu", "silu", "gelu_new"] = "gelu"
    pooler_nonlinearity: Optional[str] = None
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

    def post_init(self, context):
        super().post_init(context)

        if self.embedding_dropout_rate is None:
            self.embedding_dropout_rate = self.dropout_rate

        self.nonlinearity = self.encoder_nonlinearity

    @field_validator("label_vocab_file", mode="after")
    @classmethod
    def get_label_vocab_file(cls, label_vocab_file):
        if label_vocab_file is None:
            return label_vocab_file
        if not os.path.exists(label_vocab_file):
            raise FileNotFoundError(
                f"Label vocab file does not exist: {label_vocab_file}"
            )
        return os.path.abspath(label_vocab_file)


class BertForTokenClassification(nn.Module):
    def __init__(self, config: BertForTokenClassificationModelConfig):
        super().__init__()

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.encoder_output_dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)

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
    ):
        encoded_outputs, _ = self.bert(
            input_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        encoded_outputs = self.dropout(encoded_outputs)
        logits = self.classifier(encoded_outputs)

        return logits


class BertForTokenClassificationLoss(nn.Module):
    def __init__(self, num_labels, include_padding_in_loss, loss_weight=1.0):
        super().__init__()
        self.num_labels = num_labels
        self.loss_weight = loss_weight
        self.include_padding_in_loss = include_padding_in_loss

    def forward(self, logits, labels, attention_mask):
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1).long()
            )

            loss_weight = self.loss_weight
            if attention_mask is not None:
                # Only keep active parts of the loss
                loss = loss * attention_mask.to(dtype=logits.dtype).view(-1)
                if self.include_padding_in_loss:
                    loss_weight = self.loss_weight * (
                        1.0 / attention_mask.shape[1]
                    )

            loss = torch.sum(loss) / labels.shape[0] * loss_weight
        return loss
