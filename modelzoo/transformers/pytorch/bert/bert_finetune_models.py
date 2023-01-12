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

from modelzoo.common.pytorch.run_utils import half_dtype_instance
from modelzoo.transformers.pytorch.bert.bert_model import BertModel
from modelzoo.transformers.pytorch.transformer_utils import (
    make_key_padding_mask_broadcastable,
)


class BertForSequenceClassificationLoss(nn.Module):
    def __init__(self, num_labels, problem_type):
        super(BertForSequenceClassificationLoss, self).__init__()
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
                    logits.view(-1, self.num_labels), labels.view(-1).long(),
                )
            elif self.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels.reshape(-1))
        return loss


class BertForSequenceClassification(nn.Module):
    def __init__(
        self, num_labels, problem_type, classifier_dropout, **model_kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.bert = BertModel(**model_kwargs)
        if classifier_dropout is None:
            classifier_dropout = model_kwargs["dropout_rate"]
        self.dropout = nn.Dropout(classifier_dropout)
        hidden_size = model_kwargs["hidden_size"]
        self.classifier = nn.Linear(hidden_size, self.num_labels)

        self.loss_fn = BertForSequenceClassificationLoss(
            self.num_labels, self.problem_type
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
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
    ):

        attention_mask = make_key_padding_mask_broadcastable(attention_mask)
        _, pooled_outputs = self.bert(
            input_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        return logits


class BertForQuestionAnsweringLoss(nn.Module):
    def __init__(self):
        super(BertForQuestionAnsweringLoss, self).__init__()

    def forward(self, logits, labels, cls_label_weights):

        # [batch, max_seq_len, 2] -> [batch, 2, max_seq_len]
        logits = torch.permute(logits, [0, 2, 1])
        max_seq_len = logits.shape[-1]
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.reshape(-1, max_seq_len), labels.view(-1).long())
        return (loss * cls_label_weights.view(-1)).sum() / labels.shape[0]


class BertForQuestionAnswering(nn.Module):
    def __init__(self, **model_kwargs):
        super().__init__()

        hidden_size = model_kwargs["hidden_size"]

        self.bert = BertModel(**model_kwargs, add_pooling_layer=False)
        self.classifier = nn.Linear(hidden_size, 2)

        self.loss_fn = BertForQuestionAnsweringLoss()
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
        labels=None,
        label_weights=None,
    ):
        attention_mask = make_key_padding_mask_broadcastable(attention_mask)
        encoded_outputs, _ = self.bert(
            input_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        logits = self.classifier(encoded_outputs)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return logits, start_logits, end_logits


class BertForTokenClassificationLoss(nn.Module):
    def __init__(self, num_labels, loss_weight=1.0):
        super(BertForTokenClassificationLoss, self).__init__()
        self.num_labels = num_labels
        self.loss_weight = loss_weight

    def forward(self, logits, labels, attention_mask):
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')

            loss = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1).long()
            )
            if attention_mask is not None:
                # Only keep active parts of the loss
                loss = loss * attention_mask.to(dtype=logits.dtype).view(-1)
            loss = torch.sum(loss) / labels.shape[0] * self.loss_weight
            loss = loss.to(half_dtype_instance.half_dtype)
        return loss


class BertForTokenClassification(nn.Module):
    def __init__(
        self,
        num_labels,
        classifier_dropout=None,
        loss_weight=1.0,
        include_padding_in_loss=True,
        **model_kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.include_padding_in_loss = include_padding_in_loss

        self.bert = BertModel(**model_kwargs, add_pooling_layer=False)
        if classifier_dropout is None:
            classifier_dropout = model_kwargs["dropout_rate"]
        self.dropout = nn.Dropout(classifier_dropout)

        hidden_size = model_kwargs["hidden_size"]
        self.classifier = nn.Linear(hidden_size, num_labels)

        self.loss_fn = BertForTokenClassificationLoss(
            self.num_labels, loss_weight
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
        loss_mask=None,
        labels=None,
    ):
        attention_mask = make_key_padding_mask_broadcastable(attention_mask)
        encoded_outputs, _ = self.bert(
            input_ids,
            segment_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        encoded_outputs = self.dropout(encoded_outputs)
        logits = self.classifier(encoded_outputs)

        return logits


class BertForSummarizationLoss(nn.Module):
    def __init__(
        self, num_labels, loss_weight=1.0,
    ):
        super(BertForSummarizationLoss, self).__init__()
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
            loss = loss.to(half_dtype_instance.half_dtype)

        return loss


class BertForSummarization(nn.Module):
    def __init__(
        self, num_labels=2, loss_weight=1.0, use_cls_bias=True, **model_kwargs,
    ):
        super().__init__()
        self.num_labels = num_labels

        hidden_size = model_kwargs["hidden_size"]

        self.bert = BertModel(**model_kwargs, add_pooling_layer=False)
        self.classifier = nn.Linear(
            hidden_size, self.num_labels, bias=use_cls_bias
        )

        self.loss_fn = BertForSummarizationLoss(self.num_labels, loss_weight,)
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
        labels=None,
        cls_tokens_positions=None,
        cls_label_weights=None,
    ):
        attention_mask = make_key_padding_mask_broadcastable(attention_mask)
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
