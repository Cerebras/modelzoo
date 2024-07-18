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

from typing import Optional

import torch.nn as nn

from cerebras.modelzoo.models.nlp.bert.bert_model import BertModel, BertPooler


class ESM2SequenceClassifier(nn.Module):
    """
    Sequence Classification model using the Esm2PretrainModel as the base model.

    Args:
        hidden_size (int): The size of the hidden layer.
        num_labels (int): The number of labels for classification.
        esm2_bert_encoder (BertModel): (Pretrained) BERT encoder model.
        freeze_pretrained_model (bool, optional): If True, freeze the pretrained model parameters. Defaults to False.
        classifier_dropout (Optional[float], optional): Dropout probability for the classifier layer. Defaults to None.
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        esm2_bert_encoder: BertModel,
        freeze_pretrained_model: bool = False,
        classifier_dropout: Optional[float] = None,
        classifier_init_mean: float = 0.0,
        classifier_init_std: float = 0.02,
    ):
        super().__init__()

        self.num_labels = num_labels
        self.dropout = (
            nn.Dropout(classifier_dropout)
            if classifier_dropout is not None
            else None
        )
        self.bert_encoder = esm2_bert_encoder
        self.hidden_size = hidden_size
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.classifier_init_mean = classifier_init_mean
        self.classifier_init_std = classifier_init_std
        self.bert_pooler = BertPooler(hidden_size)
        if freeze_pretrained_model:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False

        self.__reset_parameters()

    def reset_parameters(self):
        self.bert_encoder.reset_parameters()
        self.bert_pooler.reset_parameters()
        self.__reset_parameters()

    def __reset_parameters(self):
        self.classifier.weight.data.normal_(
            mean=self.classifier_init_mean,
            std=self.classifier_init_std,
        )
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        segment_ids=None,
    ):
        """
        Forward pass of the model.

        Args:
            input_ids (Tensor, optional): Input tensor containing token IDs. Defaults to None.
            attention_mask (Tensor, optional): Attention mask tensor. Defaults to None.
            segment_ids (Tensor, optional): Token type IDs tensor. Defaults to None.

        Returns:
            Tensor: Logits tensor of shape (batch_size, num_labels).
        """

        hidden_states, _ = self.bert_encoder(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
        )
        pooled_outputs = self.bert_pooler(hidden_states)

        if self.dropout is not None:
            pooled_outputs = self.dropout(pooled_outputs)
        logits = self.classifier(pooled_outputs)

        return logits
