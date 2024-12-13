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

from cerebras.modelzoo.data_preparation.utils import get_label_id_map
from cerebras.modelzoo.models.nlp.bert.token_classifier.bert_token_classifier import (
    BertForTokenClassification,
    BertForTokenClassificationLoss,
    BertForTokenClassificationModelConfig,
)
from cerebras.pytorch.metrics import FBetaScoreMetric


class BertForTokenClassificationModel(torch.nn.Module):
    def __init__(self, config: BertForTokenClassificationModelConfig):
        super().__init__()

        self.compute_eval_metrics = config.compute_eval_metrics

        self.model = BertForTokenClassification(config)
        self.loss_fn = BertForTokenClassificationLoss(
            config.num_classes,
            config.include_padding_in_loss,
            config.loss_weight,
        )

        if self.compute_eval_metrics:
            self.label_map_id = get_label_id_map(config.label_vocab_file)
            # Ignore token labels in eval which dont
            # refer to a token beginning or inside.
            # Labels such as
            # "O", [CLS], [SEP], [PAD], "O", "X"
            # are ignored during eval
            eval_ignore_labels = []
            if self.label_map_id is not None:
                for key, label_id in self.label_map_id.items():
                    if not (key.startswith("B") or key.startswith("I")):
                        eval_ignore_labels.append(label_id)

            self.f1_metric = FBetaScoreMetric(
                num_classes=config.num_classes,
                beta=1.0,
                average_type="macro",
                ignore_labels=eval_ignore_labels,
                name="eval/f1_score",
            )

    def forward(self, data):
        logits = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            token_type_ids=data["token_type_ids"],
        )
        loss = self.loss_fn(logits, data["labels"], data["loss_mask"])
        if not self.model.training and self.compute_eval_metrics:
            labels = data["labels"].clone()
            predictions = logits.argmax(-1).int()

            self.f1_metric(
                labels=labels,
                predictions=predictions,
            )

        return loss
