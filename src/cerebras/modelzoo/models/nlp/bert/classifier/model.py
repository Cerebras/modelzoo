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

import cerebras.pytorch as cstorch
from cerebras.modelzoo.models.nlp.bert.classifier.bert_classifier_model import (
    BertForSequenceClassification,
    BertForSequenceClassificationLoss,
    BertForSequenceClassificationModelConfig,
)
from cerebras.pytorch.metrics import AccuracyMetric, FBetaScoreMetric


class BertForSequenceClassificationModel(torch.nn.Module):
    def __init__(self, config: BertForSequenceClassificationModelConfig):
        super().__init__()

        self.compute_eval_metrics = config.compute_eval_metrics
        self.num_labels = config.num_labels

        self.model = BertForSequenceClassification(config)
        self.loss_fn = BertForSequenceClassificationLoss(
            self.num_labels, config.problem_type
        )

        if self.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")
            if self.num_labels == 2:
                self.f1_metric = FBetaScoreMetric(
                    num_classes=self.num_labels,
                    beta=1.0,
                    average_type="micro",
                    name="eval/f1_score",
                )
            # Below flag helps create two more accuracy objects for
            # matched and mismatched partitions
            self.matched_accuracy_metric = AccuracyMetric(
                name="eval/accuracy_matched"
            )
            self.mismatched_accuracy_metric = AccuracyMetric(
                name="eval/accuracy_mismatched"
            )

    def forward(self, data):
        logits = self.model(
            input_ids=data["input_ids"],
            token_type_ids=data["token_type_ids"],
            attention_mask=data["attention_mask"],
        )
        loss = self.loss_fn(data["labels"], logits)
        if not self.model.training and self.compute_eval_metrics:
            labels = data["labels"].clone()

            predictions = logits.argmax(-1).int()

            metric_dtype = (
                torch.float32
                if cstorch.amp.is_cbfloat16_tensor(logits)
                else logits.dtype
            )

            self.accuracy_metric(
                labels=labels, predictions=predictions, dtype=metric_dtype
            )

            if self.num_labels == 2:
                self.f1_metric(labels=labels, predictions=predictions)

            if "is_matched" in data:
                self.matched_accuracy_metric(
                    labels=labels,
                    predictions=predictions,
                    weights=data["is_matched"],
                    dtype=metric_dtype,
                )

            if "is_mismatched" in data:
                self.mismatched_accuracy_metric(
                    labels=labels,
                    predictions=predictions,
                    weights=data["is_mismatched"],
                    dtype=metric_dtype,
                )

        return loss
