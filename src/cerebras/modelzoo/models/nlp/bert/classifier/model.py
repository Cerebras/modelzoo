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

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.models.nlp.bert.bert_finetune_models import (
    BertForSequenceClassification,
    BertForSequenceClassificationLoss,
)
from cerebras.pytorch.metrics import AccuracyMetric, FBetaScoreMetric


@registry.register_model("bert/classifier")
class BertForSequenceClassificationModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        self._model_params = params.model
        model_params = self._model_params

        model_kwargs = {
            "vocab_size": model_params.vocab_size,
            "hidden_size": model_params.hidden_size,
            "num_hidden_layers": model_params.num_hidden_layers,
            "num_heads": model_params.num_heads,
            "filter_size": model_params.filter_size,
            "nonlinearity": model_params.encoder_nonlinearity,
            "pooler_nonlinearity": model_params.pooler_nonlinearity,
            "embedding_dropout_rate": model_params.embedding_dropout_rate,
            "dropout_rate": model_params.dropout_rate,
            "attention_dropout_rate": model_params.attention_dropout_rate,
            "attention_kernel": model_params.attention_kernel,
            "max_position_embeddings": model_params.max_position_embeddings,
            "layer_norm_epsilon": model_params.layer_norm_epsilon,
        }

        self.model = BertForSequenceClassification(
            model_params.num_labels,
            model_params.problem_type,
            model_params.task_dropout,
            **model_kwargs,
        )
        self.loss_fn = BertForSequenceClassificationLoss(
            model_params.num_labels, model_params.problem_type
        )

        if model_params.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")
            if model_params.num_labels == 2:
                self.f1_metric = FBetaScoreMetric(
                    num_classes=model_params.num_labels,
                    beta=1.0,
                    average_type="micro",
                    name="eval/f1_score",
                )
            # Below flag helps create two more accuracy objects for
            # matched and mismatched partitions
            if model_params.is_mnli_dataset:
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
        model_params = self._model_params
        if not self.model.training and model_params.compute_eval_metrics:
            labels = data["labels"].clone()

            predictions = logits.argmax(-1).int()

            self.accuracy_metric(
                labels=labels, predictions=predictions, dtype=logits.dtype
            )

            if model_params.num_labels == 2:
                self.f1_metric(labels=labels, predictions=predictions)

            if model_params.is_mnli_dataset:
                self.matched_accuracy_metric(
                    labels=labels,
                    predictions=predictions,
                    weights=data["is_matched"],
                    dtype=logits.dtype,
                )
                self.mismatched_accuracy_metric(
                    labels=labels,
                    predictions=predictions,
                    weights=data["is_mismatched"],
                    dtype=logits.dtype,
                )

        return loss
