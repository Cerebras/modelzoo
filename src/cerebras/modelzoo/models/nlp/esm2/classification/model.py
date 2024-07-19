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

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.models.nlp.bert.bert_finetune_models import (
    BertForSequenceClassificationLoss,
)
from cerebras.modelzoo.models.nlp.esm2.classification.esm2_classification_models import (
    ESM2SequenceClassifier,
)
from cerebras.modelzoo.models.nlp.esm2.model import Esm2ForPreTrainingModel
from cerebras.pytorch.metrics import AccuracyMetric, FBetaScoreMetric


@registry.register_model("esm2/classification")
class ESM2ForSequenceClassificationModel(Esm2ForPreTrainingModel):
    """
    ESM2 model for sequence classification tasks.

    This model uses a pre-trained ESM2 model and fine-tunes it for sequence classification.
    It also includes an optional accuracy and F1 score evaluation metrics for binary classification.
    """

    def __init__(self, params):
        # SW-131003: skip eval metrics in Esm2ForPreTrainingModel
        compute_eval_metrics = params.model.compute_eval_metrics
        params.model.compute_eval_metrics = False
        super().__init__(params)
        params.model.compute_eval_metrics = compute_eval_metrics

        self._model_params = params.model
        self.classifier = ESM2SequenceClassifier(
            hidden_size=self._model_params.hidden_size,
            num_labels=self._model_params.num_labels,
            esm2_bert_encoder=self.model.bert_encoder,
            freeze_pretrained_model=self._model_params.freeze_pretrained_model,
            classifier_dropout=self._model_params.classifier_dropout,
        )

        self.loss_fn = BertForSequenceClassificationLoss(
            self._model_params.num_labels, self._model_params.problem_type
        )

        if self._model_params.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")
            if self._model_params.num_labels == 2:
                self.f1_metric = FBetaScoreMetric(
                    num_classes=self._model_params.num_labels,
                    beta=1.0,
                    average_type="micro",
                    name="eval/f1_score",
                )

    def forward(self, data):
        logits = self.classifier(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
        )
        loss = self.loss_fn(data["labels"], logits)
        model_params = self._model_params
        if not self.classifier.training and model_params.compute_eval_metrics:
            labels = data["labels"].clone()

            predictions = logits.argmax(-1).int()

            self.accuracy_metric(
                labels=labels, predictions=predictions, dtype=logits.dtype
            )

            if model_params.num_labels == 2:
                self.f1_metric(labels=labels, predictions=predictions)

        return loss
