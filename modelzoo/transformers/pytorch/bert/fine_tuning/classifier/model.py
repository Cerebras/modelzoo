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

from modelzoo.common.pytorch.metrics import AccuracyMetric, FBetaScoreMetric
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.pytorch.bert.utils import check_unused_model_params
from modelzoo.transformers.pytorch.huggingface_common.modeling_bert import (
    BertConfig,
    BertForSequenceClassification,
)


class BertForSequenceClassificationModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        self.num_labels = model_params.pop("num_labels")
        self.model = BertForSequenceClassification(
            BertConfig(
                vocab_size=model_params.pop("vocab_size"),
                hidden_size=model_params.pop("hidden_size"),
                num_hidden_layers=model_params.pop("num_hidden_layers"),
                num_attention_heads=model_params.pop("num_heads"),
                intermediate_size=model_params.pop("filter_size"),
                hidden_act=model_params.pop("encoder_nonlinearity"),
                hidden_dropout_prob=model_params.pop("dropout_rate"),
                attention_probs_dropout_prob=model_params.pop(
                    "attention_dropout_rate"
                ),
                max_position_embeddings=model_params.pop(
                    "max_position_embeddings"
                ),
                classifier_dropout=model_params.pop("task_dropout"),
                problem_type=model_params.pop("problem_type"),
                num_labels=self.num_labels,
                layer_norm_eps=float(model_params.pop("layer_norm_epsilon")),
            ),
        )
        self.compute_eval_metrics = model_params.pop(
            "compute_eval_metrics", False
        )
        # Below flag helps create two more accuray objects for
        # matched and mismatched partitions
        self.is_mnli_dataset = model_params.pop("is_mnli_dataset", False)
        check_unused_model_params(model_params)
        self.loss_fn = self.model.loss_fn
        if self.compute_eval_metrics:
            self.accuracy_metric = AccuracyMetric(name="eval/accuracy")
            if self.num_labels == 2:
                self.f1_metric = FBetaScoreMetric(
                    num_classes=self.num_labels,
                    beta=1.0,
                    average_type="micro",
                    name="eval/f1_score",
                )
            if self.is_mnli_dataset:
                self.matched_accuracy_metric = AccuracyMetric(
                    name="eval/accuracy_matched"
                )
                self.mismatched_accuracy_metric = AccuracyMetric(
                    name="eval/accuracy_mismatched"
                )

        super(BertForSequenceClassificationModel, self).__init__(
            params=params, model=self.model, device=device
        )

    def __call__(self, data):
        output = self.model(
            input_ids=data["input_ids"],
            token_type_ids=data["token_type_ids"],
            labels=data["labels"],
            attention_mask=data["attention_mask"],
        )
        if not self.model.training and self.compute_eval_metrics:
            labels = data["labels"].clone()

            predictions = output.logits.argmax(-1).int()

            self.accuracy_metric(labels=labels, predictions=predictions)

            if self.num_labels == 2:
                self.f1_metric(labels=labels, predictions=predictions)

            if self.is_mnli_dataset:
                self.matched_accuracy_metric(
                    labels=labels,
                    predictions=predictions,
                    weights=data["is_matched"],
                )
                self.mismatched_accuracy_metric(
                    labels=labels,
                    predictions=predictions,
                    weights=data["is_mismatched"],
                )

        return output.loss
