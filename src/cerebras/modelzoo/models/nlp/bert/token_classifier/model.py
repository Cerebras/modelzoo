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
from cerebras.modelzoo.data_preparation.utils import get_label_id_map
from cerebras.modelzoo.models.nlp.bert.bert_finetune_models import (
    BertForTokenClassification,
    BertForTokenClassificationLoss,
)
from cerebras.pytorch.metrics import FBetaScoreMetric


@registry.register_model(
    "bert/token_classifier",
    datasetprocessor=["BertTokenClassifierDataProcessor"],
)
class BertForTokenClassificationModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        self._model_params = params.model
        model_params = self._model_params
        num_classes = self._model_params.num_classes
        loss_weight = self._model_params.loss_weight

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

        self.model = BertForTokenClassification(
            num_classes,
            classifier_dropout=model_params.encoder_output_dropout_rate,
            loss_weight=loss_weight,
            include_padding_in_loss=model_params.include_padding_in_loss,
            **model_kwargs,
        )
        self.loss_fn = BertForTokenClassificationLoss(num_classes, loss_weight)

        if model_params.compute_eval_metrics:
            self.label_map_id = get_label_id_map(model_params.label_vocab_file)
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
                num_classes=num_classes,
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
        if not self.model.training and self._model_params.compute_eval_metrics:
            labels = data["labels"].clone()
            predictions = logits.argmax(-1).int()

            self.f1_metric(
                labels=labels,
                predictions=predictions,
            )

        return loss
