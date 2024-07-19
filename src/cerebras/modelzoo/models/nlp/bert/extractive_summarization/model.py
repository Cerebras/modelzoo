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
    BertForSummarization,
    BertForSummarizationLoss,
)


@registry.register_model(
    "bert/extractive_summarization",
    datasetprocessor=["BertSumCSVDataProcessor"],
)
class BertSummarizationModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        self._model_params = params.model
        model_params = self._model_params

        num_labels = 2
        loss_weight = model_params.loss_weight

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

        self.model = BertForSummarization(
            num_labels=num_labels,
            loss_weight=loss_weight,
            use_cls_bias=model_params.use_cls_bias,
            **model_kwargs,
        )
        self.loss_fn = BertForSummarizationLoss(
            num_labels,
            loss_weight,
        )

        if model_params.compute_eval_metrics:
            raise NotImplementedError(
                "RougeScoreMetric not yet supported in weight streaming"
            )

            self.rouge1_score = RougeScoreMetric(
                max_n=1,
                vocab_file=model_params.vocab_file,
                name="eval/rouge1",
            )
            self.rouge2_score = RougeScoreMetric(
                max_n=2,
                vocab_file=model_params.vocab_file,
                name="eval/rouge2",
            )

    def forward(self, data):
        logits = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            token_type_ids=data["token_type_ids"],
            cls_tokens_positions=data["cls_indices"],
        )
        loss = self.loss_fn(
            logits, data["labels"], data["cls_weights"].clone().to(logits.dtype)
        )
        if not self.model.training and self._model_params.compute_eval_metrics:
            labels = data["labels"].clone()
            predictions = logits.argmax(-1).int()
            input_ids = data["input_ids"].clone()
            cls_indices = data["cls_indices"].clone()
            cls_weights = data["cls_weights"].clone()
            self.rouge1_score(
                labels,
                predictions,
                cls_indices,
                cls_weights,
                input_ids,
            )
            self.rouge2_score(
                labels,
                predictions,
                cls_indices,
                cls_weights,
                input_ids,
            )

        return loss
