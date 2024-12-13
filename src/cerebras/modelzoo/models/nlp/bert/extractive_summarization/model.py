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

from cerebras.modelzoo.models.nlp.bert.extractive_summarization.bert_extractive_summarization import (
    BertForSummarization,
    BertForSummarizationLoss,
    BertSummarizationModelConfig,
)


class BertSummarizationModel(torch.nn.Module):
    def __init__(self, config: BertSummarizationModelConfig):
        super().__init__()

        self.compute_eval_metrics = config.compute_eval_metrics

        self.model = BertForSummarization(config)
        self.loss_fn = BertForSummarizationLoss(
            config.num_labels,
            config.loss_weight,
        )

        if self.compute_eval_metrics:
            raise NotImplementedError(
                "RougeScoreMetric not yet supported in weight streaming"
            )

            self.rouge1_score = RougeScoreMetric(
                max_n=1,
                vocab_file=config.vocab_file,
                name="eval/rouge1",
            )
            self.rouge2_score = RougeScoreMetric(
                max_n=2,
                vocab_file=config.vocab_file,
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
        if not self.model.training and self.compute_eval_metrics:
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
