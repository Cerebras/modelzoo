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

from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.transformers.pytorch.bert.bert_finetune_models import (
    BertForQuestionAnswering,
    BertForQuestionAnsweringLoss,
)
from modelzoo.transformers.pytorch.bert.utils import check_unused_model_params


class BertForQuestionAnsweringModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        dropout_rate = model_params.pop("dropout_rate")
        embedding_dropout_rate = model_params.pop(
            "embedding_dropout_rate", dropout_rate
        )

        model_kwargs = {
            "vocab_size": model_params.pop("vocab_size"),
            "hidden_size": model_params.pop("hidden_size"),
            "num_hidden_layers": model_params.pop("num_hidden_layers"),
            "num_heads": model_params.pop("num_heads"),
            "filter_size": model_params.pop("filter_size"),
            "nonlinearity": model_params.pop("encoder_nonlinearity"),
            "embedding_dropout_rate": embedding_dropout_rate,
            "dropout_rate": dropout_rate,
            "attention_dropout_rate": model_params.pop(
                "attention_dropout_rate"
            ),
            "max_position_embeddings": model_params.pop(
                "max_position_embeddings"
            ),
            "layer_norm_epsilon": float(model_params.pop("layer_norm_epsilon")),
        }

        self.model = BertForQuestionAnswering(**model_kwargs)
        self.loss_fn = BertForQuestionAnsweringLoss()

        check_unused_model_params(model_params)
        super(BertForQuestionAnsweringModel, self).__init__(
            params=params, model=self.model, device=device,
        )

    def __call__(self, data):
        logits, start_logits, end_logits = self.model(
            input_ids=data["input_ids"],
            token_type_ids=data["token_type_ids"],
            attention_mask=data["attention_mask"],
            labels=data["labels"],
            label_weights=data["label_weights"],
        )
        loss = self.loss_fn(logits, data["labels"], data["label_weights"])

        # for prediction inference
        self.outputs = {"start_logits": start_logits, "end_logits": end_logits}

        return loss
