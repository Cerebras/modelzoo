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
from modelzoo.transformers.pytorch.bert.utils import check_unused_model_params
from modelzoo.transformers.pytorch.huggingface_common.modeling_bert import (
    BertConfig,
    BertForQuestionAnswering,
)


class BertForQuestionAnsweringModel(PyTorchBaseModel):
    def __init__(self, params, device=None):
        self.params = params
        model_params = self.params["model"].copy()
        kwargs = {
            "vocab_size": model_params.pop("vocab_size"),
            "hidden_size": model_params.pop("hidden_size"),
            "num_hidden_layers": model_params.pop("num_hidden_layers"),
            "num_attention_heads": model_params.pop("num_heads"),
            "intermediate_size": model_params.pop("filter_size"),
            "hidden_act": model_params.pop("encoder_nonlinearity"),
            "hidden_dropout_prob": model_params.pop("dropout_rate"),
            "attention_probs_dropout_prob": model_params.pop(
                "attention_dropout_rate"
            ),
            "max_position_embeddings": model_params.pop(
                "max_position_embeddings"
            ),
            "layer_norm_eps": float(model_params.pop("layer_norm_epsilon")),
        }
        self.model = BertForQuestionAnswering(BertConfig(**kwargs),)
        check_unused_model_params(model_params)
        self.loss_fn = self.model.loss_fn
        super(BertForQuestionAnsweringModel, self).__init__(
            params=params, model=self.model, device=device,
        )

    def __call__(self, data):
        output = self.model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            token_type_ids=data["token_type_ids"],
            labels=data["labels"],
            label_weights=data["label_weights"],
        )
        self.outputs = output
        return output.loss
