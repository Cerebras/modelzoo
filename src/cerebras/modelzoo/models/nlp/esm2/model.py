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

from cerebras.modelzoo.models.nlp.bert.model import BertForPreTrainingModel
from cerebras.modelzoo.models.nlp.esm2.esm2_pretrain_models import (
    Esm2PretrainModel,
)


class Esm2ForPreTrainingModel(BertForPreTrainingModel):
    """
    Esm-2
    """

    def model_class(self):
        return Esm2PretrainModel

    def build_model_args(self, model_params):
        args = {
            **BertForPreTrainingModel.build_model_args(self, model_params),
            "token_dropout": model_params.pop("token_dropout", False),
            "mask_token_id": model_params.pop("mask_token_id", None),
            "use_final_layer_norm": model_params.pop(
                "use_final_layer_norm", False
            ),
            "embedding_layer_norm": model_params.pop(
                "embedding_layer_norm", True
            ),
        }
        return args
