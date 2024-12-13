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

from typing import Literal

from cerebras.modelzoo.models.nlp.bert.bert_pretrain_models import (
    BertPretrainModel,
)
from cerebras.modelzoo.models.nlp.bert.model import (
    BertForPreTrainingModelConfig,
)
from cerebras.modelzoo.models.nlp.esm2.esm2_model import (
    Esm2Model,
    Esm2ModelConfig,
)


class Esm2ForPreTrainingModelConfig(
    Esm2ModelConfig, BertForPreTrainingModelConfig
):
    name: Literal["esm2"]

    def post_init(self, context):
        super().post_init(context)

        self.add_pooling_layer = not self.disable_nsp


class Esm2PretrainModel(BertPretrainModel):
    """
    Esm-2 Model pretrain model.
    """

    def __init__(self, config: Esm2ForPreTrainingModelConfig):
        super().__init__(config)

    def build_encoder_model(self, config: Esm2ForPreTrainingModelConfig):
        return Esm2Model(config)
