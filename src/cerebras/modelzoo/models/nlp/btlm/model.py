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

from cerebras.modelzoo.models.nlp.gpt2.model import Gpt2Model, GPT2ModelConfig


class BTLMModelConfig(GPT2ModelConfig):
    name: Literal["btlm"]


class BTLMModel(Gpt2Model):
    def __init__(self, config: BTLMModelConfig):
        if isinstance(config, dict):
            config = BTLMModelConfig(**config)
        super().__init__(config)
