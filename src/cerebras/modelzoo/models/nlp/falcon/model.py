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

from cerebras.modelzoo.models.nlp.gptj.model import GptjModel, GPTJModelConfig


class FalconModelConfig(GPTJModelConfig):
    name: Literal["falcon"]


class FalconModel(GptjModel):
    def __init__(self, config: FalconModelConfig):
        if isinstance(config, dict):
            config = FalconModelConfig(**config)
        super().__init__(config)
