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
from warnings import warn

from pydantic import field_validator

from cerebras.modelzoo.models.nlp.gpt2.gpt2_model import (
    GPT2LMHeadModel,
    GPT2LMHeadModelConfig,
)
from cerebras.modelzoo.models.nlp.gpt2.model import Gpt2Model, GPT2ModelConfig


class LlamaModelConfig(GPT2ModelConfig):
    name: Literal["LlamaModel", "llama"]

    @field_validator("name", mode="after")
    def validate_name(cls, name):
        if name == "LlamaModel":
            warn(
                "Passing 'LlamaModel' as the model name is deprecated. "
                "Please use 'llama' instead.",
                category=FutureWarning,
            )
            return "llama"
        return name


class LlamaModel(Gpt2Model):
    def __init__(self, config: LlamaModelConfig):
        if isinstance(config, dict):
            config = LlamaModelConfig(**config)
        super().__init__(config)


class LlamaLMHeadModelConfig(GPT2LMHeadModelConfig):
    name: Literal["LlamaModel", "llama"]

    @field_validator("name", mode="after")
    def validate_name(cls, name):
        if name == "LlamaModel":
            warn(
                "Passing 'LlamaModel' as the model name is deprecated. "
                "Please use 'llama' instead.",
                category=FutureWarning,
            )
        return "llama"

    @property
    def __model_cls__(self):
        return LlamaLMHeadModel


class LlamaLMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: LlamaLMHeadModelConfig):
        if isinstance(config, dict):
            config = LlamaLMHeadModelConfig(**config)
        super().__init__(config)
