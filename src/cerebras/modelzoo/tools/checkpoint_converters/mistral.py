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

from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseConfigConverter,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    ConfigConverter_LLaMa_HF_CS21,
    Converter_LlamaForCausalLM_HF_CS21,
    Converter_LlamaModel_HF_CS21,
)


class Converter_MistralModel_HF_CS21(Converter_LlamaModel_HF_CS21):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Mistral_HF_CS21

    @classmethod
    def converter_note(cls) -> str:
        return (
            f"{cls.formats()[0]} MistralModel <-> {cls.formats()[1]} GPT2LMHeadModel (configured as "
            f"Mistral)\nThe HF model doesn't contain a language model head while the CS one does. "
            f"When converting to CS, the exported checkpoint will contain a language model head "
            f"initialized to default random values. When converting to HF, the language model head "
            f"will be dropped."
        ).format(cls.formats()[0], cls.formats()[1])


class Converter_MistralForCausalLM_HF_CS21(Converter_LlamaForCausalLM_HF_CS21):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Mistral_HF_CS21

    @classmethod
    def converter_note(cls) -> str:
        return "{} MistralForCausalLM <-> {} GPT2LMHeadModel (configured as Mistral)".format(
            cls.formats()[0], cls.formats()[1]
        )


class ConfigConverter_Mistral_HF_CS21(ConfigConverter_LLaMa_HF_CS21):
    def __init__(self):
        self.model_type = "mistral"
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "sliding_window", "attention_sliding_window_length"
                    )
                ],
                action=self.replaceKey,
            ),
            *self.rules,
        ]

        self.post_convert_defaults[0].update(
            {"model_type": "mistral", "architectures": ["MistralForCausalLM"]}
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )
