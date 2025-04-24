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

import logging
from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    BaseConfigConverter_HF_CS,
    FormatIndices,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.falcon_7b import (
    ConfigConverter_Falcon_7B_HF_CS19,
    Converter_Falcon_7B_Headless_HF_CS19,
    Converter_Falcon_7B_HF_CS19,
)
from cerebras.modelzoo.tools.checkpoint_converters.falcon_40b import (
    ConfigConverter_Falcon_40B_HF_CS20,
    Converter_Falcon_40B_Headless_HF_CS20,
    Converter_Falcon_40B_HF_CS20,
)
from cerebras.modelzoo.tools.checkpoint_converters.falcon_180b import (
    ConfigConverter_Falcon_180B_HF_CS20,
    ConfigConverter_Falcon_180B_HF_CS21,
    Converter_Falcon_180B_Headless_HF_CS20,
    Converter_Falcon_180B_Headless_HF_CS21,
    Converter_Falcon_180B_HF_CS20,
    Converter_Falcon_180B_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.gptj_hf_cs import (
    Converter_GPTJ_LMHeadModel_CS20_CS21,
)


class Converter_Falcon_Headless_HF_CS20(BaseCheckpointConverter_HF_CS):
    config2model_subconverters = {
        ConfigConverter_Falcon_7B_HF_CS19: Converter_Falcon_7B_Headless_HF_CS19,
        ConfigConverter_Falcon_40B_HF_CS20: Converter_Falcon_40B_Headless_HF_CS20,
        ConfigConverter_Falcon_180B_HF_CS20: Converter_Falcon_180B_Headless_HF_CS20,
    }

    def __init__(self):
        super().__init__()
        self.rules = []

    @classmethod
    def select_subconverter(
        cls,
        config,
        from_index: int,
        **kwargs,
    ):
        config_subconverter = (
            cls.get_config_converter_class().select_subconverter(
                config, from_index
            )
        )

        return cls.config2model_subconverters[config_subconverter]

    @classmethod
    def convert(cls, checkpoint, configs, converter_indices, **kwargs):
        subconverter = cls.select_subconverter(
            configs[converter_indices.direction], converter_indices.direction
        )
        instance = subconverter()
        new_checkpoint = instance.convert_helper(
            checkpoint, configs, converter_indices, **kwargs
        )
        return new_checkpoint

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9", "cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} FalconModel or RWModel <-> {} GPTJModel (configured as Falcon)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Falcon_HF_CS20


class Converter_Falcon_HF_CS20(BaseCheckpointConverter_HF_CS):
    config2model_subconverters = {
        ConfigConverter_Falcon_7B_HF_CS19: Converter_Falcon_7B_HF_CS19,
        ConfigConverter_Falcon_40B_HF_CS20: Converter_Falcon_40B_HF_CS20,
        ConfigConverter_Falcon_180B_HF_CS20: Converter_Falcon_180B_HF_CS20,
    }

    def __init__(self):
        super().__init__()
        self.rules = []

    @classmethod
    def select_subconverter(
        cls,
        config,
        from_index: int,
        **kwargs,
    ):
        config_subconverter = (
            cls.get_config_converter_class().select_subconverter(
                config, from_index
            )
        )
        return cls.config2model_subconverters[config_subconverter]

    @classmethod
    def convert(cls, checkpoint, configs, converter_indices, **kwargs):
        subconverter = cls.select_subconverter(
            configs[converter_indices.direction], converter_indices.direction
        )
        instance = subconverter()
        new_checkpoint = instance.convert_helper(
            checkpoint, configs, converter_indices, **kwargs
        )
        return new_checkpoint

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9", "cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            f"{cls.formats()[0]} FalconForCausalLM or RWForCausalLM <-> {cls.formats()[1]} "
            f"GPTJModel (configured as Falcon) with LM head. When converting "
            f"to HF, make sure that transformers>=4.41.0 to have the most "
            f"recent falcon implementation."
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Falcon_HF_CS20


class ConfigConverter_Falcon_HF_CS20(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = []

    @classmethod
    def select_subconverter(
        cls,
        config,
        from_index: int,
        **kwargs,
    ):
        logging.info("HF's Falcon 7B, 40B, and 180B use different codebases.")
        if from_index == 0:
            if config.get("model_type", "") == "falcon":
                logging.info(
                    "The model that you're using was generated using the 180B "
                    "style codebase (model=FalconModel)"
                )
                return ConfigConverter_Falcon_180B_HF_CS20
            elif "n_head_kv" not in config:  # MQA, 7b structure
                logging.info(
                    "The model that you're using was generated using the 7B "
                    "style codebase (model=RefinedWeb) which only supports "
                    "multi-query attention (not grouped query)."
                )
                return ConfigConverter_Falcon_7B_HF_CS19
            else:  # GQA, 40B structure
                logging.info(
                    "The model that you're using was generated using the 40B "
                    "style codebase (model=RefinedWeb) with grouped query "
                    "attention support"
                )
                return ConfigConverter_Falcon_40B_HF_CS20
        else:
            logging.info(
                "The output will be formatted for the official 180B style "
                "codebase (model=FalconModel) rather than the 7B or 40B style "
                "codebases (model=RefinedWeb)"
            )
            return ConfigConverter_Falcon_180B_HF_CS20

    @classmethod
    def convert(
        cls,
        model,
        config,
        converter_indices: FormatIndices,
        drop_unmatched_keys: bool = False,
        no_progress_bar: bool = True,
        debug: bool = False,
    ):
        subconverter = cls.select_subconverter(
            config, converter_indices.direction
        )
        instance = subconverter()
        return instance.convert_helper(
            model,
            config,
            converter_indices,
            drop_unmatched_keys=drop_unmatched_keys,
            no_progress_bar=no_progress_bar,
            debug=debug,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9", "cs-2.0"))


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1, and HF <> CS 2.1 converters:
###########################################################


class Converter_Falcon_CS20_CS21(Converter_GPTJ_LMHeadModel_CS20_CS21):
    def __init__(self):
        super().__init__()

    @classmethod
    def converter_note(cls) -> str:
        return "GPT2LMHeadModel class (configured as falcon)"


class ConfigConverter_Falcon_HF_CS21(ConfigConverter_Falcon_HF_CS20):
    @classmethod
    def select_subconverter(
        cls,
        config,
        from_index: int,
        **kwargs,
    ):
        sub_converter = super().select_subconverter(
            config, from_index, **kwargs
        )
        # Only CS21 is different because others don't support alibi
        if sub_converter == ConfigConverter_Falcon_180B_HF_CS20:
            sub_converter = ConfigConverter_Falcon_180B_HF_CS21
        return sub_converter

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_Falcon_Headless_HF_CS21(Converter_Falcon_Headless_HF_CS20):
    config2model_subconverters = {
        ConfigConverter_Falcon_7B_HF_CS19: Converter_Falcon_7B_Headless_HF_CS19,
        ConfigConverter_Falcon_40B_HF_CS20: Converter_Falcon_40B_Headless_HF_CS20,
        ConfigConverter_Falcon_180B_HF_CS21: Converter_Falcon_180B_Headless_HF_CS21,
    }

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Falcon_HF_CS21


class Converter_Falcon_HF_CS21(Converter_Falcon_HF_CS20):
    config2model_subconverters = {
        ConfigConverter_Falcon_7B_HF_CS19: Converter_Falcon_7B_HF_CS19,
        ConfigConverter_Falcon_40B_HF_CS20: Converter_Falcon_40B_HF_CS20,
        ConfigConverter_Falcon_180B_HF_CS21: Converter_Falcon_180B_HF_CS21,
    }

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Falcon_HF_CS21
