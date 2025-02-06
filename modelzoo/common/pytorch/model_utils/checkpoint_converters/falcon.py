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

from modelzoo.common.pytorch.model_utils.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    BaseConfigConverter_HF_CS,
    FormatVersions,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.falcon_7b import (
    ConfigConverter_Falcon_7B_HF_CS19,
    Converter_Falcon_7B_Headless_HF_CS19,
    Converter_Falcon_7B_HF_CS19,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.falcon_40b import (
    ConfigConverter_Falcon_40B_HF_CS20,
    Converter_Falcon_40B_Headless_HF_CS20,
    Converter_Falcon_40B_HF_CS20,
)


class Converter_Falcon_Headless_HF_CS20(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = []

    @classmethod
    def convert(cls, checkpoint, configs, checkpoint_from_index, **kwargs):
        if (
            cls.get_config_converter_class().config_class
            == ConfigConverter_Falcon_7B_HF_CS19
        ):
            instance = Converter_Falcon_7B_Headless_HF_CS19()
        elif (
            cls.get_config_converter_class().config_class
            == ConfigConverter_Falcon_40B_HF_CS20
        ):
            instance = Converter_Falcon_40B_Headless_HF_CS20()
        new_checkpoint = instance.convert_helper(
            checkpoint, configs, checkpoint_from_index, **kwargs
        )
        return new_checkpoint

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9", "cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} Falcon HF <-> {} GPTJModel (configured as Falcon) with LM head".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Falcon_HF_CS20


class Converter_Falcon_HF_CS20(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = []

    @classmethod
    def convert(cls, checkpoint, configs, checkpoint_from_index, **kwargs):
        if (
            cls.get_config_converter_class().config_class
            == ConfigConverter_Falcon_7B_HF_CS19
        ):
            instance = Converter_Falcon_7B_HF_CS19()
        elif (
            cls.get_config_converter_class().config_class
            == ConfigConverter_Falcon_40B_HF_CS20
        ):
            instance = Converter_Falcon_40B_HF_CS20()
        new_checkpoint = instance.convert_helper(
            checkpoint, configs, checkpoint_from_index, **kwargs
        )
        return new_checkpoint

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9", "cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} Falcon HF <-> {} GPTJModel (configured as Falcon) with LM head".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Falcon_HF_CS20


class ConfigConverter_Falcon_HF_CS20(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = []

    @classmethod
    def convert(
        cls,
        config,
        from_index: int,
        drop_unmatched_keys: bool = False,
        no_progress_bar: bool = True,
        debug: bool = False,
    ):
        if from_index == 0:
            if "n_head_kv" not in config:  # MQA, 7b structure
                cls.config_class = ConfigConverter_Falcon_7B_HF_CS19
            else:  # GQA, 40B structure
                cls.config_class = ConfigConverter_Falcon_40B_HF_CS20
        else:
            model_config = config["model"]
            group_num = model_config["extra_attention_params"]["num_kv_groups"]
            if group_num == 1:  # mqa, 7b
                cls.config_class = ConfigConverter_Falcon_7B_HF_CS19
            elif group_num > 1:  # gqa, 40b
                cls.config_class = ConfigConverter_Falcon_40B_HF_CS20
            else:
                raise ValueError(f"num_kv_groups is {group_num}")

        instance = cls.config_class()
        return instance.convert_helper(
            config,
            from_index,
            drop_unmatched_keys=drop_unmatched_keys,
            no_progress_bar=no_progress_bar,
            debug=debug,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9", "cs-2.0"))
