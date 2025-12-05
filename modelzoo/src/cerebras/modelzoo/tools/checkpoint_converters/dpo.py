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
import re
from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_CS_CS,
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    BaseConfigConverter_CS_CS,
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatIndices,
    FormatVersions,
)

SUPPORTED_DPO_MODELS = {
    "bloom",
    "falcon",
    "gpt2",
    "gpt3",
    "gptj",
    "gpt-neox",
    "lambda",
    "llama",
    "mistral",
    "mpt",
    "opt",
    "palm",
    "santacoder",
    "starcoder",
}


def get_dpoless_formats(cls, from_index):
    formats = cls.formats()
    formats = [
        f[-1].replace("-no-dpo", "").replace("-dpo", "") for f in formats
    ]
    if from_index == 1:
        formats.reverse()
    return formats


class Converter_DPO_HF_CS21(BaseCheckpointConverter_HF_CS):
    def __init__(self) -> None:
        super().__init__()
        self.rules = []

    @classmethod
    def convert(cls, checkpoint, configs, converter_indices, **kwargs):
        model_name = configs[1]["model"]["model_name"]
        formats = get_dpoless_formats(cls, converter_indices.direction)
        model_converter_class, model_converter_indices = (
            cls.make_dpo_model_converter(model_name, *formats, configs)
        )
        instance = model_converter_class()
        new_checkpoint = instance.convert_helper(
            checkpoint, configs, model_converter_indices, **kwargs
        )
        return new_checkpoint

    @classmethod
    def make_dpo_model_converter(cls, model_name, src_fmt, tgt_fmt, configs):
        # Deferred import to break circular dependency:
        from cerebras.modelzoo.tools.convert_checkpoint import (
            _select_model_and_config_converter,
        )

        (
            converter_class,
            checkpoint_from_index,
            config_converter_class,
            config_from_index,
        ) = _select_model_and_config_converter(model_name, src_fmt, tgt_fmt)
        assert converter_class is not None
        if hasattr(converter_class, "select_subconverter"):
            converter_class = converter_class.select_subconverter(
                configs[config_from_index.direction],
                checkpoint_from_index.direction,
            )

        class DPO_Converter(converter_class):
            def __init__(self):
                super().__init__()
                self.rules = [
                    ConversionRule(
                        [
                            EquivalentSubkey("", "policy_model.model."),
                            converter_class(),
                        ],
                        action=None,
                    ),
                    # Throw away reference model keys when converting to HF:
                    ConversionRule([r"ref_model\.model\..*"], action=None),
                ]

            def post_model_convert(
                self,
                old_state_dict,
                new_state_dict,
                configs,
                converter_indices,
                drop_unmatched_keys,
                key_prefix="",
            ):
                # Finalize checkpoint:
                super().post_model_convert(
                    old_state_dict,
                    new_state_dict,
                    configs,
                    converter_indices,
                    drop_unmatched_keys,
                    key_prefix=key_prefix + "policy_model.model.",
                )
                # In the HF -> CS direction, we need to create the ref_model
                # keys:
                if converter_indices.direction == 0:
                    policy_model_keys = list(new_state_dict.keys())
                    for policy_key in policy_model_keys:
                        ref_key = re.sub(
                            r"policy_model\.", "ref_model.", policy_key
                        )
                        new_state_dict[ref_key] = new_state_dict[policy_key]

        return DPO_Converter, checkpoint_from_index

    @classmethod
    def converter_note(cls) -> str:
        formats = get_dpoless_formats(cls, 0)
        return (
            f"{formats[0]} (Non-DPO) model <-> {formats[1]} DPO "
            f"model. The type of model that is trained via DPO is specified in "
            f"the config using the 'model_name' property. The following are "
            f"supported: {SUPPORTED_DPO_MODELS}. These are the same names "
            f"as those used in the checkpoint converter's --model argument."
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions(
                "cs-2.1-dpo",
                "cs-2.2-dpo",
                "cs-2.3-dpo",
                "cs-2.4-dpo",
                "cs-2.5-dpo",
            ),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DPO_HF_CS21


class ConfigConverter_DPO_HF_CS21(BaseConfigConverter_HF_CS):
    def __init__(self) -> None:
        super().__init__()
        self.rules = []

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions(
                "cs-2.1-dpo",
                "cs-2.2-dpo",
                "cs-2.3-dpo",
                "cs-2.4-dpo",
                "cs-2.5-dpo",
            ),
        )

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
        if converter_indices.direction == 0:
            if "model_type" not in config:
                raise ConfigConversionError(
                    "Cannot convert model into DPO if HF config doesn't have "
                    "model_type property"
                )

            # HF & CS model type/names may not align. The following represents
            # the HF -> CS mapping
            remap_modelnames = {"gpt_neox": "gpt-neox"}
            model_name = config["model_type"]
            if model_name in remap_modelnames:
                model_name = remap_modelnames[model_name]

            if model_name not in SUPPORTED_DPO_MODELS:
                raise ConfigConversionError(
                    f"DPO doesn't support model_type={config['model_type']}. "
                    f"The following are supported: {SUPPORTED_DPO_MODELS}"
                )

            logging.warning(
                f"Converting a non-DPO {model_name} HF checkpoint into a CS DPO"
                f" checkpoint"
            )
        else:
            if "model_name" not in config["model"]:
                raise ConfigConversionError(
                    "Cannot convert DPO model if CS config doesn't have "
                    "model_name property"
                )
            model_name = config["model"]["model_name"]
            logging.warning(
                f"Converting a CS DPO {model_name} checkpoint into a non-DPO HF"
                f" checkpoint"
            )

        formats = get_dpoless_formats(cls, converter_indices.direction)
        config_converter_class, model_converter_indices = (
            cls.make_dpo_config_converter(
                model_name,
                *formats,
                config,
            )
        )
        instance = config_converter_class()
        return instance.convert_helper(
            model,
            config,
            model_converter_indices,
            drop_unmatched_keys=drop_unmatched_keys,
            no_progress_bar=no_progress_bar,
            debug=debug,
        )

    @classmethod
    def make_dpo_config_converter(cls, model_name, src_fmt, tgt_fmt, config):
        # Deferred import to break circular dependency:
        from cerebras.modelzoo.tools.convert_checkpoint import (
            _select_model_and_config_converter,
        )

        (
            converter_class,
            checkpoint_from_index,
            config_converter_class,
            config_from_index,
        ) = _select_model_and_config_converter(model_name, src_fmt, tgt_fmt)
        assert config_converter_class is not None
        if hasattr(config_converter_class, "select_subconverter"):
            config_converter_class = config_converter_class.select_subconverter(
                config, config_from_index.direction
            )

        class DPO_Config_Converter(config_converter_class):
            def __init__(self):
                super().__init__()
                self.rules = [
                    ConversionRule(
                        ["model_name"],
                        action=BaseConfigConverter.assert_factory_fn(
                            1, model_name
                        ),
                    ),
                    ConversionRule(
                        ["dpo"],
                        action=None,
                    ),
                    *self.rules,
                ]

                self.post_convert_defaults[1].update(
                    {
                        "model_name": model_name,
                        "dpo": {"beta": 0.1, "reference_free": False},
                    }
                )

        return DPO_Config_Converter, checkpoint_from_index


class Converter_NON_DPO_TO_DPO_CS21(BaseCheckpointConverter_CS_CS):
    def __init__(self) -> None:
        super().__init__()
        self.rules = [
            ConversionRule(
                [EquivalentSubkey("", "policy_model.model."), r".*"],
                action=self.replaceKey,
            ),
            # Throw away reference model keys when converting to Non-DPO:
            ConversionRule([r"ref_model\.model\..*"], action=None),
        ]

    def post_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
    ):
        # Finalize checkpoint:
        super().post_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            converter_indices,
            drop_unmatched_keys,
        )
        # In the Non-DPO -> DPO direction, we need to create the ref_model
        # keys:
        if converter_indices.direction == 0:
            policy_model_keys = list(new_state_dict.keys())
            for policy_key in policy_model_keys:
                ref_key = re.sub(r"policy_model\.", "ref_model.", policy_key)
                new_state_dict[ref_key] = new_state_dict[policy_key]

    @classmethod
    def converter_note(cls) -> str:
        formats = get_dpoless_formats(cls, 0)
        return (
            f"{formats[0]} (Non-DPO) model <-> {formats[1]} DPO "
            f"model. The type of model that is trained via DPO is specified in "
            f"the config using the 'model_name' property. The following are "
            f"supported: {SUPPORTED_DPO_MODELS}. These are the same names"
            f"as those used in the checkpoint converter's --model argument."
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.1-no-dpo"), FormatVersions("cs-2.1-dpo"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_NON_DPO_TO_DPO_CS21


class ConfigConverter_NON_DPO_TO_DPO_CS21(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Drop dpo field in DPO -> Non-DPO
            ConversionRule(
                [r"dpo"],
                action=None,
            ),
            # Keep everything else
            ConversionRule(
                [r".*"],
                action=self.replaceKey,
            ),
        ]
        self.post_convert_defaults[1].update(
            {"dpo": {"beta": 0.1, "reference_free": False}}
        )

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        if converter_indices.direction == 0:
            if "model_name" not in config:
                raise ConfigConversionError(
                    "Converting a Non-DPO config to DPO relies on the "
                    "'model_name' property. Please add 'model_name' to the "
                    "config under the model parameters before running "
                    "conversion. The following are supported: "
                    f"{SUPPORTED_DPO_MODELS}"
                )
            elif config["model_name"] not in SUPPORTED_DPO_MODELS:
                raise ConfigConversionError(
                    f"DPO doesn't support model_name={config['model_name']}. "
                    f"The following are supported: {SUPPORTED_DPO_MODELS}"
                )
        else:
            if "model_name" not in config:
                raise ConfigConversionError(
                    "The supplied config is not a valid DPO config: it is "
                    "missing the 'model_name' property in the model parameters."
                )
            elif "dpo" not in config:
                raise ConfigConversionError(
                    "The supplied config is not a valid DPO config: it is "
                    "missing the 'dpo' property in the model parameters."
                )

        return config

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.1-no-dpo"), FormatVersions("cs-2.1-dpo"))
