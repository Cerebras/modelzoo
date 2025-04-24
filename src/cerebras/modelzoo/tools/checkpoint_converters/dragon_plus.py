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

import re
from typing import List, Tuple, Type

import torch

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter,
    BaseCheckpointConverter_HF_CS,
    BaseCheckpointConverter_UnpackedHF_PackedCS,
    BaseConfigConverter,
    BaseConfigConverter_UnpackedHF_PackedCS,
    ConversionRule,
    EquivalentSubkey,
    FormatIndices,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.bert import (
    ConfigConverter_Bert_HF_CS21,
    Converter_BertModel_WithoutOptionalModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)

Dragon_CONFIG_ERROR_MESSAGE = """
DragonConverter assumes that the input file will
be a directory that contains two sub-directories: ctx_encoder
and q_encoder. It further expects files named config.json
from each of these sub-directories.
"""


class Converter_Dragon_BertWrapper(
    Converter_BertModel_WithoutOptionalModel_HF_CS21
):
    def __init__(self, encoder_params_key):
        super().__init__()
        self.encoder_params_key = encoder_params_key

    # In BERT converter this expects the max_position_embeddings at a different
    # level in the config, so overwriting to allow the nested Dragon structure
    def position_embeddings_convert(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        self.replaceKey(
            old_key, new_key, old_state_dict, new_state_dict, from_index
        )
        if from_index == 1:
            # HF stores an register buffer with position_ids
            position_id_key = re.sub(
                "\.position_embeddings\.weight", ".position_ids", new_key
            )
            if "max_position_embeddings" in action_fn_args["configs"][0]:
                max_position_embeddings = action_fn_args["configs"][0][
                    "max_position_embeddings"
                ]
            else:
                max_position_embeddings = action_fn_args["configs"][1]["model"][
                    self.encoder_params_key
                ]["max_position_embeddings"]
            new_state_dict[position_id_key] = torch.arange(
                max_position_embeddings
            ).expand((1, -1))

    def convert_pooler_factory_fn(self):
        """
        Dragon checkpoints have pooler weights, but these are thrown away in the
        HF model code. Therefore we have to explicitly catch these weights but
        we return None to get rid of them.
        """
        return None

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DragonModel_HF_CS23


class Converter_DragonQuestionEncoder_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [r"pooler\.dense\.*"],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("", "question_encoder."),
                    Converter_Dragon_BertWrapper("question_encoder"),
                ],
                action=None,
            ),
            ConversionRule(
                ["ctx_encoder.*"],
                action=None,
            ),
        ]

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        if converter_indices.direction == 0:
            pass

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "Used within broader Dragon converter, not meant for own use"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DragonModel_HF_CS23


class Converter_DragonContextEncoder_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [r"pooler\.dense\.*"],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("", "ctx_encoder."),
                    Converter_Dragon_BertWrapper("context_encoder"),
                ],
                action=None,
            ),
            ConversionRule(["question_encoder.*"], action=None),
        ]

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # Normally this does output_checkpoint["model"] = {} and then we
        # reference output_checkpoint["model"] later in extract_model_dict.
        # We don't want to reset the output_checkpoint["model"] here though
        # because we will store the keys under the same "model" key
        # created by this function during the question-encoder conversion
        if converter_indices == 0:
            pass

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "Used within broader Dragon converter, not meant for own use"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DragonModel_HF_CS23


Converter_DragonQuestionEncoderWithOptionalModelPrefix_HF_CS = (
    Build_HF_CS_Converter_WithOptionalModel(
        "Converter_DragonQuestionEncoder_HF_CS",
        Converter_DragonQuestionEncoder_HF_CS,
        Converter_DragonQuestionEncoder_HF_CS,
    )
)

Converter_Converter_DragonContextEncoderWithOptionalModelPrefix_HF_CS = (
    Build_HF_CS_Converter_WithOptionalModel(
        "Converter_DragonQuestionEncoder_HF_CS",
        Converter_DragonContextEncoder_HF_CS,
        Converter_DragonContextEncoder_HF_CS,
    )
)


class Converter_DragonModel_HF_CS23(
    BaseCheckpointConverter_UnpackedHF_PackedCS
):
    def __init__(self):
        super().__init__()
        # rules are empty because sub-converters are used in the convert fn
        # but tests require presence of self.rules
        self.rules = []

    @staticmethod
    def converters() -> List[Type[BaseCheckpointConverter]]:
        return (
            Converter_DragonQuestionEncoderWithOptionalModelPrefix_HF_CS,
            Converter_Converter_DragonContextEncoderWithOptionalModelPrefix_HF_CS,
        )

    @staticmethod
    def component_names() -> List[str]:
        return ("q_encoder", "ctx_encoder")

    @staticmethod
    def architectures() -> Tuple[List[str], str]:
        return (
            ("BertForMaskedLM", "BertForMaskedLM"),
            "DPRModel",
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DragonModel_HF_CS23


class ConfigConverter_Dragon_HF_CS(ConfigConverter_Bert_HF_CS21):
    def __init__(self):
        self.model_type = "bert"
        super().__init__()
        self.rules.append(
            ConversionRule(
                [EquivalentSubkey("projection_dim", "add_pooling_layer")],
                action=self.convert_pooler_config,
            ),
        )

        self.post_convert_defaults[0].update(
            {
                "architectures": ["BertModel"],
            }
        )

    def convert_pooler_config(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        """
        In Dragon configs, projection_dim will be 0 if there is no pooler, and
        otherwise sets the dimension for the FFN of the pooler.
        Note: in Dragon the hidden size of pooler can be set to an arbitrary
        value, whereas our implementation only supports poolers with hidden
        size matching the model hidden size.
        """
        if from_index == 0:
            # if we have a pooler dimension in the HF config, it has to match
            # the hidden size
            if old_state_dict[old_key]:
                assert (
                    old_state_dict[old_key] == new_state_dict["hidden_size"]
                ), """
                CS pooler implementation only supports pooler dimension that
                matches the hidden size of the rest of the model
                """
            new_state_dict[new_key] = bool(old_state_dict[old_key])
        else:
            # get the hidden dimension
            new_state_dict[new_key] = (
                old_state_dict["hidden_size"] if old_state_dict[old_key] else 0
            )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


class ConfigConverter_DragonModel_HF_CS23(
    BaseConfigConverter_UnpackedHF_PackedCS
):
    def __init__(self):
        super().__init__()
        # rules are empty because sub-converters are used in the convert fn
        # but tests require presence of self.rules
        self.rules = []
        self.post_convert_defaults[1].update(
            {
                "pooler_type": "cls",
                "use_biencoder": True,
                "mutual_information": False,
                "softmax_temperature": 1,
                "compute_eval_metrics": False,
            }
        )

    @staticmethod
    def converters() -> List[Type[BaseCheckpointConverter]]:
        return (ConfigConverter_Dragon_HF_CS, ConfigConverter_Dragon_HF_CS)

    @staticmethod
    def component_names() -> List[str]:
        return ("q_encoder", "ctx_encoder")

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )
