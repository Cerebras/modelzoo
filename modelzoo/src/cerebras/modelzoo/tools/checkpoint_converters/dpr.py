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
import math
import re
from typing import List, Tuple, Type

import torch

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter,
    BaseCheckpointConverter_CS_CS,
    BaseCheckpointConverter_HF_CS,
    BaseCheckpointConverter_UnpackedHF_PackedCS,
    BaseConfigConverter,
    BaseConfigConverter_CS_CS,
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

DPR_CONFIG_ERROR_MESSAGE = """
    DPRConverter assumes that the input file will
    be a directory that contains two sub-directories: ctx_encoder
    and q_encoder. It further expects files named config.json
    from each of these sub-directories.
    """


class Converter_DPR_BertWrapper(
    Converter_BertModel_WithoutOptionalModel_HF_CS21
):
    def __init__(self, encoder_params_key):
        super().__init__()
        self.encoder_params_key = encoder_params_key

    # In BERT converter this expects the max_position_embeddings at a different
    # level in the config, so overwriting to allow the nested DPR structure
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
        DPR checkpoints have pooler weights, but these are thrown away in the
        HF model code. Therefore we have to explicitly catch these weights but
        we return None to get rid of them.
        """
        return None

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DPRModel_HF_CS22


class Converter_DPRQuestionEncoder_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    r"question_encoder\.",
                    EquivalentSubkey("bert_model.", ""),
                    Converter_DPR_BertWrapper("question_encoder"),
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
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "Used within broader DPR converter, not meant for own use"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DPRModel_HF_CS22


class Converter_DPRContextEncoder_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    r"ctx_encoder\.",
                    EquivalentSubkey("bert_model.", ""),
                    Converter_DPR_BertWrapper("context_encoder"),
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
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "Used within broader DPR converter, not meant for own use"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DPRModel_HF_CS22


Converter_DPRQuestionEncoderWithOptionalModelPrefix_HF_CS = (
    Build_HF_CS_Converter_WithOptionalModel(
        "Converter_DPRQuestionEncoder_HF_CS",
        Converter_DPRQuestionEncoder_HF_CS,
        Converter_DPRQuestionEncoder_HF_CS,
    )
)

Converter_DPRContextEncoderWithOptionalModelPrefix_HF_CS = (
    Build_HF_CS_Converter_WithOptionalModel(
        "Converter_DPRContextEncoder_HF_CS",
        Converter_DPRContextEncoder_HF_CS,
        Converter_DPRContextEncoder_HF_CS,
    )
)


class Converter_DPRModel_HF_CS22(BaseCheckpointConverter_UnpackedHF_PackedCS):
    def __init__(self):
        super().__init__()
        # rules are empty because sub-converters are used in the convert fn
        # but tests require presence of self.rules
        self.rules = []

    @staticmethod
    def converters() -> List[Type[BaseCheckpointConverter]]:
        return (
            Converter_DPRQuestionEncoderWithOptionalModelPrefix_HF_CS,
            Converter_DPRContextEncoderWithOptionalModelPrefix_HF_CS,
        )

    @staticmethod
    def component_names() -> List[str]:
        return ("q_encoder", "ctx_encoder")

    @staticmethod
    def architectures() -> Tuple[List[str], str]:
        return (("DPRQuestionEncoder", "DPRQuestionEncoder"), "DPRModel")

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DPRModel_HF_CS22


class ConfigConverter_DPR_HF_CS(ConfigConverter_Bert_HF_CS21):
    def __init__(self):
        self.model_type = "dpr"
        super().__init__()
        self.rules.append(
            ConversionRule(
                [EquivalentSubkey("projection_dim", "add_pooling_layer")],
                action=self.convert_pooler_config,
            ),
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
        In DPR configs, projection_dim will be 0 if there is no pooler, and
        otherwise sets the dimension for the FFN of the pooler.
        Note: in DPR the hidden size of pooler can be set to an arbitrary
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
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class ConfigConverterEncoder_HF_CS(ConfigConverter_DPR_HF_CS):
    def post_config_convert(
        self,
        model,
        original_config,
        old_config,
        new_config,
        converter_indices,
        drop_unmatched_keys,
    ):
        return super().post_config_convert(
            "bert_model",
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )


class ConfigConverter_DPRModel_HF_CS22(BaseConfigConverter_UnpackedHF_PackedCS):
    def __init__(self):
        super().__init__()
        # rules are empty because sub-converters are used in the convert fn
        # but tests require presence of self.rules
        self.rules = []
        self.post_convert_defaults[1].update(
            {
                "scale_similarity": False,
            }
        )

    def post_config_convert(
        self,
        model,
        original_config,
        old_config,
        new_config,
        converter_indices,
        drop_unmatched_keys,
    ):
        from_index = converter_indices.direction
        model_config = super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )
        if from_index == 1:
            model_config[0].update(
                {
                    "_name_or_path": "facebook/dpr-question_encoder-single-nq-base",
                    "architectures": ["DPRQuestionEncoder"],
                    "model_type": "dpr",
                }
            )
            model_config[1].update(
                {
                    "_name_or_path": "facebook/dpr-ctx_encoder-single-nq-base",
                    "architectures": ["DPRContextEncoder"],
                    "model_type": "dpr",
                }
            )
        return model_config

    @staticmethod
    def converters() -> List[Type[BaseCheckpointConverter]]:
        return (ConfigConverterEncoder_HF_CS, ConfigConverterEncoder_HF_CS)

    @staticmethod
    def component_names() -> List[str]:
        return ("q_encoder", "ctx_encoder")

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class ConfigConverter_DPR_HF_CS23(ConfigConverterEncoder_HF_CS):
    def __init__(self):
        self.model_type = "dpr"
        super().__init__()

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


class ConfigConverter_DPRModel_HF_CS23(ConfigConverter_DPRModel_HF_CS22):
    def __init__(self):
        super().__init__()
        self.post_convert_defaults[1].pop("scale_similarity")
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
        return (ConfigConverter_DPR_HF_CS23, ConfigConverter_DPR_HF_CS23)

    @staticmethod
    def component_names() -> List[str]:
        return ("q_encoder", "ctx_encoder")

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_DPRModel_HF_CS23(Converter_DPRModel_HF_CS22):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DPRModel_HF_CS23


class Converter_DPRModel_CS22_CS23(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()

        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-2.2"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_DPRModel_CS22_CS23


class ConfigConverter_DPRModel_CS22_CS23(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [EquivalentSubkey("scale_similarity", "softmax_temperature")],
                action=self.convert_softmax_temperature,
            ),
            ConversionRule([".*"], action=self.replaceKey),
        ]

        self.post_convert_defaults[1].update(
            {
                "pooler_type": "cls",
                "use_biencoder": True,
                "mutual_information": False,
                "softmax_temperature": 1,
                "compute_eval_metrics": False,
            }
        )

        self.post_convert_defaults[0].update(
            {
                "scale_similarity": False,
            }
        )

    def convert_softmax_temperature(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            scale_similarity = old_state_dict[old_key]
            hidden_size = old_state_dict["q_encoder"]["hidden_size"]

            if scale_similarity is True:
                new_state_dict[new_key] = math.sqrt(hidden_size)
            else:
                new_state_dict[new_key] = 1.0
        else:
            softmax_temperature = old_state_dict[old_key]
            hidden_size = old_state_dict["q_encoder"]["hidden_size"]

            if math.isclose(
                softmax_temperature,
                math.sqrt(hidden_size),
                abs_tol=1.0,
            ):
                new_state_dict[new_key] = True
            elif softmax_temperature == 1.0:
                new_state_dict[new_key] = False
            else:
                logging.warning(
                    "When converting from 2.3 to 2.2, we do not support "
                    "arbitrary `softmax_temperature` value. "
                    "We only support `softmax_temperature` that are either close to "
                    "sqrt(hidden size) or 1.0. "
                )
                new_state_dict[new_key] = False

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-2.2"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )
