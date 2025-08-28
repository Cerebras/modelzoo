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
from typing import Tuple

import torch

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.bert import (
    ConfigConverter_Bert_HF_CS18,
    Converter_BertLayerNorm_HF_CS,
    Converter_BertModel_CS16_CS17,
    Converter_BertModel_WithoutOptionalModel_HF_CS21,
    Converter_BertPretrainModel_HF_CS18,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)


class Converter_RobertaPretrainModel_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("roberta.", "bert_encoder."),
                    Converter_BertModel_CS16_CS17(),  # CS16 = HF
                ],
            ),
            # CLS:
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head.dense",
                        "bert_mlm_head.mlm_transform.ffn.ffn.0.linear_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head.",
                        "bert_mlm_head.mlm_transform.",
                    ),
                    Converter_BertLayerNorm_HF_CS("layer_norm", "ln"),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head.decoder",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head.decoder",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    r"\.bias",
                ],
                action=self.convert_cls_predictions_bias,
            ),
            ConversionRule([r"lm_head\.bias"], exists="left"),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None

    def convert_cls_predictions_bias(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        self.replaceKey(
            old_key,
            new_key,
            old_state_dict,
            new_state_dict,
            from_index,
            action_fn_args,
        )
        if from_index == 1:
            # HF stores an extra copy of the decoder bias in the predictions object itself
            bias_key = re.sub(r"\.decoder\.", ".", new_key)
            self.replaceKey(
                old_key,
                bias_key,
                old_state_dict,
                new_state_dict,
                from_index,
                action_fn_args,
            )


class Converter_RobertaPretrainModel_HF_CS18(
    Converter_BertPretrainModel_HF_CS18
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_RobertaPretrainModel_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_RobertaPretrainModel_HF_CS(),
                ],
                action=None,
            ),
        ]

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for RobertaForPreTraining".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Roberta_HF_CS18

    def post_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
        key_prefix="",
    ):
        if converter_indices.direction == 1:
            num_segments = configs[1]["model"]["num_segments"]
            if not num_segments:
                new_state_dict[
                    key_prefix
                    + "roberta.embeddings.token_type_embeddings.weight"
                ] = torch.zeros(
                    configs[0]["type_vocab_size"], configs[0]["hidden_size"]
                )
        else:
            # HF -> CS
            # sometimes HF checkpoints are missing the MLM classifier bias
            # which we need to manually initialize
            if (
                "bert_mlm_head.classifier.ffn.0.linear_layer.bias"
                not in new_state_dict
            ):
                new_state_dict[
                    "bert_mlm_head.classifier.ffn.0.linear_layer.bias"
                ] = torch.zeros(configs[0]["vocab_size"])
        super().post_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            converter_indices,
            drop_unmatched_keys,
            key_prefix=key_prefix,
        )


class ConfigConverter_Roberta_HF_CS18(ConfigConverter_Bert_HF_CS18):
    def __init__(self):
        super().__init__()
        # Override Bert's config converter with the following:

        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "roberta"),
            ),
            ConversionRule(
                ["max_position_embeddings"],
                action=self.convert_max_pos_embed,
            ),
            ConversionRule(
                [EquivalentSubkey("type_vocab_size", "num_segments")],
                action=self.convert_num_segments,
            ),
            ConversionRule(
                ["pad_token_id"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["mask_padding_in_positional_embed"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["disable_nsp"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["mlm_nonlinearity"],
                action=BaseConfigConverter.assert_factory_fn(1, "gelu"),
            ),
            *self.rules,
        ]

        self.pre_convert_defaults[0].update(
            {
                "vocab_size": 50265,
                "position_embedding_type": "absolute",
                "type_vocab_size": 2,
                "pad_token_id": 1,
            }
        )

        self.pre_convert_defaults[1].update(
            {
                "disable_nsp": False,
                "pad_token_id": 0,
                "mask_padding_in_positional_embed": False,
            }
        )

        self.post_convert_defaults[0].update({"model_type": "roberta"})
        self.post_convert_defaults[1].update(
            {
                "disable_nsp": True,
                "mask_padding_in_positional_embed": True,
            }
        )

    def convert_num_segments(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # CS allows segment embeddings to be disabled while HF doesn't
        # When it is disabled in CS, we need to enable it in HF and set the
        # embedding weight to zero
        if from_index == 1 and old_state_dict[old_key] == 0:
            new_state_dict[new_key] = 1
        else:
            new_state_dict[new_key] = old_state_dict[old_key]

    def convert_max_pos_embed(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # The number of positional embeddings = MSL + pad token offset + 1
        # HF refers to number of positional embeddings (the total) as
        # max_position_embeddings while we refer to MSL as
        # max_position_embeddings
        if from_index == 0:
            new_state_dict[new_key] = (
                old_state_dict[old_key] - old_state_dict["pad_token_id"] - 1
            )
        else:
            new_state_dict[new_key] = (
                old_state_dict[old_key] + old_state_dict["pad_token_id"] + 1
            )

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        if converter_indices.direction == 1:
            if "num_segments" not in config:
                config["num_segments"] = 0 if config["disable_nsp"] else 2

        return config

    def post_config_convert(
        self,
        model,
        original_config,
        old_config,
        new_config,
        converter_indices,
        drop_unmatched_keys,
    ):
        if converter_indices.direction == 0:
            new_config["mlm_nonlinearity"] = "gelu"

        return super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1, and HF <> CS 2.1 converters:
###########################################################


class ConfigConverter_Roberta_HF_CS21(ConfigConverter_Roberta_HF_CS18):
    "CS 2.1 config is the same as CS 2.0."

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_RobertaPretrainModel_WithoutOptionalModel_HF_CS21(
    Converter_RobertaPretrainModel_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Roberta HF implementation does not have an NSP head. It also
            # shouldn't have a pooler head. However, some open source
            # checkpoints have these weights which need to be ignored.
            ConversionRule(
                [r"roberta.pooler.dense\.(?:weight|bias)"],
                exists="left",
                action=None,
            ),
            # Proceed with conversion rules as normal:
            ConversionRule(
                [
                    EquivalentSubkey("roberta.", "bert_encoder."),
                    Converter_BertModel_WithoutOptionalModel_HF_CS21(),  # CS16 = HF
                ],
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Roberta_HF_CS21


Converter_RobertaPretrainModel_HF_CS21 = (
    Build_HF_CS_Converter_WithOptionalModel(
        "Converter_RobertaPretrainModel_HF_CS21",
        Converter_RobertaPretrainModel_WithoutOptionalModel_HF_CS21,
        derived_class=Converter_RobertaPretrainModel_HF_CS18,
        config_converter_class=ConfigConverter_Roberta_HF_CS21,
        formats=(
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        ),
    )
)
