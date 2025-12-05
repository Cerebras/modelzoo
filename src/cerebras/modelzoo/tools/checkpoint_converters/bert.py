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

import torch

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
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
    maybe_tie_lm_head,
)


class Converter_BertLayerNorm_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self, hf_name, cs_name):
        super().__init__()
        self.rules = [
            # torch.nn.LayerNorm has .weight & .bias properties
            ConversionRule(
                [
                    EquivalentSubkey(hf_name, cs_name),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # Old HF implementation uses .gamma instead of .weight
            ConversionRule(
                [
                    EquivalentSubkey(hf_name, cs_name),
                    EquivalentSubkey(".gamma", ".weight"),
                ],
                action=self.replaceKey,
            ),
            # Old HF implementation uses .beta instead of .bias
            ConversionRule(
                [
                    EquivalentSubkey(hf_name, cs_name),
                    EquivalentSubkey(".beta", ".bias"),
                ],
                action=self.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None


class Converter_BertModel_CS16_CS17(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Embedding:
            ConversionRule(
                [
                    EquivalentSubkey("embeddings", "embedding_layer"),
                    r"\.word_embeddings\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("embeddings", "embedding_layer"),
                    r"\.position_embeddings\.weight",
                ],
                action=self.position_embeddings_convert,
            ),
            ConversionRule(
                [
                    r"embedding_layer\.position_embeddings",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.token_type_embeddings",
                        "embedding_layer.segment_embeddings",
                    ),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"embeddings\.position_ids",
                ],
                exists="left",
            ),
            ConversionRule(
                [
                    EquivalentSubkey("embeddings.", ""),
                    Converter_BertLayerNorm_HF_CS("LayerNorm", "embed_ln_f"),
                ],
                action=None,
            ),
            # Encoder Layers:
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.self.query", "self_attn.proj_q_dense_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.self.key", "self_attn.proj_k_dense_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.self.value", "self_attn.proj_v_dense_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.output.dense",
                        "self_attn.proj_output_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("attention.output.", ""),
                    Converter_BertLayerNorm_HF_CS("LayerNorm", "norm1"),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "intermediate.dense", "ffn.ffn.0.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("output.dense", "ffn.ffn.1.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("output.", ""),
                    Converter_BertLayerNorm_HF_CS("LayerNorm", "norm2"),
                ],
                action=None,
            ),
            # Head:
            ConversionRule(
                [
                    r"pooler\.",
                    EquivalentSubkey("dense", "pooler.ffn.0.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_pooler_factory_fn(),
            ),
        ]

    def convert_pooler_factory_fn(self):
        """
        DPR, which uses two BERT sub-converters, requires different
        behavior of the pooler conversion, so we generalize to allow
        overriding.
        """

        def bert_pooler_convert(
            old_key,
            new_key,
            old_state_dict,
            new_state_dict,
            from_index,
            action_fn_args,
        ):
            return self.replaceKey(
                old_key,
                new_key,
                old_state_dict,
                new_state_dict,
                from_index,
                action_fn_args,
            )

        return bert_pooler_convert

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
                r"\.position_embeddings\.weight", ".position_ids", new_key
            )
            if "max_position_embeddings" in action_fn_args["configs"][0]:
                max_position_embeddings = action_fn_args["configs"][0][
                    "max_position_embeddings"
                ]
            else:
                max_position_embeddings = action_fn_args["configs"][1]["model"][
                    "max_position_embeddings"
                ]
            new_state_dict[position_id_key] = torch.arange(
                max_position_embeddings
            ).expand((1, -1))

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.7"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS17


class ConfigConverter_Bert_CS16_CS17(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Config didn't change between 1.6 and 1.7. Copy all keys.
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.7"))


class Converter_BertModel_CS16_CS18(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_BertModel_CS16_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertModel_CS16_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-1.6"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS18


class ConfigConverter_Bert_CS16_CS18(ConfigConverter_Bert_CS16_CS17):
    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)
        if converter_indices.direction == 1:
            if (
                "pooler_nonlinearity" in config
                and config["pooler_nonlinearity"]
                != config["encoder_nonlinearity"]
            ):
                raise ConfigConversionError(
                    "pooler_nonlinearity was introduced in CS 1.8. Prior to that, the pooler "
                    "nonlinearity must be the same as encoder_nonlinearity."
                )
            if "mlm_nonlinearity" in config:
                if config["mlm_nonlinearity"] != "gelu":
                    raise ConfigConversionError(
                        "mlm_nonlinearity was introduced in CS 1.8. Prior to that, the mlm "
                        "nonlinearity must be gelu."
                    )
            else:
                if config["encoder_nonlinearity"] != "gelu":
                    raise ConfigConversionError(
                        f"mlm_nonlinearity was introduced in CS 1.8. Prior to that, the mlm "
                        f"nonlinearity must be gelu. However, the input config has an "
                        f"mlm_nonlinearity which defaults to encoder_nonlinearity = "
                        f"{config['encoder_nonlinearity']}"
                    )
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
            new_config["pooler_nonlinearity"] = new_config[
                "encoder_nonlinearity"
            ]
            new_config["mlm_nonlinearity"] = "gelu"

        return super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-1.6"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )


class Converter_Bert_CS17_CS18(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Checkpoint didn't change between 1.7 and 1.8. Copy all keys.
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-1.7"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return (
            "BertForPreTraining, BertForSequenceClassification, "
            "BertForQuestionAnswering, and BertForSummarization classes"
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS17_CS18


# Config didn't change between 1.6 and 1.7. Therefore 1.7 <-> 1.8
# converter is equivalent to 1.6 <-> 1.8 converter.
class ConfigConverter_Bert_CS17_CS18(ConfigConverter_Bert_CS16_CS18):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-1.7"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )


class Converter_BertModel_HF_CS17(
    Converter_BertModel_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
    ):
        # Manually tie weights
        if (
            converter_indices.direction == 1
            and configs[1]["model"]["share_embedding_weights"]
        ):
            if (
                old_state_dict.get(
                    "bert_encoder.embedding_layer.word_embeddings.weight", 0
                )
                is None
            ):
                old_state_dict[
                    "bert_encoder.embedding_layer.word_embeddings.weight"
                ] = old_state_dict[
                    "bert_mlm_head.classifier.ffn.0.linear_layer.weight"
                ]

    def pre_checkpoint_convert(
        self,
        *args,
    ):
        return BaseCheckpointConverter_HF_CS.pre_checkpoint_convert(
            self,
            *args,
        )

    def extract_model_dict(self, *args):
        return BaseCheckpointConverter_HF_CS.extract_model_dict(self, *args)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS17


class Converter_BertModel_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_BertModel_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [EquivalentSubkey("", "model."), Converter_BertModel_HF_CS17()],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS17


class Converter_BertPretrainModel_CS16_CS17(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("bert.", "bert_encoder."),
                    Converter_BertModel_CS16_CS17(),
                ],
            ),
            # CLS:
            ConversionRule(
                [
                    EquivalentSubkey(
                        "cls.predictions.transform.dense",
                        "bert_mlm_head.mlm_transform.ffn.ffn.0.linear_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "cls.predictions.transform.",
                        "bert_mlm_head.mlm_transform.",
                    ),
                    Converter_BertLayerNorm_HF_CS("LayerNorm", "ln"),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "cls.predictions.decoder",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "cls.predictions.decoder",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    r"\.bias",
                ],
                action=self.convert_cls_predictions_bias,
            ),
            ConversionRule([r"cls\.predictions\.bias"], exists="left"),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "cls.seq_relationship",
                        "bert_cls_head.classifier.ffn.0.linear_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

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

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # Don't copy non model keys like optimizer state:
        logging.warning(
            "The Bert model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        output_checkpoint["model"] = {}

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "BertPretrainModel class"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS17


class Converter_BertPretrainModel_CS16_CS18(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_BertPretrainModel_CS16_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertPretrainModel_CS16_CS17(),
                ],
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
        # Don't copy non model keys like optimizer state:
        logging.warning(
            "The Bert model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        output_checkpoint["model"] = {}

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-1.6"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "BertPretrainModel class"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS18


class Converter_BertPretrainModel_HF_CS17(
    Converter_BertPretrainModel_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
    ):
        # Manually tie weights
        old_state_dict = dict(old_state_dict)
        if converter_indices.direction == 1 and configs[1]["model"].get(
            "share_embedding_weights", False
        ):
            if (
                old_state_dict.get(
                    "bert_encoder.embedding_layer.word_embeddings.weight", 0
                )
                is None
            ):
                old_state_dict[
                    "bert_encoder.embedding_layer.word_embeddings.weight"
                ] = old_state_dict[
                    "bert_mlm_head.classifier.ffn.0.linear_layer.weight"
                ]

    def pre_checkpoint_convert(
        self,
        *args,
    ):
        return BaseCheckpointConverter_HF_CS.pre_checkpoint_convert(
            self,
            *args,
        )

    def extract_model_dict(self, *args):
        return BaseCheckpointConverter_HF_CS.extract_model_dict(self, *args)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for BertForPreTraining".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS17


class Converter_BertPretrainModel_HF_CS18(Converter_BertPretrainModel_HF_CS17):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_BertPretrainModel_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertPretrainModel_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} for BertForPreTraining".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS18


class ConfigConverter_Bert_HF_CS17(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        # allows DPR child class to set model_type without being
        # overriden in the super().init() call
        if not hasattr(self, "model_type"):
            self.model_type = "bert"
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(
                    0, self.model_type
                ),
            ),
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(
                ["position_embedding_type"],
                action=self.convert_position_embedding_type,
            ),
            ConversionRule(
                ["max_position_embeddings"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "tie_word_embeddings", "share_embedding_weights"
                    )
                ],
                action=self.replaceKey,
            ),
            # Decoder Block
            ConversionRule(
                ["hidden_size"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("num_attention_heads", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["num_hidden_layers"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("intermediate_size", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("hidden_act", "encoder_nonlinearity")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["mlm_nonlinearity"],
                action=self.assert_mlm_nonlinearity,
            ),
            ConversionRule(
                ["pooler_nonlinearity"],
                action=BaseConfigConverter.assert_factory_fn(1, "tanh"),
            ),
            ConversionRule(
                [EquivalentSubkey("hidden_dropout_prob", "dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_probs_dropout_prob", "attention_dropout_rate"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["disable_nsp"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["type_vocab_size"],
                action=BaseConfigConverter.assert_factory_fn(0, 2),
            ),
            ConversionRule(
                ["is_decoder"],
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(
                ["add_cross_attention"],
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(
                [EquivalentSubkey("layer_norm_eps", "layer_norm_epsilon")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["attention_type"],
                action=BaseConfigConverter.assert_factory_fn(
                    1, "scaled_dot_product"
                ),
            ),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_ffn_bias_in_attention"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_ffn_bias"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_ffn_bias_in_mlm"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_output_bias_in_mlm"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(["initializer_range"], action=self.replaceKey),
        ]

        self.pre_convert_defaults[0].update(
            {
                "vocab_size": 30522,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512,
                "layer_norm_eps": 1e-12,
                "tie_word_embeddings": True,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "share_embedding_weights": True,
                "encoder_nonlinearity": "gelu",
            },
        )

        self.post_convert_defaults[0].update({"model_type": "bert"})

    def convert_position_embedding_type(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # HF supports absolute, relative_key, relative_key_query
        # CS supports learned, fixed

        embed_type = old_state_dict[old_key]

        if from_index == 0:
            if embed_type == "absolute":
                new_state_dict[new_key] = "learned"
            else:
                raise ConfigConversionError(
                    "CS model doesn't support HF's position_embedding_type={}".format(
                        embed_type
                    )
                )
        else:
            if embed_type == "learned":
                new_state_dict[new_key] = "absolute"
            else:
                raise ConfigConversionError(
                    "HF model doesn't support CS's position_embedding_type={}".format(
                        embed_type
                    )
                )

    def assert_mlm_nonlinearity(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if (
            old_state_dict[old_key] != old_state_dict["encoder_nonlinearity"]
            and old_state_dict[old_key] is not None
        ):
            raise ConfigConversionError(
                "HF model doesn't support different encoder & mlm nonlinearities"
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
        if converter_indices.direction == 0:
            if (
                "mlm_nonlinearity" not in new_config
                and "encoder_nonlinearity" in new_config
                and new_config["encoder_nonlinearity"] != "gelu"
            ):
                logging.warning(
                    f"HF used a mlm_nonlinearity of {new_config['encoder_nonlinearity']} while "
                    f"CS 1.7 is fixed to gelu. Please use CS 1.8 if you want to control "
                    f"mlm_nonlinearity."
                )
                new_config["mlm_nonlinearity"] = "gelu"

        return super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


class ConfigConverter_Bert_HF_CS18(ConfigConverter_Bert_HF_CS17):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-1.8", "cs-1.9", "cs-2.0"),
        )

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)
        if converter_indices.direction == 1:
            if "pooler_nonlinearity" not in config:
                if config["encoder_nonlinearity"] != "tanh":
                    raise ConfigConversionError(
                        f"CS Model used a pooler_nonlinearity of {config['encoder_nonlinearity']} "
                        f"according to encoder_nonlinearity. HF only supports tanh in the pooler "
                        f"nonlinearity."
                    )
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
            new_config["pooler_nonlinearity"] = "tanh"
            if "mlm_nonlinearity" not in new_config:
                new_config["mlm_nonlinearity"] = new_config[
                    "encoder_nonlinearity"
                ]

        return super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )


class Converter_Bert_CS18_CS20(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Checkpoint didn't change between 1.8/1.9 and 2.0. Handle weight tying
        # and copy all keys.
        self.rules = [
            ConversionRule(
                [
                    "(?:model.|)",
                    EquivalentSubkey(
                        "bert_encoder.embedding_layer.word_embeddings",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    "\.weight",
                ],
                action=maybe_tie_lm_head,
            ),
            ConversionRule(
                [
                    "(?:model.|)",
                    EquivalentSubkey(
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                        "bert_encoder.embedding_layer.word_embeddings",
                    ),
                    "\.weight",
                ],
                action=maybe_tie_lm_head,
            ),
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-1.8", "cs-1.9"),
            FormatVersions("cs-2.0"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return "BertForPreTraining class"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS18_CS20


# Config didn't change between 1.8/1.9 and 2.0.
class ConfigConverter_Bert_CS18_CS20(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-1.8", "cs-1.9"),
            FormatVersions("cs-2.0"),
        )


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1, and HF <> CS 2.1 converters:
###########################################################


class Converter_Bert_CS20_CS21(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Refactored embeddings (BERT only supported fixed):
            ConversionRule(
                [
                    "(?:model\.|)",
                    "(?:bert_encoder|bert)\.",
                    EquivalentSubkey(
                        "embedding_layer.position_embeddings.weight",
                        "embedding_layer.position_embeddings.embed.weight",
                    ),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:model\.|)",
                    "(?:bert_encoder|bert)\.",
                    EquivalentSubkey(
                        "embedding_layer.position_embeddings",
                        "embedding_layer.position_embeddings.fpe",
                    ),
                ],
                action=self.replaceKey,
            ),
            # Copy everything else
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @classmethod
    def converter_note(cls) -> str:
        return (
            "BertForPreTraining, BertForSequenceClassification, "
            "BertForQuestionAnswering, and BertForSummarization classes"
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.0"), FormatVersions("cs-2.1"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BertModel_CS20_CS21


class ConfigConverter_BertModel_CS20_CS21(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # No differences in config
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.0"), FormatVersions("cs-2.1"))


class ConfigConverter_Bert_HF_CS21(ConfigConverter_Bert_HF_CS18):
    "CS 2.1 config is the same as CS 2.0."

    def __init__(self):
        super().__init__()
        self.post_convert_defaults[1].update({"freeze_ffn_bias_in_glu": False})

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2"),
        )


class Converter_BertModel_WithoutOptionalModel_HF_CS21(
    Converter_BertModel_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("embeddings", "embedding_layer"),
                    "\.position_embeddings",
                    EquivalentSubkey("", ".embed"),
                    "\.weight",
                ],
                action=self.position_embeddings_convert,
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS21


class Converter_BertPretrainModel_WithoutOptionalModel_HF_CS21(
    Converter_BertPretrainModel_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("bert.", "bert_encoder."),
                    Converter_BertModel_WithoutOptionalModel_HF_CS21(),
                ],
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS21


Converter_BertPretrainModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BertPretrainModel_HF_CS21",
    Converter_BertPretrainModel_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_BertPretrainModel_WithoutOptionalModel_HF_CS21,
)


class ConfigConverter_Bert_HF_CS23(ConfigConverter_Bert_HF_CS21):
    def supports_mup_conversion(self):
        return True

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_BertModel_WithoutOptionalModel_HF_CS23(
    Converter_BertModel_WithoutOptionalModel_HF_CS21
):
    def supports_mup_conversion(self):
        return True

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS23


class Converter_BertPretrainModel_WithoutOptionalModel_HF_CS23(
    Converter_BertPretrainModel_WithoutOptionalModel_HF_CS21
):
    def supports_mup_conversion(self):
        return True

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS23


Converter_BertPretrainModel_HF_CS23 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BertPretrainModel_HF_CS23",
    Converter_BertPretrainModel_WithoutOptionalModel_HF_CS23,
    derived_class=Converter_BertPretrainModel_WithoutOptionalModel_HF_CS23,
)
