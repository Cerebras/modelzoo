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

from modelzoo.common.pytorch.model_utils.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseCheckpointConverter_PT_PT,
    BaseConfigConverter,
    BaseConfigConverter_CS_CS,
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    converter_notes,
)


class Converter_BertLayerNorm_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self, hf_name, cs_name):
        super().__init__()
        self.rules = [
            # torch.nn.LayerNorm has .weight & .bias properties
            ConversionRule(
                [EquivalentSubkey(hf_name, cs_name), "\.(?:weight|bias)",],
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
    def formats() -> Tuple[str, str]:
        return ("hf", "cs")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None


class Converter_BertModel_CS16_CS17(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Embedding:
            ConversionRule(
                [
                    EquivalentSubkey("embeddings", "embedding_layer"),
                    "\.word_embeddings\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("embeddings", "embedding_layer"),
                    "\.position_embeddings\.weight",
                ],
                action=self.position_embeddings_convert,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.token_type_embeddings",
                        "embedding_layer.segment_embeddings",
                    ),
                    "\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(["embeddings\.position_ids",], exists="left",),
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
                        "encoder.layer", "transformer_encoder.layers",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "attention.self.query", "self_attn.proj_q_dense_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer", "transformer_encoder.layers",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "attention.self.key", "self_attn.proj_k_dense_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer", "transformer_encoder.layers",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "attention.self.value", "self_attn.proj_v_dense_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer", "transformer_encoder.layers",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "attention.output.dense",
                        "self_attn.proj_output_dense_layer",
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer", "transformer_encoder.layers",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("attention.output.", ""),
                    Converter_BertLayerNorm_HF_CS("LayerNorm", "norm1"),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer", "transformer_encoder.layers",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "intermediate.dense", "ffn.ffn.0.linear_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer", "transformer_encoder.layers",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("output.dense", "ffn.ffn.1.linear_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer", "transformer_encoder.layers",
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("output.", ""),
                    Converter_BertLayerNorm_HF_CS("LayerNorm", "norm2"),
                ],
                action=None,
            ),
            # Head:
            ConversionRule(
                [
                    "pooler\.",
                    EquivalentSubkey("dense", "pooler.ffn.0.linear_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

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
            max_position_embeddings = action_fn_args["configs"][1]["model"][
                "max_position_embeddings"
            ]
            new_state_dict[position_id_key] = torch.arange(
                max_position_embeddings
            ).expand((1, -1))

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("cs-1.6", "cs-1.7")

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
    def formats() -> Tuple[str, str]:
        return ("cs-1.6", "cs-1.7")


class Converter_BertModel_CS16_CS18(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule([Converter_BertModel_CS16_CS17(),], action=None,),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertModel_CS16_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("cs-1.6", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS18


class ConfigConverter_Bert_CS16_CS18(ConfigConverter_Bert_CS16_CS17):
    def __init__(self):
        super().__init__()

    def pre_config_convert(
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)
        if from_index == 1:
            if (
                "pooler_nonlinearity" in config
                and config["pooler_nonlinearity"]
                != config["encoder_nonlinearity"]
            ):
                raise ConfigConversionError(
                    "pooler_nonlinearity was introduced in CS 1.8. Prior to that, the pooler nonlinearity must be the same as encoder_nonlinearity"
                )
            if "mlm_nonlinearity" in config:
                if config["mlm_nonlinearity"] != "gelu":
                    raise ConfigConversionError(
                        "mlm_nonlinearity was introduced in CS 1.8. Prior to that, the mlm nonlinearity must be gelu"
                    )
            else:
                if config["encoder_nonlinearity"] != "gelu":
                    raise ConfigConversionError(
                        "mlm_nonlinearity was introduced in CS 1.8. Prior to that, the mlm nonlinearity must be gelu. However, the input config has an mlm_nonlinearity which defaults to encoder_nonlinearity = {}".format(
                            config["encoder_nonlinearity"]
                        )
                    )
        return config

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            new_config["pooler_nonlinearity"] = new_config[
                "encoder_nonlinearity"
            ]
            new_config["mlm_nonlinearity"] = "gelu"

        return super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("cs-1.6", "cs-1.8")


class Converter_Bert_CS17_CS18(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        # Checkpoint didn't change between 1.7 and 1.8. Copy all keys.
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    @converter_notes(
        notes="""
    CS 1.7 <-> CS 1.8 for 
    BertForPreTraining, BertForSequenceClassification,
    BertForQuestionAnswering, and BertForSummarization classes"""
    )
    def formats() -> Tuple[str, str]:
        return ("cs-1.7", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS17_CS18


class ConfigConverter_Bert_CS17_CS18(ConfigConverter_Bert_CS16_CS18):
    def __init__(self):
        # Config didn't change between 1.6 and 1.7. Therefore 1.7 <-> 1.8
        # converter is equivalent to 1.6 <-> 1.8 converter.
        super().__init__()

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("cs-1.7", "cs-1.8")


class Converter_BertModel_HF_CS17(
    Converter_BertModel_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()

    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        from_index,
        drop_unmatched_keys,
    ):
        # Manually tie weights
        if from_index == 1 and configs[1]["model"]["share_embedding_weights"]:
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

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        return BaseCheckpointConverter_HF_CS.post_checkpoint_convert(
            self, checkpoint, from_index
        )

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS17


class Converter_BertModel_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule([Converter_BertModel_HF_CS17(),], action=None,),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [EquivalentSubkey("", "model."), Converter_BertModel_HF_CS17()],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS17


class Converter_BertPretrainModel_CS16_CS17(BaseCheckpointConverter_PT_PT):
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
                    "\.(?:weight|bias)",
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
                    "\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "cls.predictions.decoder",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    "\.bias",
                ],
                action=self.convert_cls_predictions_bias,
            ),
            ConversionRule(["cls\.predictions\.bias"], exists="left"),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "cls.seq_relationship",
                        "bert_cls_head.classifier.ffn.0.linear_layer",
                    ),
                    "\.(?:weight|bias)",
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
            bias_key = re.sub("\.decoder\.", ".", new_key)
            self.replaceKey(
                old_key,
                bias_key,
                old_state_dict,
                new_state_dict,
                from_index,
                action_fn_args,
            )

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        logging.warning(
            "The Bert model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        return {"model": checkpoint["model"]}

    @staticmethod
    @converter_notes(notes="CS 1.6 <-> CS 1.7 BertPretrainModel")
    def formats() -> Tuple[str, str]:
        return ("cs-1.6", "cs-1.7")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS17


class Converter_BertPretrainModel_CS16_CS18(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [Converter_BertPretrainModel_CS16_CS17(),], action=None,
            ),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertPretrainModel_CS16_CS17(),
                ],
                action=None,
            ),
        ]

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        logging.warning(
            "The Bert model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        return {"model": checkpoint["model"]}

    @staticmethod
    @converter_notes(notes="CS 1.6 <-> CS 1.8 BertPretrainModel")
    def formats() -> Tuple[str, str]:
        return ("cs-1.6", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_CS16_CS18


class Converter_BertPretrainModel_HF_CS17(
    Converter_BertPretrainModel_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()

    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        from_index,
        drop_unmatched_keys,
    ):
        # Manually tie weights
        if from_index == 1 and configs[1]["model"]["share_embedding_weights"]:
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

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        return BaseCheckpointConverter_HF_CS.post_checkpoint_convert(
            self, checkpoint, from_index
        )

    @staticmethod
    @converter_notes(notes="HF <-> CS 1.7 for BertForPreTraining")
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS17


class Converter_BertPretrainModel_HF_CS18(Converter_BertPretrainModel_HF_CS17):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [Converter_BertPretrainModel_HF_CS17(),], action=None,
            ),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BertPretrainModel_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    @converter_notes(notes="HF <-> CS for 1.8 BertForPreTraining")
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Bert_HF_CS18


class ConfigConverter_Bert_HF_CS17(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(
                ["position_embedding_type"],
                action=self.convert_position_embedding_type,
            ),
            ConversionRule(
                ["max_position_embeddings"], action=self.replaceKey,
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
            ConversionRule(["hidden_size"], action=self.replaceKey,),
            ConversionRule(
                [EquivalentSubkey("num_attention_heads", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(["num_hidden_layers"], action=self.replaceKey,),
            ConversionRule(
                [EquivalentSubkey("intermediate_size", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("hidden_act", "encoder_nonlinearity")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["mlm_nonlinearity"], action=self.assert_mlm_nonlinearity,
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
        if old_state_dict[old_key] != old_state_dict["encoder_nonlinearity"]:
            raise ConfigConversionError(
                "HF model doesn't support different encoder & mlm nonlinearities"
            )

    def pre_config_convert(
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)

        defaults = [
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
            },
            {"share_embedding_weights": True, "encoder_nonlinearity": "gelu",},
        ]

        # Apply defaults
        for key in defaults[from_index]:
            if key not in config:
                config[key] = defaults[from_index][key]

        return config

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            if "enable_vts" not in new_config:
                new_config["enable_vts"] = False
            if (
                "mlm_nonlinearity" not in new_config
                and "encoder_nonlinearity" in new_config
                and new_config["encoder_nonlinearity"] != "gelu"
            ):
                logging.warning(
                    "HF used a mlm_nonlinearity of {} while CS 1.7 is fixed to gelu. Please use CS 1.8 if you want to control mlm_nonlinearity".format(
                        new_config["encoder_nonlinearity"]
                    )
                )
                new_config["mlm_nonlinearity"] = "gelu"

        return super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.7")


class ConfigConverter_Bert_HF_CS18(ConfigConverter_Bert_HF_CS17):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formats() -> Tuple[str, str]:
        return ("hf", "cs-1.8")

    def pre_config_convert(
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)
        if from_index == 1:
            if "pooler_nonlinearity" not in config:
                if config["encoder_nonlinearity"] != "tanh":
                    raise ConfigConversionError(
                        "CS Model used a pooler_nonlinearity of {} according to encoder_nonlinearity. HF only supports tanh in the pooler nonlinearity".format(
                            config["encoder_nonlinearity"]
                        )
                    )
        return config

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            new_config["pooler_nonlinearity"] = "tanh"
            if "mlm_nonlinearity" not in new_config:
                new_config["mlm_nonlinearity"] = new_config[
                    "encoder_nonlinearity"
                ]

        return super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )
