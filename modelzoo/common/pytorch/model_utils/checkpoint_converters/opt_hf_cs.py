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
    BaseConfigConverter,
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)


class Converter_OPT_Attention_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("q_proj", "proj_q_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("k_proj", "proj_k_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("v_proj", "proj_v_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("out_proj", "proj_output_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None


class Converter_OPT_Headless_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # word embeddings
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embed_tokens", "embedding_layer.word_embeddings"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embed_positions", "embedding_layer.position_embeddings"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # final layer norm
            ConversionRule(
                [
                    EquivalentSubkey(
                        "final_layer_norm", "transformer_decoder.norm"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replace_final_norm,
            ),
            # attention
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("self_attn.", "self_attn."),
                    Converter_OPT_Attention_HF_CS17(),
                ],
                action=None,
            ),
            # attention norm
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("self_attn_layer_norm", "norm1"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # ffn norm
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("final_layer_norm", "norm3"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # intermediate ffn
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("fc1", "ffn.ffn.0.linear_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey("fc2", "ffn.ffn.1.linear_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(["lm_head\.(?:weight|bias)"], exists="right"),
            ConversionRule(["ln_f\.(?:weight|bias)"], exists="right"),
        ]

    def replace_final_norm(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        new_state_dict[new_key] = old_state_dict[old_key]
        # CS 1.7 has both "ln_f" and "transformer_decoder.norm"
        # we need to copy the original ("ln_f") too:
        if from_index == 0:
            ln_f_key = re.sub("transformer_decoder\.norm\.", "ln_f.", new_key)
            new_state_dict[ln_f_key] = old_state_dict[old_key]

    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            logging.warning(
                "{} OPT has a language model head (lm_head) "
                "while {} OPTModel does not. Initializing lm_head to default.".format(
                    self.formats()[1], self.formats()[0]
                )
            )

        # Manually tie weights
        if from_index == 1 and configs[1]["model"]["share_embedding_weights"]:
            if (
                old_state_dict.get("embedding_layer.word_embeddings.weight", 0)
                is None
            ):
                old_state_dict[
                    "embedding_layer.word_embeddings.weight"
                ] = old_state_dict["lm_head.weight"]

    def post_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            # We are converting from HF OPTModel (which is headless) -> CS OPTModel (which has a head)
            # We need to create 'lm_head' and init to default values
            vocab_size, embed_dim = new_state_dict[
                "embedding_layer.word_embeddings.weight"
            ].shape
            lm_head_weight = torch.zeros((vocab_size, embed_dim))
            lm_head_weight.normal_(mean=0.0, std=0.02)
            new_state_dict["lm_head.weight"] = lm_head_weight
            if configs[1]["model"]["use_bias_in_output"]:
                lm_head_bias = torch.zeros(vocab_size)
                new_state_dict["lm_head.bias"] = lm_head_bias

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} OPTModel <-> {} GPT2LMHeadModel (configured as OPT)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_OPTModel_HF_CS17


class Converter_OPT_Headless_HF_CS18(Converter_OPT_Headless_HF_CS17):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule([Converter_OPT_Headless_HF_CS17(),], action=None,),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_OPT_Headless_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} OPTModel <-> {} GPT2LMHeadModel (configured as OPT)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_OPTModel_HF_CS18


class Converter_OPT_LMHeadModel_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["lm_head\.(?:weight|bias)"], action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("model.decoder.", ""),
                    Converter_OPT_Headless_HF_CS17(),
                ],
                action=None,
            ),
        ]

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
                old_state_dict.get("embedding_layer.word_embeddings.weight", 0)
                is None
            ):
                old_state_dict[
                    "embedding_layer.word_embeddings.weight"
                ] = old_state_dict["lm_head.weight"]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} OPTForCausalLM <-> {} GPT2LMHeadModel (configured as OPT)\n"
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_OPTModel_HF_CS17


class Converter_OPT_LMHeadModel_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [Converter_OPT_LMHeadModel_HF_CS17(),], action=None,
            ),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_OPT_LMHeadModel_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} OPTForCausalLM <-> {} GPT2LMHeadModel (configured as OPT)\n"
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_OPTModel_HF_CS18


class ConfigConverter_OPTModel_HF_CS17(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(
                ["position_embedding_type"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, "learned"),
            ),
            ConversionRule(
                ["use_position_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["position_embedding_offset"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, 2),
            ),
            ConversionRule(
                [EquivalentSubkey("embd_pdrop", "embedding_dropout_rate")],
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
            ConversionRule(["hidden_size"], action=self.replaceKey,),
            ConversionRule(
                [EquivalentSubkey("num_attention_heads", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("num_hidden_layers", "num_hidden_layers")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "max_position_embeddings", "max_position_embeddings"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["attention_type"],
                action=BaseConfigConverter.assert_factory_fn(
                    1, "scaled_dot_product"
                ),
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "enable_bias", "use_projection_bias_in_attention"
                    )
                ],
                action=self.convert_bias,
            ),
            ConversionRule(
                [EquivalentSubkey("enable_bias", "use_ffn_bias_in_attention")],
                action=self.convert_bias,
            ),
            ConversionRule(
                [EquivalentSubkey("enable_bias", "use_ffn_bias")],
                action=self.convert_bias,
            ),
            ConversionRule(
                [EquivalentSubkey("ffn_dim", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("activation_function", "nonlinearity")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_dropout", "attention_dropout_rate"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("dropout", "dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["use_bias_in_output"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                [EquivalentSubkey("init_std", "initializer_range")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["embedding_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["fixed_sparse_attention"],
                action=BaseConfigConverter.assert_factory_fn(1, None),
            ),
            ConversionRule(
                ["do_layer_norm_before"],
                action=BaseConfigConverter.assert_factory_fn(
                    0, True
                ),  # False isn't supported since HF removes final layer norm
            ),
            ConversionRule(
                ["norm_first"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_ff_layer1_dropout"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["scale_attn_by_inverse_layer_idx"],
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(
                ["reorder_and_upcast_attn"],
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(["layer_norm_epsilon"], action=self.replaceKey),
            ConversionRule(
                ["word_embed_proj_dim"],
                exists="left",
                action=self.assert_word_embed_proj_dim,
            ),
            ConversionRule(
                ["layerdrop"],
                action=BaseConfigConverter.assert_factory_fn(0, 0.0),
            ),
            ConversionRule(
                ["layer_norm_elementwise_affine"],
                action=BaseConfigConverter.assert_factory_fn(0, True),
            ),
            ConversionRule(
                ["_remove_final_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(
                ["attention_module"],
                action=BaseConfigConverter.assert_factory_fn(
                    1, "aiayn_attention"
                ),
            ),
            ConversionRule(
                ["use_rms_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
        ]

    def convert_attention_type(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            new_state_dict[new_key] = (
                "scaled_dot_product"
                if old_state_dict[old_key]
                else "dot_product"
            )
        else:
            assert (
                old_state_dict[old_key] == "scaled_dot_product"
                or old_state_dict[old_key] == "dot_product"
            )
            new_state_dict[new_key] = old_state_dict[old_key].startswith(
                "scaled_"
            )

    def assert_word_embed_proj_dim(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if old_state_dict[old_key] != old_state_dict["hidden_size"]:
            raise ConfigConversionError(
                "CS only supports word_embed_proj_dim = hidden_size"
            )

    def convert_bias(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            # enable_bias in HF controls all three of the following:
            new_state_dict["use_projection_bias_in_attention"] = old_state_dict[
                old_key
            ]
            new_state_dict["use_ffn_bias_in_attention"] = old_state_dict[
                old_key
            ]
            new_state_dict["use_ffn_bias"] = old_state_dict[old_key]
        else:
            if (
                new_key in new_state_dict
                and new_state_dict[new_key] != old_state_dict[old_key]
            ):
                # We have already set 'enable_bias' and see a param that conflicts
                # with this setting:
                raise ConfigConversionError(
                    "The following params must all be the set the same when \
                     converting to HF: use_projection_bias_in_attention, \
                     use_ffn_bias_in_attention, use_ffn_bias"
                )
            else:
                new_state_dict[new_key] = old_state_dict[old_key]

    def pre_config_convert(
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)

        defaults = [
            {
                "vocab_size": 50272,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "ffn_dim": 3072,
                "num_attention_heads": 12,
                "activation_function": "relu",
                "max_position_embeddings": 2048,
                "do_layer_norm_before": True,
                "dropout": 0.1,
                "attention_dropout": 0.0,
                "init_std": 0.02,
                "layer_norm_epsilon": 1e-5,
                "tie_word_embeddings": True,
                "enable_bias": True,
            },
            {
                "max_position_embeddings": 1024,
                "position_embedding_offset": 0,
                "share_embedding_weights": True,
                "dropout_rate": 0.1,
                "nonlinearity": "gelu",
                "layer_norm_epsilon": 1.0e-5,
                "use_ffn_bias": True,
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "initializer_range": 0.02,
                "norm_first": True,
            },
        ]

        # Apply defaults
        for key in defaults[from_index]:
            if key not in config:
                config[key] = defaults[from_index][key]

        if from_index == 0:
            if "ffn_dim" not in config or config["ffn_dim"] is None:
                config["ffn_dim"] = 4 * config["hidden_size"]

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
            new_config["use_bias_in_output"] = False
            if "attention_type" not in new_config:
                new_config["attention_type"] = "scaled_dot_product"
            if "position_embedding_offset" not in new_config:
                new_config["position_embedding_offset"] = 2

        return super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


class ConfigConverter_OPTModel_HF_CS18(ConfigConverter_OPTModel_HF_CS17):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))
