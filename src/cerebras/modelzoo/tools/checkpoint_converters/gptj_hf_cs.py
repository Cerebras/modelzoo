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
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    convert_use_biasless_layer_norm_helper,
    maybe_tie_lm_head,
    tie_none_weights,
)


class Converter_GPTJ_Attention_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("q_proj", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("k_proj", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("v_proj", "proj_v_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("out_proj", "proj_output_dense_layer"),
                    r"\.(?:weight|bias)",
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


class Converter_GPTJ_Headless_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # word embeddings
            ConversionRule(
                [
                    EquivalentSubkey("wte", "embedding_layer.word_embeddings"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # final layer norm
            ConversionRule(
                [
                    EquivalentSubkey("ln_f", "transformer_decoder.norm"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replace_final_norm,
            ),
            # attention
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("attn.", "self_attn."),
                    Converter_GPTJ_Attention_HF_CS17(),
                ],
                action=None,
            ),
            # attention norm
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("ln_1", "norm1"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # norm3 layers should be set to None if they exist:
            ConversionRule(
                [
                    r"transformer_decoder.layers\.\d+\.norm3\.(?:weight|bias)",
                ],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, None),
            ),
            # intermediate ffn
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.fc_in", "ffn.ffn.0.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.fc_out", "ffn.ffn.1.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule([r"lm_head\.(?:weight|bias)"], exists="right"),
            ConversionRule([r"ln_f\.(?:weight|bias)"], exists="right"),
            ConversionRule(
                [
                    r"h\.\d+\.attn\.(?:masked_bias|bias)",
                ]
            ),
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
            ln_f_key = re.sub(r"transformer_decoder\.norm\.", "ln_f.", new_key)
            new_state_dict[ln_f_key] = old_state_dict[old_key]

    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
    ):
        if converter_indices.direction == 0:
            logging.warning(
                "{} GPTJ has a language model head (lm_head) "
                "while {} GPTJModel does not. Initializing lm_head to default.".format(
                    *self.formats()
                )
            )

        # Manually tie weights
        if (
            converter_indices.direction == 1
            and configs[1]["model"]["share_embedding_weights"]
        ):
            if (
                old_state_dict.get("embedding_layer.word_embeddings.weight", 0)
                is None
            ):
                old_state_dict["embedding_layer.word_embeddings.weight"] = (
                    old_state_dict["lm_head.weight"]
                )

    def post_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
        key_prefix="",
    ):
        if converter_indices.direction == 0:
            # We are converting from HF GPTJModel (which is headless) -> CS GPTJModel (which has
            # a head). We need to create 'lm_head' and init to default values.
            hf_config = configs[0]
            cs_config = configs[1]
            use_bias_in_output = cs_config["model"].get(
                "use_bias_in_output", False
            )
            vocab_size = cs_config["model"]["vocab_size"]
            embed_dim = cs_config["model"]["hidden_size"]
            if hf_config["tie_word_embeddings"]:
                lm_head_weight = old_state_dict['wte.weight']
            else:
                lm_head_weight = torch.zeros((vocab_size, embed_dim))
                lm_head_weight.normal_(mean=0.0, std=0.02)
            new_state_dict[key_prefix + "lm_head.weight"] = lm_head_weight
            if use_bias_in_output:
                lm_head_bias = torch.zeros(vocab_size)
                new_state_dict[key_prefix + "lm_head.bias"] = lm_head_bias

        super().post_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            converter_indices,
            drop_unmatched_keys,
            key_prefix=key_prefix,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} GPTJModel <-> {} GPTJModel(with head)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS17


class Converter_GPTJ_Headless_HF_CS18(Converter_GPTJ_Headless_HF_CS17):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_GPTJ_Headless_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_GPTJ_Headless_HF_CS17(),
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
            "{} GPTJModel <-> {} GPTJModel(with head)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS18


class Converter_GPTJ_LMHeadModel_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [r"lm_head\.(?:weight|bias)"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("transformer.", ""),
                    Converter_GPTJ_Headless_HF_CS17(),
                ],
                action=None,
            ),
        ]

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
                old_state_dict.get("embedding_layer.word_embeddings.weight", 0)
                is None
            ):
                old_state_dict["embedding_layer.word_embeddings.weight"] = (
                    old_state_dict["lm_head.weight"]
                )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPTJForCausalLM <-> {} GPTJModel(with head)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS17


class Converter_GPTJ_LMHeadModel_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_GPTJ_LMHeadModel_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_GPTJ_LMHeadModel_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPTJForCausalLM <-> {} GPTJModel(with head)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS18


class ConfigConverter_GPTJModel_HF_CS17(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "gptj"),
            ),
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(["rotary_dim"], action=self.replaceKey),
            ConversionRule(
                [EquivalentSubkey("rotary", "position_embedding_type")],
                exists="right",
                action=self.convert_position_embedding_type,
            ),
            ConversionRule(
                ["use_position_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
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
            ConversionRule(
                [EquivalentSubkey("n_embd", "hidden_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_head", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_layer", "num_hidden_layers")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_positions", "max_position_embeddings")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["scale_attn_weights"],
                action=BaseConfigConverter.assert_factory_fn(0, True),
            ),
            ConversionRule(
                ["attention_type"],
                action=BaseConfigConverter.assert_factory_fn(
                    1, "scaled_dot_product"
                ),
            ),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["use_ffn_bias_in_attention"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["use_ffn_bias"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                [EquivalentSubkey("n_inner", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("activation_function", "nonlinearity")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("attn_pdrop", "attention_dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("resid_pdrop", "residual_dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["layer_norm_epsilon"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["use_bias_in_output"],
                action=self.replaceKey,
            ),
            ConversionRule(["initializer_range"], action=self.replaceKey),
            ConversionRule(
                ["embedding_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["fixed_sparse_attention"],
                action=BaseConfigConverter.assert_factory_fn(1, None),
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
                ["use_untied_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
        ]

        self.pre_convert_defaults[0].update(
            {
                "vocab_size": 50400,
                "n_positions": 2048,
                "n_embd": 4096,
                "n_layer": 28,
                "n_head": 16,
                "rotary_dim": 64,
                "activation_function": "gelu_new",
                "resid_pdrop": 0.1,
                "embd_pdrop": 0.1,
                "attn_pdrop": 0.1,
                "initializer_range": 0.02,
                "layer_norm_epsilon": 1e-5,
                "tie_word_embeddings": False,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "max_position_embeddings": 1024,
                "embedding_dropout_rate": 0.1,
                "share_embedding_weights": True,
                "residual_dropout_rate": 0.1,
                "nonlinearity": "gelu",
                "layer_norm_epsilon": 1.0e-5,
                "use_ffn_bias": False,
                "use_untied_layer_norm": False,
                "attention_dropout_rate": 0.1,
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "initializer_range": 0.02,
                "use_bias_in_output": False,
                "norm_first": True,
            }
        )

        self.post_convert_defaults[0].update({"model_type": "gptj"})
        self.post_convert_defaults[1].update(
            {
                "use_untied_layer_norm": False,
                "use_ffn_bias_in_attention": False,
                "use_projection_bias_in_attention": False,
                "use_ffn_bias": True,
                "use_bias_in_output": True,
                "attention_type": "scaled_dot_product",
            },
        )

    def convert_position_embedding_type(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            if old_state_dict[old_key] != True:
                raise ConfigConversionError(
                    "HF GPT-J must use rotary embeddings, but got {}={}".format(
                        old_key, old_state_dict[old_key]
                    )
                )
            new_state_dict[new_key] = "rotary"
        else:
            if old_state_dict[old_key] != "rotary":
                raise ConfigConversionError(
                    "CS GPT-J must use rotary embeddings, but got {}={}".format(
                        old_key, old_state_dict[old_key]
                    )
                )
            new_state_dict[new_key] = True

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

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        if converter_indices.direction == 0:
            if "n_inner" not in config or config["n_inner"] is None:
                config["n_inner"] = 4 * config["n_embd"]
        return config

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


class ConfigConverter_GPTJModel_HF_CS18(ConfigConverter_GPTJModel_HF_CS17):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))


class Converter_GPTJ_LMHeadModel_CS18_CS20(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Model didn't change between 1.8 and 1.9. Copy all keys.
        self.rules = [
            ConversionRule(
                [
                    "(?:model.|)",
                    EquivalentSubkey(
                        "lm_head", "embedding_layer.word_embeddings"
                    ),
                    "\.weight",
                ],
                action=maybe_tie_lm_head,
            ),
            ConversionRule(
                [
                    "(?:model.|)",
                    EquivalentSubkey(
                        "embedding_layer.word_embeddings",
                        "lm_head",
                    ),
                    "\.weight",
                ],
                action=maybe_tie_lm_head,
            ),
            ConversionRule(
                [
                    "(?:model.|)",
                    EquivalentSubkey("transformer_decoder.norm", "ln_f"),
                    "\.(?:weight|bias)",
                ],
                action=tie_none_weights,
            ),
            ConversionRule(
                [
                    "(?:model.|)",
                    EquivalentSubkey("ln_f", "transformer_decoder.norm"),
                    "\.(?:weight|bias)",
                ],
                action=tie_none_weights,
            ),
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} GPTJModel".format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.8", "cs-1.9"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_CS18_CS20


class ConfigConverter_GPTJModel_CS18_CS20(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Only difference between 1.8/1.9 and 2.0 is introduction of norm_type
        self.rules = [
            ConversionRule(
                [EquivalentSubkey("use_biasless_norm", "norm_type")],
                action=self.convert_use_biasless_layer_norm,
            ),
            ConversionRule([".*"], action=self.replaceKey),
        ]

    def convert_use_biasless_layer_norm(self, *args):
        convert_use_biasless_layer_norm_helper(self, *args)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.8", "cs-1.9"), FormatVersions("cs-2.0"))


class Converter_GPTJ_Headless_HF_CS20(Converter_GPTJ_Headless_HF_CS18):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.0", "cs-2.1", "cs-2.2"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS20


# Despite the embedding layer being refactored in release 2.1, the 2.0
# HF <> CS converter also works for 2.1 and 2.2. This is because the HF
# GPT-J style models only support RoPE embeddings which don't store any
# learnable params in the checkpoint."
class Converter_GPTJ_LMHeadModel_HF_CS20(Converter_GPTJ_LMHeadModel_HF_CS18):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.0", "cs-2.1", "cs-2.2"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS20


class ConfigConverter_GPTJModel_HF_CS20(ConfigConverter_GPTJModel_HF_CS18):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["norm_type"],
                action=BaseConfigConverter.assert_factory_fn(1, "layernorm"),
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.0", "cs-2.1", "cs-2.2"),
        )


class Converter_GPTJ_Headless_HF_CS23(Converter_GPTJ_Headless_HF_CS20):
    def supports_mup_conversion(self):
        return True

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS23

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_GPTJ_LMHeadModel_HF_CS23(Converter_GPTJ_LMHeadModel_HF_CS20):
    def supports_mup_conversion(self):
        return True

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_HF_CS23

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


class ConfigConverter_GPTJModel_HF_CS23(ConfigConverter_GPTJModel_HF_CS20):
    def supports_mup_conversion(self):
        return True

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1. We don't need a separate HF <> CS 2.1 converters since
# HF only supports RoPE which doesn't produce any checkpoint keys.
###########################################################


class Converter_GPTJ_LMHeadModel_CS20_CS21(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Refactored embeddings:
            ConversionRule(
                [
                    "(?:model\.|)",
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
                    "embedding_layer\.",
                    EquivalentSubkey(
                        "position_embeddings",
                        "position_embeddings.fpe",
                    ),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:model\.|)",
                    EquivalentSubkey(
                        "relative_pe_helper.relative_attention_bias",
                        "embedding_layer.position_embed_helper.relative_attention_bias",
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:model\.|)",
                    EquivalentSubkey(
                        "relative_pe_helper.slopes",
                        "embedding_layer.position_embed_helper.slopes",
                    ),
                ],
                action=self.replaceKey,
            ),
            # Copy everything else
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @classmethod
    def converter_note(cls) -> str:
        return "GPTJLMHeadModel class"

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.0"), FormatVersions("cs-2.1"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPTJModel_CS20_CS21


class ConfigConverter_GPTJModel_CS20_CS21(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # No differences in config
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.0"), FormatVersions("cs-2.1"))
