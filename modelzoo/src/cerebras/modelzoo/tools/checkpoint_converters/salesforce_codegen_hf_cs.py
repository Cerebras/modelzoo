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
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.gptj_hf_cs import (
    ConfigConverter_GPTJModel_CS18_CS20,
    Converter_GPTJ_LMHeadModel_CS18_CS20,
    Converter_GPTJ_LMHeadModel_CS20_CS21,
)


class Converter_Codegen_Attention_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.mp_num = 4
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("qkv_proj", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.qkv_converter,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("qkv_proj", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.assert_already_converted,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("qkv_proj", "proj_v_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.assert_already_converted,
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

    def qkv_converter(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            self.qkv_converter_hf_to_cs17(
                old_key,
                new_key,
                old_state_dict,
                new_state_dict,
                action_fn_args,
            )
        else:
            self.qkv_converter_cs17_to_hf(
                old_key,
                new_key,
                old_state_dict,
                new_state_dict,
                action_fn_args,
            )

    def qkv_converter_hf_to_cs17(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        action_fn_args,
    ):
        # HF represents Q, K, and V in a packed format (torch.Size(3*hidden, hidden)). We need to
        # unpack the weight and bias tensor for CS 1.7 format.
        q_key = new_key
        k_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_k_dense_layer.", q_key)
        v_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_v_dense_layer.", q_key)

        if new_key.endswith(".bias"):
            assert (
                False
            ), "Codegen model doesn't support bias with attention projection"
        elif new_key.endswith(".weight"):
            packed_dim, embed_dim = old_state_dict[old_key].shape
            assert 3 * embed_dim == packed_dim, (
                f"Invalid tensor shape {old_state_dict[old_key].shape} at {old_key}. The first "
                f"dimension (packed_dim) should be 3x the second dimension (embed_dim) since "
                f"Q, K, and V are packed"
            )

            packed_dim = old_state_dict[old_key].shape[0]
            dim = old_state_dict[old_key].shape[1]
            split_by_mp_num = old_state_dict[old_key].reshape(
                self.mp_num, -1, dim
            )

            local_dim = dim // self.mp_num
            query, value, key = torch.split(split_by_mp_num, local_dim, dim=1)
            query = query.reshape(-1, dim)
            value = value.reshape(-1, dim)
            key = key.reshape(-1, dim)
            new_state_dict[q_key] = query
            new_state_dict[k_key] = key
            new_state_dict[v_key] = value
        else:
            raise ValueError("Invalid key after conversion: {}".format(new_key))

    def qkv_converter_cs17_to_hf(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        action_fn_args,
    ):
        # HF represents Q, K, and V in a packed format. It also contains
        # special ".bias" and ".masked_bias" register buffers that need to be
        # initalized
        q_key = old_key
        k_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_k_dense_layer.", q_key)
        v_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_v_dense_layer.", q_key)

        assert (
            k_key in old_state_dict
        ), "Expected the following key to exist! {}".format(k_key)
        assert (
            v_key in old_state_dict
        ), "Expected the following key to exist! {}".format(v_key)

        query = old_state_dict[q_key]
        value = old_state_dict[v_key]
        key = old_state_dict[k_key]

        hidden_size = old_state_dict[q_key].shape[-1]

        query = query.reshape(self.mp_num, -1, old_state_dict[q_key].shape[-1])
        value = value.reshape(self.mp_num, -1, old_state_dict[v_key].shape[-1])
        key = key.reshape(self.mp_num, -1, old_state_dict[k_key].shape[-1])

        cat_dim = 1
        packed_qkv = torch.cat(
            (
                query,
                value,
                key,
            ),
            dim=cat_dim,
        )
        packed_qkv = packed_qkv.reshape(-1, hidden_size)

        new_state_dict[new_key] = packed_qkv

        if new_key.endswith(".bias"):
            cs_config = action_fn_args["configs"][1]
            max_positions = cs_config["model"]["max_position_embeddings"]
            rotary_dim = cs_config["model"]["rotary_dim"]
            hf_config = action_fn_args["configs"][0]
            rotary_emb_base = hf_config["rotary_emb_base"]

            attn_bias_key = re.sub(r"\.query_key_value\.", ".", new_key)
            new_state_dict[attn_bias_key] = torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.uint8)
            ).view(1, 1, max_positions, max_positions)

            masked_bias_key = re.sub(
                r"\.query_key_value\.", ".masked_", new_key
            )
            new_state_dict[masked_bias_key] = torch.tensor(-1e9)

            inv_freq_key = re.sub(
                r"\.query_key_value\.bias", ".rotary_emb.inv_freq", new_key
            )
            new_state_dict[inv_freq_key] = 1.0 / (
                rotary_emb_base
                ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
            )

    def assert_already_converted(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            # We should never hit this case as this key should have been matched
            # already
            assert False, "Invalid key: {}".format(old_key)
        else:
            # When we convert from CS -> HF, the proj_q_dense_layer should also handle
            # conversion of proj_k_dense_layer and proj_v_dense_layer since HF
            # represents these three layers in a packed format. We simply need
            # to test that the key containing the packed format has already
            # been converted.
            assert (
                new_key in new_state_dict
            ), "Key should've been already converted: {} -> {}".format(
                old_key, new_key
            )


class Converter_Codegen_Headless_HF_CS17(BaseCheckpointConverter_HF_CS):
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
                    Converter_Codegen_Attention_HF_CS17(),
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
                ],
                exists="left",
            ),
            ConversionRule(
                [
                    r"h\.\d+\.attn\.causal_mask",
                ],
                exists="left",
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
                "while {} CodeGenModel does not. Initializing lm_head to default.".format(
                    self.formats()[1], self.formats()[0]
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
            # We are converting from HF CodeGenModel (which is headless) -> CS GPTJModel
            # (which has a head). We need to create 'lm_head' and init to default values
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
            "{} CodeGenModel <-> {} GPTJModel (configured as codegen)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Codegen_Model_HF_CS17


class Converter_Codegen_Headless_HF_CS18(Converter_Codegen_Headless_HF_CS17):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_Codegen_Headless_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_Codegen_Headless_HF_CS17(),
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
            "{} CodeGenModel <-> {} GPTJModel (configured as codegen)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Codegen_Model_HF_CS18


class Converter_Codegen_LMHeadModel_HF_CS17(BaseCheckpointConverter_HF_CS):
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
                    Converter_Codegen_Headless_HF_CS17(),
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
        return "{} CodeGenForCausalLM <-> {} GPTJModel (configured as codegen)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Codegen_Model_HF_CS17


class Converter_Codegen_LMHeadModel_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_Codegen_LMHeadModel_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_Codegen_LMHeadModel_HF_CS17(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} CodeGenForCausalLM <-> {} GPTJModel (configured as codegen)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Codegen_Model_HF_CS18


class ConfigConverter_Codegen_Model_HF_CS17(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "codegen"),
            ),
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(["rotary_dim"], action=self.replaceKey),
            ConversionRule(
                ["position_embedding_type"],
                action=BaseConfigConverter.assert_factory_fn(1, "rotary"),
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
                ["use_bias_in_output"],
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
                "resid_pdrop": 0.0,
                "embd_pdrop": 0.0,
                "attn_pdrop": 0.0,
                "initializer_range": 0.02,
                "layer_norm_epsilon": 1.0e-5,
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
            },
        )

        self.post_convert_defaults[0].update({"model_type": "codegen"})
        self.post_convert_defaults[1].update(
            {
                "use_ffn_bias_in_attention": False,
                "use_projection_bias_in_attention": False,
                "use_ffn_bias": True,
                "use_bias_in_output": True,
                "attention_type": "scaled_dot_product",
                "use_untied_layer_norm": False,
            },
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
            if config["n_head"] < 4:
                raise ConfigConversionError(
                    "HF Model doesn't support n_head < 4 because of hardcoded TPU constraint"
                )
        else:
            if config["num_heads"] < 4:
                raise ConfigConversionError(
                    "HF Model doesn't support n_head < 4 because of hardcoded TPU constraint"
                )
        return config

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


class ConfigConverter_Codegen_Model_HF_CS18(
    ConfigConverter_Codegen_Model_HF_CS17
):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))


class Converter_Codegen_LMHeadModel_CS18_CS20(
    Converter_GPTJ_LMHeadModel_CS18_CS20
):
    r"""
    Codegen uses the GPTJ backbone.
    """

    @classmethod
    def converter_note(cls) -> str:
        return "GPTJModel (configured as codegen)"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Codegen_Model_CS18_CS20


class ConfigConverter_Codegen_Model_CS18_CS20(
    ConfigConverter_GPTJModel_CS18_CS20
):
    r"""
    Codegen uses the GPTJ backbone.
    """


class Converter_Codegen_Headless_HF_CS20(Converter_Codegen_Headless_HF_CS18):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions(
                "cs-2.0", "cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"
            ),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Codegen_Model_HF_CS20


class Converter_Codegen_LMHeadModel_HF_CS20(
    Converter_Codegen_LMHeadModel_HF_CS18
):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions(
                "cs-2.0", "cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"
            ),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Codegen_Model_HF_CS20


class ConfigConverter_Codegen_Model_HF_CS20(
    ConfigConverter_Codegen_Model_HF_CS18
):
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
            FormatVersions(
                "cs-2.0", "cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"
            ),
        )


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1. We don't need a separate HF <> CS 2.1 converters since
# HF only supports RoPE which doesn't produce any checkpoint keys.
###########################################################


class Converter_Codegen_LMHeadModel_CS20_CS21(
    Converter_GPTJ_LMHeadModel_CS20_CS21
):
    def __init__(self):
        super().__init__()

    @classmethod
    def converter_note(cls) -> str:
        return "GPTJLMHeadModel class (configured as codegen)"
