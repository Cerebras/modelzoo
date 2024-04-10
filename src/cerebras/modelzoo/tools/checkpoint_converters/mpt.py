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
from cerebras.modelzoo.tools.checkpoint_converters.gpt2_hf_cs import (
    Converter_GPT2LMHeadModel_CS20_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)


class Converter_MPTAttention_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("Wqkv", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_qkv,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("Wqkv", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.assert_already_converted,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("Wqkv", "proj_v_dense_layer"),
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

    def convert_qkv(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            self.convert_qkv_hf_to_cs(
                old_key, new_key, old_state_dict, new_state_dict, action_fn_args
            )
        else:
            self.convert_qkv_cs_to_hf(
                old_key, new_key, old_state_dict, new_state_dict, action_fn_args
            )

    def convert_qkv_hf_to_cs(
        self, old_key, new_key, old_state_dict, new_state_dict, action_fn_args
    ):
        # HF represents Q, K, and V in a packed format. We need to unpack the
        # weight and bias tensor for CS 1.7 format.
        q_key = new_key
        k_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_k_dense_layer.", q_key)
        v_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_v_dense_layer.", q_key)

        (
            new_state_dict[q_key],
            new_state_dict[k_key],
            new_state_dict[v_key],
        ) = torch.chunk(old_state_dict[old_key], 3, dim=0)

    def convert_qkv_cs_to_hf(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        action_fn_args,
    ):
        # HF represents Q, K, and V in a packed format. It also contains
        # special ".bias" and ".masked_bias" register buffers that need to be
        # initialized
        q_key = old_key
        k_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_k_dense_layer.", q_key)
        v_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_v_dense_layer.", q_key)

        assert (
            k_key in old_state_dict
        ), "Expected the following key to exist! {}".format(k_key)
        assert (
            v_key in old_state_dict
        ), "Expected the following key to exist! {}".format(v_key)

        new_state_dict[new_key] = torch.cat(
            (
                old_state_dict[q_key],
                old_state_dict[k_key],
                old_state_dict[v_key],
            ),
            dim=0,
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


class Converter_MPTModel_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.cs_slopes_key = "relative_pe_helper.slopes"
        self.rules = [
            # word embeddings
            ConversionRule(
                [
                    EquivalentSubkey("wte", "embedding_layer.word_embeddings"),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_word_embeddings,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "wpe", "embedding_layer.position_embeddings"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # final layer norm
            ConversionRule(
                [
                    EquivalentSubkey("norm_f", "transformer_decoder.norm"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replace_final_norm,
            ),
            # attention
            ConversionRule(
                [
                    EquivalentSubkey("blocks", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("attn", "self_attn"),
                    r"\.",
                    Converter_MPTAttention_HF_CS(),
                ],
                action=None,
            ),
            # attention norm
            ConversionRule(
                [
                    EquivalentSubkey("blocks", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("norm_1", "norm1"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("blocks", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("norm_2", "norm3"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # intermediate ffn
            ConversionRule(
                [
                    EquivalentSubkey("blocks", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("ffn.up_proj", "ffn.ffn.0.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("blocks", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("ffn.down_proj", "ffn.ffn.1.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule([r"lm_head\.(?:weight|bias)"], exists="right"),
            ConversionRule([r"ln_f\.(?:weight|bias)"], exists="right"),
            ConversionRule([r"relative_pe_helper\.slopes"], exists="right"),
        ]

    def convert_word_embeddings(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        new_state_dict[new_key] = old_state_dict[old_key]
        if from_index == 0:
            lm_head_key = re.sub(
                r"embedding_layer\.word_embeddings", "lm_head", new_key
            )
            new_state_dict[lm_head_key] = old_state_dict[old_key]

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

    @staticmethod
    def get_alibi_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some a. This function has
        # some good properties that only occur when the input is a power of 2. To maintain that even
        # when the number of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            slopes_list = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes_list = (
                get_slopes_power_of_2(closest_power_of_2)
                + Converter_MPTModel_HF_CS.get_alibi_slopes(
                    2 * closest_power_of_2
                )[0::2][: n - closest_power_of_2]
            )
        return torch.tensor(slopes_list).unsqueeze(-1)

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
            # We are converting from HF MPTModel (which is headless) ->
            # CS GPT2LMHeadModel configured as MPT (which has a head)
            #
            # convert_word_embeddings action_fn already initialized lm_head
            # we just need to warn the user.

            logging.warning(
                "{} has a language model head (lm_head) "
                "while {} does not. Initializing to same as word embeddings "
                "(tied embeddings)".format(self.formats()[1], self.formats()[0])
            )

            # Need to initialize alibi slopes:
            cs_config = configs[1]
            if cs_config["model"]["position_embedding_type"] == "alibi":
                new_state_dict[
                    key_prefix + self.cs_slopes_key
                ] = Converter_MPTModel_HF_CS.get_alibi_slopes(
                    cs_config["model"]["num_heads"]
                )
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
        return (FormatVersions("hf"), FormatVersions("cs"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None


class Converter_MPTModel_HF_CS20(Converter_MPTModel_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_MPTModel_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from deprecated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_MPTModel_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} MPTModel <-> {} GPT2LMHeadModel (configured as MPT)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MPT_HF_CS20


class Converter_MPTForCausalLM_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.cs_slopes_key = "relative_pe_helper.slopes"
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("transformer.", ""),
                    Converter_MPTModel_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None

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
            # Need to initialize alibi slopes:
            cs_config = configs[1]
            if cs_config["model"]["position_embedding_type"] == "alibi":
                new_state_dict[
                    key_prefix + self.cs_slopes_key
                ] = Converter_MPTModel_HF_CS.get_alibi_slopes(
                    cs_config["model"]["num_heads"]
                )
        super().post_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            converter_indices,
            drop_unmatched_keys,
            key_prefix=key_prefix,
        )


class Converter_MPTForCausalLM_HF_CS20(Converter_MPTForCausalLM_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_MPTForCausalLM_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from deprecated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_MPTForCausalLM_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} MPTForCausalLM <-> {} GPT2LMHeadModel (configured as MPT)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MPT_HF_CS20


class ConfigConverter_MPT_HF_CS20(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "mpt"),
            ),
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(
                ["use_position_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                [EquivalentSubkey("emb_pdrop", "embedding_dropout_rate")],
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
            ConversionRule(
                ["embedding_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            # Decoder Block
            ConversionRule(
                [EquivalentSubkey("d_model", "hidden_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_heads", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_layers", "num_hidden_layers")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("max_seq_len", "max_position_embeddings")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["attn_config"],
                exists="left",
                action=self.convert_attention_config,
            ),
            ConversionRule(
                ["attention_module"],
                exists="right",
                action=self.convert_attention_config,
            ),
            ConversionRule(
                ["attention_type"],
                exists="right",
                action=self.convert_attention_config,
            ),
            ConversionRule(
                ["attention_dropout_rate"],
                exists="right",
                action=self.convert_attention_config,
            ),
            ConversionRule(
                ["position_embedding_type"],
                exists="right",
                action=self.convert_attention_config,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "no_bias", "use_projection_bias_in_attention"
                    )
                ],
                action=self.convert_no_bias,
            ),
            ConversionRule(
                [EquivalentSubkey("no_bias", "use_ffn_bias_in_attention")],
                action=self.convert_no_bias,
            ),
            ConversionRule(
                [EquivalentSubkey("no_bias", "use_ffn_bias")],
                action=self.convert_no_bias,
            ),
            ConversionRule(
                [EquivalentSubkey("expansion_ratio", "filter_size")],
                action=self.convert_expansion_ratio,
            ),
            ConversionRule(
                ["nonlinearity"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, "gelu"),
            ),
            ConversionRule(
                [EquivalentSubkey("resid_pdrop", "dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["layer_norm_epsilon"],
                action=BaseConfigConverter.assert_factory_fn(1, 1.0e-5),
            ),
            ConversionRule(
                ["use_bias_in_output"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(["initializer_range"], action=self.replaceKey),
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
                ["norm_type"],
                action=self.convert_norm_type,
            ),
            ConversionRule(
                ["embedding_fraction"],
                action=BaseConfigConverter.assert_factory_fn(0, 1.0),
            ),
            ConversionRule(
                ["logit_scale"],
                action=BaseConfigConverter.assert_factory_fn(0, None),
            ),
        ]

        self.pre_convert_defaults[0].update(
            {
                "d_model": 2048,
                "n_heads": 16,
                "n_layers": 24,
                "expansion_ratio": 4,
                "max_seq_len": 2048,
                "vocab_size": 50368,
                "resid_pdrop": 0.0,
                "emb_pdrop": 0.0,
                "learned_pos_emb": True,
                "attn_config": {
                    "attn_type": "multihead_attention",
                    "attn_pdrop": 0.0,
                    "attn_impl": "triton",
                    "qk_ln": False,
                    "clip_qkv": None,
                    "softmax_scale": None,
                    "prefix_lm": False,
                    "attn_uses_sequence_id": False,
                    "alibi": False,
                    "alibi_bias_max": 8,
                },
                "logit_scale": None,
                "no_bias": False,
                "embedding_fraction": 1.0,
                "norm_type": "low_precision_layernorm",
                "use_cache": False,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "share_embedding_weights": True,
                "norm_type": "layernorm",
                "max_position_embeddings": 1024,
                "position_embedding_type": "learned",
                "layer_norm_epsilon": 1.0e-5,
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "nonlinearity": "gelu",
                "use_ffn_bias": True,
                "use_bias_in_output": False,
                "norm_first": True,
            },
        )

        self.post_convert_defaults[0].update(
            {
                "model_type": "mpt",
                "attn_config": {
                    "attn_type": "multihead_attention",
                    "attn_pdrop": 0.0,
                    "attn_impl": "torch",
                    "qk_ln": False,
                    "clip_qkv": None,
                    "softmax_scale": None,
                    "prefix_lm": False,
                    "attn_uses_sequence_id": False,
                    "alibi": False,
                    "alibi_bias_max": 8,
                },
            }
        )

        self.post_convert_defaults[1].update(
            {
                "use_position_embedding": True,
                "position_embedding_type": "learned",
                "embedding_dropout_rate": 0.0,
                "embedding_layer_norm": False,
                "attention_type": "scaled_dot_product",
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "use_ffn_bias": True,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.0,
                "use_bias_in_output": False,
                "norm_first": True,
                "use_ff_layer1_dropout": False,
                "share_embedding_weights": True,
                "nonlinearity": "gelu",
            },
        )

    def convert_attention_config(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            attention_module = old_state_dict[old_key]["attn_type"]
            if attention_module == "multihead_attention":
                attention_module = "aiayn_attention"
            new_state_dict["attention_module"] = attention_module
            new_state_dict["attention_dropout_rate"] = old_state_dict[old_key][
                "attn_pdrop"
            ]
            softmax_scale = old_state_dict[old_key]["softmax_scale"]

            if softmax_scale is None:
                new_state_dict["attention_type"] = "scaled_dot_product"
            elif softmax_scale == 1.0:
                new_state_dict["attention_type"] = "dot_product"
            else:
                raise ConfigConversionError(
                    "CS model only supports softmax_scale of 1.0 or None"
                )

            if old_state_dict[old_key]["alibi"]:
                new_state_dict["position_embedding_type"] = "alibi"
            else:
                new_state_dict["position_embedding_type"] = "learned"

            if old_state_dict[old_key].get("alibi_bias_max", 8) != 8:
                raise ConfigConversionError("CS only supports alibi_bias_max=8")

        else:
            if "attn_config" not in new_state_dict:
                new_state_dict["attn_config"] = {}

            if old_key == "attention_module":
                attention_module = old_state_dict[old_key]
                if attention_module == "aiayn_attention":
                    attention_module = "multihead_attention"
                elif attention_module == "multiquery_attention":
                    pass
                else:
                    raise ConfigConversionError(
                        "MPT model does not support attention_module={}".format(
                            attention_module
                        )
                    )
                new_state_dict["attn_config"]["attn_type"] = attention_module
            elif old_key == "attention_dropout_rate":
                new_state_dict["attn_config"]["attn_pdrop"] = old_state_dict[
                    old_key
                ]
            elif old_key == "attention_type":
                attention_type = old_state_dict[old_key]

                if attention_type == "scaled_dot_product":
                    softmax_scale = None
                elif attention_type == "dot_product":
                    softmax_scale = 1.0
                else:
                    raise ConfigConversionError(
                        "attention_type {} isn't supported in MPT models".format(
                            attention_type
                        )
                    )
                new_state_dict["attn_config"]["softmax_scale"] = softmax_scale
            elif old_key == "position_embedding_type":
                position_embedding_type = old_state_dict[old_key]
                if position_embedding_type == "alibi":
                    new_state_dict["attn_config"]["alibi"] = True
                elif position_embedding_type == "learned":
                    new_state_dict["attn_config"]["alibi"] = False
                else:
                    raise ConfigConversionError(
                        "MPT model only supports alibi or learned position embeddings"
                    )

    def convert_expansion_ratio(
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
                old_state_dict[old_key] * old_state_dict["d_model"]
            )
        else:
            expansion_ratio = (
                old_state_dict[old_key] / old_state_dict["hidden_size"]
            )
            if not expansion_ratio.is_integer():
                raise ConfigConversionError(
                    "expansion_ratio (filter_size / hidden_size) must be an integer"
                )
            new_state_dict[new_key] = int(expansion_ratio)

    def convert_no_bias(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            new_state_dict[
                "use_projection_bias_in_attention"
            ] = not old_state_dict[old_key]
            new_state_dict["use_ffn_bias_in_attention"] = not old_state_dict[
                old_key
            ]
            new_state_dict["use_ffn_bias"] = not old_state_dict[old_key]

        else:
            if (
                new_key in new_state_dict
                and new_state_dict[new_key] == old_state_dict[old_key]
            ):
                raise ConfigConversionError(
                    "use_projection_bias_in_attention, use_ffn_bias_in_attention, and "
                    "use_ffn_bias must all be the same in MPT models."
                )
            new_state_dict[new_key] = not old_state_dict[old_key]

    def convert_norm_type(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        norm_type = old_state_dict[old_key]
        if from_index == 0:
            if norm_type.startswith("low_precision_"):
                logging.warning(
                    "CS doesn't support low precision layer norm. Using non-low-precision "
                    "implementation."
                )
                norm_type = norm_type[len("low_precision_") :]
            if norm_type == "layernorm" and old_state_dict["no_bias"]:
                norm_type = "biasless-layernorm"
        else:
            if norm_type == "biasless-layernorm":
                if (
                    old_state_dict["use_projection_bias_in_attention"]
                    or old_state_dict["use_ffn_bias_in_attention"]
                    or old_state_dict["use_ffn_bias"]
                ):
                    raise ConfigConversionError(
                        "use_projection_bias_in_attention, use_ffn_bias_in_attention, and "
                        "use_ffn_bias must all be False when using biasless-layernorm in MPT "
                        "models."
                    )
                new_state_dict["no_bias"] = True
                norm_type = "layernorm"
        new_state_dict[new_key] = norm_type

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1, and HF <> CS 2.1 converters:
###########################################################


class Converter_MPTForCausalLM_CS20_CS21(Converter_GPT2LMHeadModel_CS20_CS21):
    def __init__(self):
        super().__init__()

    @classmethod
    def converter_note(cls) -> str:
        return "GPT2LMHeadModel class (configured as MPT)"


class ConfigConverter_MPT_HF_CS21(ConfigConverter_MPT_HF_CS20):
    "CS 2.1 config is the same as CS 2.0"

    def __init__(self):
        super().__init__()
        del self.post_convert_defaults[1]["use_position_embedding"]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.1", "cs-2.2"))


class Converter_MPTModel_WithoutOptionalModel_HF_CS21(Converter_MPTModel_HF_CS):
    def __init__(self):
        super().__init__()
        self.cs_slopes_key = "embedding_layer.position_embed_helper.slopes"  # used in post_model_convert fn
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "wpe", "embedding_layer.position_embeddings.embed"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [r"embedding_layer\.position_embed_helper\.slopes"],
                exists="right",
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.1", "cs-2.2"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} MPTModel <-> {} GPT2LMHeadModel (configured as MPT)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MPT_HF_CS21


Converter_MPTModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_MPTModel_HF_CS21",
    Converter_MPTModel_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_MPTModel_WithoutOptionalModel_HF_CS21,
)


class Converter_MPTForCausalLM_WithoutOptionalModel_HF_CS21(
    Converter_MPTForCausalLM_HF_CS
):
    def __init__(self):
        super().__init__()
        self.cs_slopes_key = "embedding_layer.position_embed_helper.slopes"  # used in post_model_convert fn
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("transformer.", ""),
                    Converter_MPTModel_WithoutOptionalModel_HF_CS21(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.1", "cs-2.2"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} MPTForCausalLM <-> {} GPT2LMHeadModel (configured as MPT)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MPT_HF_CS21


Converter_MPTForCausalLM_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_MPTForCausalLM_HF_CS21",
    Converter_MPTForCausalLM_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_MPTForCausalLM_WithoutOptionalModel_HF_CS21,
)
