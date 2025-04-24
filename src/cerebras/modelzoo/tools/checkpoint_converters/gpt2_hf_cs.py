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
    Build_HF_CS_Converter_WithOptionalModel,
    convert_use_rms_layer_norm_helper,
    maybe_tie_lm_head,
    tie_none_weights,
    transpose_key_if_2D,
)

#########################################################
# GPT2 HF <> CS17
#########################################################


class Converter_GPT2_Attention_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self, generate_hf_biases=True):
        super().__init__()
        self.generate_hf_biases = generate_hf_biases
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("c_proj", "proj_output_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=transpose_key_if_2D,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("c_attn", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.c_attn_converter,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("q_attn", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.assert_already_converted,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("c_attn", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.assert_already_converted,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("c_attn", "proj_v_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.assert_already_converted,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None

    def c_attn_converter(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            self.c_attn_converter_hf_to_cs17(
                old_key, new_key, old_state_dict, new_state_dict, action_fn_args
            )
        else:
            self.c_attn_converter_cs17_to_hf(
                old_key, new_key, old_state_dict, new_state_dict, action_fn_args
            )

    def c_attn_converter_hf_to_cs17(
        self, old_key, new_key, old_state_dict, new_state_dict, action_fn_args
    ):
        # HF represents Q, K, and V in a packed format. We need to unpack the
        # weight and bias tensor for CS 1.7 format.
        q_key = new_key
        k_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_k_dense_layer.", q_key)
        v_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_v_dense_layer.", q_key)

        if new_key.endswith(".bias"):
            assert len(old_state_dict[old_key].shape) == 1
            packed_dim = old_state_dict[old_key].shape[0]
            embed_dim = packed_dim // 3
            assert 3 * embed_dim == packed_dim, (
                f"Invalid tensor shape {old_state_dict[old_key].shape} at {old_key}. Bias should "
                f"be divisible by 3 since Q, K, and V are packed."
            )

            (
                new_state_dict[q_key],
                new_state_dict[k_key],
                new_state_dict[v_key],
            ) = torch.chunk(old_state_dict[old_key], 3, dim=0)
        elif new_key.endswith(".weight"):
            embed_dim, packed_dim = old_state_dict[old_key].shape
            assert 3 * embed_dim == packed_dim, (
                f"Invalid tensor shape {old_state_dict[old_key].shape} at {old_key}. The second "
                f"dimension should be 3x the first dimension (embed_dim) since Q, K, and V are "
                f"packed."
            )
            (
                new_state_dict[q_key],
                new_state_dict[k_key],
                new_state_dict[v_key],
            ) = torch.chunk(
                torch.transpose(old_state_dict[old_key], 0, 1), 3, dim=0
            )
        else:
            raise ValueError("Invalid key after conversion: {}".format(new_key))

    def c_attn_converter_cs17_to_hf(
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

        # Need to transpose to convert from Linear.weight -> Conv1D.weight
        if len(new_state_dict[new_key].shape) == 2:
            new_state_dict[new_key] = torch.transpose(
                new_state_dict[new_key], 0, 1
            )

        if new_key.endswith(".bias") and self.generate_hf_biases:
            max_position_embeddings = action_fn_args["configs"][1]["model"][
                "max_position_embeddings"
            ]
            attn_bias_key = re.sub(r"\.c_attn\.", ".", new_key)
            new_state_dict[attn_bias_key] = torch.tril(
                torch.ones(
                    (max_position_embeddings, max_position_embeddings),
                    dtype=torch.uint8,
                )
            ).view(1, 1, max_position_embeddings, max_position_embeddings)
            masked_bias_key = re.sub(r"\.c_attn\.", ".masked_", new_key)
            new_state_dict[masked_bias_key] = torch.tensor(-1e4)

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


class Converter_GPT2Model_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("wte", "embedding_layer.word_embeddings"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
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
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("attn.", "self_attn."),
                    self.attention_converter_class(),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("ln_1", "norm1"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("ln_2", "norm3"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.c_fc", "ffn.ffn.0.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.ffn_converter(),
            ),
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.c_proj", "ffn.ffn.1.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.ffn_converter(),
            ),
            ConversionRule(
                [
                    EquivalentSubkey("ln_f", "transformer_decoder.norm"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replace_final_norm,
            ),
            ConversionRule([r"lm_head\.(?:weight|bias)"], exists="right"),
            ConversionRule([r"ln_f\.(?:weight|bias)"], exists="right"),
            ConversionRule(
                [
                    r"h\.\d+\.attn\.(?:masked_bias|bias)",
                ],
                exists="left",
            ),
        ]

    def attention_converter_class(self):
        # Allows other checkpoint converters to inherit from
        # this main converter but can overide this function with
        # different types of attention converters (i.e. MQA)
        return Converter_GPT2_Attention_HF_CS17()

    def ffn_converter(self):
        # similar to above, allows overriding method for other models
        # that use mostly GPT-2, but with slight changes
        return transpose_key_if_2D

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
                "{} GPT2 has a language model head (lm_head) "
                "while {} GPT2Model does not. Initializing lm_head to default.".format(
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
            # We are converting from HF GPT2Model (which is headless) -> CS GPT2LMHeadModel
            # We need to create 'lm_head' and init to default values
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
            "{} GPT2Model <-> {} GPT2LMHeadModel\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_HF_CS17


class Converter_GPT2LMHeadModel_HF_CS17(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["lm_head\.(?:weight|bias)"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("transformer.", ""),
                    Converter_GPT2Model_HF_CS17(),
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
        return "{} GPT2LMHeadModel <-> {} GPT2LMHeadModel".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_HF_CS17


class ConfigConverter_GPT2Model_HF_CS17(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "gpt2"),
            ),
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
            ConversionRule(
                ["embedding_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
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
                [EquivalentSubkey("scale_attn_weights", "attention_type")],
                action=self.convert_attention_type,
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
                [EquivalentSubkey("resid_pdrop", "dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(["rotary_dim"], action=self.replaceKey),
            ConversionRule(
                ["layer_norm_epsilon"],
                action=self.replaceKey,
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
                ["scale_attn_by_inverse_layer_idx"],
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(
                ["reorder_and_upcast_attn"],
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
        ]

        self.pre_convert_defaults[0].update(
            {
                "tie_word_embeddings": True,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "share_embedding_weights": True,
            },
        )
        self.post_convert_defaults[0].update({"model_type": "gpt2"})
        self.post_convert_defaults[1].update({"use_bias_in_output": False})

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
            if (
                old_state_dict[old_key] != "scaled_dot_product"
                and old_state_dict[old_key] != "dot_product"
            ):
                raise ConfigConversionError(
                    "Can't convert config with {}={}. Only {} is supported.".format(
                        old_key,
                        old_state_dict[old_key],
                        "scaled_dot_product and dot_product",
                    )
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
        else:
            if "embedding_dropout_rate" not in config:
                config["embedding_dropout_rate"] = config["dropout_rate"]
            if "attention_dropout_rate" not in config:
                config["attention_dropout_rate"] = config["dropout_rate"]
        return config

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))


#########################################################
# GPT2 HF <> CS18, CS19
#########################################################


class ConfigConverter_GPT2Model_HF_CS18(ConfigConverter_GPT2Model_HF_CS17):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

    def supports_mup_conversion(self):
        return True


Converter_GPT2Model_HF_CS18 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_GPT2Model_HF_CS18",
    Converter_GPT2Model_HF_CS17,
    derived_class=Converter_GPT2Model_HF_CS17,
    config_converter_class=ConfigConverter_GPT2Model_HF_CS18,
    formats=(FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9")),
    converter_note_fn=lambda cls: (
        "{} GPT2Model <-> {} GPT2LMHeadModel\n"
        "The HF model doesn't contain a language model head while the CS "
        "one does. When converting to CS, the exported checkpoint will "
        "contain a language model head initialized to default random "
        "values. When converting to HF, the language model head will be "
        "dropped."
    ).format(cls.formats()[0], cls.formats()[1]),
)


class Converter_GPT2LMHeadModel_HF_CS17(BaseCheckpointConverter_HF_CS):
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
                    Converter_GPT2Model_HF_CS17(),
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
        return "{} GPT2LMHeadModel <-> {} GPT2LMHeadModel".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_HF_CS17


class Converter_GPT2LMHeadModel_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_GPT2LMHeadModel_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_GPT2LMHeadModel_HF_CS17(),
                ],
                action=None,
            ),
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
        if converter_indices.direction == 0:
            lm_head_weight_key = key_prefix + "lm_head.weight"
            embed_key = key_prefix + "transformer.wte.weight"
            if lm_head_weight_key not in new_state_dict:
                new_state_dict[lm_head_weight_key] = old_state_dict[embed_key]
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
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPT2LMHeadModel <-> {} GPT2LMHeadModel".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_HF_CS18

    def supports_mup_conversion(self) -> bool:
        return True


###########################################################
# In CS 2.0, we changed introduced norm_type in the config.
# CS 1.8, CS 1.9 <> CS 2.0, and HF <> CS 2.0 converters:
###########################################################


class Converter_GPT2LMHeadModel_CS18_CS20(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Model didn't change between 1.8/1.9 and 2.0. Copy all keys.
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
        return "GPT2LMHeadModel class"

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.8", "cs-1.9"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_CS18_CS20


class ConfigConverter_GPT2Model_CS18_CS20(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Only difference between 1.8/1.9 and 2.0 is introduction of norm_type
        self.rules = [
            ConversionRule(
                [EquivalentSubkey("use_rms_norm", "norm_type")],
                action=self.convert_use_rms_layer_norm,
            ),
            ConversionRule([".*"], action=self.replaceKey),
        ]

        self.pre_convert_defaults[0]["use_rms_norm"] = False
        self.pre_convert_defaults[1]["norm_type"] = "layernorm"

    def convert_use_rms_layer_norm(self, *args):
        convert_use_rms_layer_norm_helper(self, *args)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.8", "cs-1.9"), FormatVersions("cs-2.0"))


class Converter_GPT2Model_HF_CS20(Converter_GPT2Model_HF_CS18):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_HF_CS20


class Converter_GPT2LMHeadModel_HF_CS20(Converter_GPT2LMHeadModel_HF_CS18):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_HF_CS20


class ConfigConverter_GPT2Model_HF_CS20(ConfigConverter_GPT2Model_HF_CS18):
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
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1, and HF <> CS 2.1 converters:
###########################################################


class Converter_GPT2LMHeadModel_CS20_CS21(BaseCheckpointConverter_CS_CS):
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
        return "GPT2LMHeadModel class"

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.0"), FormatVersions("cs-2.1"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_CS20_CS21


class ConfigConverter_GPT2Model_CS20_CS21(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.0"), FormatVersions("cs-2.1"))


class ConfigConverter_GPT2Model_HF_CS21(ConfigConverter_GPT2Model_HF_CS20):
    "CS 2.1 config is the same as CS 2.0."

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_GPT2Model_WithoutOptionalModel_HF_CS21(
    Converter_GPT2Model_HF_CS17
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "wpe", "embedding_layer.position_embeddings.embed"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            *self.rules,
        ]

    def supports_mup_conversion(self) -> bool:
        return True


Converter_GPT2Model_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_GPT2Model_HF_CS21",
    Converter_GPT2Model_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_GPT2Model_WithoutOptionalModel_HF_CS21,
    config_converter_class=ConfigConverter_GPT2Model_HF_CS21,
    formats=(
        FormatVersions("hf"),
        FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
    ),
)


class Converter_GPT2LMHeadModel_WithoutOptionalModel_HF_CS21(
    BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["lm_head\.(?:weight|bias)"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("transformer.", ""),
                    Converter_GPT2Model_WithoutOptionalModel_HF_CS21(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_HF_CS21

    def supports_mup_conversion(self) -> bool:
        return True


Converter_GPT2LMHeadModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_GPT2LMHeadModel_HF_CS21",
    Converter_GPT2LMHeadModel_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_GPT2LMHeadModel_WithoutOptionalModel_HF_CS21,
    converter_note_fn=lambda cls: "{} GPT2LMHeadModel <-> {} GPT2LMHeadModel".format(
        cls.formats()[0], cls.formats()[1]
    ),
)


class Converter_GPT2LMHeadModel_CS22_CS23(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @classmethod
    def converter_note(cls) -> str:
        return "GPT2LMHeadModel class"

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-2.2"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2Model_CS22_CS23


class ConfigConverter_GPT2Model_CS22_CS23(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # The following params were deprecated:
            ConversionRule(["use_position_embedding"], action=None),
            ConversionRule(["alibi_implementation"], action=None),
            ConversionRule(["weight_initialization_seed"], action=None),
            # Remaining params can be copied:
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("cs-2.2"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )
