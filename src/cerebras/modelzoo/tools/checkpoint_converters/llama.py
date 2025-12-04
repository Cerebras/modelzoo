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

import copy
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
from cerebras.modelzoo.tools.checkpoint_converters.gpt2_hf_cs import (
    ConfigConverter_GPT2Model_CS18_CS20,
    Converter_GPT2LMHeadModel_CS18_CS20,
    Converter_GPT2LMHeadModel_CS20_CS21,
)


class Converter_LlamaAttention_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("q_proj", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_with_interleaving_query,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("k_proj", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_with_interleaving_key,
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
                    EquivalentSubkey("o_proj", "proj_output_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    def convert_with_interleaving_query(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # Query & Keys should be interleaved since HF and CS RoPE differ
        cs_config = action_fn_args["configs"][1]
        tensor = old_state_dict[old_key]
        initial_shape = tensor.size()
        num_heads = cs_config["model"]["num_heads"]

        if from_index == 0:
            if len(tensor.size()) == 2:
                tensor = tensor.view(
                    num_heads, tensor.size(0) // num_heads, tensor.size(-1)
                )
            elif len(tensor.size()) == 1:
                tensor = tensor.view(num_heads, tensor.size(0) // num_heads)
            tensor = self.interleave_helper(tensor, cs_config)
        else:
            tensor = self.reverse_interleave_helper(
                tensor, cs_config, num_heads
            )
        tensor = tensor.view(*initial_shape)
        new_state_dict[new_key] = tensor

    def convert_with_interleaving_key(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # Query & Keys should be interleaved since HF and CS RoPE differ
        cs_config = action_fn_args["configs"][1]

        if (
            cs_config["model"].get("attention_module", "aiayn_attention")
            == "aiayn_attention"
        ):
            self.convert_with_interleaving_query(
                old_key,
                new_key,
                old_state_dict,
                new_state_dict,
                from_index,
                action_fn_args,
            )
            return
        elif cs_config["model"]["attention_module"] == "multiquery_attention":
            tensor = old_state_dict[old_key]
            initial_shape = tensor.size()
            num_group = cs_config["model"]["extra_attention_params"][
                "num_kv_groups"
            ]

            if from_index == 0:
                if len(tensor.size()) == 2:
                    tensor = tensor.view(
                        num_group, tensor.size(0) // num_group, tensor.size(-1)
                    )
                elif len(tensor.size()) == 1:
                    tensor = tensor.view(num_group, tensor.size(0) // num_group)
                tensor = self.interleave_helper(tensor, cs_config)
            else:
                tensor = self.reverse_interleave_helper(
                    tensor, cs_config, num_group
                )
            tensor = tensor.view(*initial_shape)
            new_state_dict[new_key] = tensor
        else:
            assert False, (
                f"attention_module {cs_config['model']['attention_module']} is not supported for "
                f"llama"
            )

    def interleave_helper(self, t, cs_config):
        rotary_dim = cs_config["model"]["rotary_dim"]
        if len(t.shape) == 3:
            to_rotate = t[:, :rotary_dim, :]
            to_pass = t[:, rotary_dim:, :]
            to_rotate = (
                to_rotate.reshape(t.shape[0], 2, -1, t.shape[-1])
                .permute(0, 2, 1, 3)
                .reshape(t.shape[0], -1, t.shape[-1])
            )
            interleaved = torch.cat((to_rotate, to_pass), dim=1)
        elif len(t.shape) == 2:
            to_rotate = t[:, :rotary_dim]
            to_pass = t[:, rotary_dim:]
            to_rotate = (
                to_rotate.reshape(t.shape[0], 2, -1)
                .permute(0, 2, 1)
                .reshape(t.shape[0], -1)
            )
            interleaved = torch.cat((to_rotate, to_pass), dim=1)
        else:
            assert False, (
                "shape of query, key, value projection tensor has to have shape of length 2 "
                "(biases) or 3 (weights) when converting from HF to CS."
            )
        return interleaved

    def reverse_interleave_helper(self, t, cs_config, num_heads):
        rotary_dim = cs_config["model"]["rotary_dim"]
        if len(t.shape) == 2:
            t = t.reshape(num_heads, -1, t.shape[-1])
            to_rotate = t[:, :rotary_dim, :]
            to_pass = t[:, rotary_dim:, :]
            # pylint: disable=redefined-builtin
            reversed = (
                to_rotate.reshape(num_heads, -1, 2, t.shape[-1])
                .permute(0, 2, 1, 3)
                .reshape(num_heads, rotary_dim, t.shape[-1])
            )
            reversed = torch.cat((reversed, to_pass), dim=1)
        elif len(t.shape) == 1:
            t = t.reshape(num_heads, -1)
            to_rotate = t[:, :rotary_dim]
            to_pass = t[:, rotary_dim:]
            reversed = (
                to_rotate.reshape(num_heads, -1, 2)
                .permute(0, 2, 1)
                .reshape(num_heads, -1)
            )
            reversed = torch.cat((reversed, to_pass), dim=1)
        else:
            assert False, (
                "shape of query, key, value projection tensor has to have shape of length 1 "
                "(biases) or 2 (weights) when converting from CS to HF."
            )
        return reversed

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None


class Converter_LlamaModel_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # word embeddings
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embed_tokens", "embedding_layer.word_embeddings"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # final layer norm
            ConversionRule(
                [
                    EquivalentSubkey("norm", "transformer_decoder.norm"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replace_final_norm,
            ),
            # attention
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.self_attn\.",
                    Converter_LlamaAttention_HF_CS(),
                ],
                action=None,
            ),
            # Rotary embedding
            ConversionRule(
                [r"layers\.\d+\.self_attn\.rotary_emb\.inv_freq"],
                exists="left",
                action=None,
            ),
            # attention norm
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("input_layernorm", "norm1"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("post_attention_layernorm", "norm3"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # intermediate ffn
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.up_proj", "ffn.ffn.0.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "mlp.gate_proj", "ffn.ffn.0.linear_layer_for_glu"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.down_proj", "ffn.ffn.1.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule([r"lm_head\.(?:weight|bias)"], exists="right"),
            ConversionRule([r"ln_f\.(?:weight|bias)"], exists="right"),
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
            # We are converting from HF LlamaModel (which is headless) ->
            # CS GPT2LMHeadModel configured as llama (which has a head)
            # We need to create 'lm_head' and init to default values
            logging.warning(
                f"{self.formats()[1]} has a language model head (lm_head) "
                f"while {self.formats()[0]} does not. Initializing lm_head to default."
            )
            hf_config = configs[0]
            cs_config = configs[1]
            use_bias_in_output = cs_config["model"].get(
                "use_bias_in_output", False
            )
            vocab_size = cs_config["model"]["vocab_size"]
            embed_dim = cs_config["model"]["hidden_size"]
            if hf_config["tie_word_embeddings"]:
                lm_head_weight = old_state_dict['embed_tokens.weight']
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
        return (FormatVersions("hf"), FormatVersions("cs"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None


class Converter_LlamaModel_HF_CS19(Converter_LlamaModel_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_LlamaModel_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_LlamaModel_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            f"{cls.formats()[0]} LlamaModel <-> {cls.formats()[1]} GPT2LMHeadModel (configured as "
            f"Llama)\nThe HF model doesn't contain a language model head while the CS one does. "
            f"When converting to CS, the exported checkpoint will contain a language model head "
            f"initialized to default random values. When converting to HF, the language model head "
            f"will be dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaMa_HF_CS19


class Converter_LlamaForCausalLM_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [r"lm_head\.(?:weight|bias)"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("model.", ""),
                    Converter_LlamaModel_HF_CS(),
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


class Converter_LlamaForCausalLM_HF_CS19(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_LlamaForCausalLM_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_LlamaForCausalLM_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} LlamaForCausalLM <-> {} GPT2LMHeadModel (configured as Llama)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaMa_HF_CS19


class ConfigConverter_LLaMa_HF_CS19(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        if not hasattr(self, "model_type"):
            self.model_type = "llama"
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
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, "rotary"),
            ),
            ConversionRule(
                ["use_position_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["embedding_dropout_rate"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, 0.0),
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
                ["max_position_embeddings"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["attention_type"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(
                    1, "scaled_dot_product"
                ),
            ),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                exists="right",
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
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                [EquivalentSubkey("intermediate_size", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("hidden_act", "nonlinearity")],
                action=self.convert_nonlinearity,
            ),
            ConversionRule(
                ["attention_dropout_rate"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, 0.0),
            ),
            ConversionRule(
                ["dropout_rate"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, 0.0),
            ),
            ConversionRule(
                ["rotary_dim"], exists="right", action=self.assert_rotary_dim
            ),
            ConversionRule(["rope_theta"], action=self.replaceKey),
            ConversionRule(
                [EquivalentSubkey("rms_norm_eps", "layer_norm_epsilon")],
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
                ["use_rms_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
        ]

        self.pre_convert_defaults[0].update(
            {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "hidden_act": "silu",
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": False,
                "max_position_embeddings": 2048,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "share_embedding_weights": True,
                "use_rms_norm": False,
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

        self.post_convert_defaults[0].update({"model_type": "llama"})
        self.post_convert_defaults[1].update(
            {
                "use_position_embedding": True,
                "position_embedding_type": "rotary",
                "embedding_dropout_rate": 0.0,
                "embedding_layer_norm": False,
                "attention_type": "scaled_dot_product",
                "use_projection_bias_in_attention": False,
                "use_ffn_bias_in_attention": False,
                "use_ffn_bias": False,
                "attention_dropout_rate": 0.0,
                "dropout_rate": 0.0,
                "use_bias_in_output": False,
                "norm_first": True,
                "use_ff_layer1_dropout": False,
                "use_rms_norm": True,
            },
        )

    def convert_nonlinearity(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        activation = old_state_dict[old_key]
        if from_index == 0:
            gated_hf2cs = {"silu": "swiglu", "relu": "reglu", "gelu": "geglu"}
            if activation not in gated_hf2cs:
                raise ConfigConversionError(
                    "{} is not a GLU-able activation in CS".format(activation)
                )
            activation = gated_hf2cs[activation]
        elif from_index == 1:
            gated_cs2hf = {"swiglu": "silu", "reglu": "relu", "geglu": "gelu"}
            if activation not in gated_cs2hf:
                raise ConfigConversionError(
                    "{} is not a supported GLU activation in HF".format(
                        activation
                    )
                )
            activation = gated_cs2hf[activation]

        new_state_dict[new_key] = activation

    def assert_rotary_dim(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert from_index == 1, "{} should only exist in CS config".format(
            old_key
        )
        if (
            old_state_dict[old_key]
            != old_state_dict["hidden_size"] // old_state_dict["num_heads"]
        ):
            raise ConfigConversionError(
                "rotary_dim must be hidden_size // num_heads in order to be compatible with HF"
            )

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        if converter_indices.direction == 1 and (
            "rotary_dim" not in config or config["rotary_dim"] is None
        ):
            raise ConfigConversionError("rotary_dim must be specified")

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
            new_config["rotary_dim"] = (
                new_config["hidden_size"] // new_config["num_heads"]
            )

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
        return (FormatVersions("hf"), FormatVersions("cs-1.9"))


class Converter_LlamaForCausalLM_CS19_CS20(Converter_GPT2LMHeadModel_CS18_CS20):
    r"""
    Llama uses the GPT2 backbone.
    """

    @classmethod
    def converter_note(cls) -> str:
        return "GPT2LMHeadModel class (configured as Llama)"

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.9"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LlamaModel_CS19_CS20


class ConfigConverter_LlamaModel_CS19_CS20(ConfigConverter_GPT2Model_CS18_CS20):
    r"""
    Llama uses the GPT2 backbone.
    """

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.9"), FormatVersions("cs-2.0"))


class Converter_LlamaModel_HF_CS20(Converter_LlamaModel_HF_CS19):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaMa_HF_CS20


class Converter_LlamaForCausalLM_HF_CS20(Converter_LlamaForCausalLM_HF_CS19):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaMa_HF_CS20


class ConfigConverter_LLaMa_HF_CS20(ConfigConverter_LLaMa_HF_CS19):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["norm_type"],
                action=BaseConfigConverter.assert_factory_fn(1, "rmsnorm"),
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "num_key_value_heads", "extra_attention_params"
                    )
                ],
                action=self.convert_gqa,
            ),
            *self.rules,
        ]

        del self.pre_convert_defaults[1]["use_rms_norm"]
        del self.post_convert_defaults[1]["use_rms_norm"]
        self.pre_convert_defaults[1]["norm_type"] = "layernorm"
        self.post_convert_defaults[1]["norm_type"] = "rmsnorm"
        del self.post_convert_defaults[1]["use_position_embedding"]

    def convert_gqa(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            # check mha or gqa
            if old_state_dict[old_key] == old_state_dict["num_attention_heads"]:
                new_state_dict["attention_module"] = "aiayn_attention"
            else:
                assert (
                    old_state_dict["num_attention_heads"]
                    % old_state_dict[old_key]
                    == 0
                ), (
                    f"number of attention heads should be divisible by num_key_value_heads but "
                    f"got {old_state_dict['num_attention_heads']} and {old_state_dict[old_key]},"
                )
                extra = {"num_kv_groups": old_state_dict[old_key]}
                new_state_dict[new_key] = extra
                new_state_dict["attention_module"] = "multiquery_attention"
        elif from_index == 1:
            if (
                old_state_dict.get("attention_module", "aiayn_attention")
                == "aiayn_attention"
            ):
                assert (
                    old_key not in old_state_dict
                    or "num_kv_groups" not in old_state_dict[old_key]
                ), "Conflict between use of multi-query and multi-head attention"
                new_state_dict[new_key] = old_state_dict["num_heads"]
            elif old_state_dict["attention_module"] == "multiquery_attention":
                num_heads = old_state_dict["num_heads"]
                num_kv_groups = old_state_dict[old_key]["num_kv_groups"]
                assert num_heads % num_kv_groups == 0, (
                    f"number of attention heads should be divisible by num_key_value_heads but "
                    f"got {num_heads} and {num_kv_groups}."
                )
                new_state_dict[new_key] = old_state_dict[old_key][
                    "num_kv_groups"
                ]
            else:
                assert False, (
                    f"attention_module {old_state_dict['attention_module']} is not supported for "
                    f"llama"
                )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1. We don't need a separate HF <> CS 2.1 converters since
# HF only supports RoPE which doesn't produce any checkpoint keys.
###########################################################


class Converter_LlamaForCausalLM_CS20_CS21(Converter_GPT2LMHeadModel_CS20_CS21):
    @classmethod
    def converter_note(cls) -> str:
        return "GPT2LMHeadModel class (configured as Llama)"


class Converter_LlamaModel_HF_CS21(Converter_LlamaModel_HF_CS20):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaMa_HF_CS21


class Converter_LlamaForCausalLM_HF_CS21(Converter_LlamaForCausalLM_HF_CS20):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaMa_HF_CS21

    def supports_mup_conversion(self):
        return True


class ConfigConverter_LLaMa_HF_CS21(ConfigConverter_LLaMa_HF_CS20):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [EquivalentSubkey("rope_scaling", "pos_scaling_factor")],
                action=self.convert_pi,
            ),
            ConversionRule(
                [EquivalentSubkey("", "pos_scaling_extra_args")],
                action=None,
            ),
            ConversionRule(
                [EquivalentSubkey("", "pos_scaling_type")],
                action=None,
            ),
            *self.rules,
        ]

        self.pre_convert_defaults[0].update(
            {
                "rope_scaling": None,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "pos_scaling_factor": 1.0,
            },
        )

    def convert_pi(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            if old_state_dict[old_key] is None:
                new_state_dict[new_key] = 1.0
            else:
                if "type" in old_state_dict[old_key]:
                    scaling_type = old_state_dict[old_key]["type"].lower()
                else:
                    scaling_type = old_state_dict[old_key]["rope_type"].lower()
                if scaling_type not in ["linear", "yarn", "llama3", "longrope"]:
                    raise ConfigConversionError(
                        f"Only `rope_scaling` type `linear`, `yarn`,`llama3` or 'longrope' are currently supported, "
                        f"but got type `{scaling_type}`."
                    )
                new_state_dict[new_key] = old_state_dict[old_key]["factor"]
                new_state_dict["pos_scaling_type"] = scaling_type
                if scaling_type == "yarn":
                    new_state_dict["pos_scaling_extra_args"] = dict(
                        {
                            "original_max_position_embeddings": old_state_dict[
                                old_key
                            ]["original_max_position_embeddings"],
                        }
                    )
                elif scaling_type == "llama3":
                    pos_scaling_extra_args = copy.deepcopy(
                        old_state_dict[old_key]
                    )
                    pos_scaling_extra_args.pop("rope_type")
                    pos_scaling_extra_args.pop("factor")
                    new_state_dict["pos_scaling_extra_args"] = (
                        pos_scaling_extra_args
                    )
                elif scaling_type == "longrope":
                    # Create a copy of the remaining extra params of original dictionary and add to pos_scaling_extra_args
                    # longrope uses param names which are HF compatible
                    extra_args = {
                        **{
                            k: v
                            for k, v in old_state_dict[old_key].items()
                            if k not in ["type", "factor"]
                        }
                    }
                    new_state_dict["pos_scaling_extra_args"] = extra_args

        else:
            if old_state_dict[old_key] == 1.0:
                new_state_dict[new_key] = None
            else:
                type_key = "type"
                if old_state_dict["pos_scaling_type"] == "llama3":
                    type_key = "rope_type"
                new_state_dict[new_key] = {
                    type_key: old_state_dict.get("pos_scaling_type", "linear"),
                    "factor": old_state_dict[old_key],
                }
                if "pos_scaling_extra_args" in old_state_dict:
                    new_state_dict[new_key].update(
                        old_state_dict["pos_scaling_extra_args"]
                    )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    def supports_mup_conversion(self):
        return True
