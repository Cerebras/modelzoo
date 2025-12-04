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
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    ConfigConverter_LLaMa_HF_CS21,
    Converter_LlamaAttention_HF_CS,
)

MAGIC_STR = "__"


class Converter_MixtralModel_HF_CS(BaseCheckpointConverter_HF_CS):
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
            # moe ffn
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("block_sparse_moe.gate", "ffn.gate"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            *self.moe_rules(),
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

    def moe_rules(self):
        return self.moe_optimized_impl_rules()

    def moe_functional_impl_rules(self):
        return [
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts", "ffn.experts.experts"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("w1", "ffn.0.linear_layer_for_glu"),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts", "ffn.experts.experts"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("w3", "ffn.0.linear_layer"),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts", "ffn.experts.experts"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("w2", "ffn.1.linear_layer"),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
        ]

    def moe_optimized_impl_rules(self):
        return [
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts",
                        "ffn.experts",
                    ),
                    EquivalentSubkey(
                        ".0.w1.weight",
                        ".fused_ffns.0.linear_layer_for_glu.expert_weights",
                    ),
                ],
                action=self.convert_expert_weights,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts",
                        f"ffn.experts{MAGIC_STR}",
                    ),
                    r"\.\d+",
                    EquivalentSubkey(
                        ".w1.weight",
                        ".fused_ffns.0.linear_layer_for_glu.expert_weights",
                    ),
                ],
                action=self.assert_already_converted,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts",
                        "ffn.experts",
                    ),
                    EquivalentSubkey(
                        ".0.w3.weight",
                        ".fused_ffns.0.linear_layer.expert_weights",
                    ),
                ],
                action=self.convert_expert_weights,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts",
                        f"ffn.experts{MAGIC_STR}",
                    ),
                    r"\.\d+",
                    EquivalentSubkey(
                        ".w3.weight",
                        ".fused_ffns.0.linear_layer.expert_weights",
                    ),
                ],
                action=self.assert_already_converted,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts",
                        "ffn.experts",
                    ),
                    EquivalentSubkey(
                        ".0.w2.weight",
                        ".fused_ffns.1.linear_layer.expert_weights",
                    ),
                ],
                action=self.convert_expert_weights,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "block_sparse_moe.experts",
                        f"ffn.experts{MAGIC_STR}",
                    ),
                    r"\.\d+",
                    EquivalentSubkey(
                        ".w2.weight",
                        ".fused_ffns.0.linear_layer.expert_weights",
                    ),
                ],
                action=self.assert_already_converted,
            ),
        ]

    def convert_expert_weights(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        num_experts = action_fn_args['configs'][1]['model']['moe'][
            'num_experts'
        ]
        if from_index == 0:
            # Fuse weights across experts.
            expert_weights = []
            for expert in range(num_experts):
                curr_old_key = re.sub(
                    r"experts\.0", f"experts.{expert}", old_key
                )
                expert_weights.append(old_state_dict[curr_old_key])

            expert_weights = [v.unsqueeze(1) for v in expert_weights]
            new_state_dict[new_key] = torch.concat(expert_weights, dim=1)
        else:
            # Unfuse weights.
            for expert in range(num_experts):
                curr_new_key = re.sub(
                    r"experts\.0", f"experts.{expert}", new_key
                )
                new_state_dict[curr_new_key] = old_state_dict[old_key][
                    :, expert
                ]

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
            new_key = re.sub(f"{MAGIC_STR}.\d+", "", new_key)
            assert (
                new_key in new_state_dict
            ), f"Expected {new_key} to be in new_state_dict"
        else:
            assert False, "Unreachable"

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


class Converter_MixtralForCausalLM_HF_CS(BaseCheckpointConverter_HF_CS):
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
                    Converter_MixtralModel_HF_CS(),
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


class Converter_MixtralModel_WithoutOptionalModel_HF_CS23(
    Converter_MixtralModel_HF_CS
):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Mixtral_HF_CS23

    @classmethod
    def converter_note(cls) -> str:
        return (
            f"{cls.formats()[0]} MixtralModel <-> {cls.formats()[1]} GPT2LMHeadModel (configured as "
            f"Mixtral)\nThe HF model doesn't contain a language model head while the CS one does. "
            f"When converting to CS, the exported checkpoint will contain a language model head "
            f"initialized to default random values. When converting to HF, the language model head "
            f"will be dropped."
        ).format(cls.formats()[0], cls.formats()[1])


class Converter_MixtralLMHeadModel_WithoutOptionalModel_HF_CS23(
    Converter_MixtralForCausalLM_HF_CS
):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Mixtral_HF_CS23

    @classmethod
    def converter_note(cls) -> str:
        return "{} MixtralForCausalLM <-> {} GPT2LMHeadModel (configured as Mixtral)".format(
            cls.formats()[0], cls.formats()[1]
        )


class ConfigConverter_Mixtral_HF_CS23(ConfigConverter_LLaMa_HF_CS21):
    def __init__(self):
        self.model_type = "mixtral"
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "sliding_window", "attention_sliding_window_length"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("num_local_experts", "moe")],
                action=self.convert_moe_params,
            ),
            *self.rules,
        ]

        self.post_convert_defaults[0].update({"model_type": "mixtral"})

    def convert_moe_params(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            if "moe" not in new_state_dict:
                new_state_dict["moe"] = {}
            new_state_dict["moe"]["num_experts"] = old_state_dict[
                "num_local_experts"
            ]
            new_state_dict["moe"]["top_k"] = old_state_dict[
                "num_experts_per_tok"
            ]
            new_state_dict["moe"]["load_balancing_loss_coef"] = old_state_dict[
                "router_aux_loss_coef"
            ]
        else:
            new_state_dict["num_local_experts"] = old_state_dict["moe"][
                "num_experts"
            ]
            new_state_dict["num_experts_per_tok"] = old_state_dict["moe"][
                "top_k"
            ]
            new_state_dict["router_aux_loss_coef"] = old_state_dict["moe"][
                "load_balancing_loss_coef"
            ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


Converter_MixtralModel_HF_CS23 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_MixtralModel_HF_CS23",
    Converter_MixtralModel_WithoutOptionalModel_HF_CS23,
    derived_class=Converter_MixtralModel_WithoutOptionalModel_HF_CS23,
)

Converter_MixtralForCausalLM_HF_CS23 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_MixtralForCausalLM_HF_CS23",
    Converter_MixtralLMHeadModel_WithoutOptionalModel_HF_CS23,
    derived_class=Converter_MixtralLMHeadModel_WithoutOptionalModel_HF_CS23,
)
