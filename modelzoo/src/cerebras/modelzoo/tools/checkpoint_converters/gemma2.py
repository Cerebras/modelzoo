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
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    Converter_LlamaAttention_HF_CS,
)

#########################################################
# Gemma2 HF <> CS2.3.1
#########################################################


class Converter_Gemma2_WithoutOptionalModel_HF_CS23(
    BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.embed_tokens", "embedding_layer.word_embeddings"
                    ),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.layers", "transformer_decoder.layers"
                    ),
                    r"\.\d+\.self_attn\.",
                    Converter_LlamaAttention_HF_CS(),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.layers", "transformer_decoder.layers"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "mlp.gate_proj", "ffn.ffn.0.linear_layer_for_glu"
                    ),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.layers", "transformer_decoder.layers"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.up_proj", "ffn.ffn.0.linear_layer"),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.layers", "transformer_decoder.layers"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.down_proj", "ffn.ffn.1.linear_layer"),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.layers", "transformer_decoder.layers"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("input_layernorm", "norm1"),
                    r"\.weight",
                ],
                action=self.convert_layer_norm,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.layers", "transformer_decoder.layers"
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("post_attention_layernorm", "norm1_post"),
                    r"\.weight",
                ],
                action=self.convert_layer_norm,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.layers", "transformer_decoder.layers"
                    ),
                    "\.\d+\.",
                    EquivalentSubkey("pre_feedforward_layernorm", "norm3"),
                    r"\.weight",
                ],
                action=self.convert_layer_norm,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.layers", "transformer_decoder.layers"
                    ),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "post_feedforward_layernorm", "norm3_post"
                    ),
                    r"\.weight",
                ],
                action=self.convert_layer_norm,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("model.norm", "transformer_decoder.norm"),
                    r"\.weight",
                ],
                action=self.convert_layer_norm,
            ),
            ConversionRule(
                [
                    r"lm_head\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule([r"ln_f\.weight"]),
        ]

    def convert_layer_norm(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        weight = old_state_dict[old_key]

        # Gemma2 HF implementation has a constant (1) offset unlike other
        # implementations. See https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modeling_gemma2.py#L70
        # for details. The constant offset can be folded/unfolded into the
        # weight in order to avoid a model implementation change.
        if from_index == 0:
            weight = weight + torch.ones_like(weight)
        else:
            weight = weight - torch.ones_like(weight)

        new_state_dict[new_key] = weight

        # Since CS 1.7, our model implementations store the final layernorm
        # twice ("ln_f" and "transformer_decoder.norm"). As a result, we need to
        # copy "ln_f".
        if from_index == 0 and new_key.find("layers") == -1:
            ln_f_key = re.sub(r"transformer_decoder\.norm\.", "ln_f.", new_key)
            new_state_dict[ln_f_key] = weight

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} Gemma2ForCausalLM <-> {} GPT2LMHeadModel (configured as Gemma2)"
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Gemma2_HF_CS23


class ConfigConverter_Gemma2_HF_CS23(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=self.assert_factory_fn(0, "gemma2"),
            ),
            # Parameters that are in both HF and CS:
            ConversionRule(
                [EquivalentSubkey("vocab_size", "vocab_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("hidden_size", "hidden_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("num_hidden_layers", "num_hidden_layers")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("num_attention_heads", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "num_key_value_heads", "extra_attention_params"
                    )
                ],
                action=self.convert_gqa,
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
                [EquivalentSubkey("rms_norm_eps", "layer_norm_epsilon")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("rope_theta", "rope_theta")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "final_logit_softcapping", "final_logit_softcapping"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attn_logit_softcapping", "attention_logit_softcapping"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "sliding_window", "attention_sliding_window_length"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("hidden_activation", "nonlinearity")],
                action=self.convert_nonlinearity,
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
                [EquivalentSubkey("intermediate_size", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("head_dim", "attention_inner_dim")],
                action=self.convert_head_dim,
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
                [
                    EquivalentSubkey(
                        "attention_bias", "use_projection_bias_in_attention"
                    )
                ],
                exists="left",
                action=self.convert_attention_bias,
            ),
            # Parameters that are only in HF:
            ConversionRule(
                ["eos_token_id"],
                exists="left",
                action=None,
            ),
            ConversionRule(
                ["bos_token_id"],
                exists="left",
                action=None,
            ),
            ConversionRule(
                ["cache_implementation"],
                exists="left",
                action=None,
            ),
            ConversionRule(
                ["use_cache"],
                exists="left",
                action=None,
            ),
            ConversionRule(
                ["pad_token_id"],
                exists="left",
                action=None,
            ),
            ConversionRule(
                ["initializer_range"],
                exists="left",
                action=None,
            ),
            # Parameters that are only in CS:
            ConversionRule(
                ["position_embedding_type"],
                exists="right",
                action=self.assert_factory_fn(1, "rotary"),
            ),
            ConversionRule(
                ["rotary_dim"],
                exists="right",
                action=self.assert_rotary_dim,
            ),
            ConversionRule(
                ["dropout_rate"],
                exists="right",
                action=self.assert_factory_fn(1, 0.0),
            ),
            ConversionRule(
                ["sliding_window_every_other_decoder_layer"],
                exists="right",
                action=self.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["norm_first"],
                exists="right",
                action=self.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["norm_first_sandwich"],
                exists="right",
                action=self.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_bias_in_output"],
                exists="right",
                action=self.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["embedding_dropout_rate"],
                exists="right",
                action=self.assert_factory_fn(1, [None, 0.0]),
            ),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                exists="right",
                action=self.assert_factory_fn(1, True),  # Verify this!
            ),
            ConversionRule(
                ["position_embedding_offset"],
                exists="right",
                action=self.assert_factory_fn(1, 0),
            ),
            ConversionRule(
                ["use_ffn_bias"],
                exists="right",
                action=self.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["embedding_layer_norm"],
                exists="right",
                action=self.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["attention_type"],
                exists="right",
                action=self.assert_factory_fn(1, 'scaled_dot_product'),
            ),
            ConversionRule(
                ["norm_type"],
                exists="right",
                action=self.assert_factory_fn(1, 'rmsnorm'),
            ),
            ConversionRule(
                ["use_ff_layer1_dropout"],
                exists="right",
                action=self.assert_factory_fn(1, False),  # Verify this!
            ),
        ]

        # HF config class defaults:
        self.pre_convert_defaults[0].update(
            {
                "vocab_size": 256000,
                "hidden_size": 3072,
                "intermediate_size": 24576,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "head_dim": 256,
                "hidden_activation": "gelu_pytorch_tanh",
                "max_position_embeddings": 8192,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-06,
                "use_cache": True,
                "pad_token_id": 0,
                "eos_token_id": 1,
                "bos_token_id": 2,
                "tie_word_embeddings": True,
                "rope_theta": 10000.0,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "final_logit_softcapping": 30.0,
                "attn_logit_softcapping": 50.0,
                "query_pre_attn_scalar": 224,
                "sliding_window": 4096,
            }
        )
        self.post_convert_defaults[0].update({"model_type": "gemma2"})

        # CS config class defaults:
        self.pre_convert_defaults[1].update(
            {
                "embeddings_scale": 1.0,
                "embedding_layer_norm": False,
                "embedding_dropout_rate": None,
                "share_embedding_weights": True,
                "position_embedding_type": 'learned',
                "max_position_embeddings": 1024,
                "position_embedding_offset": 0,
                "num_relative_attention_buckets": 32,
                "rotary_dim": None,
                "rope_theta": 10000,
                "alibi_trainable_slopes": False,
                "pos_scaling_factor": 1.0,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "dropout_rate": 0.1,
                "norm_type": 'layernorm',
                "layer_norm_epsilon": 1e-5,
                "norm_first": True,
                "norm_first_sandwich": False,
                "num_heads": 12,
                "attention_module": 'aiayn_attention',
                "extra_attention_params": {},
                "attention_type": 'scaled_dot_product',
                "attention_dropout_rate": None,
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "attention_sliding_window_length": None,
                "sliding_window_every_other_decoder_layer": False,
                "attention_sink_tokens": None,
                "attention_qk_norm_layer": None,
                "attention_qk_norm_eps": 1e-5,
                "attention_inner_dim": None,
                "scale_qk_dot_by_layer_idx": False,
                "attention_logit_softcapping": None,
                "filter_size": 3072,
                "nonlinearity": 'gelu',
                "use_ffn_bias": True,
                "use_bias_in_output": False,
                "use_ff_layer1_dropout": False,
                "final_logit_softcapping": None,
                "attention_logits_alpha": 1,
            }
        )
        self.post_convert_defaults[1].update(
            {
                "norm_first": True,
                "norm_first_sandwich": True,
                "sliding_window_every_other_decoder_layer": True,
                "position_embedding_type": "rotary",
                "norm_type": "rmsnorm",
                "use_ffn_bias": False,
                "dropout_rate": 0.0,
            }
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

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
                    f"gemma2"
                )

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
            != old_state_dict["attention_inner_dim"]
            // old_state_dict["num_heads"]
        ):
            raise ConfigConversionError(
                "rotary_dim must be attention_inner_dim // num_heads in order to be compatible with HF"
            )

    def convert_head_dim(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            # attention_inner_dim = head_dim * num_attention_heads
            new_state_dict[new_key] = (
                old_state_dict[old_key] * old_state_dict["num_attention_heads"]
            )
        else:
            # head_dim = attention_inner_dim // num_attention_heads
            attention_inner_dim = old_state_dict[old_key]
            if attention_inner_dim is None:
                attention_inner_dim = old_state_dict["hidden_size"]

            if attention_inner_dim % old_state_dict["num_heads"] != 0:
                raise ConfigConversionError(
                    "attention_inner_dim must be divisible by num_heads"
                )
            new_state_dict[new_key] = (
                old_state_dict[old_key] // old_state_dict["num_heads"]
            )

    def convert_attention_bias(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            # attention_bias -> use_projection_bias_in_attention, use_ffn_bias_in_attention
            new_state_dict["use_projection_bias_in_attention"] = old_state_dict[
                old_key
            ]
            new_state_dict["use_ffn_bias_in_attention"] = old_state_dict[
                old_key
            ]
        else:
            # use_projection_bias_in_attention, use_ffn_bias_in_attention -> attention_bias
            assert (
                old_state_dict["use_ffn_bias_in_attention"]
                == old_state_dict["use_projection_bias_in_attention"]
            )
            new_state_dict[new_key] = old_state_dict[old_key]

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
            if activation.startswith("gelu_"):
                activation = "gelu"
            gated_hf2cs = {
                "silu": "swiglu",
                "relu": "reglu",
                "gelu": "geglu",
                "gelu": "geglu",
            }
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

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        if converter_indices.direction == 1:
            exception = None
            try:
                torch.testing.assert_close(
                    config["embeddings_scale"],
                    config["hidden_size"] ** 0.5,
                    rtol=2.0e-7,
                    atol=1e-6,
                )
            except Exception as e:
                exception = e
            # Reraise the exception as a config conversion error
            if exception:
                raise ConfigConversionError(
                    "embeddings_scale must be equal to hidden_size**0.5\n"
                    + str(exception)
                )

        if converter_indices.direction == 1 and (
            "attention_inner_dim" not in config
            or config["attention_inner_dim"] is None
        ):
            config["attention_inner_dim"] = config["hidden_size"]

        if converter_indices.direction == 1 and (
            "rotary_dim" not in config or config["rotary_dim"] is None
        ):
            raise ConfigConversionError("rotary_dim must be specified")

        if (
            converter_indices.direction == 1
            and config["attention_dropout_rate"] is None
        ):
            config["attention_dropout_rate"] = config["dropout_rate"]

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
            new_config["rotary_dim"] = old_config["head_dim"]
            new_config["embeddings_scale"] = new_config["hidden_size"] ** 0.5

            attention_head_size = (
                new_config["attention_inner_dim"] / new_config["num_heads"]
            )
            if attention_head_size != old_config["query_pre_attn_scalar"]:
                new_config["attention_logits_alpha"] = (
                    attention_head_size / old_config["query_pre_attn_scalar"]
                ) ** 0.5
        else:
            attention_head_size = (
                old_config["attention_inner_dim"] / old_config["num_heads"]
            )
            new_config["query_pre_attn_scalar"] = (
                attention_head_size / old_config["attention_logits_alpha"] ** 2
            )

        return super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )


Converter_Gemma2ForCausalLM_HF_CS23 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_Gemma2ForCausalLM_HF_CS23",
    Converter_Gemma2_WithoutOptionalModel_HF_CS23,
    derived_class=Converter_Gemma2_WithoutOptionalModel_HF_CS23,
    config_converter_class=ConfigConverter_Gemma2_HF_CS23,
    formats=(
        FormatVersions("hf"),
        FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
    ),
)
