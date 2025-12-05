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

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.falcon_40b import (
    Converter_Falcon_40B_Headless_WithoutModelPrefix_HF_CS20,
    Converter_Falcon_40B_HF_CS20,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)


class Converter_Falcon_180B_Headless_WithoutModelPrefix_HF_CS20(
    Converter_Falcon_40B_Headless_WithoutModelPrefix_HF_CS20
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Drop alibi slopes
            ConversionRule(
                [r"relative_pe_helper\.slopes"],
                exists="right",
                action=None,
            ),
            # 180B specific layernorms:
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("input_layernorm.", "norm1."),
                    r"(?:weight|bias)",
                ],
                action=self.replaceNorm1,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("post_attention_layernorm.", "norm3."),
                    r"(?:weight|bias)",
                ],
                action=self.replaceNorm3,
            ),
            *self.rules,
        ]

    def replaceNorm1(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            new_state_dict[new_key] = old_state_dict[old_key]
        else:
            # key is either ln_attn or input_layernorm depending on the
            # following params: parallel_attn, new_decoder_architecture, and
            # num_ln_in_parallel_attn
            hf_config = action_fn_args["configs"][0]

            is_input_layernorm = (not hf_config["parallel_attn"]) or (
                hf_config["num_ln_in_parallel_attn"] != 2
            )

            if not is_input_layernorm:
                new_key = re.sub(r"\.input_layernorm\.", ".ln_attn.", new_key)
            new_state_dict[new_key] = old_state_dict[old_key]

    def replaceNorm3(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            new_state_dict[new_key] = old_state_dict[old_key]
        else:
            # key is either ln_mlp or post_attention_layernorm depending on the
            # following params: parallel_attn, new_decoder_architecture, and
            # num_ln_in_parallel_attn
            hf_config = action_fn_args["configs"][0]

            if not hf_config["parallel_attn"]:
                raise ConfigConversionError(
                    f"This model cannot be converted to HuggingFace's Falcon "
                    f"implementation. Your CS model is using "
                    f"use_untied_layer_norm=True (i.e. the checkpoint contains"
                    f" norm1 and norm3). The equivalent in HF requires setting"
                    f" parallel_attn=True..."
                )

            is_post_attention_layernorm = not hf_config["parallel_attn"]

            if not is_post_attention_layernorm:
                new_key = re.sub(
                    r"\.post_attention_layernorm\.", ".ln_mlp.", new_key
                )
            new_state_dict[new_key] = old_state_dict[old_key]

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Falcon_180B_HF_CS20


class ConfigConverter_Falcon_180B_HF_CS20(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "falcon"),
            ),
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(
                [EquivalentSubkey("alibi", "position_embedding_type")],
                action=self.convert_position_embedding_type,
            ),
            ConversionRule(
                ["rope_theta"],
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
                action=self.convert_hidden_size,
            ),
            ConversionRule(
                [EquivalentSubkey("num_attention_heads", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("num_kv_heads", "extra_attention_params")],
                action=self.convert_num_head_groups,
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
                ["nonlinearity"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, "gelu"),
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
                [EquivalentSubkey("hidden_dropout", "residual_dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["layer_norm_epsilon"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["use_bias_in_output"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["initializer_range"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["bias"],
                exists="left",
                action=self.convert_bias,
            ),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                exists="right",
                action=self.convert_bias,
            ),
            ConversionRule(
                ["use_ffn_bias_in_attention"],
                exists="right",
                action=self.convert_bias,
            ),
            ConversionRule(
                ["use_ffn_bias"],
                exists="right",
                action=self.convert_bias,
            ),
            ConversionRule(
                ["alibi"],
                exists="left",
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(
                ["new_decoder_architecture"],
                exists="left",
                action=self.assert_new_decoder_arch_and_parallel,
            ),
            ConversionRule(
                ["multi_query"], exists="left", action=self.convert_multi_query
            ),
            ConversionRule(
                ["parallel_attn"],
                exists="left",
                action=self.assert_new_decoder_arch_and_parallel,
            ),
            ConversionRule(
                ["use_untied_layer_norm"], exists="right", action=None
            ),
            ConversionRule(
                ["attention_module"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(
                    1, "multiquery_attention"
                ),
            ),
            ConversionRule(
                ["attention_type"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(
                    1, "scaled_dot_product"
                ),
            ),
        ]

        self.pre_convert_defaults[0].update(
            {
                "vocab_size": 65024,
                "hidden_size": 4544,
                "num_hidden_layers": 32,
                "num_attention_heads": 71,
                "layer_norm_epsilon": 1e-5,
                "initializer_range": 0.02,
                "use_cache": True,
                "hidden_dropout": 0.0,
                "attention_dropout": 0.0,
                "num_kv_heads": None,
                "alibi": False,
                "new_decoder_architecture": False,
                "multi_query": True,
                "parallel_attn": True,
                "bias": False,
                "max_position_embeddings": 2048,
                "rope_theta": 10000.0,
                "rope_scaling": None,
                "bos_token_id": 11,
                "eos_token_id": 11,
                "num_ln_in_parallel_attn": 2,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "position_embedding_type": "rotary",
                "rope_theta": 10000.0,
                "embedding_dropout_rate": 0.1,
                "share_embedding_weights": True,
                "nonlinearity": "gelu",
                "max_position_embeddings": 1024,
                "attention_module": "aiayn_attention",
                "attention_type": "scaled_dot_product",
                "use_untied_layer_norm": False,
                "extra_attention_params": {"num_kv_groups": 1},
            }
        )

        self.post_convert_defaults[0].update(
            {
                "model_type": "falcon",
                "new_decoder_architecture": True,
                "multi_query": True,
            }
        )
        self.post_convert_defaults[1].update(
            {
                "embedding_dropout_rate": 0.0,
                "share_embedding_weights": True,
                "nonlinearity": "gelu",
                "max_position_embeddings": 2048,
                "attention_module": "multiquery_attention",
                "attention_type": "scaled_dot_product",
                "use_untied_layer_norm": True,
                "extra_attention_params": {"num_kv_groups": 1},
                "loss_scaling": "num_tokens",
                "use_projection_bias_in_attention": False,
                "use_ffn_bias_in_attention": False,
                "use_ffn_bias": False,
                "use_bias_in_output": False,
            }
        )

    def assert_new_decoder_arch_and_parallel(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert from_index == 0, f"'{old_key}' should only be HF config"
        if (
            not old_state_dict["new_decoder_architecture"]
            and not old_state_dict["parallel_attn"]
        ):
            raise ConfigConversionError(
                "HF config must have either new_decoder_architecture or parallel_attn as True"
            )

    def convert_multi_query(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert from_index == 0, f"'{old_key}' should only be HF config"
        if (
            not old_state_dict["new_decoder_architecture"]
            and old_state_dict["multi_query"]
        ):
            if "extra_attention_params" not in new_state_dict:
                new_state_dict["extra_attention_params"] = {}
            new_state_dict["extra_attention_params"].update(
                {"num_kv_groups": 1}
            )

    def convert_num_head_groups(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            kv_groups = old_state_dict[old_key]
            if (
                not old_state_dict["new_decoder_architecture"]
                and old_state_dict["multi_query"]
            ):
                kv_groups = 1
            elif (
                not old_state_dict["new_decoder_architecture"]
                and not old_state_dict["multi_query"]
            ):
                kv_groups = old_state_dict["num_attention_heads"]
            new_state_dict[new_key] = {"num_kv_groups": kv_groups}
        elif from_index == 1:
            new_state_dict[new_key] = old_state_dict[old_key]["num_kv_groups"]

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
            new_state_dict["use_projection_bias_in_attention"] = old_state_dict[
                old_key
            ]
            new_state_dict["use_ffn_bias_in_attention"] = old_state_dict[
                old_key
            ]
            new_state_dict["use_ffn_bias"] = old_state_dict[old_key]
        else:
            if (
                old_state_dict["use_projection_bias_in_attention"]
                != old_state_dict["use_ffn_bias_in_attention"]
                or old_state_dict["use_ffn_bias_in_attention"]
                != old_state_dict["use_ffn_bias"]
            ):
                raise ConfigConversionError(
                    "All of the following CS parameters must be set the same in order to convert "
                    "to HF: use_projection_bias_in_attention, use_ffn_bias_in_attention, "
                    "use_ffn_bias"
                )
            new_state_dict[new_key] = old_state_dict[old_key]

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
            if old_state_dict[old_key] == True:
                raise ConfigConversionError(
                    "CS model doesn't support falcon with position_embedding_type = alibi"
                )
            new_state_dict[new_key] = "rotary"
        else:
            if old_state_dict[old_key] not in ["rotary"]:
                raise ConfigConversionError(
                    f"HF model doesn't support falcon with position_embedding_type = "
                    f"{old_state_dict[old_key]}"
                )
            new_state_dict[new_key] = old_state_dict[old_key] == "alibi"

    def convert_hidden_size(
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
            # Falcon uses 4 * hidden as intermediate size
            new_state_dict["filter_size"] = old_state_dict[old_key] * 4
        else:
            assert (
                old_state_dict[old_key] * 4 == old_state_dict["filter_size"]
            ), "HF model only supports filter_size = 4 * hidden_size"

    def parallel_attn_convert(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if old_state_dict[old_key] != True:
            raise ConfigConversionError(
                "parallel attention has to be enabled for falcon-180B"
            )
        new_state_dict[new_key] = True

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
            # falcon uses rotary_dim == head_dim
            new_config["rotary_dim"] = (
                old_config["hidden_size"] // old_config["num_attention_heads"]
            )

            new_config["use_untied_layer_norm"] = (
                old_config["new_decoder_architecture"]
                and old_config["num_ln_in_parallel_attn"] == 2
            ) or not old_config["parallel_attn"]

        else:
            # embedding dropout check
            assert (
                old_config["embedding_dropout_rate"] == 0.0
            ), "Falcon has no embedding dropout"

            # rotary check
            assert (
                old_config["rotary_dim"]
                == old_config["hidden_size"] // old_config["num_heads"]
            ), "rotary dimension of falcon is equal to head_dim"

            new_config["parallel_attn"] = True
            if not old_config["use_untied_layer_norm"]:
                new_config["new_decoder_architecture"] = False
                if new_config["num_kv_heads"] == 1:
                    new_config["multi_query"] = True
                elif (
                    new_config["num_kv_heads"]
                    == new_config["num_attention_heads"]
                ):
                    new_config["multi_query"] = False
                else:
                    raise ConfigConversionError(
                        "HF's falcon doesn't support use_untied_layer_norm=False"
                        "with grouped query attention (i.e. num_kv_groups != "
                        "num_heads and num_kv_groups != 1"
                    )
            else:
                new_config["new_decoder_architecture"] = True
                new_config["multi_query"] = new_config["num_kv_heads"] == 1

            if "num_ln_in_parallel_attn" not in new_config:
                if new_config["new_decoder_architecture"]:
                    new_config["num_ln_in_parallel_attn"] = 2
                else:
                    new_config["num_ln_in_parallel_attn"] = None

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
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))


Converter_Falcon_180B_Headless_HF_CS20 = (
    Build_HF_CS_Converter_WithOptionalModel(
        "Converter_Falcon_180B_Headless_HF_CS20",
        Converter_Falcon_180B_Headless_WithoutModelPrefix_HF_CS20,
        derived_class=Converter_Falcon_180B_Headless_WithoutModelPrefix_HF_CS20,
        config_converter_class=ConfigConverter_Falcon_180B_HF_CS20,
        formats=(
            FormatVersions("hf"),
            FormatVersions("cs-2.0"),
        ),
    )
)


class Converter_Falcon_180B_WithoutModelPrefix_HF_CS20(
    Converter_Falcon_40B_HF_CS20
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    "lm_head",
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("transformer.", ""),
                    Converter_Falcon_180B_Headless_WithoutModelPrefix_HF_CS20(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Falcon_180B_HF_CS20


Converter_Falcon_180B_HF_CS20 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_Falcon_180B_HF_CS20",
    Converter_Falcon_180B_WithoutModelPrefix_HF_CS20,
    derived_class=Converter_Falcon_180B_WithoutModelPrefix_HF_CS20,
    config_converter_class=ConfigConverter_Falcon_180B_HF_CS20,
    formats=(
        FormatVersions("hf"),
        FormatVersions("cs-2.0"),
    ),
)

###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1, and HF <> CS 2.1 converters:
###########################################################


class ConfigConverter_Falcon_180B_HF_CS21(ConfigConverter_Falcon_180B_HF_CS20):
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
                scaling_type = old_state_dict[old_key]["type"].lower()
                if scaling_type not in ["linear", "yarn", "longrope"]:
                    raise ConfigConversionError(
                        f"Only `rope_scaling` type `linear, `yarn` or 'longrope' is currently supported, "
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
                new_state_dict[new_key] = {
                    "type": old_state_dict.get("pos_scaling_type", "linear"),
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


class Converter_Falcon_180B_Headless_WithoutOptionalModel_HF_CS21(
    Converter_Falcon_180B_Headless_WithoutModelPrefix_HF_CS20
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["embedding_layer\.position_embed_helper\.slopes"],
                exists="right",
                action=None,
            ),
            *self.rules,
        ]


Converter_Falcon_180B_Headless_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_Falcon_180B_Headless_HF_CS21",
    Converter_Falcon_180B_Headless_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_Falcon_180B_Headless_WithoutOptionalModel_HF_CS21,
    config_converter_class=ConfigConverter_Falcon_180B_HF_CS21,
    formats=(
        FormatVersions("hf"),
        FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
    ),
)


class Converter_Falcon_180B_WithoutOptionalModel_HF_CS21(
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
                    Converter_Falcon_180B_Headless_WithoutOptionalModel_HF_CS21(),
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
        return ConfigConverter_Falcon_180B_HF_CS21


Converter_Falcon_180B_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_Falcon_180B_HF_CS21",
    Converter_Falcon_180B_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_Falcon_180B_WithoutOptionalModel_HF_CS21,
)
