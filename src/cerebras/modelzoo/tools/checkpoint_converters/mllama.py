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
from collections import OrderedDict
from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatIndices,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    ConfigConverter_LLaMa_HF_CS21,
    Converter_LlamaModel_HF_CS,
)

USE_CONV_PATCHIFIED_EMBEDDING = False


class Converter_MLlamaSelfAttention_HF_CS(BaseCheckpointConverter_HF_CS):
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
                    EquivalentSubkey("o_proj", "proj_output_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MLLaMa_HF_CS24

    def supports_mup_conversion(self):
        return True


class Converter_MLlamaCrossAttention_HF_CS(Converter_MLlamaSelfAttention_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules += [
            ConversionRule(
                [
                    EquivalentSubkey("q_norm", "q_norm"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("k_norm", "k_norm"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]


class Converter_MLlamaTextModel_HF_CS(Converter_LlamaModel_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules += [
            # cross-attention
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("cross_attn", "multihead_attn"),
                    r"\.",
                    Converter_MLlamaCrossAttention_HF_CS(),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "cross_attn_attn_gate", "cross_attn_attn_gate"
                    ),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layers", "transformer_decoder.layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "cross_attn_mlp_gate", "cross_attn_mlp_gate"
                    ),
                ],
                action=self.replaceKey,
            ),
        ]


class Converter_MLlamaVisionModel_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("class_embedding", "class_embedding"),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("patch_embedding", "patch_embedding"),
                    r"\.(?:weight)",
                ],
                action=(
                    self.replaceKey
                    if USE_CONV_PATCHIFIED_EMBEDDING
                    else self.linear_projection_convert
                ),
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layernorm_pre", "layernorm_pre"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("layernorm_post", "layernorm_post"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "gated_positional_embedding",
                        "gated_positional_embedding",
                    ),
                    r"\.(?:gate|embedding|tile_embedding.weight)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "pre_tile_positional_embedding",
                        "pre_tile_positional_embedding",
                    ),
                    r"\.(?:gate|embedding.weight)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "post_tile_positional_embedding",
                        "post_tile_positional_embedding",
                    ),
                    r"\.(?:gate|embedding.weight)",
                ],
                action=self.replaceKey,
            ),
            # self-attention
            ConversionRule(
                [
                    r"(?:transformer\.|global_transformer\.)",
                    EquivalentSubkey("layers", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("self_attn", "self_attn"),
                    r"\.",
                    Converter_MLlamaSelfAttention_HF_CS(),
                ],
                action=None,
            ),
            # attention norm
            ConversionRule(
                [
                    r"(?:transformer\.|global_transformer\.)",
                    EquivalentSubkey("layers", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("input_layernorm", "norm1"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:transformer\.|global_transformer\.)",
                    EquivalentSubkey("layers", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("post_attention_layernorm", "norm2"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:transformer\.|global_transformer\.)",
                    EquivalentSubkey("layers", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "mlp.fc1",
                        "ffn.ffn.0.linear_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:transformer\.|global_transformer\.)",
                    EquivalentSubkey("layers", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "mlp.fc2",
                        "ffn.ffn.1.linear_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "global_transformer.layers", "global_transformer.layers"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("gate_attn", "gate_attn_weight"),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "global_transformer.layers", "global_transformer.layers"
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("gate_ffn", "gate_ffn_weight"),
                ],
                action=self.replaceKey,
            ),
        ]

    def linear_projection_convert(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        hidden_size = old_state_dict[old_key].shape[0]
        if from_index == 0:
            # HF -->> CS
            new_state_dict[new_key] = (
                old_state_dict[old_key]
                .permute(0, 2, 3, 1)
                .reshape(hidden_size, -1)
            )
        else:
            # CS -->> HF
            patch_size = action_fn_args["configs"][1]["model"][
                "vision_config_patch_size"
            ]
            num_channels = action_fn_args["configs"][1]["model"][
                "vision_config_num_channels"
            ]
            new_state_dict[new_key] = (
                old_state_dict[old_key]
                .reshape(
                    hidden_size, patch_size[0], patch_size[1], num_channels
                )
                .permute(0, 3, 1, 2)
            )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MLLaMa_HF_CS24

    def supports_mup_conversion(self):
        return True


class Converter_MLlamaForCausalLM_HF_CS24(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "language_model.lm_head", "text_model.model.lm_head"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "language_model.model.", "text_model.model."
                    ),
                    Converter_MLlamaTextModel_HF_CS(),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("vision_model.", "image_model."),
                    Converter_MLlamaVisionModel_HF_CS(),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("multi_modal_projector", "projection"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MLLaMa_HF_CS24

    def supports_mup_conversion(self):
        return True

    def post_model_convert(
        self,
        old_state_dict: OrderedDict,
        new_state_dict: OrderedDict,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
        drop_unmatched_keys: bool,
        key_prefix: str = "",
    ):
        if converter_indices.direction == 0:
            # HF -> CS
            config = copy.deepcopy(configs[1])
            config["model"]["image_model"] = {}
            config["model"]["text_model"] = {}
            for key in configs[1]["model"].keys():
                if key.startswith("vision_config_"):
                    config["model"]["image_model"][key[14:]] = config[
                        "model"
                    ].pop(key)
                else:
                    config["model"]["text_model"][key] = config["model"].pop(
                        key
                    )
            config["model"]["text_model"]["name"] = "llama"
            config["model"]["text_model"][
                "use_projection_bias_in_attention"
            ] = False
            config["model"]["text_model"]["use_ffn_bias_in_attention"] = False
            config["model"]["text_model"]["use_ffn_bias"] = False
            config["model"]["text_model"]["use_bias_in_output"] = False
            config["model"]["text_model"]["num_extra_input_vocab_tokens"] = 8
            config["model"]["image_model"]["name"] = "MllamaVisionModel"
            config["model"]["image_model"][
                "use_projection_bias_in_attention"
            ] = True
            config["model"]["image_model"]["use_ffn_bias_in_attention"] = False
            config["model"]["image_model"]["use_ffn_bias"] = True
            config["model"]["image_model"][
                "use_conv_patchified_embedding"
            ] = USE_CONV_PATCHIFIED_EMBEDDING
            configs[1].clear()
            configs[1].update(config)
        else:
            # CS -> HF
            config = copy.deepcopy(configs[0])
            config["vision_config"] = {}
            config["text_config"] = {}
            for key in configs[0].keys():
                if key.startswith("vision_config_"):
                    config["vision_config"][key[14:]] = config.pop(key)
                else:
                    config["text_config"][key] = config.pop(key)

            configs[0].clear()
            configs[0].update(config)

        super().post_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            converter_indices,
            drop_unmatched_keys,
            key_prefix=key_prefix,
        )


class ConfigConverter_MLLaMa_HF_CS24(ConfigConverter_LLaMa_HF_CS21):
    def __init__(self):
        self.model_type = "llama"
        super().__init__()
        self.rules += [
            ConversionRule(
                ["cross_attention_layers"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["pad_token_id"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_model_type"],
                action=BaseConfigConverter.assert_factory_fn(
                    0, "mllama_vision_model"
                ),
            ),
            ConversionRule(
                ["vision_config_hidden_size"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_num_hidden_layers"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_num_global_layers"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "vision_config_norm_eps",
                        "vision_config_layer_norm_epsilon",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "vision_config_attention_heads",
                        "vision_config_num_heads",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_supported_aspect_ratios"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_intermediate_layers_indices"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_vision_output_dim"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "vision_config_hidden_act", "vision_config_nonlinearity"
                    )
                ],
                action=self.convert_nonlinearity,
            ),
            ConversionRule(
                ["vision_config_use_projection_bias_in_attention"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["vision_config_use_ffn_bias_in_attention"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "vision_config_intermediate_size",
                        "vision_config_filter_size",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_use_ffn_bias"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["vision_config_initializer_range"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_image_size"],
                action=self.convert_image_patch_size,
            ),
            ConversionRule(
                ["vision_config_num_channels"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["vision_config_patch_size"],
                action=self.convert_image_patch_size,
            ),
            ConversionRule(
                ["vision_config_use_conv_patchified_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(
                    1, USE_CONV_PATCHIFIED_EMBEDDING
                ),
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.4", "cs-2.5"))

    def convert_image_patch_size(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            size = old_state_dict[old_key]
            new_state_dict[new_key] = [size, size]
        else:
            width, height = old_state_dict[old_key]
            if width != height:
                raise ConfigConversionError(
                    "Can't convert config with {}={}. Image width and height need to match.".format(
                        old_key, old_state_dict[old_key]
                    )
                )
            new_state_dict[new_key] = width

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
            gated_hf2cs = {
                "silu": "swiglu",
                "gelu_pytorch_tanh": "gelu_new",
                "quick_gelu": "quick_gelu",
            }
            if activation in gated_hf2cs:
                activation = gated_hf2cs[activation]
        elif from_index == 1:
            gated_cs2hf = {
                "swiglu": "silu",
                "gelu_new": "gelu_pytorch_tanh",
                "quick_gelu": "quick_gelu",
            }
            if activation in gated_cs2hf:
                activation = gated_cs2hf[activation]

        new_state_dict[new_key] = activation

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        """
        config: List[dicts] if converter_indices = 0 (HF-> CS) else dict (CS->HF).
        """

        if converter_indices.direction == 0:
            # HF -> CS
            text_config = config.pop("text_config")
            vision_config = config.pop("vision_config")
            for key in vision_config:
                config["vision_config_" + key] = vision_config[key]
            for key in text_config:
                config[key] = text_config[key]
            config["model_type"] = "llama"
            config.pop("pad_token_id")
        else:
            # CS -> HF
            # For v2 yaml compatibility
            if "trainer" in config:
                config["model"] = config["trainer"]["init"]["model"]
                config["model"]["text_model"]["pad_token_id"] = config[
                    "trainer"
                ]["fit"]["train_dataloader"]["pad_token_id"]
            elif "train_input" in config:
                config["model"]["text_model"]["pad_token_id"] = config[
                    "train_input"
                ]["pad_token_id"]
            else:
                config["model"]["text_model"]["pad_token_id"] = 0

            text_config = config["model"].pop("text_model")
            vision_config = config["model"].pop("image_model")
            for key in vision_config:
                config["model"]["vision_config_" + key] = vision_config[key]
            for key in text_config:
                config["model"][key] = text_config[key]

        return super().pre_config_convert(model, config, converter_indices)
