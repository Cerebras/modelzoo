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
from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_CS_CS,
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    BaseConfigConverter_CS_CS,
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatIndices,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
    convert_use_rms_layer_norm_helper,
)
from cerebras.modelzoo.tools.checkpoint_converters.mup import (
    ConfigConverter_T5_sP_muP,
    Converter_T5_sP_muP,
)


class Converter_T5_CS16_CS17(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    r"(?:encoder|decoder)_",
                    EquivalentSubkey(
                        "token_embedding", "embeddings.word_embeddings"
                    ),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.embed_tokens\.weight",
                ],
                exists="left",
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)",
                    EquivalentSubkey(
                        ".absolute_position_embedding",
                        "_embeddings.position_embeddings",
                    ),
                    # Fixed position embeddings don't have a .weight suffix while learned absolute
                    # does
                    r"(?:\.weight|)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("layer.0.SelfAttention", "self_attn"),
                    r"\.",
                    EquivalentSubkey("q", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("layer.0.SelfAttention", "self_attn"),
                    r"\.",
                    EquivalentSubkey("k", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("layer.0.SelfAttention", "self_attn"),
                    r"\.",
                    EquivalentSubkey("v", "proj_v_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("layer.0.SelfAttention", "self_attn"),
                    r"\.",
                    EquivalentSubkey("o", "proj_output_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("layer.0.layer_norm", "norm1"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("layer.1.layer_norm", "norm2"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey("layer.2.layer_norm", "norm3"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"decoder\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.EncDecAttention", "multihead_attn"
                    ),
                    r"\.",
                    EquivalentSubkey("q", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"decoder\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.EncDecAttention", "multihead_attn"
                    ),
                    r"\.",
                    EquivalentSubkey("k", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"decoder\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.EncDecAttention", "multihead_attn"
                    ),
                    r"\.",
                    EquivalentSubkey("v", "proj_v_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"decoder\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.EncDecAttention", "multihead_attn"
                    ),
                    r"\.",
                    EquivalentSubkey("o", "proj_output_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    # pylint: disable=line-too-long
                    r"(?:encoder|decoder)\.block\.\d+\.layer\.0\.SelfAttention\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="left",
                action=self.convert_relative_attention_bias_cs16_to_cs17,
            ),
            ConversionRule(
                [
                    # pylint: disable=line-too-long
                    r"relative_position_(?:encoder|decoder)\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="right",
                action=self.convert_relative_attention_bias_cs17_to_cs16,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.DenseReluDense.wi", "ffn.ffn.0.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.DenseReluDense.wi_0",
                        "ffn.ffn.0.linear_layer_for_glu",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.DenseReluDense.wi_1", "ffn.ffn.0.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.DenseReluDense.wo", "ffn.ffn.1.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.2.DenseReluDense.wi", "ffn.ffn.0.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.2.DenseReluDense.wi_0",
                        "ffn.ffn.0.linear_layer_for_glu",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.2.DenseReluDense.wi_1", "ffn.ffn.0.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "layer.2.DenseReluDense.wo", "ffn.ffn.1.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    r"(?:encoder|decoder)\.",
                    EquivalentSubkey("final_layer_norm", "norm"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [r"lm_head\.(?:weight|bias)"],
                action=self.replaceKey,
            ),
        ]

    def convert_dense_layer(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            self.replaceKey(
                old_key, new_key, old_state_dict, new_state_dict, from_index
            )
        else:
            # When going from CS -> HF, we need to figure out if cross
            # attention is enabled or not

            old_key_split = old_key.split(".")
            multihead_q_key = (
                '.'.join(old_key_split[: old_key_split.index("ffn")])
                + ".multihead_attn.proj_q_dense_layer.weight"
            )
            cross_attention_enabled = multihead_q_key in old_state_dict

            if cross_attention_enabled:
                layer_names = new_key.split(".")
                layer_names[layer_names.index("layer") + 1] = "2"
                new_key = '.'.join(layer_names)

            # The ".linear_layer" needs to be mapped differently if we are using
            # a gated attention:
            is_gated_act = (
                action_fn_args["configs"][0]
                .get("feed_forward_proj", "relu")
                .startswith("gated-")
            )
            if is_gated_act:
                wi_position = new_key.find(".wi.")
                if wi_position != -1:
                    new_key = (
                        new_key[:wi_position]
                        + ".wi_1."
                        + new_key[wi_position + len(".wi.") :]
                    )

            new_state_dict[new_key] = old_state_dict[old_key]

    def convert_relative_attention_bias_cs16_to_cs17(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert (
            from_index == 0
        ), "Shouldn't have matched the following key: {}".format(old_key)
        if old_key.find(".block.0.") != -1:
            module = old_key[: old_key.find(".")]  # encoder or decoder
            layer_type = old_key[old_key.rfind(".") + 1 :]  # bias or weight
            key_prefix = new_key[: new_key.find(module)]
            new_key = (
                "{}relative_position_{}.relative_attention_bias.{}".format(
                    key_prefix, module, layer_type
                )
            )
            new_state_dict[new_key] = old_state_dict[old_key]

    def convert_relative_attention_bias_cs17_to_cs16(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert (
            from_index == 1
        ), "Shouldn't have matched the following key: {}".format(old_key)
        # CS 16 stored relative attention bias on every single transformer block event though they
        # weren't used. Extract the text after 'relative_position_' and before the following '.'
        # into module.
        relative_position_start = old_key.find("relative_position_")
        assert relative_position_start != -1, "Invalid key: {}".format(old_key)
        module = old_key[
            relative_position_start
            + len("relative_position_") : old_key.find(
                ".", relative_position_start
            )
        ]
        layer_type = old_key[old_key.rfind(".") + 1 :]

        num_layers = 0
        while (
            "{}encoder.layers.{}.self_attn.proj_q_dense_layer.weight".format(
                old_key[:relative_position_start], num_layers
            )
            in old_state_dict
        ):
            num_layers += 1

        for idx in range(num_layers):
            new_key = "{}.block.{}.layer.0.SelfAttention.relative_attention_bias.{}".format(
                module, idx, layer_type
            )
            new_state_dict[new_key] = old_state_dict[old_key]

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # Don't copy non model keys like optimizer state:
        logging.warning(
            "The T5 model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        output_checkpoint["model"] = {}

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "T5ForConditionalGeneration class"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_CS16_CS17


class Converter_T5_CS17_CS18(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [r"(?!model\.).*"],
                action=self.replaceKey,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [EquivalentSubkey("", "model."), ".*"],
                action=self.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.7"), FormatVersions("cs-1.8"))

    @classmethod
    def converter_note(cls) -> str:
        return "T5ForConditionalGeneration class"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_CS17_CS18


class Converter_T5_CS16_CS18(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_T5_CS16_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [EquivalentSubkey("", "model."), Converter_T5_CS16_CS17()],
                action=None,
            ),
        ]

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # Don't copy non model keys like optimizer state:
        logging.warning(
            "The T5 model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        output_checkpoint["model"] = {}

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.8"))

    @classmethod
    def converter_note(cls) -> str:
        return "T5ForConditionalGeneration class"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_CS16_CS18


class ConfigConverter_T5_CS16_CS17(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Config didn't change between 1.6 and 1.7. Copy all keys.
        self.rules = [
            ConversionRule([".*"], action=BaseConfigConverter.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.7"))


class ConfigConverter_T5_CS17_CS18(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Only thing that changed between 1.7 and 1.8 is flipped
        # use_pre_encoder_decoder_layer_norm
        self.rules = [
            ConversionRule(
                ["use_pre_encoder_decoder_layer_norm"],
                action=self.flip_use_pre_encoder_decoder_layer_norm,
            ),
            ConversionRule([".*"], action=BaseConfigConverter.replaceKey),
        ]

    @classmethod
    def flip_use_pre_encoder_decoder_layer_norm(
        cls,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        new_state_dict[new_key] = not old_state_dict[old_key]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.7"), FormatVersions("cs-1.8"))


class ConfigConverter_T5_CS16_CS18(
    ConfigConverter_T5_CS16_CS17,
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["use_pre_encoder_decoder_layer_norm"],
                action=ConfigConverter_T5_CS17_CS18.flip_use_pre_encoder_decoder_layer_norm,
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.8"))


### T5ForConditional Generation HF <-> CS1.7
class Converter_T5_HF_CS17(
    Converter_T5_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    r"(?:encoder|decoder)",
                    EquivalentSubkey(
                        ".embed_tokens", "_embeddings.word_embeddings"
                    ),
                    r"\.weight",
                ],
                action=self.convert_embeddings,
            ),
            ConversionRule([r"shared\.weight"], exists="left"),
            ConversionRule(
                [
                    # pylint: disable=line-too-long
                    r"relative_position_(?:encoder|decoder)\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="right",
                action=self.convert_relative_attention_bias_cs17_to_hf,
            ),
            ConversionRule(
                [
                    # pylint: disable=line-too-long
                    r"decoder\.block\.\d+\.layer\.1\.EncDecAttention\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="left",
                action=None,
            ),
            *self.rules,
        ]

    def convert_embeddings(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        self.replaceKey(
            old_key, new_key, old_state_dict, new_state_dict, from_index
        )
        if from_index == 1:
            # HF stores a copy of the word embeddings at the top level in a variable named 'shared'
            self.replaceKey(
                old_key,
                "shared.weight",
                old_state_dict,
                new_state_dict,
                from_index,
            )

    def convert_relative_attention_bias_cs17_to_hf(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert (
            from_index == 1
        ), "Shouldn't have matched the following key: {}".format(old_key)
        # CS 16 stored relative attention bias on every single transformer block event though they
        # weren't used
        relative_position_start = old_key.find("relative_position_")
        assert relative_position_start != -1, "Invalid key: {}".format(old_key)
        module = old_key[
            relative_position_start
            + len("relative_position_") : old_key.find(
                ".", relative_position_start
            )
        ]
        layer_type = old_key[old_key.rfind(".") + 1 :]

        new_key = "{}.block.0.layer.0.SelfAttention.relative_attention_bias.{}".format(
            module, layer_type
        )
        new_state_dict[new_key] = old_state_dict[old_key]

    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
    ):
        if converter_indices.direction == 1:
            assert (
                "encoder_embeddings.position_embeddings.weight"
                not in old_state_dict
                and "decoder_embeddings.position_embeddings.weight"
                not in old_state_dict
            ), (
                "Cannot convert to HF because it doesn't support "
                "position_embedding_type=\"learned_absolute\""
            )
            assert (
                "encoder_embeddings.position_embeddings" not in old_state_dict
                and "decoder_embeddings.position_embeddings"
                not in old_state_dict
            ), "Cannot convert to HF because it doesn't support position_embedding_type=\"fixed\""

            assert (
                "relative_position_encoder.relative_attention_bias.weight"
                in old_state_dict
                and "relative_position_decoder.relative_attention_bias.weight"
                in old_state_dict
            ), "Cannot convert to HF because it doesn't support position_embedding_type=None"

            if (
                configs[1]["model"]["share_embedding_weights"]
                and configs[1]["model"]["share_encoder_decoder_embedding"]
                and old_state_dict.get(
                    "encoder_embeddings.word_embeddings.weight", 0
                )
                is None
            ):
                old_state_dict["encoder_embeddings.word_embeddings.weight"] = (
                    old_state_dict["lm_head.weight"]
                )
            if (
                configs[1]["model"]["share_embedding_weights"]
                and configs[1]["model"]["share_encoder_decoder_embedding"]
                and old_state_dict.get(
                    "decoder_embeddings.word_embeddings.weight", 0
                )
                is None
            ):
                old_state_dict["decoder_embeddings.word_embeddings.weight"] = (
                    old_state_dict["lm_head.weight"]
                )

    def pre_checkpoint_convert(
        self,
        *args,
    ):
        return BaseCheckpointConverter_HF_CS.pre_checkpoint_convert(
            self,
            *args,
        )

    def extract_model_dict(self, *args):
        return BaseCheckpointConverter_HF_CS.extract_model_dict(self, *args)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} T5ForConditionalGeneration".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_HF_CS17


class Converter_T5_HF_CS18(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_T5_HF_CS17(),
                ],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [EquivalentSubkey("", "model."), Converter_T5_HF_CS17()],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} <-> {} T5ForConditionalGeneration".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_HF_CS18


class ConfigConverter_T5_HF_CS17(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "t5"),
            ),
            # Embedding
            ConversionRule(
                [EquivalentSubkey("vocab_size", "src_vocab_size")],
                action=self.replaceKey,
            ),
            ConversionRule(["d_model"], action=self.replaceKey),
            ConversionRule(["d_kv"], action=self.replaceKey),
            ConversionRule(["d_ff"], action=self.replaceKey),
            ConversionRule(
                [EquivalentSubkey("num_layers", "encoder_num_hidden_layers")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "num_decoder_layers", "decoder_num_hidden_layers"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(["num_heads"], action=self.replaceKey),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["relative_attention_num_buckets"],
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
                ["is_encoder_decoder"],
                action=BaseConfigConverter.assert_factory_fn(0, True),
            ),
            ConversionRule(
                ["relative_attention_max_distance"],
                action=BaseConfigConverter.assert_factory_fn(0, 128),
            ),
            ConversionRule(["dropout_rate"], action=self.replaceKey),
            ConversionRule(
                ["layer_norm_epsilon"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("feed_forward_proj", "encoder_nonlinearity")],
                action=self.convert_nonlinearity,
            ),
            ConversionRule(
                ["decoder_nonlinearity"],
                action=self.assert_decoder_nonlinearity,
            ),
            ConversionRule(
                ["position_embedding_type"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, "relative"),
            ),
            ConversionRule(
                ["(?:src|tgt)_max_position_embeddings"],
                exists="right",
                action=None,
            ),
            ConversionRule(
                ["use_dropout_outside_residual_path"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["share_encoder_decoder_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_pre_encoder_decoder_dropout"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            # use_pre_encoder_decoder_layer_norm=False in CS <= 1.7. This flag
            # was flipped in 1.8
            ConversionRule(
                ["use_pre_encoder_decoder_layer_norm"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["use_ffn_bias"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["use_transformer_initialization"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
        ]

        self.pre_convert_defaults[0].update(
            {
                "vocab_size": 32128,
                "d_model": 512,
                "d_kv": 64,
                "d_ff": 2048,
                "num_layers": 6,
                "num_heads": 8,
                "relative_attention_num_buckets": 32,
                "relative_attention_max_distance": 128,
                "dropout_rate": 0.1,
                "layer_norm_epsilon": 1e-6,
                "initializer_factor": 1,
                "feed_forward_proj": "relu",
                "tie_word_embeddings": True,
            }
        )

        self.pre_convert_defaults[1].update(
            {
                "use_projection_bias_in_attention": False,
                "relative_attention_num_buckets": 32,
                "share_embedding_weights": True,
                "use_t5_layer_norm": True,
                "layer_norm_epsilon": 1.0e-5,
                "position_embedding_type": "relative",
                "use_dropout_outside_residual_path": True,
                "share_encoder_decoder_embedding": True,
                "use_pre_encoder_decoder_dropout": False,
                "use_pre_encoder_decoder_layer_norm": False,
                "use_ffn_bias": False,
                "use_transformer_initialization": False,
            },
        )

        self.post_convert_defaults[0].update({"model_type": "t5"})
        self.post_convert_defaults[1].update(
            {
                "src_max_position_embeddings": 512,
                "tgt_max_position_embeddings": 512,
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
        if activation.startswith("gated-"):
            activation = activation[6:]
        if from_index == 0 and old_state_dict[old_key].startswith("gated-"):
            gated_hf2cs = {"silu": "swiglu", "relu": "reglu", "gelu": "geglu"}
            assert activation in gated_hf2cs
            activation = gated_hf2cs[activation]
        elif from_index == 1 and activation.endswith("glu"):
            gated_cs2hf = {
                "swiglu": "gated-silu",
                "reglu": "gated-relu",
                "geglu": "gated-gelu",
            }
            assert activation in gated_cs2hf
            activation = gated_cs2hf[activation]

        new_state_dict[new_key] = activation
        if from_index == 0:
            new_state_dict["decoder_nonlinearity"] = activation

    def assert_decoder_nonlinearity(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if old_state_dict["encoder_nonlinearity"] != old_state_dict[old_key]:
            raise ConfigConversionError(
                f"Encoder & Decoder nonlinearities must be the same in HF model. "
                f"Got: {old_state_dict['encoder_nonlinearity']} vs {old_state_dict[old_key]}"
            )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        if converter_indices.direction == 1:
            if "tgt_vocab_size" in config:
                if (
                    config["tgt_vocab_size"] is not None
                    and config["tgt_vocab_size"] != config["src_vocab_size"]
                ):
                    raise ConfigConversionError(
                        "HF implementation doesn't allow tgt_vocab_size != src_vocab_size"
                    )
            if "relu_dropout_rate" in config:
                if (
                    config["relu_dropout_rate"] is not None
                    and config["relu_dropout_rate"] != config["dropout_rate"]
                ):
                    raise ConfigConversionError(
                        "HF implementation doesn't allow relu_dropout_rate != dropout_rate"
                    )

        return config


class ConfigConverter_T5_HF_CS18(ConfigConverter_T5_HF_CS17):
    def __init__(self):
        super().__init__()
        self.rules = [
            # CS 1.8 flipped the use_pre_encoder_decoder_layer_norm param
            ConversionRule(
                ["use_pre_encoder_decoder_layer_norm"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            *self.rules,
        ]
        self.pre_convert_defaults[1][
            "use_pre_encoder_decoder_layer_norm"
        ] = True

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8"))


class Converter_T5_CS18_CS20(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Model didn't change between 1.8/1.9 and 2.0. Copy all keys.
        self.rules = [
            # Don't copy encoder/decoder word embeddings or lm_head due to tying
            # These props will be handled in the `post_model_convert`
            ConversionRule(
                [
                    "(?:model.|)",
                    "lm_head",
                    "\.weight",
                ],
                action=None,
            ),
            ConversionRule(
                [
                    "(?:model.|)",
                    "encoder_embeddings.word_embeddings",
                    "\.weight",
                ],
                action=None,
            ),
            ConversionRule(
                [
                    "(?:model.|)",
                    "decoder_embeddings.word_embeddings",
                    "\.weight",
                ],
                action=None,
            ),
            ConversionRule([".*"], action=self.replaceKey),
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
        cs_config = configs[1]

        if "decoder_embeddings.word_embeddings.weight" in old_state_dict:
            model_prefix = ""
        elif (
            "model.decoder_embeddings.word_embeddings.weight" in old_state_dict
        ):
            model_prefix = "model."
        else:
            raise KeyError(
                "Unable to find decoder_embeddings.word_embeddings.weight in checkpoint"
            )

        lm_head = old_state_dict[f"{model_prefix}lm_head.weight"]
        decoder_embed = old_state_dict[
            f"{model_prefix}decoder_embeddings.word_embeddings.weight"
        ]
        encoder_embed = old_state_dict[
            f"{model_prefix}encoder_embeddings.word_embeddings.weight"
        ]

        if cs_config["model"].get(
            "share_encoder_decoder_embedding", True
        ) and cs_config["model"].get("share_embedding_weights", True):
            not_none = list(
                filter(
                    lambda e: e is not None,
                    [lm_head, decoder_embed, encoder_embed],
                )
            )
            if not all(not_none[0].equal(e) for e in not_none):
                logging.warning(
                    "When encoder-decoder embeddings & lm_head are tied, all "
                    "tensors should be the same. However, there is a conflict "
                    "between some of the tensors. As a result, the output "
                    "checkpoint may be inconsistent."
                )

            if len(not_none) > 0:
                lm_head = not_none[0]
                decoder_embed = not_none[0]
                encoder_embed = not_none[0]

        if cs_config["model"].get("share_embedding_weights", True):
            if lm_head is None:
                lm_head = decoder_embed
            if decoder_embed is None:
                decoder_embed = lm_head

        if cs_config["model"].get("share_encoder_decoder_embedding", True):
            if encoder_embed is None:
                encoder_embed = decoder_embed
            if decoder_embed is None:
                decoder_embed = encoder_embed

        new_state_dict[f"{model_prefix}lm_head.weight"] = lm_head
        new_state_dict[
            f"{model_prefix}decoder_embeddings.word_embeddings.weight"
        ] = decoder_embed
        new_state_dict[
            f"{model_prefix}encoder_embeddings.word_embeddings.weight"
        ] = encoder_embed

        # Finalize checkpoint:
        super().post_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            converter_indices,
            drop_unmatched_keys,
            key_prefix=key_prefix,
        )

    @classmethod
    def converter_note(cls) -> str:
        return "T5ForConditionalGeneration class"

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.8", "cs-1.9"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_CS18_CS20


class ConfigConverter_T5_CS18_CS20(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # Only difference between 1.8/1.9 and 2.0 is introduction of norm_type
        self.rules = [
            ConversionRule(
                [EquivalentSubkey("use_t5_layer_norm", "norm_type")],
                action=self.convert_use_t5_layer_norm,
            ),
            ConversionRule([".*"], action=self.replaceKey),
        ]

    def convert_use_t5_layer_norm(self, *args):
        convert_use_rms_layer_norm_helper(self, *args)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.8", "cs-1.9"), FormatVersions("cs-2.0"))


class Converter_T5_HF_CS20(Converter_T5_HF_CS18):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_HF_CS20


class ConfigConverter_T5_HF_CS20(ConfigConverter_T5_HF_CS18):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["norm_type"],
                action=BaseConfigConverter.assert_factory_fn(1, "rmsnorm"),
            ),
            *self.rules,
        ]
        del self.pre_convert_defaults[1]["use_t5_layer_norm"]
        self.pre_convert_defaults[1]["norm_type"] = "rmsnorm"
        self.post_convert_defaults[1]["norm_type"] = "rmsnorm"

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1, and HF <> CS 2.1 converters:
###########################################################


class Converter_T5_CS20_CS21(BaseCheckpointConverter_CS_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Refactored embeddings:
            ConversionRule(
                [
                    "(?:model\.|)",
                    "(?:encoder|decoder)_embeddings\.",
                    EquivalentSubkey(
                        "position_embeddings.weight",
                        "position_embeddings.embed.weight",
                    ),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:model\.|)",
                    "(?:encoder|decoder)_embeddings\.",
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
                        "relative_position_encoder",
                        "encoder_embeddings.position_embed_helper",
                    ),
                    "\.relative_attention_bias",
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:model\.|)",
                    EquivalentSubkey(
                        "relative_position_decoder",
                        "decoder_embeddings.position_embed_helper",
                    ),
                    "\.relative_attention_bias",
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # T5 <= CS 2.0 didn't support ALIBI or RoPE
            # Copy everything else
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @classmethod
    def converter_note(cls) -> str:
        return "T5ForConditionalGeneration class"

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.0"), FormatVersions("cs-2.1"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_CS20_CS21


class ConfigConverter_T5_CS20_CS21(BaseConfigConverter_CS_CS):
    def __init__(self):
        super().__init__()
        # No differences in config
        self.rules = [
            ConversionRule([".*"], action=self.replaceKey),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-2.0"), FormatVersions("cs-2.1"))


class ConfigConverter_T5_HF_CS21(ConfigConverter_T5_HF_CS20):
    "CS 2.1 config is the same as CS 2.0."

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2"),
        )


class Converter_T5_WithoutOptionalModel_HF_CS21(Converter_T5_HF_CS17):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    "(?:encoder|decoder)\.block\.\d+\.layer\.0\.SelfAttention\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="left",
                action=self.convert_relative_attention_bias_hf_to_cs21,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)_embeddings\.position_embed_helper\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="right",
                action=self.convert_relative_attention_bias_cs17_to_hf,
            ),
            *self.rules,
        ]

    def convert_relative_attention_bias_hf_to_cs21(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert (
            from_index == 0
        ), "Shouldn't have matched the following key: {}".format(old_key)
        if old_key.find(".block.0.") != -1:
            module = old_key[: old_key.find(".")]  # encoder or decoder
            layer_type = old_key[old_key.rfind(".") + 1 :]  # bias or weight
            key_prefix = new_key[: new_key.find(module)]
            new_key = "{}{}_embeddings.position_embed_helper.relative_attention_bias.{}".format(
                key_prefix, module, layer_type
            )
            new_state_dict[new_key] = old_state_dict[old_key]

    def convert_relative_attention_bias_cs17_to_hf(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert (
            from_index == 1
        ), "Shouldn't have matched the following key: {}".format(old_key)
        # HF stored relative attention bias on every single transformer block event though they weren't used
        if old_key.find("encoder_embeddings.position_embed_helper.") != -1:
            module = "encoder"
        elif old_key.find("decoder_embeddings.position_embed_helper.") != -1:
            module = "decoder"
        else:
            assert False, "Invalid key: {}".format(old_key)

        layer_type = old_key[old_key.rfind(".") + 1 :]

        new_key = "{}.block.0.layer.0.SelfAttention.relative_attention_bias.{}".format(
            module, layer_type
        )
        new_state_dict[new_key] = old_state_dict[old_key]

    # CS 17 converter has custom pre_model_convert logic which CS21 doesn't need
    def pre_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
    ):
        pass

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_HF_CS21


class ConfigConverter_T5_HF_CS23(ConfigConverter_T5_HF_CS21):
    def supports_mup_conversion(self):
        return True

    def get_mup_converter(self):
        return ConfigConverter_T5_sP_muP

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_T5_WithoutOptionalModel_HF_CS23(
    Converter_T5_WithoutOptionalModel_HF_CS21
):
    def supports_mup_conversion(self):
        return True

    def get_mup_converter(self):
        return Converter_T5_sP_muP

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_HF_CS23


Converter_T5_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_T5_HF_CS21",
    Converter_T5_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_T5_WithoutOptionalModel_HF_CS21,
)
Converter_T5_HF_CS23 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_T5_HF_CS23",
    Converter_T5_WithoutOptionalModel_HF_CS23,
    derived_class=Converter_T5_WithoutOptionalModel_HF_CS23,
)
