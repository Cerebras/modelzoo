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

from modelzoo.common.pytorch.model_utils.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseCheckpointConverter_PT_PT,
    BaseConfigConverter,
    BaseConfigConverter_CS_CS,
    BaseConfigConverter_HF_CS,
    ConfigConversionError,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)


class Converter_T5_CS16_CS17(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    "(?:encoder|decoder)_",
                    EquivalentSubkey(
                        "token_embedding", "embeddings.word_embeddings"
                    ),
                    "\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["(?:encoder|decoder)\.embed_tokens\.weight",], exists="left"
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)",
                    EquivalentSubkey(
                        ".absolute_position_embedding",
                        "_embeddings.position_embeddings",
                    ),
                    "(?:\.weight|)",  # Fixed position embeddings don't have a .weight suffix while learned absolute does
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey("layer.0.SelfAttention", "self_attn"),
                    "\.",
                    EquivalentSubkey("q", "proj_q_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey("layer.0.SelfAttention", "self_attn"),
                    "\.",
                    EquivalentSubkey("k", "proj_k_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey("layer.0.SelfAttention", "self_attn"),
                    "\.",
                    EquivalentSubkey("v", "proj_v_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey("layer.0.SelfAttention", "self_attn"),
                    "\.",
                    EquivalentSubkey("o", "proj_output_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey("layer.0.layer_norm", "norm1"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey("layer.1.layer_norm", "norm2"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey("layer.2.layer_norm", "norm3"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "decoder\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.EncDecAttention", "multihead_attn"
                    ),
                    "\.",
                    EquivalentSubkey("q", "proj_q_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "decoder\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.EncDecAttention", "multihead_attn"
                    ),
                    "\.",
                    EquivalentSubkey("k", "proj_k_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "decoder\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.EncDecAttention", "multihead_attn"
                    ),
                    "\.",
                    EquivalentSubkey("v", "proj_v_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "decoder\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.EncDecAttention", "multihead_attn"
                    ),
                    "\.",
                    EquivalentSubkey("o", "proj_output_dense_layer"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.block\.\d+\.layer\.0\.SelfAttention\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="left",
                action=self.convert_relative_attention_bias_cs16_to_cs17,
            ),
            ConversionRule(
                [
                    "relative_position_(?:encoder|decoder)\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="right",
                action=self.convert_relative_attention_bias_cs17_to_cs16,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.DenseReluDense.wi", "ffn.ffn.0.linear_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.DenseReluDense.wi_0",
                        "ffn.ffn.0.linear_layer_for_glu",
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.DenseReluDense.wi_1", "ffn.ffn.0.linear_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.1.DenseReluDense.wo", "ffn.ffn.1.linear_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.2.DenseReluDense.wi", "ffn.ffn.0.linear_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.2.DenseReluDense.wi_0",
                        "ffn.ffn.0.linear_layer_for_glu",
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.2.DenseReluDense.wi_1", "ffn.ffn.0.linear_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.convert_dense_layer,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("block", "layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "layer.2.DenseReluDense.wo", "ffn.ffn.1.linear_layer"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    "(?:encoder|decoder)\.",
                    EquivalentSubkey("final_layer_norm", "norm"),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["lm_head\.(?:weight|bias)"], action=self.replaceKey,
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
            if action_fn_args["configs"][0].get("is_gated_act", False):
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
            new_key = "relative_position_{}.relative_attention_bias.{}".format(
                module, layer_type
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
        # CS 16 stored relative attention bias on every single transformer block event though they weren't used
        # Extract the text after 'relative_position_' and before the following '.' into module.
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

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        logging.warning(
            "The T5 model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        return {"model": checkpoint["model"]}

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.7"))

    @classmethod
    def converter_note(cls) -> str:
        return "T5ForConditionalGeneration class"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_CS16_CS17


class Converter_T5_CS17_CS18(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(["(?!model\.).*"], action=self.replaceKey,),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [EquivalentSubkey("", "model."), ".*"], action=self.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.7"), FormatVersions("cs-1.8", "cs-1.9"))

    @classmethod
    def converter_note(cls) -> str:
        return "T5ForConditionalGeneration class"

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_T5_CS17_CS18


class Converter_T5_CS16_CS18(BaseCheckpointConverter_PT_PT):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule([Converter_T5_CS16_CS17(),], action=None,),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [EquivalentSubkey("", "model."), Converter_T5_CS16_CS17()],
                action=None,
            ),
        ]

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        logging.warning(
            "The T5 model changed significantly between {} and {}. As a result, the"
            " optimizer state won't be included in the converted checkpoint.".format(
                *self.formats()
            )
        )
        return {"model": checkpoint["model"]}

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.8", "cs-1.9"))

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
        return (FormatVersions("cs-1.7"), FormatVersions("cs-1.8", "cs-1.9"))


class ConfigConverter_T5_CS16_CS18(ConfigConverter_T5_CS16_CS17,):
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
        return (FormatVersions("cs-1.6"), FormatVersions("cs-1.8", "cs-1.9"))


### T5ForConditional Generation HF <-> CS1.7
class Converter_T5_HF_CS17(
    Converter_T5_CS16_CS17, BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    "(?:encoder|decoder)",
                    EquivalentSubkey(
                        ".embed_tokens", "_embeddings.word_embeddings"
                    ),
                    "\.weight",
                ],
                action=self.convert_embeddings,
            ),
            ConversionRule(["shared\.weight"], exists="left"),
            ConversionRule(
                [
                    "relative_position_(?:encoder|decoder)\.relative_attention_bias\.(?:weight|bias)"
                ],
                exists="right",
                action=self.convert_relative_attention_bias_cs17_to_hf,
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
        # CS 16 stored relative attention bias on every single transformer block event though they weren't used
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
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 1:
            assert (
                "encoder_embeddings.position_embeddings.weight"
                not in old_state_dict
                and "decoder_embeddings.position_embeddings.weight"
                not in old_state_dict
            ), "Cannot convert to HF because it doesn't support position_embedding_type=\"learned_absolute\""
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
                    "encoder_embeddings.position_embeddings.weight", 0
                )
                is None
            ):
                old_state_dict[
                    "encoder_embeddings.position_embeddings.weight"
                ] = old_state_dict["lm_head.weight"]
            if (
                configs[1]["model"]["share_embedding_weights"]
                and configs[1]["model"]["share_encoder_decoder_embedding"]
                and old_state_dict.get(
                    "decoder_embeddings.position_embeddings.weight", 0
                )
                is None
            ):
                old_state_dict[
                    "decoder_embeddings.position_embeddings.weight"
                ] = old_state_dict["lm_head.weight"]

    def post_checkpoint_convert(
        self, checkpoint, from_index: int,
    ):
        return BaseCheckpointConverter_HF_CS.post_checkpoint_convert(
            self, checkpoint, from_index
        )

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
            ConversionRule([Converter_T5_HF_CS17(),], action=None,),
            # Catch checkpoints from depricated PyTorchBaseModel
            ConversionRule(
                [EquivalentSubkey("", "model."), Converter_T5_HF_CS17()],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))

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
                ["relative_attention_num_buckets"], action=self.replaceKey,
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
            ConversionRule(["layer_norm_epsilon"], action=self.replaceKey,),
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
        is_gated = False
        if from_index == 0 and old_state_dict.get("is_gated_act", False):
            is_gated = True
            gated_hf2cs = {"silu": "swiglu", "relu": "reglu", "gelu": "geglu"}
            assert activation in gated_hf2cs.keys()
            activation = gated_hf2cs[activation]
        elif from_index == 1 and activation.endswith("glu"):
            is_gated = True
            gated_cs2hf = {"swiglu": "silu", "reglu": "relu", "geglu": "gelu"}
            assert activation in gated_cs2hf.keys()
            activation = gated_cs2hf[activation]

        new_state_dict[new_key] = activation
        if from_index == 0:
            new_state_dict["decoder_nonlinearity"] = activation
        else:
            new_state_dict["is_gated_act"] = is_gated

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
                "Encoder & Decoder nonlinearities must be the same in HF model. Got: {} vs {}".format(
                    old_state_dict["encoder_nonlinearity"],
                    old_state_dict[old_key],
                )
            )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.7"))

    def pre_config_convert(
        self, config, from_index,
    ):
        config = super().pre_config_convert(config, from_index)

        defaults = [
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
            },
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
        ]

        # Apply defaults
        for key in defaults[from_index]:
            if key not in config:
                config[key] = defaults[from_index][key]

        if from_index == 1:
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

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):
        if from_index == 0:
            if "src_max_position_embeddings" not in new_config:
                new_config["src_max_position_embeddings"] = 512
            if "tgt_max_position_embeddings" not in new_config:
                new_config["tgt_max_position_embeddings"] = 512
        return super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )


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

    def pre_config_convert(
        self, config, from_index,
    ):
        if from_index == 1:
            if "use_pre_encoder_decoder_layer_norm" not in config["model"]:
                config["model"]["use_pre_encoder_decoder_layer_norm"] = True
        return super().pre_config_convert(config, from_index)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-1.8", "cs-1.9"))
