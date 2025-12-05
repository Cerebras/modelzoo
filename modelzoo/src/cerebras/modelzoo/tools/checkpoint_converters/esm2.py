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
from cerebras.modelzoo.tools.checkpoint_converters.bert import (  # To CS 1.7; To CS 1.8
    ConfigConverter_Bert_HF_CS21,
    Converter_BertLayerNorm_HF_CS,
    Converter_BertModel_WithoutOptionalModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)


class Converter_Esm2Model_WithoutOptionalModel_HF_CS21(
    Converter_BertModel_WithoutOptionalModel_HF_CS21
):
    def __init__(self) -> None:
        super().__init__()
        self.rules = [
            # Encoder Layers:
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.self.query", "self_attn.proj_q_dense_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_with_interleaving_query_key,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.self.key", "self_attn.proj_k_dense_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.convert_with_interleaving_query_key,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("embeddings.", ""),
                    Converter_BertLayerNorm_HF_CS("layer_norm", "embed_ln_f"),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    r"encoder\.layer\.\d+\.attention\.self\.rotary_embeddings"
                    r"\.inv_freq",
                ],
                exists="left",
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("attention.", ""),
                    Converter_BertLayerNorm_HF_CS("LayerNorm", "norm1"),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    Converter_BertLayerNorm_HF_CS("LayerNorm", "norm2"),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.",
                        "transformer_encoder.",
                    ),
                    Converter_BertLayerNorm_HF_CS(
                        "emb_layer_norm_after", "norm"
                    ),
                ],
                action=None,
            ),
            *self.rules,
        ]

    def convert_with_interleaving_query_key(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        cs_config = action_fn_args["configs"][1]
        if cs_config["model"]["position_embedding_type"] != "rotary":
            new_state_dict[new_key] = old_state_dict[old_key]
        else:
            # Query & Keys should be interleaved since HF and CS RoPE differ
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

    def position_embeddings_convert(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if not (
            from_index == 0
            and action_fn_args["configs"][0]["position_embedding_type"]
            != "absolute"
        ):
            self.replaceKey(
                old_key, new_key, old_state_dict, new_state_dict, from_index
            )

        if from_index == 1:
            # HF stores an register buffer with position_ids
            position_id_key = re.sub(
                r"\.position_embeddings\.weight", ".position_ids", new_key
            )
            if "max_position_embeddings" in action_fn_args["configs"][0]:
                max_position_embeddings = action_fn_args["configs"][0][
                    "max_position_embeddings"
                ]
            else:
                max_position_embeddings = action_fn_args["configs"][1]["model"][
                    "max_position_embeddings"
                ]
            new_state_dict[position_id_key] = torch.arange(
                max_position_embeddings
            ).expand((1, -1))


class Converter_Esm2PretrainModel_WithoutOptionalModel_HF_CS21(
    BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("esm.", "bert_encoder."),
                    Converter_Esm2Model_WithoutOptionalModel_HF_CS21(),
                ],
            ),
            # Language Model Head:
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head.dense",
                        "bert_mlm_head.mlm_transform.ffn.ffn.0.linear_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head.",
                        "bert_mlm_head.mlm_transform.",
                    ),
                    Converter_BertLayerNorm_HF_CS("layer_norm", "ln"),
                ],
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head.decoder",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    r"\.weight",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    r"\.bias",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "lm_head",
                        "bert_mlm_head.classifier.ffn.0.linear_layer",
                    ),
                    r"\.bias",
                ],
                action=self.replaceKey,
            ),
            # Contact Head:
            ConversionRule(
                [r"esm\.contact_head\.regression\.(?:weight|bias)"],
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
        return ConfigConverter_Esm2_HF_CS21

    @classmethod
    def converter_note(cls) -> str:
        return "{} EsmForMaskedLM <-> {} for Esm2ForPreTrainingModel".format(
            cls.formats()[0], cls.formats()[1]
        )


Converter_Esm2PretrainModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_Esm2PretrainModel_HF_CS21",
    Converter_Esm2PretrainModel_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_Esm2PretrainModel_WithoutOptionalModel_HF_CS21,
)


class ConfigConverter_Esm2_HF_CS21(ConfigConverter_Bert_HF_CS21):
    def __init__(self) -> None:
        if not hasattr(self, "model_type"):
            self.model_type = "esm"
        super().__init__()
        self.rules = [
            ConversionRule(
                ["max_position_embeddings"],
                action=self.convert_max_pos_embed,
            ),
            ConversionRule(
                ["encoder_nonlinearity"],
                action=BaseConfigConverter.assert_factory_fn(1, "gelu"),
            ),
            ConversionRule(
                ["mlm_nonlinearity"],
                action=BaseConfigConverter.assert_factory_fn(1, "gelu"),
            ),
            ConversionRule(
                ["use_final_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "emb_layer_norm_before", "embedding_layer_norm"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["token_dropout"],
                action=self.replace_token_dropout,
            ),
            ConversionRule(
                ["mask_token_id"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["pad_token_id"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["disable_nsp"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["rotary_dim"], exists="right", action=self.assert_rotary_dim
            ),
            *self.rules,
        ]
        self.pre_convert_defaults[0].update(
            {
                "mask_token_id": None,
                "pad_token_id": None,
                "token_dropout": False,
                "emb_layer_norm_before": False,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "disable_nsp": False,
                "pad_token_id": 0,
                "mask_padding_in_positional_embed": False,
                "use_final_layer_norm": False,
                "token_dropout": False,
                "embedding_layer_norm": True,
            }
        )
        self.post_convert_defaults[0].update(
            {
                "is_folding_model": False,
                "esmfold_config": None,
            }
        )
        self.post_convert_defaults[1].update(
            {
                "use_final_layer_norm": True,
                "disable_nsp": True,
            }
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
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
            old_state_dict["position_embedding_type"] == "rotary"
            and old_state_dict[old_key]
            != old_state_dict["hidden_size"] // old_state_dict["num_heads"]
        ):
            raise ConfigConversionError(
                "rotary_dim must be hidden_size // num_heads in order to be compatible with HF"
            )

    def replace_token_dropout(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        token_dropout = old_state_dict[old_key]
        if token_dropout and old_state_dict.get("mask_token_id") is None:
            raise ConfigConversionError(
                "mask_token_id must be provided when token_dropout is enabled"
            )
        new_state_dict[new_key] = token_dropout

    def convert_max_pos_embed(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # The following only applies to learned embeddings. There is no effect
        # on RoPE
        # The number of positional embeddings = MSL + pad token offset + 1
        # HF refers to number of positional embeddings (the total) as
        # max_position_embeddings while we refer to MSL as
        # max_position_embeddings

        if (
            from_index == 0
            and old_state_dict["position_embedding_type"] == "absolute"
        ):
            new_state_dict[new_key] = (
                old_state_dict[old_key] - old_state_dict["pad_token_id"] - 1
            )
        elif (
            from_index == 1
            and old_state_dict["position_embedding_type"] == "learned"
        ):
            new_state_dict[new_key] = (
                old_state_dict[old_key] + old_state_dict["pad_token_id"] + 1
            )
        else:
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
        # HF supports absolute, relative_key, relative_key_query, rotary
        # CS supports learned, fixed, rotary

        embed_type = old_state_dict[old_key]
        if embed_type == "rotary":
            new_state_dict[new_key] = embed_type
        elif from_index == 0:
            if embed_type == "absolute":
                new_state_dict[new_key] = "learned"
                new_state_dict["mask_padding_in_positional_embed"] = True
            else:
                raise ConfigConversionError(
                    "CS model doesn't support HF's position_embedding_type={}".format(
                        embed_type
                    )
                )
        else:
            if embed_type == "learned":
                if (
                    old_state_dict.get("mask_padding_in_positional_embed")
                    != True
                ):
                    raise ConfigConversionError(
                        "ESM-2 trained in CS with learned embeddings must have "
                        "mask_padding_in_positional_embed=True in order to "
                        "convert to HF"
                    )
                new_state_dict[new_key] = "absolute"
            else:
                raise ConfigConversionError(
                    "HF model doesn't support CS's position_embedding_type={}".format(
                        embed_type
                    )
                )

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        return BaseConfigConverter_HF_CS.pre_config_convert(
            self, model, config, converter_indices
        )

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
            if new_config["position_embedding_type"] == "rotary":
                new_config["rotary_dim"] = (
                    new_config["hidden_size"] // new_config["num_heads"]
                )

        return BaseConfigConverter_HF_CS.post_config_convert(
            self,
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )
