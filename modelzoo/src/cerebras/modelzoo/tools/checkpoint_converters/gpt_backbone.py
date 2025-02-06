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
from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_CS_CS,
    BaseConfigConverter,
    BaseConfigConverter_CS_CS,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)


class Converter_GPT2LMHeadModel_GPTBackboneLMHeadModel_CS24(
    BaseCheckpointConverter_CS_CS
):
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
        return (FormatVersions("cs-2.4"), FormatVersions("cs-2.4-backbone"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_GPT2LMHeadModel_GPTBackboneLMHeadModel_CS24


class ConfigConverter_GPT2LMHeadModel_GPTBackboneLMHeadModel_CS24(
    BaseConfigConverter_CS_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule([r'name'], action=self.convert_name),
            ConversionRule([r'hidden_size'], action=self.replaceKey),
            ConversionRule(
                [EquivalentSubkey("vocab_size", "embedding_layer.vocab_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "max_position_embeddings",
                        "embedding_layer.max_position_embeddings",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("embd_pdrop", "embedding_layer.embd_pdrop")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "position_embedding_type",
                        "embedding_layer.position_embedding_type",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "constant_pos_embedding",
                        "embedding_layer.constant_pos_embedding",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "position_embedding_offset",
                        "embedding_layer.position_embedding_offset",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "share_embedding_weights",
                        "embedding_layer.share_embedding_weights",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embedding_layer_norm",
                        "embedding_layer.embedding_layer_norm",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "num_relative_attention_buckets",
                        "embedding_layer.num_relative_attention_buckets",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("rotary_dim", "embedding_layer.rotary_dim")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("rope_theta", "embedding_layer.rope_theta")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "fold_rope_consts", "embedding_layer.fold_rope_consts"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "alibi_trainable_slopes",
                        "embedding_layer.alibi_trainable_slopes",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "pos_scaling_factor",
                        "embedding_layer.pos_scaling_factor",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "pos_scaling_type", "embedding_layer.pos_scaling_type"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "pos_scaling_extra_args",
                        "embedding_layer.pos_scaling_extra_args",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "rel_distance_mode", "embedding_layer.rel_distance_mode"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "rel_distance_extra_args",
                        "embedding_layer.rel_distance_extra_args",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embedding_initializer",
                        "embedding_layer.embedding_initializer",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "num_heads",
                        "transformer_decoder.layers.self_attn.num_heads",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_type",
                        "transformer_decoder.layers.self_attn.attention_type",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_module",
                        "transformer_decoder.layers.self_attn.attention_module",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_sliding_window_length",
                        "transformer_decoder.layers.self_attn.attention_sliding_window_length",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "scale_qk_dot_by_layer_idx",
                        "transformer_decoder.layers.self_attn.scale_qk_dot_by_layer_idx",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_sink_tokens",
                        "transformer_decoder.layers.self_attn.attention_sink_tokens",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_vertical_column_spacing",
                        "transformer_decoder.layers.self_attn.attention_vertical_column_spacing",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_vertical_column_width",
                        "transformer_decoder.layers.self_attn.attention_vertical_column_width",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_chunk_size",
                        "transformer_decoder.layers.self_attn.attention_chunk_size",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_qk_norm_layer",
                        "transformer_decoder.layers.self_attn.attention_qk_norm_layer",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_qk_norm_eps",
                        "transformer_decoder.layers.self_attn.attention_qk_norm_eps",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "extra_attention_params",
                        "transformer_decoder.layers.self_attn.extra_attention_params",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "extra_ffn_params",
                        "transformer_decoder.layers.self_attn.extra_ffn_params",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_inner_dim",
                        "transformer_decoder.layers.self_attn.attention_inner_dim",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "use_projection_bias_in_attention",
                        "transformer_decoder.layers.self_attn.use_projection_bias_in_attention",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "use_ffn_bias_in_attention",
                        "transformer_decoder.layers.self_attn.use_ffn_bias_in_attention",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_dropout_rate",
                        "transformer_decoder.layers.self_attn.attention_dropout_rate",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_softmax_fp32",
                        "attention_softmax_fp32",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_kernel",
                        "transformer_decoder.layers.self_attn.attention_kernel",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_logit_softcapping",
                        "transformer_decoder.layers.self_attn.attention_logit_softcapping",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "fixed_sparse_attention",
                        "transformer_decoder.layers.self_attn.fixed_sparse_attention",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "num_memory_tokens_per_chunk",
                        "transformer_decoder.layers.self_attn.num_memory_tokens_per_chunk",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attn_memory_chunk_size",
                        "transformer_decoder.layers.self_attn.attn_memory_chunk_size",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "filter_size", "transformer_decoder.layers.filter_size"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "nonlinearity",
                        "transformer_decoder.layers.nonlinearity",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "use_ffn_bias",
                        "transformer_decoder.layers.use_ffn_bias",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "norm_first", "transformer_decoder.layers.norm_first"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "norm_first_sandwich",
                        "transformer_decoder.layers.norm_first_sandwich",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "use_ff_layer1_dropout",
                        "transformer_decoder.layers.use_ff_layer1_dropout",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "num_hidden_layers",
                        "transformer_decoder.num_hidden_layers",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "dropout_rate", "transformer_decoder.dropout_rate"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "norm_type", "transformer_decoder.norm_type"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "layer_norm_epsilon",
                        "transformer_decoder.layer_norm_epsilon",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule([r'use_bias_in_output'], action=self.replaceKey),
            ConversionRule([r'initializer_range'], action=self.replaceKey),
            ConversionRule([r'initializer'], action=self.replaceKey),
            ConversionRule(
                [r'output_layer_initializer'], action=self.replaceKey
            ),
            ConversionRule([r'ffn_initializer'], action=self.replaceKey),
            ConversionRule(
                [r'ffn_output_layer_initializer'], action=self.replaceKey
            ),
            ConversionRule([r'lr_adjustment_groups'], action=self.replaceKey),
            ConversionRule([r'mup_base_hidden_size'], action=self.replaceKey),
            ConversionRule([r'mup_base_filter_size'], action=self.replaceKey),
            ConversionRule([r'embeddings_scale'], action=self.replaceKey),
            ConversionRule([r'scale_qk_dot_by_d'], action=self.replaceKey),
            ConversionRule([r'attention_logits_alpha'], action=self.replaceKey),
            ConversionRule(
                [r'scale_output_logits_by_d'], action=self.replaceKey
            ),
            ConversionRule([r'output_logits_alpha'], action=self.replaceKey),
            ConversionRule([r'mup_verification'], action=self.replaceKey),
            ConversionRule([r'output_logits_scale'], action=self.replaceKey),
            ConversionRule(
                [r'final_logit_softcapping'], action=self.replaceKey
            ),
            ConversionRule([r'moe_params'], action=self.replaceKey),
            ConversionRule([r'dtype'], action=self.replaceKey),
            ConversionRule([r'boundary_casting'], action=self.replaceKey),
            ConversionRule([r'loss_scaling'], action=self.replaceKey),
            ConversionRule([r'loss_weight'], action=self.replaceKey),
            ConversionRule([r'label_smoothing'], action=self.replaceKey),
            ConversionRule([r'z_loss_eps'], action=self.replaceKey),
        ]

    def convert_name(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            new_state_dict[new_key] = "gpt_backbone"
        else:
            new_state_dict[new_key] = "gpt2"

    def pre_config_convert(self, model, config, converter_indices):
        new_config = super().pre_config_convert(
            model, config, converter_indices
        )
        if converter_indices.direction == 1:

            def roll_dict(d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        for k2, v2 in roll_dict(v):
                            yield k + "." + k2, v2
                    else:
                        yield k, v

            new_config = {k: v for k, v in roll_dict(new_config)}
        return new_config

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
            unrolled_config = {}
            for k, v in new_config.items():
                split_k = k.split(".")
                subdict = unrolled_config
                for subk in split_k[:-1]:
                    if subk not in subdict:
                        subdict[subk] = {}
                    subdict = subdict[subk]
                subdict[split_k[-1]] = v
            new_config = unrolled_config

            if old_config.get(
                "sliding_window_every_other_decoder_layer", False
            ):
                # Need to explicitly make SWA every other decoder
                decoder_config_stack = [
                    new_config["transformer_decoder"]["layers"],
                    copy.deepcopy(new_config["transformer_decoder"]["layers"]),
                ]
                decoder_config_stack[0]["self_attn"][
                    "attention_sliding_window_length"
                ] = None

                new_config["transformer_decoder"]["layers"] = [
                    decoder_config_stack[i % 2]
                    for i in range(
                        new_config["transformer_decoder"]["num_hidden_layers"]
                    )
                ]

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
        return (FormatVersions("cs-2.4"), FormatVersions("cs-2.4-backbone"))
