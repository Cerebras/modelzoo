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

from typing import Tuple

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseConfigConverter,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    ConfigConverter_LLaMa_HF_CS21,
    Converter_LlamaForCausalLM_HF_CS21,
    Converter_LlamaModel_HF_CS21,
)


class Converter_Qwen2Model_HF_CS25(Converter_LlamaModel_HF_CS21):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Qwen2_HF_CS25

    @classmethod
    def converter_note(cls) -> str:
        return (
            f"{cls.formats()[0]} Qwen2Model <-> {cls.formats()[1]} GPT2LMHeadModel (configured as "
            f"Qwen2)\nThe HF model doesn't contain a language model head while the CS one does. "
            f"When converting to CS, the exported checkpoint will contain a language model head "
            f"initialized to default random values. When converting to HF, the language model head "
            f"will be dropped."
        ).format(cls.formats()[0], cls.formats()[1])


class Converter_Qwen2ForCausalLM_HF_CS25(Converter_LlamaForCausalLM_HF_CS21):
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_Qwen2_HF_CS25

    @classmethod
    def converter_note(cls) -> str:
        return "{} Qwen2ForCausalLM <-> {} GPT2LMHeadModel (configured as Qwen2)".format(
            cls.formats()[0], cls.formats()[1]
        )


class ConfigConverter_Qwen2_HF_CS25(ConfigConverter_LLaMa_HF_CS21):
    def __init__(self):
        self.model_type = "qwen2"
        super().__init__()
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
                action=BaseConfigConverter.assert_factory_fn(1, [True, False]),
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

        self.post_convert_defaults[0].update(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
            }
        )

        self.post_convert_defaults[1].update(
            {
                "use_projection_bias_in_attention": True,  # Qwen2 uses projection bias
            }
        )

    def handle_sliding_window(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if (
            'use_sliding_window' in old_state_dict
            and old_state_dict['use_sliding_window']
        ):
            new_state_dict[new_key] = old_state_dict[new_key]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.5"),
        )
