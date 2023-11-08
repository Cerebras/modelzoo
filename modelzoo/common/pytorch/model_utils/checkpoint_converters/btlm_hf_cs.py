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
    BaseConfigConverter,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from modelzoo.common.pytorch.model_utils.checkpoint_converters.gpt2_hf_cs import (
    ConfigConverter_GPT2Model_HF_CS20,
    Converter_GPT2Model_HF_CS17,
)


class Converter_BTLMModel_WithoutModelPrefix_HF_CS20(
    Converter_GPT2Model_HF_CS17
):
    def __init__(self) -> None:
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "relative_pe.slopes", "relative_pe_helper.slopes"
                    ),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("h", "transformer_decoder.layers"),
                    "\.\d+\.",
                    EquivalentSubkey(
                        "mlp.c_fc2", "ffn.ffn.0.linear_layer_for_glu"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.ffn_converter(),
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("hf", "cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} BTLMModel <-> {} GPT2LMHeadModel (configured as BTLM)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BTLMModel_HF_CS20


class Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS20(
    BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["lm_head\.(?:weight|bias)"], action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("transformer.", ""),
                    Converter_BTLMModel_WithoutModelPrefix_HF_CS20(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("hf", "cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} BTLMLMHeadModel <-> {} GPT2LMHeadModel (configured as BTLM)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BTLMModel_HF_CS20


class Converter_BTLMModel_HF_CS20(
    Converter_BTLMModel_WithoutModelPrefix_HF_CS20
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [Converter_BTLMModel_WithoutModelPrefix_HF_CS20(),],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BTLMModel_WithoutModelPrefix_HF_CS20(),
                ],
                action=None,
            ),
        ]


class Converter_BTLMLMHeadModel_HF_CS20(
    Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS20
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS20(),],
                action=None,
            ),
            # Catch checkpoints from 1.7/1.8
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS20(),
                ],
                action=None,
            ),
        ]


class ConfigConverter_BTLMModel_HF_CS20(ConfigConverter_GPT2Model_HF_CS20):
    def __init__(self) -> None:
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "btlm"),
            ),
            # alibi and mup parameters
            ConversionRule(
                [
                    EquivalentSubkey(
                        "position_embedding_type", "position_embedding_type"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("mup_output_alpha", "output_logits_scale")],
                action=self.convert_mup_output,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "mup_scale_qk_dot_by_d", "scale_qk_dot_by_d"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("mup_embeddings_scale", "embeddings_scale")],
                action=self.replaceKey,
            ),
            *self.rules,
        ]

        self.pre_convert_defaults[0].update({"mup_width_scale": 1.0})
        self.pre_convert_defaults[1].update({"decoder_kernel": 1.0})

        self.post_convert_defaults[0].update(
            {
                "model_type": "btlm",
                "use_cache": True,
                "auto_map": {
                    "AutoConfig": "cerebras/btlm-3b-8k-base--configuration_btlm.BTLMConfig",
                    "AutoModel": "cerebras/btlm-3b-8k-base--modeling_btlm.BTLMModel",
                    "AutoModelForCausalLM": "cerebras/btlm-3b-8k-base--modeling_btlm.BTLMLMHeadModel",
                },
            }
        )

    def pre_config_convert(
        self, config, from_index,
    ):
        if from_index == 1:
            if (
                "optimizer" not in config
                or "adjust_learning_rate" not in config["optimizer"]
                or "decoder_kernel"
                not in config["optimizer"]["adjust_learning_rate"]
            ):
                logging.warn(
                    "The provided config is missing the following muP parameter"
                    " (which is required for BTLM):\noptimizer:\n"
                    "\tadjust_learning_rate:\n\t\tdecoder_kernel\n"
                    "Please make sure that you're using a muP config.\n"
                    "Proceeding with a default value of decoder_kernel={}".format(
                        self.pre_convert_defaults[1]["decoder_kernel"]
                    )
                )
            else:
                config["model"]["decoder_kernel"] = config["optimizer"][
                    "adjust_learning_rate"
                ]["decoder_kernel"]

        config = super().pre_config_convert(config, from_index)

        return config

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        from_index,
        drop_unmatched_keys,
    ):

        final_config = super().post_config_convert(
            original_config,
            old_config,
            new_config,
            from_index,
            drop_unmatched_keys,
        )

        # We need to inject muP decoder_kernel parameter which is stored in
        # optimizer rather than model:
        if from_index == 0:
            final_config["optimizer"] = {
                "adjust_learning_rate": {
                    "decoder_kernel": old_config["mup_width_scale"]
                }
            }

        return final_config

    def convert_mup_output(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            new_state_dict[new_key] = (
                old_state_dict[old_key] * old_state_dict["mup_width_scale"]
            )
            # mup_width_scale -> decoder_kernel handled in post_config_convert
        else:
            wscale = old_state_dict["decoder_kernel"]
            new_state_dict[new_key] = old_state_dict[old_key] / wscale
            new_state_dict["mup_width_scale"] = wscale
