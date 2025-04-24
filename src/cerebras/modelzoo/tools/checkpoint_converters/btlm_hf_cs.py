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
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.gpt2_hf_cs import (
    ConfigConverter_GPT2Model_HF_CS20,
    Converter_GPT2_Attention_HF_CS17,
    Converter_GPT2LMHeadModel_CS20_CS21,
    Converter_GPT2Model_HF_CS17,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
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
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "mlp.c_fc2", "ffn.ffn.0.linear_layer_for_glu"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.ffn_converter(),
            ),
            *self.rules,
        ]

    def attention_converter_class(self):
        return Converter_GPT2_Attention_HF_CS17(generate_hf_biases=False)

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

    def attempt_mup_to_sp(self) -> bool:
        return False


class Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS20(
    BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [r"lm_head\.(?:weight|bias)"],
                action=self.replaceKey,
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

    def attempt_mup_to_sp(self) -> bool:
        return False


Converter_BTLMModel_HF_CS20 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BTLMModel_HF_CS20",
    Converter_BTLMModel_WithoutModelPrefix_HF_CS20,
    derived_class=Converter_BTLMModel_WithoutModelPrefix_HF_CS20,
)


Converter_BTLMLMHeadModel_HF_CS20 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BTLMLMHeadModel_HF_CS20",
    Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS20,
    derived_class=Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS20,
)


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
            ConversionRule(
                [
                    EquivalentSubkey(
                        "scale_attn_by_inverse_layer_idx",
                        "scale_qk_dot_by_layer_idx",
                    )
                ],
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
                    "AutoModelForCausalLM": (
                        "cerebras/btlm-3b-8k-base--modeling_btlm.BTLMLMHeadModel"
                    ),
                },
            }
        )

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        if converter_indices.direction == 1:
            if (
                "optimizer" not in config
                or "adjust_learning_rate" not in config["optimizer"]
                or "decoder_kernel"
                not in config["optimizer"]["adjust_learning_rate"]
            ):
                logging.warning(
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

        config = super().pre_config_convert(model, config, converter_indices)

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
        final_config = super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )

        # We need to inject muP decoder_kernel parameter which is stored in
        # optimizer rather than model:
        if converter_indices.direction == 0:
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

    def attempt_mup_to_sp(self) -> bool:
        return False


###########################################################
# In CS 2.1, we refactored the embedding layer.
# CS 2.0 <> CS 2.1. We don't need a separate HF <> CS 2.1 converters since
# HF only supports RoPE which doesn't produce any checkpoint keys.
###########################################################


class Converter_BTLMLMHeadModel_CS20_CS21(Converter_GPT2LMHeadModel_CS20_CS21):
    @classmethod
    def converter_note(cls) -> str:
        return "GPT2LMHeadModel class (configured as BTLM)"


class ConfigConverter_BTLMModel_HF_CS21(ConfigConverter_BTLMModel_HF_CS20):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [EquivalentSubkey("alibi_scaling", "pos_scaling_factor")],
                action=self.convert_pi,
            ),
            *self.rules,
        ]

        self.pre_convert_defaults[0].update(
            {
                "alibi_scaling": None,
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
                train_seq_len = old_state_dict[old_key].get(
                    "train_seq_len", None
                )
                if train_seq_len is not None:
                    raise ValueError(
                        f"Only `alibi_scaling` fixed linear scaling is currently supported, "
                        f"but got train_seq_len is `{train_seq_len}` which requires support "
                        f"for dynamic linear scaling."
                    )
                new_state_dict[new_key] = old_state_dict[old_key]["factor"]
        else:
            if old_state_dict[old_key] == 1.0:
                new_state_dict[new_key] = None
            else:
                new_state_dict[new_key] = {
                    "type": "linear",
                    "factor": old_state_dict[old_key],
                }

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class Converter_BTLMModel_WithoutModelPrefix_HF_CS21(
    Converter_BTLMModel_WithoutModelPrefix_HF_CS20
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "wpe", "embedding_layer.position_embeddings.embed"
                    ),
                    "\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "relative_pe.slopes",
                        "embedding_layer.position_embed_helper.slopes",
                    ),
                ],
                action=self.replaceKey,
            ),
            *self.rules,
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_BTLMModel_HF_CS21


Converter_BTLMModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BTLMModel_HF_CS21",
    Converter_BTLMModel_WithoutModelPrefix_HF_CS21,
    derived_class=Converter_BTLMModel_WithoutModelPrefix_HF_CS21,
)


class Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS21(
    Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS20
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
                    Converter_BTLMModel_WithoutModelPrefix_HF_CS21(),
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
        return ConfigConverter_BTLMModel_HF_CS21


Converter_BTLMLMHeadModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_BTLMLMHeadModel_HF_CS21",
    Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS21,
    derived_class=Converter_BTLMLMHeadModel_WithoutModelPrefix_HF_CS21,
)
