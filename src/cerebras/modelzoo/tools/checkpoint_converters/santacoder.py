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
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.gpt2_hf_cs import (
    ConfigConverter_GPT2Model_HF_CS20,
    Converter_GPT2LMHeadModel_CS20_CS21,
    Converter_GPT2Model_HF_CS17,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
    transpose_key_if_2D,
)


class Converter_Santacoder_Attention_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("c_proj", "proj_output_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=transpose_key_if_2D,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("q_attn", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=transpose_key_if_2D,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("kv_attn", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.kv_attn_converter,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("kv_attn", "proj_v_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.assert_already_converted,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-X.X"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return None

    def kv_attn_converter(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            self.kv_attn_converter_hf_to_cs(
                old_key, new_key, old_state_dict, new_state_dict, action_fn_args
            )
        else:
            self.kv_attn_converter_cs_to_hf(
                old_key, new_key, old_state_dict, new_state_dict, action_fn_args
            )

    def kv_attn_converter_hf_to_cs(
        self, old_key, new_key, old_state_dict, new_state_dict, action_fn_args
    ):
        # HF represents K and V in a packed format. We need to unpack the
        # weight and bias tensor for CS format.
        k_key = new_key
        v_key = re.sub(r"\.proj_k_dense_layer\.", ".proj_v_dense_layer.", k_key)
        if new_key.endswith(".bias"):
            assert len(old_state_dict[old_key].shape) == 1
            (
                new_state_dict[k_key],
                new_state_dict[v_key],
            ) = torch.chunk(old_state_dict[old_key], 2, dim=0)
        elif new_key.endswith(".weight"):
            (
                new_state_dict[k_key],
                new_state_dict[v_key],
            ) = torch.chunk(
                torch.transpose(old_state_dict[old_key], 0, 1), 2, dim=0
            )
        else:
            raise ValueError("Invalid key after conversion: {}".format(new_key))

    def kv_attn_converter_cs_to_hf(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        action_fn_args,
    ):
        # HF represents K and V in a packed format. It also contains
        # special ".bias" and ".masked_bias" register buffers that need to be
        # initialize
        k_key = old_key
        v_key = re.sub(r"\.proj_k_dense_layer\.", ".proj_v_dense_layer.", k_key)

        assert (
            k_key in old_state_dict
        ), "Expected the following key to exist! {}".format(k_key)
        assert (
            v_key in old_state_dict
        ), "Expected the following key to exist! {}".format(v_key)

        new_state_dict[new_key] = torch.cat(
            (
                old_state_dict[k_key],
                old_state_dict[v_key],
            ),
            dim=0,
        )

        # Need to transpose to convert from Linear.weight -> Conv1D.weight
        if len(new_state_dict[new_key].shape) == 2:
            new_state_dict[new_key] = torch.transpose(
                new_state_dict[new_key], 0, 1
            )

        if new_key.endswith(".bias"):
            max_position_embeddings = action_fn_args["configs"][1]["model"][
                "max_position_embeddings"
            ]
            attn_bias_key = re.sub(r"\.kv_attn\.", ".", new_key)
            new_state_dict[attn_bias_key] = torch.tril(
                torch.ones(
                    (max_position_embeddings, max_position_embeddings),
                    dtype=torch.uint8,
                )
            ).view(1, 1, max_position_embeddings, max_position_embeddings)
            masked_bias_key = re.sub(r"\.kv_attn\.", ".masked_", new_key)
            new_state_dict[masked_bias_key] = torch.tensor(-1e4)

    def assert_already_converted(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            # We should never hit this case as this key should have been matched
            # already
            assert False, "Invalid key: {}".format(old_key)
        else:
            # When we convert from CS -> HF, the proj_q_dense_layer should also handle
            # conversion of proj_k_dense_layer and proj_v_dense_layer since HF
            # represents these three layers in a packed format. We simply need
            # to test that the key containing the packed format has already
            # been converted.
            assert (
                new_key in new_state_dict
            ), "Key should've been already converted: {} -> {}".format(
                old_key, new_key
            )


# This is a base converter for Santacoder that inherits from GPT-2
# CS17 converter that contains most of the rules necessary for
# converting GPT-2 checkpoints. This class is meant to be used as
# an action within the rules of the CS-2.0 converter below,
# that catches checkpoints from Pytorch 2.0 API and PyTorchBaseModel.
# It is not meant for use on its own, because this model was not
# included in the codebase before release 2.0. Note that we include a
# a formats() method in this class and the SantacoderLMHeadModel
# converter below because it is a required method, due to the
# declaration as an @abstractmethod in the BaseDictionaryConverter.
# The cs-X.X in the formats() method is meant to call this to attention
class Converter_SantacoderModel_HF_CS(Converter_GPT2Model_HF_CS17):
    def attention_converter_class(self):
        return Converter_Santacoder_Attention_HF_CS()

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-X.X"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_SantacoderModel_HF_CS20


class Converter_SantacoderLMHeadModel_HF_CS(BaseCheckpointConverter_HF_CS):
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
                    Converter_SantacoderModel_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-X.X"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_SantacoderModel_HF_CS20


class Converter_SantacoderModel_HF_CS20(Converter_SantacoderModel_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_SantacoderModel_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from deprecated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_SantacoderModel_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} GPT2CustomModel <-> {} GPT2LMHeadModel (configured as SantaCoder)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_SantacoderModel_HF_CS20


class Converter_SantacoderLMHeadModel_HF_CS20(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_SantacoderLMHeadModel_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from deprecated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_SantacoderLMHeadModel_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPT2LMHeadCustomModel <-> {} GPT2LMHeadModel (configured as SantaCoder)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_SantacoderModel_HF_CS20


class ConfigConverter_SantacoderModel_HF_CS20(
    ConfigConverter_GPT2Model_HF_CS20
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey(
                        "scale_attn_by_inverse_layer_idx",
                        "scale_qk_dot_by_layer_idx",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["attention_head_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "multiquery"),
            ),
            ConversionRule(
                ["attention_module"],
                action=BaseConfigConverter.assert_factory_fn(
                    1, "multiquery_attention"
                ),
            ),
            ConversionRule(
                ["extra_attention_params"],
                action=BaseConfigConverter.assert_factory_fn(
                    1, {"num_kv_groups": 1}
                ),
            ),
            *self.rules,
        ]
        self.post_convert_defaults[0].update(
            {
                "architectures": ["GPT2LMHeadCustomModel"],
                "attention_head_type": "multiquery",
                "scale_attn_by_inverse_layer_idx": False,
                "scale_attn_weight": True,
                "auto_map": {
                    "AutoConfig": "configuration_gpt2_mq.GPT2CustomConfig",
                    "AutoModelForCausalLM": "modeling_gpt2_mq.GPT2LMHeadCustomModel",
                },
                "model_type": "gpt2",
                "reorder_and_upcast_attn": False,
                "summary_activation": None,
                "summary_first_dropout": 0.1,
                "summary_proj_to_labels": True,
                "summary_type": "cls_index",
                "summary_use_proj": True,
                "torch_dtype": "float32",
                "transformers_version": "4.24.0",
                "use_cache": True,
            },
        )
        self.post_convert_defaults[1].update(
            {
                "position_embedding_type": "learned",
                "attention_module": "multiquery_attention",
                "softmax_dtype_fp32": False,
                "scale_by_layer_index": False,
                "extra_attention_params": {"num_kv_groups": 1},
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "use_ffn_bias": True,
                "loss_scaling": "num_tokens",
                "use_bfloat16": True,
            },
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))


###########################################################
# In CS 2.1, we refactored the embedding layer.
###########################################################


class Converter_SantacoderLMHeadModel_CS20_CS21(
    Converter_GPT2LMHeadModel_CS20_CS21
):
    @classmethod
    def converter_note(cls) -> str:
        return "GPT2LMHeadModel class (configured as Santacoder)"


class ConfigConverter_SantacoderModel_HF_CS21(
    ConfigConverter_SantacoderModel_HF_CS20
):
    def __init__(self) -> None:
        super().__init__()
        del self.post_convert_defaults[1]["use_bfloat16"]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    def supports_mup_conversion(self):
        return True


class Converter_SantacoderModel_WithoutOptionalModel_HF_CS21(
    Converter_SantacoderModel_HF_CS
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
        return ConfigConverter_SantacoderModel_HF_CS21

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} GPT2CustomModel <-> {} GPT2LMHeadModel (configured as SantaCoder)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])


Converter_SantacoderModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_SantacoderModel_HF_CS21",
    Converter_SantacoderModel_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_SantacoderModel_WithoutOptionalModel_HF_CS21,
)


class Converter_SantacoderLMHeadModel_WithoutOptionalModel_HF_CS21(
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
                    Converter_SantacoderModel_WithoutOptionalModel_HF_CS21(),
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
        return ConfigConverter_SantacoderModel_HF_CS21

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPT2LMHeadCustomModel <-> {} GPT2LMHeadModel (configured as SantaCoder)".format(
            cls.formats()[0], cls.formats()[1]
        )

    def supports_mup_conversion(self):
        return True


Converter_SantacoderLMHeadModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_SantacoderLMHeadModel_HF_CS21",
    Converter_SantacoderLMHeadModel_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_SantacoderLMHeadModel_WithoutOptionalModel_HF_CS21,
)
