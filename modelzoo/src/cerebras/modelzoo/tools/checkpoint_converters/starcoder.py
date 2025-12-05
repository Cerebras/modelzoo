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
from cerebras.modelzoo.tools.checkpoint_converters.gpt2_hf_cs import (
    Converter_GPT2LMHeadModel_CS20_CS21,
    Converter_GPT2Model_HF_CS17,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)


class Converter_Starcoder_Attention_HF_CS(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("c_proj", "proj_output_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("c_attn", "proj_q_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.c_attn_converter,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("c_attn", "proj_k_dense_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.assert_already_converted,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("c_attn", "proj_v_dense_layer"),
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

    def c_attn_converter(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            self.c_attn_converter_hf_to_cs(
                old_key, new_key, old_state_dict, new_state_dict, action_fn_args
            )
        else:
            self.c_attn_converter_cs_to_hf(
                old_key, new_key, old_state_dict, new_state_dict, action_fn_args
            )

    def c_attn_converter_hf_to_cs(
        self, old_key, new_key, old_state_dict, new_state_dict, action_fn_args
    ):
        # For both MHA and MQA, the c_attn weights are packed,
        # but the weight matrix for each is a different shape.
        # MHA: weight --> 3 * embed_dim x embed_dim
        # MQA: weight --> (embed_dim + 2 * head_dim) x embed_dim
        # where embed_dim is for the Queries, and each of the 2 head_dim is
        # for one of Keys and Values
        q_key = new_key
        k_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_k_dense_layer.", q_key)
        v_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_v_dense_layer.", q_key)
        hf_config = action_fn_args["configs"][0]
        is_multiquery = hf_config["multi_query"]
        embed_dim = hf_config["n_embd"]
        n_head = hf_config["n_head"]
        d_head = int(embed_dim / n_head)
        # Note that nn.Linear stores matrices with shape [out_dim x in_dim]
        packed_dim = old_state_dict[old_key].shape[0]
        if is_multiquery:
            assert packed_dim == embed_dim + 2 * d_head, (
                f"Invalid tensor shape {old_state_dict[old_key].shape} at {old_key}. The second "
                f"dimension should be the first dimension (embed_dim) plus 2x the head_dim since "
                f"Q, K, and V are packed"
            )
            # the ellipsis handles both weight and bias. indexes all of the 2nd dim for weight and
            # no-op for bias
            q_weight, kv_weight = (
                old_state_dict[old_key][:embed_dim, ...],
                old_state_dict[old_key][embed_dim:, ...],
            )
            k_weight, v_weight = kv_weight.chunk(2, dim=0)
            (
                new_state_dict[q_key],
                new_state_dict[k_key],
                new_state_dict[v_key],
            ) = (q_weight, k_weight, v_weight)
        else:
            assert 3 * embed_dim == packed_dim, (
                f"Invalid tensor shape {old_state_dict[old_key].shape} at {old_key}. The second "
                f"dimension should be 3x the first dimension (embed_dim) since Q, K, and V are "
                f"packed"
            )
            packed_weight = old_state_dict[old_key]

            query_indices = [
                i + j
                for i in range(0, packed_dim, 3 * d_head)
                for j in range(d_head)
                if i + j < packed_dim
            ]
            key_indices = [
                i + j
                for i in range(d_head, packed_dim, 3 * d_head)
                for j in range(d_head)
                if i + j < packed_dim
            ]
            value_indices = [
                i + j
                for i in range(2 * d_head, packed_dim, 3 * d_head)
                for j in range(d_head)
                if i + j < packed_dim
            ]

            query = packed_weight[query_indices, ...]
            key = packed_weight[key_indices, ...]
            value = packed_weight[value_indices, ...]

            new_state_dict[q_key] = query
            new_state_dict[k_key] = key
            new_state_dict[v_key] = value

    def c_attn_converter_cs_to_hf(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        action_fn_args,
    ):
        # HF represents Q, K, and V in a packed format
        q_key = old_key
        k_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_k_dense_layer.", q_key)
        v_key = re.sub(r"\.proj_q_dense_layer\.", ".proj_v_dense_layer.", q_key)

        assert (
            k_key in old_state_dict
        ), "Expected the following key to exist! {}".format(k_key)
        assert (
            v_key in old_state_dict
        ), "Expected the following key to exist! {}".format(v_key)
        hf_config = action_fn_args["configs"][0]
        embed_dim = hf_config["n_embd"]
        n_head = hf_config["n_head"]
        d_head = int(embed_dim / n_head)
        is_multiquery = hf_config["multi_query"]
        # Note that nn.Linear stores matrices with shape [out_dim x in_dim]
        packed_dim = 3 * embed_dim

        if is_multiquery:
            new_state_dict[new_key] = torch.cat(
                (
                    old_state_dict[q_key],
                    old_state_dict[k_key],
                    old_state_dict[v_key],
                ),
                dim=0,
            )

        else:
            query_indices = [
                i + j
                for i in range(0, packed_dim, 3 * d_head)
                for j in range(d_head)
                if i + j < packed_dim
            ]
            key_indices = [
                i + j
                for i in range(d_head, packed_dim, 3 * d_head)
                for j in range(d_head)
                if i + j < packed_dim
            ]
            value_indices = [
                i + j
                for i in range(2 * d_head, packed_dim, 3 * d_head)
                for j in range(d_head)
                if i + j < packed_dim
            ]
            is_weight = len(old_state_dict[q_key].shape) > 1
            packed_weights = (
                torch.zeros(packed_dim, embed_dim)
                if is_weight
                else torch.zeros(packed_dim)
            )
            packed_weights[query_indices, ...] = old_state_dict[q_key]
            packed_weights[key_indices, ...] = old_state_dict[k_key]
            packed_weights[value_indices, ...] = old_state_dict[v_key]
            new_state_dict[new_key] = packed_weights

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


# This is a base converter for Starcoder that inherits from GPT-2
# CS17 converter that contains most of the rules necessary for
# converting GPT-2 checkpoints. This class is meant to be used as
# an action within the rules of the CS-2.0 converter below,
# that catches checkpoints from Pytorch 2.0 API and PyTorchBaseModel.
# It is not meant for use on its own, because this model was not
# included in the codebase before release 2.0. Note that we include a
# a formats() method in this class and the StarcoderForCausalLM
# converter below because it is a required method, due to the
# declaration as an @abstractmethod in the BaseDictionaryConverter.
# The cs-X.X in the formats() method is meant to call this to attention
class Converter_StarcoderModel_HF_CS(Converter_GPT2Model_HF_CS17):
    def attention_converter_class(self):
        return Converter_Starcoder_Attention_HF_CS()

    def ffn_converter(self):
        return self.replaceKey

    # see note above
    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-X.X"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_StarcoderModel_HF_CS20


class Converter_StarcoderForCausalLM_HF_CS(BaseCheckpointConverter_HF_CS):
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
                    Converter_StarcoderModel_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-X.X"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_StarcoderModel_HF_CS20


class Converter_StarcoderModel_HF_CS20(Converter_StarcoderModel_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_StarcoderModel_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from deprecated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_StarcoderModel_HF_CS(),
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
            "{} GPTBigCodeModel <-> {} GPT2ForCausalLM (configured as Starcoder)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_StarcoderModel_HF_CS20


class Converter_StarcoderForCausalLM_HF_CS20(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Catch checkpoints from Pytorch 2.0 API
            ConversionRule(
                [
                    Converter_StarcoderForCausalLM_HF_CS(),
                ],
                action=None,
            ),
            # Catch checkpoints from deprecated PyTorchBaseModel
            ConversionRule(
                [
                    EquivalentSubkey("", "model."),
                    Converter_StarcoderForCausalLM_HF_CS(),
                ],
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPTBigCodeForCausalLM <-> {} GPT2ForCausalLM (configured as Starcoder)".format(
            cls.formats()[0], cls.formats()[1]
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_StarcoderModel_HF_CS20


class ConfigConverter_StarcoderModel_HF_CS20(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["norm_type"],
                action=BaseConfigConverter.assert_factory_fn(1, "layernorm"),
            ),
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "gpt_bigcode"),
            ),
            # Embedding
            ConversionRule(["vocab_size"], action=self.replaceKey),
            ConversionRule(
                ["position_embedding_type"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, "learned"),
            ),
            ConversionRule(
                ["use_position_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                [EquivalentSubkey("embd_pdrop", "embedding_dropout_rate")],
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
                ["embedding_layer_norm"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            # Decoder Block
            ConversionRule(
                [EquivalentSubkey("n_embd", "hidden_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_head", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_layer", "num_hidden_layers")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("n_positions", "max_position_embeddings")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("scale_attn_weights", "attention_type")],
                action=self.convert_attention_type,
            ),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_ffn_bias_in_attention"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_ffn_bias"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                [EquivalentSubkey("n_inner", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("activation_function", "nonlinearity")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("attn_pdrop", "attention_dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("resid_pdrop", "dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(["rotary_dim"], action=self.replaceKey),
            ConversionRule(
                ["layer_norm_epsilon"],
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
                [
                    EquivalentSubkey(
                        "attention_softmax_in_fp32",
                        "attention_softmax_fp32",
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["scale_qk_dot_by_layer_idx"],
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
        ]

        # HF pre/post updates
        self.pre_convert_defaults[0].update(
            {
                "tie_word_embeddings": True,
                "multi_query": True,
                "attn_pdrop": 0.0,
                "scale_attn_weights": True,
                "resid_pdrop": 0.0,
                "embd_pdrop": 0.0,
                "n_inner": 24576,
                "n_embd": 6144,
                "n_head": 48,
                "n_layer": 40,
                "vocab_size": 49152,
                "n_positions": 8192,
            }
        )
        self.post_convert_defaults[0].update(
            {
                "model_type": "gpt_bigcode",
                "architectures": ["GPTBigCodeForCausalLM"],
                "validate_runner_input": True,
                "use_cache": True,
                "transformers_version": "4.28.1",
                "summary_use_proj": True,
                "summary_type": "cls_index",
                "inference_runner": 0,
                "eos_token_id": 0,
                "bos_token_id": 0,
                "max_sequence_length": None,
                "max_batch_size": None,
            }
        )

        # CS pre/post updates
        self.pre_convert_defaults[1].update(
            {
                "share_embedding_weights": True,
                "attention_dropout_rate": 0.0,
                "attention_module": "multiquery_attention",
                "attention_type": "scaled_dot_product",
                "scale_qk_dot_by_layer_idx": False,
                "dropout_rate": 0.0,
                "embedding_dropout_rate": 0.0,
                "filter_size": 24576,
                "hidden_size": 6144,
                "max_position_embeddings": 8192,
                "num_heads": 48,
                "num_hidden_layers": 40,
                "vocab_size": 49152,
            },
        )
        self.post_convert_defaults[1].update(
            {
                "position_embedding_type": "learned",
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "use_ffn_bias": True,
                "nonlinearity": "gelu",
                "use_bias_in_output": False,
                "loss_scaling": "num_tokens",
            }
        )

    def convert_attention_type(
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
                "scaled_dot_product"
                if old_state_dict[old_key]
                else "dot_product"
            )
            new_state_dict["attention_module"] = (
                "multiquery_attention"
                if old_state_dict["multi_query"]
                else "aiayn_attention"
            )

            if old_state_dict["multi_query"]:
                new_state_dict["extra_attention_params"] = {"num_kv_groups": 1}
        else:
            if (
                old_state_dict[old_key] != "scaled_dot_product"
                and old_state_dict[old_key] != "dot_product"
            ):
                raise ConfigConversionError(
                    "Can't convert config with {}={}. Only {} is supported.".format(
                        old_key,
                        old_state_dict[old_key],
                        "scaled_dot_product and dot_product",
                    )
                )
            new_state_dict[new_key] = old_state_dict[old_key].startswith(
                "scaled_"
            )
            is_multiquery = (
                old_state_dict["attention_module"] == "multiquery_attention"
            )
            new_state_dict["multi_query"] = is_multiquery

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        config = super().pre_config_convert(model, config, converter_indices)

        if converter_indices.direction == 0:
            if "n_inner" not in config or config["n_inner"] is None:
                config["n_inner"] = 4 * config["n_embd"]
        else:
            if "embedding_dropout_rate" not in config:
                config["embedding_dropout_rate"] = config["dropout_rate"]
            if "attention_dropout_rate" not in config:
                config["attention_dropout_rate"] = config["dropout_rate"]
        return config

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.0"))


###########################################################
# In CS 2.1, we refactored the embedding layer.
###########################################################


class Converter_StarcoderLMHeadModel_CS20_CS21(
    Converter_GPT2LMHeadModel_CS20_CS21
):
    @classmethod
    def converter_note(cls) -> str:
        return "GPT2LMHeadModel class (configured as Starcoder)"


class ConfigConverter_StarcoderModel_HF_CS21(
    ConfigConverter_StarcoderModel_HF_CS20
):
    "CS 2.1 config is the same as CS 2.0."

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    def supports_mup_conversion(self):
        return True


class Converter_StarcoderModel_WithoutOptionalModel_HF_CS21(
    Converter_StarcoderModel_HF_CS
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
        return ConfigConverter_StarcoderModel_HF_CS21

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} GPTBigCodeModel <-> {} GPT2ForCausalLM (configured as Starcoder)\n"
            "The HF model doesn't contain a language model head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a language model head initialized to default random "
            "values. When converting to HF, the language model head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])


Converter_StarcoderModel_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_StarcoderModel_HF_CS21",
    Converter_StarcoderModel_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_StarcoderModel_WithoutOptionalModel_HF_CS21,
)


class Converter_StarcoderForCausalLM_WithoutOptionalModel_HF_CS21(
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
                    Converter_StarcoderModel_WithoutOptionalModel_HF_CS21(),
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
        return ConfigConverter_StarcoderModel_HF_CS21

    @classmethod
    def converter_note(cls) -> str:
        return "{} GPTBigCodeForCausalLM <-> {} GPT2ForCausalLM (configured as Starcoder)".format(
            cls.formats()[0], cls.formats()[1]
        )

    def supports_mup_conversion(self):
        return True


Converter_StarcoderForCausalLM_HF_CS21 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_StarcoderForCausalLM_HF_CS21",
    Converter_StarcoderForCausalLM_WithoutOptionalModel_HF_CS21,
    derived_class=Converter_StarcoderForCausalLM_WithoutOptionalModel_HF_CS21,
)
