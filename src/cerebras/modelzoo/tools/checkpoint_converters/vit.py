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


class Converter_ViT_Core_HF_CS21(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Embedding:
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.cls_token", "embedding_layer.cls_embedding"
                    ),
                ],
                action=self.cls_embedding_convert,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.position_embeddings",
                        "embedding_layer.position_embeddings.weight",
                    ),
                ],
                action=self.position_embeddings_convert,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.patch_embeddings.projection",
                        "embedding_layer.linear_proj",
                    ),
                    r"\.(?:weight)",
                ],
                action=self.linear_projection_convert,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.patch_embeddings.projection",
                        "embedding_layer.linear_proj",
                    ),
                    r"\.(?:bias)",
                ],
                action=self.replaceKey,
            ),
            # Encoder:
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.attention.query",
                        "self_attn.proj_q_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.attention.key",
                        "self_attn.proj_k_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.attention.value",
                        "self_attn.proj_v_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "attention.output.dense",
                        "self_attn.proj_output_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "intermediate.dense", "ffn.ffn.0.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("output.dense", "ffn.ffn.1.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("layernorm_before", "norm1"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layer",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("layernorm_after", "norm2"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "layernorm",
                        "encoder.transformer_encoder.norm",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # pooler
            ConversionRule(
                [
                    EquivalentSubkey(
                        "pooler.dense",
                        "encoder.pooler.pooler.ffn.0.linear_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    def cls_embedding_convert(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:
            new_state_dict[new_key] = old_state_dict[old_key].squeeze()
        else:
            new_state_dict[new_key] = old_state_dict[old_key].reshape(1, 1, -1)

    # This allows other models using ViT backbone to subclass and override this method
    # to handle different ways of converting linear projection weights.
    def linear_projection_convert(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        new_state_dict[new_key] = old_state_dict[old_key]

    def position_embeddings_convert(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        assert (
            action_fn_args["configs"][1]["model"]["position_embedding_type"]
            == "learned"
        ), "Only learned embeddings are supported"
        # cs vit pe puts cls token at last by default but hf put at index 0
        if from_index == 0:
            new_state_dict[new_key] = torch.cat(
                [
                    old_state_dict[old_key][0, 1:, :],
                    old_state_dict[old_key][0, :1, :],
                ],
                dim=0,
            )
        else:
            new_state_dict[new_key] = torch.cat(
                [
                    old_state_dict[old_key][-1:, :],
                    old_state_dict[old_key][:-1, :],
                ],
                dim=0,
            ).unsqueeze(0)

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_ViT_HF_CS21


class Converter_ViT_Headless_HF_CS21(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # ViTModel has a pooling layer, ViTModelForImageClassification doesn't
            ConversionRule(
                [
                    r"pooler.dense\.(?:weight|bias)",
                ],
                exists="left",
                action=None,
            ),
            # for HF without head
            ConversionRule(
                [
                    EquivalentSubkey("", "vit_model."),
                    Converter_ViT_Core_HF_CS21(),
                ],
            ),
            # drop classifier during CS -> HF
            ConversionRule(
                [
                    r"classifier.classifier.ffn.0.linear_layer\.(?:weight|bias)",
                ],
                exists="right",
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} ViTModel <-> {} ViTClassificationModel\n"
            "The HF model doesn't contain a classifier head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a classifier head initialized to default random "
            "values. When converting to HF, the classifier head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_ViT_HF_CS21

    def post_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
        key_prefix="",
    ):
        if converter_indices.direction == 0:
            # We are converting from HF ViTModel (headless) to our ViTForClassificationModel
            # We need to create 'classifier' and init to default values
            cs_config = configs[1]
            use_bias_in_output = cs_config["model"].get(
                "use_bias_in_output", False
            )
            num_classes = cs_config["model"]["num_classes"]
            embed_dim = cs_config["model"]["hidden_size"]
            classifier_weight = torch.zeros((num_classes, embed_dim))
            classifier_weight.normal_(mean=0.0, std=0.02)
            new_state_dict[
                key_prefix + "classifier.classifier.ffn.0.linear_layer.weight"
            ] = classifier_weight
            if use_bias_in_output:
                lm_head_bias = torch.zeros(num_classes)
                new_state_dict[
                    key_prefix + "classifier.classifier.ffn.0.linear_layer.bias"
                ] = lm_head_bias

        super().post_model_convert(
            old_state_dict,
            new_state_dict,
            configs,
            converter_indices,
            drop_unmatched_keys,
            key_prefix=key_prefix,
        )


class Converter_ViT_HF_CS21(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # for HF with head
            ConversionRule(
                [
                    EquivalentSubkey("vit.", "vit_model."),
                    Converter_ViT_Core_HF_CS21(),
                ],
            ),
            # classifier
            ConversionRule(
                [
                    EquivalentSubkey(
                        "classifier", "classifier.classifier.ffn.0.linear_layer"
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} ViTForImageClassification <-> {} ViTClassificationModel".format(
                cls.formats()[0], cls.formats()[1]
            )
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_ViT_HF_CS21


class ConfigConverter_ViT_Core_HF_CS21(BaseConfigConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(0, "vit"),
            ),
            ConversionRule(
                ["use_post_embed_layer_norm"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["hidden_size"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["num_hidden_layers"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("layer_norm_eps", "layer_norm_epsilon")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("num_attention_heads", "num_heads")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["attention_type"],
                action=BaseConfigConverter.assert_factory_fn(
                    1, "scaled_dot_product"
                ),
            ),
            ConversionRule(
                [EquivalentSubkey("hidden_dropout_prob", "dropout_rate")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [EquivalentSubkey("hidden_act", "nonlinearity")],
                action=self.convert_nonlinearity,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_probs_dropout_prob", "attention_dropout_rate"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["use_projection_bias_in_attention"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_ffn_bias_in_attention"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                [EquivalentSubkey("intermediate_size", "filter_size")],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["use_ffn_bias"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["initializer_range"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["image_size"],
                action=self.convert_image_patch_size,
            ),
            ConversionRule(
                ["num_channels"],
                action=self.replaceKey,
            ),
            ConversionRule(
                ["patch_size"],
                action=self.convert_image_patch_size,
            ),
            ConversionRule(
                ["use_conv_patchified_embedding"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
        ]

        self.pre_convert_defaults[0].update(
            {
                "attention_probs_dropout_prob": 0.0,
                "encoder_stride": 16,
                "hidden_act": "gelu",
                "hidden_dropout_prob": 0.0,
                "hidden_size": 768,
                "image_size": 224,
                "initializer_range": 0.02,
                "intermediate_size": 3072,
                "layer_norm_eps": 1e-12,
                "model_type": "vit",
                "num_attention_heads": 12,
                "num_channels": 3,
                "num_hidden_layers": 12,
                "patch_size": 16,
                "qkv_bias": True,
            }
        )
        self.pre_convert_defaults[1].update(
            {
                "use_conv_patchified_embedding": True,
                "prepend_cls_token": True,
                "use_encoder_pooler_layer": True,
                "position_embedding_type": "learned",
                "num_classes": 2,
            },
        )

        self.post_convert_defaults[0].update(
            {
                "model_type": "vit",
            }
        )
        self.post_convert_defaults[1].update(
            {
                "use_conv_patchified_embedding": True,
                "prepend_cls_token": True,
                "use_encoder_pooler_layer": True,
                "position_embedding_type": "learned",
                "num_classes": 2,
                "use_bias_in_output": True,
            }
        )

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
        config = super().pre_config_convert(model, config, converter_indices)

        if (
            converter_indices.direction == 0
            and "encoder_stride" in config
            and config["encoder_stride"] != config["patch_size"]
        ):
            raise ConfigConversionError(
                f"{self.formats()[1]} model only supports encoder_stride == patch_size"
            )

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
        if converter_indices.direction == 1:
            if "encoder_stride" not in new_config:
                new_config["encoder_stride"] = new_config["patch_size"]
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
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1", "cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )


class ConfigConverter_ViT_HF_CS21(ConfigConverter_ViT_Core_HF_CS21):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [EquivalentSubkey("num_labels", "num_classes")],
                action=self.replaceKey,
            ),
            *self.rules,
        ]

        self.pre_convert_defaults[1].update(
            {
                "use_encoder_pooler_layer": False,
                "num_classes": 2,
            },
        )

        self.post_convert_defaults[1].update(
            {
                "use_encoder_pooler_layer": False,
                "num_classes": 2,
            }
        )
