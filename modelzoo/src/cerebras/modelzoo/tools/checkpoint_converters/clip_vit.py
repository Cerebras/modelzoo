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

# CLIPVision Model Checkpoint and ConfigConvertor

from typing import Tuple

import torch

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseConfigConverter,
    ConversionRule,
    EquivalentSubkey,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.vit import (
    ConfigConverter_ViT_HF_CS21,
)


# Checkpoint Converters
# Mapping HF `CLIPVisionTransformer` <-> CS `ViTModel.py/ViTModel`
class Converter_CLIPViT_Core_HF_CS21(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # Embedding:
            ConversionRule(
                [r"embeddings\.position_ids"], exists="left", action=None
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.class_embedding",
                        "embedding_layer.cls_embedding",
                    ),
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.position_embedding.weight",
                        "embedding_layer.position_embeddings.weight",
                    ),
                ],
                action=self.position_embeddings_convert,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "embeddings.patch_embedding",
                        "embedding_layer.linear_proj",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "pre_layrnorm",
                        "embedding_layer.post_embed_ln",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            # Encoder
            # Sample mapping for layer `0`:
            # HF: encoder.layers.0.self_attn.q_proj.weight
            # <->
            # CS: encoder.transformer_encoder.layers.0.self_attn.proj_q_dense_layer.weight
            #
            # HF: encoder.layers.0.self_attn.q_proj.bias
            # <->
            # CS: encoder.transformer_encoder.layers.0.self_attn.proj_q_dense_layer.bias
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layers",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "self_attn.q_proj",
                        "self_attn.proj_q_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layers",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "self_attn.k_proj",
                        "self_attn.proj_k_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layers",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "self_attn.v_proj",
                        "self_attn.proj_v_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layers",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey(
                        "self_attn.out_proj",
                        "self_attn.proj_output_dense_layer",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layers",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.fc1", "ffn.ffn.0.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layers",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("mlp.fc2", "ffn.ffn.1.linear_layer"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layers",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("layer_norm1", "norm1"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "encoder.layers",
                        "encoder.transformer_encoder.layers",
                    ),
                    r"\.\d+\.",
                    EquivalentSubkey("layer_norm2", "norm2"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "post_layernorm",
                        "encoder.transformer_encoder.norm",
                    ),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
        ]

    def position_embeddings_convert(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        # cs vit pe puts cls token at last by default but hf put at index 0
        if from_index == 0:
            new_state_dict[new_key] = torch.cat(
                [
                    old_state_dict[old_key][1:, :],
                    old_state_dict[old_key][:1, :],
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
            )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_CLIPViT_HF_CS21


# HF `CLIPVisionModelWithProjection` <-> CS `ViTClassificationModel`
class Converter_CLIPViT_Projection_HF_CS21(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # for HF with Projection Layer
            # First replace `vision_model.` with `vit_model.` and proceed
            # for the remaining string using rules from chained convertor
            # Prefix converted key in chained convertor with `vit_model.`
            # Sample mapping for layer `0`:
            # HF: vision_model.encoder.layers.0.self_attn.q_proj.weight
            # <->
            # CS: vit_model.encoder.transformer_encoder.layers.0.self_attn.proj_q_dense_layer.weight
            ConversionRule(
                [
                    EquivalentSubkey("vision_model.", "vit_model."),
                    Converter_CLIPViT_Core_HF_CS21(),
                ],
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "visual_projection",
                        "classifier.classifier.ffn.0.linear_layer",
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
            "{} CLIPVisionModelWithProjection <-> {} ViTClassificationModel.\n"
            "We map the projection layer in "
            " `CLIPVisionModelWithProjection` model to "
            "classifier layer in CS"
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_CLIPViT_HF_CS21


# HF `CLIPVisionModel` <-> CS `ViTClassificationModel`
class Converter_CLIPViT_HF_CS21(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            # for HF without Projection Layer
            ConversionRule(
                [
                    EquivalentSubkey("vision_model.", "vit_model."),
                    Converter_CLIPViT_Core_HF_CS21(),
                ],
            ),
            # Drop Classifier weights in CS ckpt during conversion fron CS -> HF
            ConversionRule(
                [
                    "classifier.classifier.ffn.0.linear_layer",
                    r"\.(?:weight|bias)",
                ],
                exists="right",
                action=None,
            ),
        ]

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.1"),
        )

    @classmethod
    def converter_note(cls) -> str:
        return (
            "{} CLIPVisionModel <-> {} ViTClassificationModel\n"
            "The HF model doesn't contain a classifier head while the CS "
            "one does. When converting to CS, the exported checkpoint will "
            "contain a classifier head initialized to default random "
            "values. When converting to HF, the classifier head will be "
            "dropped."
        ).format(cls.formats()[0], cls.formats()[1])

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_CLIPViT_HF_CS21

    def post_model_convert(
        self,
        old_state_dict,
        new_state_dict,
        configs,
        converter_indices,
        drop_unmatched_keys,
        key_prefix="",
    ):
        if converter_indices.direction == 0:  # HF -> CS
            # We are converting from HF CLIPViTModel (does not include projection layer
            # to our ViTForClassificationModel
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


# Config Converters  HF CLIPVisionConfig <-> CS ViTClassificationWrapperModel config
class ConfigConverter_CLIPViT_HF_CS21(ConfigConverter_ViT_HF_CS21):
    def __init__(self):
        super().__init__()

        clip_vision_rules = [
            ConversionRule(
                ["model_type"],
                action=BaseConfigConverter.assert_factory_fn(
                    0, "clip_vision_model"
                ),
            ),
            ConversionRule(
                ["use_post_embed_layer_norm"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, True),
            ),
            ConversionRule(
                ["use_embed_proj_bias"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                ["use_bias_in_output"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, False),
            ),
            ConversionRule(
                [EquivalentSubkey("projection_dim", "num_classes")],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "attention_dropout", "attention_dropout_rate"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(["dropout_rate"], exists="right", action=None),
            ConversionRule(
                ["position_embedding_type"],
                exists="right",
                action=BaseConfigConverter.assert_factory_fn(1, "learned"),
            ),
        ]

        # Since rule matching stops when the first match occurs,
        # `model_type` would get checked against `clip_vision_model` instead
        # of parent class `vit`
        self.rules = clip_vision_rules + self.rules

        del self.pre_convert_defaults[0]["attention_probs_dropout_prob"]
        del self.pre_convert_defaults[0]["encoder_stride"]
        del self.pre_convert_defaults[0]["hidden_dropout_prob"]
        del self.pre_convert_defaults[0]["qkv_bias"]

        self.pre_convert_defaults[0].update(
            {
                "attention_dropout": 0.0,
                "hidden_act": "quick_gelu",
                "initializer_factor": 1.0,
                "layer_norm_eps": 1.0e-05,
                "model_type": "clip_vision_model",
                "patch_size": 32,
                "projection_dim": 512,
            }
        )  # HF

        self.pre_convert_defaults[1].update(
            {
                "use_post_embed_layer_norm": True,
                "dropout_rate": 0.0,
                "use_embed_proj_bias": False,
                "use_bias_in_output": False,
                "use_encoder_pooler_layer": False,
                "attention_type": "scaled_dot_product",
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "use_ffn_bias": True,
                "num_classes": 512,
            }
        )  # CS

        self.post_convert_defaults[0].update(
            {"model_type": "clip_vision_model"}
        )  # HF

        self.post_convert_defaults[1].update(
            {
                "use_post_embed_layer_norm": True,
                "num_classes": 512,
                "dropout_rate": 0.0,
                "use_embed_proj_bias": False,
                "use_bias_in_output": False,
                "attention_type": "scaled_dot_product",
                "use_encoder_pooler_layer": False,
                "position_embedding_type": "learned",
                "norm_first": True,
                "use_projection_bias_in_attention": True,
                "use_ffn_bias_in_attention": True,
                "use_ffn_bias": True,
            }
        )  # CS

    def post_config_convert(
        self,
        model,
        original_config,
        old_config,
        new_config,
        converter_indices,
        drop_unmatched_keys,
    ):
        model_config = super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )
        # Since super method adds `encoder_stride`
        if converter_indices.direction == 1:
            if "encoder_stride" in new_config:
                del new_config["encoder_stride"]

        return model_config
