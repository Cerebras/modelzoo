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
import math
import os
import re
from collections import OrderedDict
from typing import List, Tuple

import torch

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseCheckpointConverter_HF_CS,
    BaseCheckpointConverter_UnpackedHF_PackedCS,
    BaseConfigConverter,
    BaseConfigConverter_HF_CS,
    BaseConfigConverter_UnpackedHF_PackedCS,
    ConversionRule,
    EquivalentSubkey,
    FormatIndices,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.clip_vit import (
    ConfigConverter_CLIPViT_HF_CS21,
    Converter_CLIPViT_Core_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.helper import (
    Build_HF_CS_Converter_WithOptionalModel,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    ConfigConverter_LLaMa_HF_CS21,
    Converter_LlamaModel_HF_CS,
)


# HF `CLIPVisionModel` <-> CS `modeling_llava.LLaVA.image_model`
class Converter_LLaVA_CLIPViT_WithoutModel_HF_CS22(
    BaseCheckpointConverter_HF_CS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("vision_model.", "image_model."),
                    Converter_CLIPViT_Core_HF_CS21(),
                ],
            ),
            # To handle cases where the ckpt corresponds to CLIPModel instead of CLIPVisionModel
            ConversionRule(["text_model.*"], action=None),
            ConversionRule(["logit_scale.*"], exists="left", action=None),
            # visual_projection and text_projection in HF
            ConversionRule(
                [r"visual_projection\.(?:weight|bias)"],
                action=None,
            ),
            ConversionRule(
                [
                    r"text_projection\.(?:weight|bias)",
                ],
                action=None,
            ),
            ConversionRule(
                ["projector_image_model.*"], exists="right", action=None
            ),
        ]

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # Normally this does output_checkpoint["model"] = {} and then we
        # reference output_checkpoint["model"] later in extract_model_dict.
        # We don't want to reset the output_checkpoint["model"] here though
        # because we will store the keys under the same "model" key
        # created by this function during the component conversion
        if converter_indices.direction == 0:
            pass

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaVA_HF_CS22


class Converter_LLaVA_LLaMA_WithoutModel_HF_CS22(BaseCheckpointConverter_HF_CS):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule([r"image_model.*"], exists="right", action=None),
            # match LM head here
            ConversionRule(
                [
                    EquivalentSubkey("", "text_model."),
                    r"lm_head\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("model.", "text_model."),
                    Converter_LlamaModel_HF_CS(),
                ],
            ),
            # projector_image_model
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.mm_projector", "projector_image_model.ffn"
                    ),
                    r"\.\d+",
                    EquivalentSubkey(".", ".linear_layer."),
                    r"(?:weight|bias)",
                ],
                action=self.convert_projector,
            ),
            # Ignore vision_tower keys if present in LLaVA-LLaMA checkpoint
            # since we are using separate checkpoints
            # i.e a pretrained checkpoint for vision_tower
            # and a separate checkpoint for LLM and projector parts
            ConversionRule(
                [
                    r"model.vision_tower.*",
                ],
                exists="left",
                action=None,
            ),
            # projector_text_model if exists
            ConversionRule(
                [r"projector_text_model.*"], exists="right", action=None
            ),
        ]

    def convert_projector(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        layer_num_old = re.findall("\d+", old_key)
        layer_num_new = re.findall("\d+", new_key)
        assert (
            len(layer_num_old) == 1
        ), f"Cannot have nested Sequential in model"
        assert (
            len(layer_num_new) == 1
        ), f"Cannot have nested Sequential in model"

        if from_index == 0:
            new_key = new_key.replace(
                layer_num_new[0], str(int(layer_num_old[0]) // 2)
            )
        else:
            new_key = new_key.replace(
                layer_num_new[0], str(int(layer_num_old[0]) * 2)
            )

        new_state_dict[new_key] = old_state_dict[old_key]

    def pre_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # Normally this does output_checkpoint["model"] = {} and then we
        # reference output_checkpoint["model"] later in extract_model_dict.
        # We don't want to reset the output_checkpoint["model"] here though
        # because we will store the keys under the same "model" key
        # created by this function during the component conversion
        if converter_indices.direction == 0:
            pass

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaVA_HF_CS22


Converter_LLaVA_CLIPViT_HF_CS22 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_LLaVA_CLIPViT_HF_CS22",
    Converter_LLaVA_CLIPViT_WithoutModel_HF_CS22,
    derived_class=Converter_LLaVA_CLIPViT_WithoutModel_HF_CS22,
)

Converter_LLaVA_LLaMA_HF_CS22 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_LLaVA_LLaMA_HF_CS22",
    Converter_LLaVA_LLaMA_WithoutModel_HF_CS22,
    derived_class=Converter_LLaVA_LLaMA_WithoutModel_HF_CS22,
)


class Converter_LLaVA_WithoutModel_HF_CS22(
    BaseCheckpointConverter_UnpackedHF_PackedCS
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(["image_model.*"], exists="right", action=None),
            ConversionRule(["text_model.*"], exists="right", action=None),
            ConversionRule(
                ["projector_image_model.*"], exists="right", action=None
            ),
            ConversionRule(
                ["projector_text_model.*"], exists="right", action=None
            ),
        ]

    @staticmethod
    def converters():
        return (
            Converter_LLaVA_CLIPViT_HF_CS22,
            Converter_LLaVA_LLaMA_HF_CS22,
        )

    @staticmethod
    def component_names():
        return ("image_model", "text_model")

    def post_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        if converter_indices.direction == 0:  # HF -> CS
            # We are converting from HF
            # to our LLaVA model. We need to create the visual token `projection`
            # layer and init to default values for phase 1
            is_projector_exists = any(
                [
                    "projector_image_model" in k
                    for k in output_checkpoint["model"].keys()
                ]
            )
            if not is_projector_exists:
                logging.info(
                    f"---- HF checkpoint does not have projector weight, initializing defaults"
                )
                cs_config = configs[1]

                im_proj_config = cs_config["model"]["projector"]["image_model"]

                input_unit = im_proj_config["input_unit"]
                layers_units = im_proj_config["layers_units"]
                use_bias = im_proj_config["use_bias"]

                input_ = [input_unit] + layers_units[:-1]
                output_ = layers_units
                for i, (inp, out) in enumerate(zip(input_, output_)):
                    scale = math.sqrt(1.0 / inp)
                    projection_weight = torch.zeros(out, inp)
                    projection_weight.uniform_(-scale, scale)
                    output_checkpoint["model"][
                        f"projector_image_model.ffn.{i}.linear_layer.weight"
                    ] = projection_weight
                    if use_bias:
                        projection_bias = torch.zeros(out)
                        projection_bias.uniform_(-scale, scale)
                        output_checkpoint["model"][
                            f"projector_image_model.ffn.{i}.linear_layer.bias"
                        ] = projection_bias

            super().post_checkpoint_convert(
                input_checkpoint,
                output_checkpoint,
                configs,
                converter_indices,
            )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def architectures() -> Tuple[List[str], str]:
        return (
            (
                "ViTModel",
                "LlamaModel",
            ),
            "LLaVAModel",
        )

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_LLaVA_HF_CS22

    @classmethod
    def converter_note(cls) -> str:
        note = super().converter_note()
        return (
            note + f"LLaVA convertor for CLIP-ViT and LLaMA backbones "
            f"for `image_model` and `text_model`. "
            f"Inorder to use the convertor {cls.formats()[0]} -> {cls.formats()[1]}, "
            f"the CLIP-ViT checkpoint, config and preprocessor_config should be "
            f"saved under `image_model` directory and LLaMA checkpoint including tokenizer files "
            f"should be saved under `text_model` directory. "
            f"Also, the convertor from {cls.formats()[0]} -> {cls.formats()[1]} "
            f"expects the `config.json` file for the `text_model` to include LLaVA specific "
            f"config parameters. The easy way is to download the LLaVA `config.json` and "
            f"modify the necessary parameters that reflect the LLaMA checkpoint being used."
            f"Please refer to modelzoo/models/multimodal/llava/README.md "
            f"for an example setup."
        )


Converter_LLaVA_HF_CS22 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_LLaVA_HF_CS22",
    Converter_LLaVA_WithoutModel_HF_CS22,
    derived_class=Converter_LLaVA_WithoutModel_HF_CS22,
)


class ConfigConverter_LLaVA_HF_CS22(BaseConfigConverter_UnpackedHF_PackedCS):
    # HF preprocessor config
    preprocessor_config_defaults = {
        "crop_size": 224,
        "do_center_crop": True,
        "do_normalize": True,
        "do_resize": True,
        "feature_extractor_type": "CLIPFeatureExtractor",
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "resample": 3,
        "size": 224,
    }

    def __init__(self):
        super().__init__()
        self.rules = []

        # CS config
        self.post_convert_defaults[1].update(
            {
                "loss_weight": 1.0,
                "loss_scaling": "num_tokens",
                "freeze": ['^image_model'],
                "label_smoothing": 0.0,
                "z_loss_eps": 0.0,
                "image_start_idx": 1,
                "image_feature_select_mode": "patch",
            }
        )

    @classmethod
    def converter_note(cls) -> str:
        return (
            f"LLaVA convertor for CLIP-ViT and LLaMA backbones "
            f"for `image_model` and `text_model`. "
            f"Inorder to use the convertor {cls.formats()[0]} -> {cls.formats()[1]}, "
            f"the CLIP-ViT checkpoint, config and preprocessor_config should be "
            f"saved under `image_model` directory and LLaMA checkpoint including tokenizer files "
            f"should be saved under `text_model` directory. "
            f"Also, the convertor from {cls.formats()[0]} -> {cls.formats()[1]} "
            f"expects the `config.json` file for the `text_model` to include LLaVA specific "
            f"config parameters. The easy way is to download the LLaVA `config.json` and "
            f"modify the necessary parameters that reflect the LLaMA checkpoint being used."
            f"Please refer to modelzoo/models/multimodal/llava/README.md "
            f"for an example setup."
        )

    @classmethod
    def save(
        cls,
        file_without_ext: str,
        config: OrderedDict,
        converter_indices: FormatIndices,
        **kwargs,
    ) -> str:
        # saving CS requires only saving once
        if converter_indices.direction == 0:  # HF -> CS
            return super().save(
                file_without_ext, config, converter_indices, **kwargs
            )
        # saving HF requires separating encoders and saving both
        else:
            save_files = []
            dir = os.path.dirname(file_without_ext)
            for i, name in enumerate(cls.component_names()):
                path = os.path.join(dir, name, "config")
                if not os.path.exists(os.path.join(dir, name)):
                    os.mkdir(os.path.join(dir, name))
                if name == "text_model":
                    # add path to folder containing
                    # image model in text_model config
                    config[i]["mm_vision_tower"] = os.path.dirname(
                        save_files[0]
                    )
                if name == "image_model":
                    preprocess_path = path.replace(
                        "config", "preprocessor_config"
                    )
                    # Save preprocessor config after the dir is created
                    BaseConfigConverter_HF_CS.save(
                        preprocess_path,
                        cls.preprocessor_config_defaults,
                        converter_indices,
                        **kwargs,
                    )
                save_file = BaseConfigConverter_HF_CS.save(
                    path, config[i], converter_indices, **kwargs
                )
                save_files.append(save_file)

            return save_files

    def post_config_convert(
        self,
        model,
        original_config,
        old_config,
        new_config,
        converter_indices,
        drop_unmatched_keys,
    ):
        """
        new_config: List[Dict] if converter_indices = 1 (CS -> HF)
        else Dict if converter_indices = 0 (HF -> CS).
        """
        new_config = super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )

        if converter_indices.direction == 0:  # src_fmt:HF -> tgt_fmt:CS
            # old_config = List[configs] where index i
            # corresponds to ith entry in component_names
            new_image_config = new_config["model"]["image_model"]
            new_image_config["name"] = "ViTModel"
            # remove non-kwargs

            new_text_config = new_config["model"]["text_model"]
            new_text_config["name"] = "LlamaModel"

            new_config["model"]["image_feature_select_mode"] = (
                new_text_config.pop("image_feature_select_mode")
            )

            # We are doing this to get "projector_image_model" under "projector" key in CS yaml
            # Convert `mm_projector_type` here since we depend on other values in the config
            mm_hidden_size = new_text_config.pop("mm_hidden_size")
            assert (
                mm_hidden_size == new_image_config["hidden_size"]
            ), f"`mm_hidden_size should be same as the hidden_dim of mm_vision_tower"
            new_projector_config = new_text_config.pop("projector")
            hf_projector_type = new_projector_config.pop("hf_type")
            num_linear = int(
                re.match("mlp(\d+)x_gelu", hf_projector_type).group(1)
            )
            new_im_proj_config = new_projector_config["image_model"]
            new_im_proj_config["name"] = "FeedForwardNetwork"

            new_im_proj_config["input_unit"] = mm_hidden_size  # image_model
            new_im_proj_config["layers_units"] = [
                new_text_config["hidden_size"]
            ] * num_linear  # text_model

            # we write `gelu` here since the input HF config
            # has `mlp2x_gelu` and LLaVA hardcodes `gelu`
            new_im_proj_config["layers_activation"] = ["gelu"] * (
                num_linear - 1
            ) + [None]
            new_im_proj_config["use_bias"] = True
            new_config["model"]["projector"] = new_projector_config

            # Add other params at `model` level for CS
            new_config["model"]["freeze"] = new_text_config.pop("freeze")

            new_config["model"]["image_feature_select_layer_idx"] = (
                new_text_config.pop("image_feature_select_layer_idx")
            )
            new_config["model"]["image_model"].pop("fp16_type", None)

        else:  # CS -> HF
            # new_config is the HF config = List[configs] where index i
            # corresponds to ith entry in component_names:
            # LLaVA model init on HF works only when
            # there is a preprocessor config
            self.preprocessor_config_defaults.update(
                {
                    "crop_size": {
                        "height": old_config["image_model"]["image_size"][0],
                        "width": old_config["image_model"]["image_size"][1],
                    },
                    "size": old_config["image_model"]["image_size"][0],
                }
            )

        return new_config

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def converters():
        return (
            ConfigConverter_CLIPViT_HF_CS21,
            ConfigConverter_LLaMaProjector_HF_CS22,
        )

    @staticmethod
    def component_names():
        return (
            "image_model",
            "text_model",
        )

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        """
        config: List[dicts] if converter_indices = 0 (HF-> CS) else dict (CS->HF).
        """

        if converter_indices.direction == 0:
            # HF -> CS
            # To avoid asserts with BaseConfigConverter.assert_factory_fn
            config[0]["model_type"] = "clip_vision_model"
            config[1]["model_type"] = "llama"
            if "vision_config" in config[0]:
                config[0] = config[0]["vision_config"]
        else:
            # CS -> HF
            # Move projector config into text_model config
            # for CS inorder to match keys
            projector_config = config["model"].pop("projector")
            config["model"]["text_model"]["projector"] = projector_config
            config["model"]["text_model"]["freeze"] = config["model"].pop(
                "freeze"
            )
            config["model"]["text_model"]["image_feature_select_layer_idx"] = (
                config["model"].pop("image_feature_select_layer_idx")
            )
            config["model"]["text_model"]["image_feature_select_mode"] = config[
                "model"
            ].pop("image_feature_select_mode")
            config["model"]["text_model"]["mm_hidden_size"] = config["model"][
                "image_model"
            ]["hidden_size"]
        return super().pre_config_convert(model, config, converter_indices)


class ConfigConverter_LLaMaProjector_HF_CS22(ConfigConverter_LLaMa_HF_CS21):
    def __init__(self):
        super().__init__()
        projector_rules = [
            ConversionRule(
                [EquivalentSubkey("mm_projector_type", "projector")],
                action=self.convert_mm_projector_type,
            ),
            ConversionRule(
                ["mm_hidden_size"],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "mm_vision_select_feature", "image_feature_select_mode"
                    )
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "mm_vision_select_layer",
                        "image_feature_select_layer_idx",
                    )
                ],
                action=self.convert_mm_vision_select_feature,
            ),
            ConversionRule(
                ["mm_use_im_start_end"],
                exists="left",
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(
                ["mm_use_im_patch_token"],
                exists="left",
                action=BaseConfigConverter.assert_factory_fn(0, False),
            ),
            ConversionRule(
                [EquivalentSubkey("tune_mm_mlp_adapter", "freeze")],
                action=self.convert_tune_mm_mlp_adapter,
            ),
        ]
        self.rules = self.rules + projector_rules

        # HF
        self.pre_convert_defaults[0].update(
            {
                "mm_vision_select_feature": "patch",
                "mm_use_im_patch_token": False,
                "mm_use_im_start_end": False,
                "tie_word_embeddings": False,
                "rope_scaling": None,
                "unfreeze_mm_vision_tower": False,
                "tune_mm_vision_resampler": False,
                "tune_mm_mlp_adapter": False,
                "mm_vision_select_layer": -2,
                "mm_projector_type": "mlp2x_gelu",
                "mm_hidden_size": 64,
            }
        )

        # CS
        # text model
        self.pre_convert_defaults[1].update(
            {
                "share_embedding_weights": False,
                "use_bias_in_output": False,
            }
        )

        # HF
        self.post_convert_defaults[0].update(
            {
                "mm_use_im_patch_token": False,
                "mm_use_im_start_end": False,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "image_aspect_ratio": "pad",
                "freeze_mm_mlp_adapter": False,
                "freeze_mm_vision_resampler": False,
                "model_type": "llava",
                "architectures": ["LlavaLlamaForCausalLM"],
                "pad_token_id": 0,
                "tune_mm_mlp_adapter": False,
                "tune_mm_vision_resampler": False,
                "unfreeze_mm_vision_tower": False,
                "use_cache": True,
            }
        )

    def convert_mm_vision_select_feature(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        old_val = old_state_dict[old_key]
        if old_val < 0:
            new_state_dict[new_key] = old_val
        else:
            if from_index == 0:  # HF -> CS
                # When HF outputs hidden states, it also includes embeddings
                # https://github.com/huggingface/transformers/blob/v4.38.1/src/transformers/models/clip/modeling_clip.py#L79
                # Also, LLava directly uses this value
                # https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_encoder/clip_encoder.py#L36
                assert old_val != 0, f" value = 0 will get embeddings"
                new_state_dict[new_key] = old_val - 1
            else:
                new_state_dict[new_key] = old_val + 1

    def convert_mm_projector_type(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:  # HF -> CS
            assert (
                re.match("mlp(\d+)x_gelu", old_state_dict["mm_projector_type"])
                is not None
            ), "Convertor only valid when `mm_projector_type` is of pattern `mlp(\d+)x_gelu`, got {}".format(
                old_state_dict["mm_projector_type"]
            )
            # old_state_dict would be list with index i
            # corresponding to component i in
            # `ConfigConverter_LLaVA_HF_CS21`
            new_state_dict[new_key] = {}
            new_state_dict[new_key]["image_model"] = {}
            new_state_dict[new_key]["hf_type"] = old_state_dict[old_key]

        else:  # CS-> HF
            assert (
                len(old_state_dict[old_key]["image_model"]) != 0
            ), f"CS model should have non-empty `projector.image_model`"
            proj_name = old_state_dict[old_key]["image_model"]["name"]
            _msg = (
                f"CS model projector.image_model.name should be of type "
                f"`FeedForwardNetwork inorder to convert to HF, got {proj_name}"
            )
            assert proj_name == "FeedForwardNetwork", _msg
            act = old_state_dict[old_key]["image_model"]["layers_activation"]
            expected_act = ["gelu"] * (len(act) - 1) + [None]
            assert (
                act == expected_act
            ), f"Cannot support {act}, expected value = {expected_act}"
            new_state_dict[new_key] = "mlp{}x_gelu".format(len(act))

    def convert_tune_mm_mlp_adapter(
        self,
        old_key,
        new_key,
        old_state_dict,
        new_state_dict,
        from_index,
        action_fn_args,
    ):
        if from_index == 0:  # HF -> CS
            # Freeze modules appropriately
            old_key_val = old_state_dict[old_key]
            new_state_dict[new_key] = ['^image_model']
            if old_key_val:
                new_state_dict[new_key].append('^text_model')

        else:
            # HF: `tune_mm_mlp_adapter`: True -> CS `freeze`: ["image_model", "text_model"]
            # HF: `tune_mm_mlp_adapter`: False -> CS `freeze`: ["image_model"]
            old_val = old_state_dict[old_key]
            if "text_model" in old_val:
                new_state_dict[new_key] = True
            else:
                new_state_dict[new_key] = False

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.2", "cs-2.3", "cs-2.4", "cs-2.5"),
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
        return super().post_config_convert(
            model,
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )
