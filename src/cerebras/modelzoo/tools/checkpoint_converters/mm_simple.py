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
from collections import OrderedDict
from typing import Tuple

import torch

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseConfigConverter,
    BaseConfigConverter_HF_CS,
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
    Converter_LlamaModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.llava import (
    ConfigConverter_LLaMaProjector_HF_CS22,
    ConfigConverter_LLaVA_HF_CS22,
    Converter_LLaVA_CLIPViT_WithoutModel_HF_CS22,
    Converter_LLaVA_LLaMA_WithoutModel_HF_CS22,
    Converter_LLaVA_WithoutModel_HF_CS22,
)


class Converter_MMSimple_LLaVA_LLaMA_WithoutModel_HF_CS23(
    Converter_LLaVA_LLaMA_WithoutModel_HF_CS22
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [
                    EquivalentSubkey("lm_head", "text_model.lm_head"),
                    r"\.(?:weight|bias)",
                ],
                action=self.replaceKey,
            ),
            ConversionRule(
                [
                    EquivalentSubkey("model.", "text_model."),
                    Converter_LlamaModel_HF_CS21(),
                ],
            ),
            ConversionRule(
                [
                    r"image_model.image_model_list.0.0.*",
                ],
                exists="right",
                action=None,
            ),
            # projector_image_model
            ConversionRule(
                [
                    EquivalentSubkey(
                        "model.mm_projector", "image_model.projection.ffn"
                    ),
                    r"\.\d+",
                    EquivalentSubkey(".", ".linear_layer."),
                    r"(?:weight|bias)",
                ],
                action=self.convert_projector,
            ),
            *self.rules,
        ]

    @classmethod
    def converter_note(cls) -> str:
        note = super().converter_note()
        return (
            note
            + f"MMSimple LLaVA converter using CLIP-ViT and LLaMA backbones."
        )


class Converter_MMSimple_LLaVA_CLIPViT_WithoutModel_HF_CS23(
    Converter_LLaVA_CLIPViT_WithoutModel_HF_CS22
):
    def __init__(self):
        super().__init__()
        self.rules = [
            # This is ignored since it's handled in Vision model
            # i.e Converter_MMSimple_LLaVA_LLaMA_WithoutModel_HF_CS23
            ConversionRule(
                [r"image_model.projection.*"],
                exists="right",
                action=None,
            ),
            ConversionRule(
                [
                    EquivalentSubkey(
                        "vision_model.",
                        "image_model.image_model_list.0.0.",
                    ),
                    Converter_CLIPViT_Core_HF_CS21(),
                ],
            ),
            *self.rules,
        ]

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MMSimple_LLaVA_CLIPViT_HF_CS23


Converter_MMSimple_LLaVA_CLIPViT_HF_CS23 = (
    Build_HF_CS_Converter_WithOptionalModel(
        "Converter_MMSimple_LLaVA_CLIPViT_HF_CS23",
        Converter_MMSimple_LLaVA_CLIPViT_WithoutModel_HF_CS23,
        derived_class=Converter_MMSimple_LLaVA_CLIPViT_WithoutModel_HF_CS23,
    )
)

Converter_MMSimple_LLaVA_LLaMA_HF_CS23 = (
    Build_HF_CS_Converter_WithOptionalModel(
        "Converter_MMSimple_LLaVA_LLaMA_HF_CS23",
        Converter_MMSimple_LLaVA_LLaMA_WithoutModel_HF_CS23,
        derived_class=Converter_MMSimple_LLaVA_LLaMA_WithoutModel_HF_CS23,
    )
)


class Converter_MMSimple_LLaVA_WithoutModel_HF_CS24(
    Converter_LLaVA_WithoutModel_HF_CS22
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def converters():
        return (
            Converter_MMSimple_LLaVA_CLIPViT_HF_CS23,
            Converter_MMSimple_LLaVA_LLaMA_HF_CS23,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.4", "cs-2.5"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MMSimple_LLaVA_HF_CS24

    def post_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        if converter_indices.direction == 0:  # HF -> CS
            # We are converting from HF
            # to our Multimodal-Simple model. We need to create the visual token `projection`
            # layer and init to default values for phase 1

            # Check if there was a mapping of HF projector weights to CS namespace,
            # if not, initialize defaults
            is_projector_exists = any(
                [
                    "image_model.projection" in k
                    for k in output_checkpoint["model"].keys()
                ]
            )
            if not is_projector_exists:
                logging.info(
                    f"---- HF checkpoint does not have projector weight, initializing defaults"
                )
                cs_config = configs[1]

                # im_proj_config = cs_config["model"]["projector"]["image_model"]
                im_proj_config = cs_config["model"]["image_model_list"][
                    "global_image_projection"
                ]

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
                        f"image_model.projection.ffn.{i}.linear_layer.weight"
                    ] = projection_weight
                    if use_bias:
                        projection_bias = torch.zeros(out)
                        projection_bias.uniform_(-scale, scale)
                        output_checkpoint["model"][
                            f"image_model.projection.ffn.{i}.linear_layer.bias"
                        ] = projection_bias

            super(
                Converter_LLaVA_WithoutModel_HF_CS22, self
            ).post_checkpoint_convert(
                input_checkpoint,
                output_checkpoint,
                configs,
                converter_indices,
            )


Converter_MMSimple_LLaVA_HF_CS24 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_MMSimple_LLaVA_HF_CS24",
    Converter_MMSimple_LLaVA_WithoutModel_HF_CS24,
    derived_class=Converter_MMSimple_LLaVA_WithoutModel_HF_CS24,
)


class Converter_MMSimple_LLaVA_WithoutModel_HF_CS23(
    Converter_MMSimple_LLaVA_WithoutModel_HF_CS24
):
    def __init__(self):
        super().__init__()

    @staticmethod
    def converters():
        return (
            Converter_MMSimple_LLaVA_CLIPViT_HF_CS23,
            Converter_MMSimple_LLaVA_LLaMA_HF_CS23,
        )

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (FormatVersions("hf"), FormatVersions("cs-2.3"))

    @staticmethod
    def get_config_converter_class() -> BaseConfigConverter:
        return ConfigConverter_MMSimple_LLaVA_HF_CS23


Converter_MMSimple_LLaVA_HF_CS23 = Build_HF_CS_Converter_WithOptionalModel(
    "Converter_MMSimple_LLaVA_HF_CS23",
    Converter_MMSimple_LLaVA_WithoutModel_HF_CS23,
    derived_class=Converter_MMSimple_LLaVA_WithoutModel_HF_CS23,
)


class ConfigConverter_MMSimple_LLaVA_CLIPViT_HF_CS23(
    ConfigConverter_CLIPViT_HF_CS21
):
    def __init__(self):
        super().__init__()


class ConfigConverter_MMSimple_LLaVA_LLaMa_HF_CS23(
    ConfigConverter_LLaMaProjector_HF_CS22
):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(["extra_ffn_params.*"], exists="right", action=None),
            *self.rules,
        ]


class ConfigConverter_MMSimple_LLaVA_HF_CS24(ConfigConverter_LLaVA_HF_CS22):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.4", "cs-2.5"),
        )

    @staticmethod
    def converters():
        return (
            ConfigConverter_MMSimple_LLaVA_CLIPViT_HF_CS23,
            ConfigConverter_MMSimple_LLaVA_LLaMa_HF_CS23,
        )

    @staticmethod
    def component_names():
        return (
            "image_model",
            "text_model",
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
            # Pop `image_model` dict here, since popping in post_config_convert will break the
            # checkpoint_convertor
            config["model"].pop("image_model")
            return super().save(
                file_without_ext, config, converter_indices, **kwargs
            )
        # saving HF requires separating encoders and saving both
        else:
            save_files = []
            dir = os.path.dirname(file_without_ext)
            for i, name in enumerate(cls.component_names()):
                path = os.path.join(dir, name, "config")
                print(dir)
                print(path)
                if not os.path.exists(os.path.join(dir, name)):
                    os.mkdir(os.path.join(dir, name))
                if name == "text_model":
                    # add path to folder containing
                    # image model in text_model config
                    config[i]["mm_vision_tower"] = os.path.dirname(
                        save_files[0]
                    )
                if name == "image_model":
                    preprocess_path = os.path.join(
                        dir, name, "preprocessor_config"
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

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        """
        config: List[dicts] if converter_indices = 0 (HF-> CS) else dict (CS->HF)
        """
        orig_config = config

        if converter_indices.direction == 1:  # CS -> HF
            # CS -> HF
            # Move projector config into text_model config
            # for CS inorder to match keys
            # config["model"] = config["trainer"]["init"].pop("model")
            if ("trainer" in config) and ("model" not in config):
                config = config["trainer"]["init"]

            projector_config = config["model"]["image_model_list"].pop(
                "global_image_projection"
            )

            config["model"]["projector"] = {"image_model": projector_config}
            image_model_list = config["model"].pop("image_model_list")
            config["model"]["image_feature_select_layer_idx"] = (
                image_model_list["image_models"][0]["image_encoder"][
                    "image_layer_idx"
                ]
            )
            config["model"]["image_feature_select_mode"] = image_model_list.pop(
                "image_feature_select_mode"
            )

            config["model"]["image_model"] = image_model_list["image_models"][
                0
            ]["image_encoder"]

        return super().pre_config_convert(model, config, converter_indices)

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
        if converter_indices.direction == 0:  # HF -> CS
            v_cfg = model_config["model"]["image_model"]
            t_cfg = model_config["model"]["text_model"]

            # # ConversionRule doesn"t work for this key -> bug in the config?
            # if "embedding_dropout_rate" in t_cfg:
            #     t_cfg["embd_pdrop"] = t_cfg["embedding_dropout_rate"]
            #     del t_cfg["embedding_dropout_rate"]

            # move some sub-dicts around
            # LLaVA convertor post_config_convert_defaults adds `image_start_idx`,
            # so we remove it here
            model_config["model"].pop("image_start_idx")

            v_cfg["image_layer_idx"] = model_config["model"].pop(
                "image_feature_select_layer_idx"
            )

            model_config["model"]["image_model_list"] = {
                "image_models": [{"image_encoder": v_cfg}]
            }
            model_config["model"]["image_model_list"][
                "image_feature_select_mode"
            ] = model_config["model"].pop("image_feature_select_mode")

            img_projector = model_config["model"].pop("projector")
            img_projector = img_projector.pop("image_model")
            model_config["model"]["image_model_list"][
                "global_image_projection"
            ] = img_projector
        return model_config


class ConfigConverter_MMSimple_LLaVA_HF_CS23(ConfigConverter_LLaVA_HF_CS22):
    def __init__(self):
        super().__init__()

    @staticmethod
    def formats() -> Tuple[FormatVersions, FormatVersions]:
        return (
            FormatVersions("hf"),
            FormatVersions("cs-2.3"),
        )

    @staticmethod
    def converters():
        return (
            ConfigConverter_MMSimple_LLaVA_CLIPViT_HF_CS23,
            ConfigConverter_MMSimple_LLaVA_LLaMa_HF_CS23,
        )

    @staticmethod
    def component_names():
        return (
            "image_model",
            "text_model",
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
            # Pop `image_model` dict here, since popping in post_config_convert will break the
            # checkpoint_convertor
            config["model"].pop("image_model")
            return super().save(
                file_without_ext, config, converter_indices, **kwargs
            )
        # saving HF requires separating encoders and saving both
        else:
            save_files = []
            dir = os.path.dirname(file_without_ext)
            for i, name in enumerate(cls.component_names()):
                path = os.path.join(dir, name, "config")
                print(dir)
                print(path)
                if not os.path.exists(os.path.join(dir, name)):
                    os.mkdir(os.path.join(dir, name))
                if name == "text_model":
                    # add path to folder containing
                    # image model in text_model config
                    config[i]["mm_vision_tower"] = os.path.dirname(
                        save_files[0]
                    )
                if name == "image_model":
                    preprocess_path = os.path.join(
                        dir, name, "preprocessor_config"
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

    def pre_config_convert(
        self,
        model,
        config,
        converter_indices,
    ):
        """
        config: List[dicts] if converter_indices = 0 (HF-> CS) else dict (CS->HF)
        """
        orig_config = config

        if converter_indices.direction == 1:  # CS -> HF
            # CS -> HF
            # Move projector config into text_model config
            # for CS inorder to match keys
            # config["model"] = config["trainer"]["init"].pop("model")
            if ("trainer" in config) and ("model" not in config):
                config = config["trainer"]["init"]

            print(config["model"]["image_model_list"])

            projector_config = config["model"]["image_model_list"].pop(
                "global_image_projection"
            )

            config["model"]["projector"] = {"image_model": projector_config}
            image_model_list = config["model"].pop("image_model_list")
            config["model"]["image_feature_select_layer_idx"] = (
                image_model_list["image_models"][0]["image_model"][0][
                    "image_layer_idx"
                ]
            )
            config["model"]["image_feature_select_mode"] = image_model_list.pop(
                "image_feature_select_mode"
            )

            config["model"]["image_model"] = image_model_list["image_models"][
                0
            ]["image_model"][0]

        return super().pre_config_convert(model, config, converter_indices)

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
        if converter_indices.direction == 0:  # HF -> CS
            v_cfg = model_config["model"]["image_model"]
            t_cfg = model_config["model"]["text_model"]

            # ConversionRule doesn"t work for this key -> bug in the config?
            # if "embedding_dropout_rate" in t_cfg:
            #     t_cfg["embd_pdrop"] = t_cfg["embedding_dropout_rate"]
            #     del t_cfg["embedding_dropout_rate"]

            # move some sub-dicts around
            # LLaVA convertor post_config_convert_defaults adds `image_start_idx`,
            # so we remove it here
            model_config["model"].pop("image_start_idx")
            v_cfg["image_layer_idx"] = model_config["model"].pop(
                "image_feature_select_layer_idx"
            )

            model_config["model"]["image_model_list"] = {
                "image_models": [{"image_model": [v_cfg]}]
            }
            model_config["model"]["image_model_list"][
                "image_feature_select_mode"
            ] = model_config["model"].pop("image_feature_select_mode")

            img_projector = model_config["model"].pop("projector")
            img_projector = img_projector.pop("image_model")
            model_config["model"]["image_model_list"][
                "global_image_projection"
            ] = img_projector
        return model_config
