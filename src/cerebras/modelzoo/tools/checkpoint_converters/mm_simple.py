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

from cerebras.modelzoo.tools.checkpoint_converters.base_converter import (
    BaseConfigConverter,
    ConversionRule,
    EquivalentSubkey,
    FormatIndices,
    FormatVersions,
)
from cerebras.modelzoo.tools.checkpoint_converters.clip_vit import (
    ConfigConverter_CLIPViT_HF_CS21,
    Converter_CLIPViT_Core_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.llama import (
    Converter_LlamaModel_HF_CS21,
)
from cerebras.modelzoo.tools.checkpoint_converters.llava import (
    ConfigConverter_LLaMaProjector_HF_CS22,
    ConfigConverter_LLaVA_HF_CS22,
    Converter_LLaVA_CLIPViT_HF_CS22,
    Converter_LLaVA_HF_CS22,
    Converter_LLaVA_LLaMA_WithoutModel_HF_CS22,
)


class Converter_MMSimple_LLaVA_LLaMA_HF_CS23(
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
                exists="left",
                action=None,
            ),
            ConversionRule(
                [
                    r"model.mm_projector.*",
                ],
                exists="right",
                action=self.convert_projector,
            ),
            ConversionRule(
                [r"image_model.projection.*"],
                exists="left",
                action=self.convert_projector,
            ),
            *self.rules,
        ]

    def convert_projector(
        self,
        old_key: str,
        new_key: str,
        old_state_dict,
        new_state_dict,
        from_index: int,
        action_fn_args=None,
    ) -> None:
        state = next(iter(re.findall("weight|bias", old_key)))
        if from_index == 0:
            p = re.compile('image_model.+layers.(\d).ffn')
            num_layers = len(
                set(
                    p.search(k).group(1)
                    for k in new_state_dict.keys()
                    if p.match(k)
                )
            )
            for i in range(num_layers):
                new_key = f'image_model.projection.ffn.{i}.linear_layer.{state}'
                new_state_dict[new_key] = old_state_dict[old_key]
        else:
            new_state_dict[f'model.mm_projector.{state}'] = old_state_dict[
                old_key
            ]

    @classmethod
    def converter_note(cls) -> str:
        note = super().converter_note()
        return (
            note
            + f"MMSimple LLaVA converter using CLIP-ViT and LLaMA backbones."
        )


class Converter_MMSimple_LLaVA_CLIPViT_HF_CS23(Converter_LLaVA_CLIPViT_HF_CS22):
    def __init__(self):
        super().__init__()
        self.rules = [
            ConversionRule(
                [r"image_model.projection.*"],
                exists="left",
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


class Converter_MMSimple_LLaVA_HF_CS23(Converter_LLaVA_HF_CS22):
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

    def post_checkpoint_convert(
        self,
        input_checkpoint,
        output_checkpoint,
        configs: Tuple[dict, dict],
        converter_indices: FormatIndices,
    ):
        # do not call super() to not initialize projector defaults
        if converter_indices.direction == 0:
            img_projector_config = configs[1]['model']['projector'][
                'image_model'
            ]
            # set default projector name
            if 'name' not in img_projector_config:
                img_projector_config['name'] = "FeedForwardNetwork"


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

    def post_config_convert(
        self,
        original_config,
        old_config,
        new_config,
        converter_indices,
        drop_unmatched_keys,
    ):
        model_config = super().post_config_convert(
            original_config,
            old_config,
            new_config,
            converter_indices,
            drop_unmatched_keys,
        )
        if converter_indices.direction == 0:
            v_cfg = model_config['model']['image_model']
            t_cfg = model_config['model']['text_model']

            # ConversionRule doesn't work for this key -> bug in the config?
            if 'embedding_dropout_rate' in t_cfg:
                t_cfg['embd_pdrop'] = t_cfg['embedding_dropout_rate']
                del t_cfg['embedding_dropout_rate']

            # move some sub-dicts around
            img_projector = model_config['model']['projector']['image_model']
            model_config['model']['global_image_projector'] = img_projector
            model_config['model']['image_model_list'] = {'image_model': [v_cfg]}
            layer_idx = model_config['model']['image_feature_select_layer_idx']
            v_cfg['image_layer_idx'] = layer_idx

        return model_config
