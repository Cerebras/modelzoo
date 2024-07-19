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

from abc import ABC, abstractmethod
from dataclasses import asdict
from enum import Enum
from typing import Any, Dict

from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
)
from cerebras.modelzoo.models.multimodal.multimodal_utils import (
    get_model_handle_from_mapping,
    get_projector_model_handle_from_mapping,
)


class ModalityType(Enum):
    IMAGE = "image_model"
    TEXT = "text_model"

    @classmethod
    def values(cls):
        return [b.value for b in ModalityType]

    @classmethod
    def get(cls, blk):
        if isinstance(blk, str):
            return ModalityType(blk)
        elif isinstance(blk, Enum):
            return blk
        else:
            raise ValueError(
                f"Unsupported type {type(blk)}, supported are `str` and `Enum`"
            )


class MultimodalBaseModelWrapper(ABC):
    def __init__(self):
        super(MultimodalBaseModelWrapper, self).__init__()

    @property
    @abstractmethod
    def modalities(self):
        pass

    @abstractmethod
    def _post_device_transfer(self):
        pass

    def _init_component_model(self, component_params: Dict) -> object:
        component_name = component_params.pop("name")
        component_handle = get_model_handle_from_mapping(component_name)
        # Remove unused params.
        component_model = component_handle(**component_params)
        return component_model

    def build_modality_models(self, model_params: Any) -> Dict[str, object]:
        modality_models = {}
        for mmode in self.modalities:
            mmode_value_present = False
            if isinstance(model_params, Dict):
                if mmode.value in model_params.keys():
                    modality_models[mmode.value] = self._init_component_model(
                        model_params[mmode.value]
                    )
                    mmode_value_present = True

            elif isinstance(model_params, BaseConfig):
                if hasattr(model_params, mmode.value):
                    modality_models[mmode.value] = self._init_component_model(
                        asdict(getattr(model_params, mmode.value))
                    )
                    mmode_value_present = True

            else:
                raise ValueError(
                    f"Unsupported type {type(model_params)}, supported are `Dict` and `BaseConfig`"
                )

            if not mmode_value_present:
                # No model for this modality
                modality_models[mmode.value] = None
        return modality_models

    def build_projectors(self, model_params: Any) -> object:
        projector_models = {}
        projector_params = model_params.projector

        for mmode in self.modalities:
            _k = f"projector_{mmode.value}"
            if projector_params is not None:
                if mmode.value in projector_params.keys():
                    params = projector_params[mmode.value]
                    proj_name = projector_params[mmode.value].pop("name")
                    proj_mdl_handle = get_projector_model_handle_from_mapping(
                        proj_name
                    )
                    projector_models[_k] = proj_mdl_handle(**params)
            else:
                # No projector for this modality
                projector_models[_k] = None

        return projector_models
