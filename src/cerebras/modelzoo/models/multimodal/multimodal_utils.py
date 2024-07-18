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

import importlib
import logging
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, List

import torch


@dataclass
class MultimodalModelMapping:
    model_name: str
    module_import_path: str


@dataclass
class ProjectorMapping:
    projector_name: str
    module_import_path: str
    config_class: str


# SUPPORTED PROJECTORS
# Dict[projector_name, ProjectorMapping]
PROJECTOR_NAME_MODULE_DEFAULTFCN_MAPPING = OrderedDict(
    [
        (
            "MLP",
            ProjectorMapping(
                "MLP",
                "cerebras.modelzoo.layers.FeedForwardNetwork",
                "FeedForwardNetworkConfig",
            ),
        ),
        (
            "FeedForwardNetwork",
            ProjectorMapping(
                "FeedForwardNetwork",
                "cerebras.modelzoo.layers.FeedForwardNetwork",
                "FeedForwardNetworkConfig",
            ),
        ),
    ]
)

# SUPPORTED MODELS FOR MODALITIES
# Dict[model_name, MultimodalModelMapping]
MODEL_MODULE_DEFAULTFCN_MAPPING = OrderedDict(
    [
        (
            "ViTModel",
            MultimodalModelMapping(
                "ViTModel",
                "cerebras.modelzoo.models.vision.vision_transformer.ViTModel.ViTModel",
            ),
        ),
        (
            "T5ForConditionalGeneration",
            MultimodalModelMapping(
                "T5ForConditionalGeneration",
                "cerebras.modelzoo.models.nlp.t5.t5_model.T5ForConditionalGeneration",
            ),
        ),
        (
            "LlamaModel",
            MultimodalModelMapping(
                "LlamaModel",
                "cerebras.modelzoo.models.nlp.gpt2.gpt2_model.GPT2LMHeadModel",
            ),
        ),
        (
            "FeedForwardNetwork",
            MultimodalModelMapping(
                "FeedForwardNetwork",
                (
                    "cerebras.modelzoo.layers.FeedForwardNetwork",
                    "cerebras.modelzoo.layers.FeedForwardNetworkConfig",
                ),
            ),
        ),
    ]
)


def _import_helper(import_module_path, import_object_name):
    try:
        module = importlib.import_module(import_module_path)
        obj = getattr(module, import_object_name)
    except Exception as e:
        raise ImportError(
            f"Unable to import {import_object_name} from {import_module_path}"
        )
    return obj


def get_per_model_handle(model_name: str) -> object:
    _model_map = model_name.split(".")
    _module_path, _modelclass = ".".join(_model_map[:-1]), _model_map[-1]
    model_handle = _import_helper(_module_path, _modelclass)
    return model_handle


def get_model_handle_from_mapping(model_name: str) -> object:
    _model_map = MODEL_MODULE_DEFAULTFCN_MAPPING[model_name].module_import_path
    if type(_model_map) is tuple or type(_model_map) is list:
        model_handle_list = [get_per_model_handle(x) for x in _model_map]
        return model_handle_list
    else:
        return get_per_model_handle(_model_map)


def get_projector_model_handle_from_mapping(projector_name: str) -> object:
    _proj_mapping = PROJECTOR_NAME_MODULE_DEFAULTFCN_MAPPING[projector_name]
    _proj_model_map = _proj_mapping.module_import_path.split(".")
    _module_path, _modelclass = (
        ".".join(_proj_model_map[:-1]),
        _proj_model_map[-1],
    )
    model_class = _import_helper(_module_path, _modelclass)
    config_class = _import_helper(_module_path, _proj_mapping.config_class)

    def handle_from_config_args(*args, **kwargs):
        return model_class(config_class(*args, **kwargs))

    return handle_from_config_args


def freeze_modules(
    model: torch.nn.Module,
    module_name_patterns: List[str],
):
    """
    Freeze modules if their name matches a regex pattern.

    Args:
        model: The model to be frozen.
        module_name_patterns: Filter to select which parameters are frozen
            Note that regex patterns should be specified as single quotes
            in the yaml for escape codes.
    """
    leaf_modules = [
        (n, m)
        for n, m in model.named_modules()
        if not next(m.children(), False)
    ]
    patterns = list(map(re.compile, module_name_patterns))

    for pattern in patterns:
        module_list = [
            (name, param)
            for name, param in leaf_modules
            if pattern.search(name)
        ]

        if len(module_list) == 0:
            raise ValueError(f"{pattern} did not match any module names!")

        for _, m in module_list:
            m.eval()
            m.requires_grad_(False)

        logging.info(
            f"The following modules are frozen due to pattern: {pattern.pattern}: "
            f"{[n for n, _ in module_list]}"
        )

        # Additional pass through parameters since a module may have child modules
        # but also child parameters. For example, the classification token in an embedding
        # layer.
        for n, p in model.named_parameters():
            if pattern.search(n):
                p.requires_grad_(False)

    logging.info(
        f"The follow parameters are being trained: "
        f"{[n for n, p in model.named_parameters() if p.requires_grad]}"
    )


def init_component_model(component_params: Any) -> object:
    if is_dataclass(component_params):
        component_params = asdict(component_params)
    component_name = component_params.pop("name")
    component_handle = get_model_handle_from_mapping(component_name)
    if type(component_handle) is list:
        component_model = component_handle[0](
            component_handle[1](**component_params)
        )
    else:
        component_model = component_handle(**component_params)

    return component_model
