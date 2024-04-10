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
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class MultimodalModelMapping:
    model_name: str
    module_import_path: str
    defaultsfcn_import_path: str


@dataclass
class ProjectorMapping:
    projector_name: str
    module_import_path: str
    defaultsfcn_import_path: str = None


# SUPPORTED PROJECTORS
# Dict[projector_name, ProjectorMapping]
PROJECTOR_NAME_MODULE_DEFAULTFCN_MAPPING = OrderedDict(
    [
        (
            "MLP",
            ProjectorMapping(
                "MLP", "cerebras.modelzoo.layers.FeedForwardNetwork", None
            ),
        ),
        (
            "FeedForwardNetwork",
            ProjectorMapping(
                "FeedForwardNetwork",
                "cerebras.modelzoo.layers.FeedForwardNetwork",
                None,
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
                "cerebras.modelzoo.models.vision.vision_transformer.utils.set_defaults",
            ),
        ),
        (
            "T5ForConditionalGeneration",
            MultimodalModelMapping(
                "T5ForConditionalGeneration",
                "cerebras.modelzoo.models.nlp.t5.t5_model.T5ForConditionalGeneration",
                "cerebras.modelzoo.models.nlp.t5.utils.set_defaults",
            ),
        ),
        (
            "LlamaModel",
            MultimodalModelMapping(
                "LlamaModel",
                "cerebras.modelzoo.models.nlp.gpt2.gpt2_model.GPT2LMHeadModel",
                "cerebras.modelzoo.models.nlp.gpt2.utils.set_defaults",
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


def get_model_handle_from_mapping(model_name: str) -> object:
    _model_map = MODEL_MODULE_DEFAULTFCN_MAPPING[
        model_name
    ].module_import_path.split(".")
    _module_path, _modelclass = ".".join(_model_map[:-1]), _model_map[-1]
    model_handle = _import_helper(_module_path, _modelclass)
    return model_handle


def get_model_defaults_fcn_handle_from_mapping(model_name: str) -> object:
    _defaults_fcn_map = MODEL_MODULE_DEFAULTFCN_MAPPING[
        model_name
    ].defaultsfcn_import_path
    _defaults_fcn_map = _defaults_fcn_map.split(".")
    _module_path, _defaultsfcn = (
        ".".join(_defaults_fcn_map[:-1]),
        _defaults_fcn_map[-1],
    )
    default_fcn_handle = _import_helper(_module_path, _defaultsfcn)
    return default_fcn_handle


def get_projector_model_handle_from_mapping(projector_name: str) -> object:
    _proj_model_map = PROJECTOR_NAME_MODULE_DEFAULTFCN_MAPPING[
        projector_name
    ].module_import_path.split(".")
    _module_path, _modelclass = (
        ".".join(_proj_model_map[:-1]),
        _proj_model_map[-1],
    )
    proj_model_handle = _import_helper(_module_path, _modelclass)
    return proj_model_handle


def get_projector_defaults_fcn_handle_from_mapping(
    projector_name: str,
) -> object:
    _defaults_fcn_map = PROJECTOR_NAME_MODULE_DEFAULTFCN_MAPPING[
        projector_name
    ].defaultsfcn_import_path

    # passthrough
    if _defaults_fcn_map is None:
        return lambda x: x

    # TODO: Note : there might be some problems here since some `utils.py` dont return params
    _defaults_fcn_map = _defaults_fcn_map.split(".")
    _module_path, _defaultsfcn = (
        ".".join(_defaults_fcn_map[:-1]),
        _defaults_fcn_map[-1],
    )
    default_fcn_handle = _import_helper(_module_path, _defaultsfcn)
    return default_fcn_handle


def _get_defaults_fcn(name_str, is_projector=False):
    if is_projector:
        fcn_handle = get_projector_defaults_fcn_handle_from_mapping(name_str)
    else:
        fcn_handle = get_model_defaults_fcn_handle_from_mapping(name_str)
    return fcn_handle


def set_model_defaults_for_components(params, model_keys):
    # model_keys refers to the name of keys which have models
    # and associated projectors if needed
    # So if model_keys = ["image_model", "image_model_1"],
    # we have params["model"]["image_model"], params["model"]["image_model_1"]
    # and projector params["model"]["projector"]["image_model"] and
    # projector params["model"]["projector"]["image_model_1"]

    mparams = params["model"]
    mpparams = params["model"]["projector"]
    mixed_precision = mparams["mixed_precision"]
    fp16_type = mparams["fp16_type"]

    for component in model_keys:
        for cparams, tgt_params, is_projector in [
            (params.copy(), mparams, False),
            (params.copy(), mpparams, True),
        ]:
            if tgt_params and tgt_params.get(component, None):
                # Add under `model` key since
                # set_defaults expects that
                cparams["model"] = tgt_params[component]
                cparams["model"]["mixed_precision"] = mixed_precision
                cparams["model"]["fp16_type"] = fp16_type

                defaults_fcn = _get_defaults_fcn(
                    tgt_params[component]["name"], is_projector
                )
                defaults_fcn(cparams)

                cparams["model"].pop("mixed_precision")
                cparams["model"].pop("fp16_type")

                tgt_params[component] = cparams["model"]


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
