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

"""
 Utility to load the config from yaml or .py config file
"""

import logging
from dataclasses import asdict
from typing import Union

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
)


def flatten_sparsity_params(kwargs):
    """
    Config classes package sparsity related params in a sub dict.
    ALthough, if we use native yaml config, they come unrolled.
    This utility unwraps the sparsity related params(if present)
    into an unroller sparsity param dict for consistency.
    Args:
        kwargs : Input args
    Returns:
        Flattened dict
    """

    if isinstance(kwargs, (int, float, list, tuple)):
        return kwargs

    if 'groups' in kwargs:
        kwargs = kwargs.pop('groups', {})
    else:
        return kwargs  # No need to flatten if no groups present

    if isinstance(kwargs, dict):
        additional_dict = kwargs.pop('params', {})
        flattened_dict = kwargs.copy()

        for key, value in additional_dict.items():
            new_key = f"{key}"
            flattened_dict[new_key] = value

        return flattened_dict
    elif isinstance(kwargs, list):
        param_list = []
        for param in kwargs:
            additional_dict = param.pop('params', {})
            flattened_dict = param.copy()

            for key, value in additional_dict.items():
                new_key = f"{key}"
                flattened_dict[new_key] = value

            param_list.append(flattened_dict)
        return param_list
    else:
        return kwargs


def flatten_optimizer_params(kwargs):
    """
    Config classes package optimizer related params in a sub dict.
    ALthough, if we use native yaml config, they come unrolled.
    This utility unwraps the optimizer related params(if present)
    into an unroller optimizer param dict for consistency.
    Args:
        kwargs : Input args dict
    Returns:
        flattened_args: Flattened dict
    """
    additional_dict = kwargs.pop('optim_params', {})
    flattened_dict = kwargs.copy()

    for key, value in additional_dict.items():
        new_key = f"{key}"
        flattened_dict[new_key] = value

    return flattened_dict


def flatten_data_params(kwargs):
    """
    Config classes package data related params in a sub dict.
    ALthough, if we use native yaml config, they come unrolled.
    This utility unwraps the data processor related params(if present)
    into an unrolled data param dict for consistency.
    Args:
        kwargs : Input args dict
    Returns:
        flattened_args: Flattened dict
    """
    additional_dict = kwargs.pop('params', {})
    flattened_dict = kwargs.copy()

    for key, value in additional_dict.items():
        flattened_dict[f"{key}"] = value

    return flattened_dict


def convert_to_dict(params_or_obj: Union[BaseConfig, dict]):
    "Utility to convert config object to dict"
    if isinstance(params_or_obj, dict):
        return params_or_obj

    params = asdict(params_or_obj)
    if params.get("sparsity") is not None:
        params["sparsity"] = flatten_sparsity_params(params["sparsity"])
    if params.get("optimizer") is not None:
        params["optimizer"] = flatten_optimizer_params(params["optimizer"])
    if params.get("train_input") is not None:
        params["train_input"] = flatten_data_params(params["train_input"])
    if params.get("eval_input") is not None:
        params["eval_input"] = flatten_data_params(params["eval_input"])
    if params.get("inference_input") is not None:
        params["inference_input"] = flatten_data_params(
            params["inference_input"]
        )
    return params


def create_config_obj_from_params(
    params_dict: dict, model_name: str
) -> Union[BaseConfig]:
    """Convert given dictionary to a config object based on the model name.

    Args:
        params_dict: The config params passed as a dict.
        model_name: The model name to be used for getting the config class from registry.
    Returns:
        The validated config class instance.
    """
    config_class = registry.get_config_class(model_name)
    if config_class is None:
        raise RuntimeError(
            f"Could not find a model in registry with name {model_name}. "
            f"Available models are: {','.join(registry.list_models())}"
        )
    # Pop the description field as we dont have a corresponding config class member
    if descr := params_dict.pop("description", None):
        logging.info(f"Loading config: {descr}")

    logging.info(f"Loading config class {config_class} to validate parameters.")
    try:
        config = config_class(**params_dict)
        config.__validate__()
    except Exception as e:  # pylint: disable=broad-except
        raise ValueError(
            f"Config validation using {type(config_class)} failed due to: {e}"
        ) from e

    logging.info(f"Config has been validated using config class.")

    return config
