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

import importlib
import importlib.util
import logging
import os
from dataclasses import asdict

import yaml

from cerebras.modelzoo.common.registry import registry

# TODO : This is a bit ugly and should be removed.
# This is an ignore list of some QA params that we find in our configs.
# These are injected via test params or similar before run.py call so we find them here.
# If we find these, we ignore them for now.
ignore_list = [
    'verify_determinism',
    'batch_size',
    'use_fake_data',
    'disable_convergence_checks',
]


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


def process_config(config, config_class, params_conf):
    """
    Perform config mapping and validation
    Args:
        config: The config class object
        config_class: The clas the config belongs to
        params_conf: Dictionary of params
    """
    # Disabled by default, enable for internal test trains
    # that might have some unused params left to be cleaned
    allow_config_class_validation_failures = int(
        os.environ.get('CONFIG_CLASS_VALIDATION_FALLBACK', 0)
    )

    if allow_config_class_validation_failures == 1:
        logging.info(
            "Config class validation failure fallback is enabled for the run which is intended for"
            "internal runs only, unset CONFIG_CLASS_VALIDATION_FALLBACK env variable if you want a"
            "strict validaiton enforced for configs using config class"
        )
        try:
            config = config_class(**params_conf)
            config.validate()
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(
                f"CONFIG WARNING: Falling back to default flow because of config class error: {e}"
                "config could not be validated via config class, proceed if this is expected"
            )
            # invalidate the config class object
            config = None
    else:
        try:
            config = config_class(**params_conf)
            config.validate()
        except Exception as e:  # pylint: disable=broad-except
            raise ValueError(
                f"CONFIG ERROR : Invalid param configuration supplied. Please fix error : {e}  or "
                "contact Cerebras support"
            )
    return config


def validate_config_params(params_conf, model_name):
    """
    Load the config class and run validation
    check on the config based on parameter constraints
    Args:
        params_conf: The config params passed as a dict
        model_name: The model key name used by config map to check what class of config to use
    """

    # Pop the description field as we dont have a corresponding config class member
    if "description" in params_conf:
        descr = params_conf["description"]
        logging.info(f"Loading config : {descr}")
        params_conf.pop("description")

    ignored_keys = {}
    # Pop the keys to be ignored and store them in a separate dictionary
    for key in ignore_list:
        if key in params_conf:
            ignored_keys[key] = params_conf.pop(key)
            logging.warning(
                f"CONFIG WARNING: Config class ignored usage of param {key} ."
                "Please note this type of usage is not permitted, please modify your config."
            )

    config_class = registry.get_config_class(model_name)

    config = None
    if config_class is not None:
        logging.info(f"Loading config class : {config_class}")
        config = process_config(config, config_class, params_conf)
        logging.info(f"Config has been validated using config class")
    else:
        # TODO: Add an error comment here once the config classes are ready and implemented.
        #  For now silently default to old path
        logging.warning(
            f"Config loaded using yaml path without using config class"
        )
    if config:
        params = asdict(config)
    else:
        params = params_conf

    if params.get("sparsity") is not None:
        params["sparsity"] = flatten_sparsity_params(params["sparsity"])
    if params.get("optimizer") is not None:
        params["optimizer"] = flatten_optimizer_params(params["optimizer"])
    # Insert the ignored keys back into the dictionary
    params.update(ignored_keys)
    return params


def get_config_from_yaml(yaml_path, model_name):
    """
    Get the config object after reading input yaml. Also runs validation
    check on the config based on parameter constraints

    Args:
        yaml_path: The path to the config yaml file
        model_name: The model key name used by config map to check what class of config to use
    """

    with open(yaml_path, 'r') as file:
        params_conf = yaml.safe_load(file)
    config_class = registry.get_config_class(model_name)

    config = None
    if config_class is not None:
        logging.info(f"Loading config class{config_class} from yaml path")
        config = process_config(config, config_class, params_conf)
        logging.info(
            f"Config {config} has been validated using config class from {yaml_path}"
        )
    else:
        # TODO: Add an error comment here once the config classes are ready and implemented.
        #  For now silently default to old path
        logging.warning(
            f"Config loaded using yaml path without using config class"
        )

    params = {}
    if config:
        params = asdict(config)
    else:
        params = params_conf

    return params


def read_from_config_class_file(file_path, function_name):
    """
    Read the config class file and call the config generator function to get config object

    Args:
        file_path: The path to the config .py file
        function_name: The config creation function defined in the config class creator .py file
    """
    # Dynamically import the module fromt he config class file
    config_module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(config_module_name, file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    # Get the config creation function from the module
    get_config = getattr(config_module, function_name, None)

    if get_config and callable(get_config):
        # Override the existing function with the same name to make sure we call the correct one
        locals()[function_name] = get_config
        # Call the imported function
        return get_config()
    else:
        logging.warning(
            f"Could not find a valid config class creator in {file_path},"
        )
        return None


def get_config_from_class(config_class_file):
    """
    Read the config class file and returns a config object

    Args:
        config_class_file: The path to the config .py file
    """
    config = read_from_config_class_file(
        config_class_file, "get_variant_config"
    )
    if config is not None:
        config.validate()
        logging.info(
            f"Config {config_class_file} has been validated using config class"
        )
    return config
