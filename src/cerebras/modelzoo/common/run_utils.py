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

"""Utilities for running Cerebras Pytorch Models."""

import argparse
import inspect
import logging
import os
import subprocess
import sys
from shutil import which
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import yaml

from cerebras.appliance.log import (
    collect_wsc_log_settings,
    get_level_name,
    wsc_logger,
)
from cerebras.modelzoo.common.pytorch_utils import RunConfigParamsValidator
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.common.run_cstorch_flow import run_cstorch_flow
from cerebras.modelzoo.common.utils.run.cli_parser import get_params_from_args
from cerebras.modelzoo.common.utils.run.utils import DeviceType
from cerebras.modelzoo.config_manager.config_loader import (
    convert_to_dict,
    create_config_obj_from_params,
)

DATA_FN_TYPE = Callable[[dict], torch.utils.data.DataLoader]


def torchrun(filename: str, arguments: List[str]):
    """Starts a distributed GPU run using torchrun."""
    torchrun_cmd = [
        which("torchrun", path=os.path.dirname(sys.executable)),
        "--nnodes=1",
        f"--nproc_per_node={torch.cuda.device_count()}",
        filename,
        *arguments,
    ]
    try:
        print(
            f"Starting distributed GPU run using torchrun:\n"
            f"{' '.join(torchrun_cmd)}"
        )
        subprocess.run(torchrun_cmd, check=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to spawn distributed GPU run using torchrun"
        ) from e


def run(
    model_fn: Union[Callable[[dict], torch.nn.Module], str],
    train_data_fn: Optional[DATA_FN_TYPE] = None,
    eval_data_fn: Optional[DATA_FN_TYPE] = None,
    default_params_fn: Optional[Callable[[dict], dict]] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """
    Entry point to running pytorch models including CLI argument parsing.
    """
    parent = inspect.getouterframes(inspect.currentframe())[1]
    params = get_params_from_args(extra_args_parser_fn)
    if default_params_fn:
        params = default_params_fn(params) or params

    main(params, model_fn, train_data_fn, eval_data_fn, script=parent.filename)


def main(
    params: Dict[str, Any],
    model_fn: Union[Callable[[dict], torch.nn.Module], str],
    train_data_fn: Optional[DATA_FN_TYPE] = None,
    eval_data_fn: Optional[DATA_FN_TYPE] = None,
    script: Optional[str] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """Entry point to running pytorch models."""
    if not script:
        parent = inspect.getouterframes(inspect.currentframe())[1]
        script = parent.filename

    if (
        "runconfig" in params
        # If using distributed GPU with experimental API
        and params["runconfig"]["target_device"] == DeviceType.GPU
        and params["runconfig"].get("enable_distributed", False)
        # If this is already set, we've already launched distributed training
        and os.environ.get("LOCAL_RANK") is None
    ):
        # use torchrun to launch distributed training
        torchrun(script, sys.argv[1:])
        return None

    return run_with_params(
        params,
        model_fn,
        train_data_fn,
        eval_data_fn,
        extra_args_parser_fn=extra_args_parser_fn,
    )


def run_with_params(
    params: Dict[str, Any],
    model_fn: Union[Callable[[dict], torch.nn.Module], str],
    train_data_fn: Optional[DATA_FN_TYPE] = None,
    eval_data_fn: Optional[DATA_FN_TYPE] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """
    Runs a full end-to-end CS/non-CS workflow for a given model.

    Args:
        params: The parsed YAML config dictionary.
        model_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a torch.nn.Module
        train_data_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a PyTorch dataloader
            corresponding to the training dataset
        eval_data_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a PyTorch dataloader
            corresponding to the evaluation dataset
        extra_args_parser_fn: An optional callable that adds any
            extra parser args not covered in `get_parser` fn.
    """
    runconfig_params = params["runconfig"]
    RunConfigParamsValidator(extra_args_parser_fn).validate(runconfig_params)
    mode = runconfig_params["mode"]

    if not os.environ.get("RUN_LEGACY_CSTORCH_FLOW"):
        logging.info("Running Trainer Flow")

        from cerebras.modelzoo.trainer.utils import run_trainer

        # Recursively update the params with the runconfig
        if "runconfig" in params and "trainer" in params:
            # runconfig was injected into params by the CLI parser
            # and we need to merge it with the trainer params
            from cerebras.modelzoo.trainer.utils import (
                convert_legacy_params_to_trainer_params,
                merge_trainer_params,
            )

            # Convert and filter out None values
            converted = convert_legacy_params_to_trainer_params(
                {"runconfig": params.pop("runconfig")}
            )

            trainers = params["trainer"]
            if isinstance(trainers, (list, tuple)):
                params["trainer"] = [
                    merge_trainer_params(trainer, converted)
                    for trainer in trainers
                ]
            else:
                params = merge_trainer_params(params, converted)

        # We will be getting a params_obj if we are using config classes
        return run_trainer(
            mode,
            params,
            model_fn,
            train_data_fn,
            eval_data_fn,
        )

    # Save the params
    summary_dir = os.path.join(runconfig_params["model_dir"], mode)
    os.makedirs(summary_dir, exist_ok=True)
    with open(os.path.join(summary_dir, f"params_{mode}.yaml"), "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)

    wsc_log_level = params["runconfig"].get("wsc_log_level") or {}
    set_wsc_log_level(wsc_log_level)

    if isinstance(model_fn, str):
        model_name = model_fn
        params_obj = create_config_obj_from_params(params, model_name)
        model_fn = registry.get_model_class(model_name)
        params = convert_to_dict(params_obj)
    else:
        params_obj = None

    return run_cstorch_flow(
        params, params_obj, model_fn, train_data_fn, eval_data_fn
    )


def set_wsc_log_level(log_levels: Union[List[str], Dict[str, str]]):
    """Assert the list of log levels is valid."""
    if isinstance(log_levels, dict):
        for task, level in log_levels.items():
            level = int(level) if level.isdigit() else get_level_name(level)
            if task:
                wsc_logger.getChild(task).setLevel(level)
            else:
                wsc_logger.setLevel(level)
    else:
        raise ValueError("Invalid log levels. Input must be a dict.")

    # validate log level setting
    collect_wsc_log_settings()
