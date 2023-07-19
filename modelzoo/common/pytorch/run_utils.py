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

"""Utilities for running Cerebras Pytorch Models"""
import argparse
import inspect
import math
import os
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional

import torch
import yaml

import modelzoo.common.pytorch.half_dtype as half_dtype
from modelzoo import CSOFT_PACKAGE, CSoftPackage
from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import is_ltc_mlir_mode_enabled, modes
from modelzoo.common.pytorch.pytorch_base_runner import PyTorchBaseRunner
from modelzoo.common.pytorch.utils import (
    RunConfigParamsValidator,
    get_checkpoints,
    setup_logging,
)
from modelzoo.common.run_utils.cli_parser import get_params_from_args
from modelzoo.common.run_utils.utils import DeviceType

DATA_FN_TYPE = Callable[[dict], torch.utils.data.DataLoader]
half_dtype_instance = half_dtype.half_dtype_instance


def arg_filter(arg: str, keyword: str) -> bool:
    """Checks if a given arg matches the given keyword"""
    arg = arg.strip()
    return (
        arg.startswith(f"--{keyword}=")
        or arg.startswith(f"-{keyword}=")
        or arg == f"--{keyword}"
        or arg == f"-{keyword}"
    )


def update_sideband_mode_arg(
    arguments: List[str], new_mode_arg: str, old_mode: str
) -> List[str]:
    """Updates sideband arguments to a different mode"""
    # filter out args with the name of the old mode provided they
    # have "mode" or "m" preceding
    offset_arguments = [None] + arguments
    updated_args = [
        an_arg
        for an_arg, prev_arg in zip(arguments, offset_arguments)
        if an_arg != old_mode
        or not (arg_filter(prev_arg, "mode") or arg_filter(prev_arg, "m"))
    ]

    # filter and add the new mode
    updated_args = [
        new_mode_arg
        if arg_filter(an_arg, "mode") or arg_filter(an_arg, "m")
        else an_arg
        for an_arg in updated_args
    ]

    return updated_args


def sideband_eval_all(
    filename: str, arguments: List[str], params: Dict[Any, Any]
):
    """Temporary support for running eval multiple times via subprocess"""
    eval_mode = "--mode=eval"

    if any(arg_filter(an_arg, "checkpoint_path") for an_arg in arguments):
        raise ValueError(
            "Checkpoint path cannot be provided with eval_all. Checkpoints inferred from model_dir"
        )

    updated_args = update_sideband_mode_arg(
        arguments, eval_mode, modes.EVAL_ALL
    )

    # Gather all checkpoints
    checkpoint_path = None
    updated_args.append(checkpoint_path)
    checkpoints = get_checkpoints(params['runconfig']['model_dir'],)
    if len(checkpoints) == 0:
        raise ValueError(
            f"No checkpoints found at {params['runconfig']['model_dir']}"
        )
    for a_chkpt in checkpoints:
        checkpoint_path = f"--checkpoint_path={a_chkpt}"
        updated_args[-1] = checkpoint_path
        # By just calling this from the top each run will be a separate logdir
        single_run = [sys.executable, filename]
        single_run.extend(updated_args)
        subprocess.run(single_run, check=True)


def sideband_train_eval_all(
    filename: str, arguments: List[str], params: Dict[Any, Any]
):
    """Temporary support for running train and eval multiple times via subprocess"""
    train_mode = "--mode=train"
    eval_mode = "--mode=eval"

    train_args = update_sideband_mode_arg(
        arguments, train_mode, modes.TRAIN_AND_EVAL
    )
    eval_args = update_sideband_mode_arg(
        arguments, eval_mode, modes.TRAIN_AND_EVAL
    )

    runconfig = params['runconfig']
    if runconfig.get('num_steps', None) is not None:
        if runconfig.get('num_epochs', None) is not None:
            raise ValueError(
                "num_steps and num_epochs cannot both be specified "
                "in the runconfig section of params"
            )
        if runconfig.get('steps_per_epoch', None) is not None:
            raise ValueError(
                "num_steps and steps_per_epoch cannot both be specified "
                "in the runconfig section of params"
            )
        if runconfig.get('eval_frequency', None) is None:
            raise ValueError(
                "if num_steps is specified, eval_frequency is needed "
                "to dictate how many train steps before each eval"
            )
        total_steps = int(runconfig['num_steps'])
        train_steps = int(runconfig['eval_frequency'])
        num_iters = math.ceil(total_steps / train_steps)

        last_steps = total_steps % train_steps

        # add num_steps overwrite
        last_train_args = train_args + [f"--num_steps={last_steps}"]
        train_args.append(f"--num_steps={train_steps}")

    elif runconfig.get('num_epochs', None) is not None:
        num_iters = int(runconfig['num_epochs'])

        # add num_epochs overwrite
        train_args.append("--num_epochs=1")
        last_steps = 0
    else:
        raise ValueError(
            "For train_and_eval mode, one of `num_steps` or `num_epochs` "
            " must be specified and not be None."
        )

    single_run = [sys.executable, filename]
    train_cmd = single_run + train_args
    eval_cmd = single_run + eval_args
    for i in range(num_iters):
        # TRAIN
        if i == num_iters - 1 and last_steps > 0:
            train_cmd = single_run + last_train_args
        try:
            subprocess.run(train_cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Training at iteration {i} failed.") from e

        # EVAL
        try:
            subprocess.run(eval_cmd, check=True)
        except Exception as e:
            raise RuntimeError(f"Evaluate at iteration {i} failed.") from e


def run(
    model_fn: Callable[[dict], torch.nn.Module],
    train_data_fn: Optional[DATA_FN_TYPE] = None,
    eval_data_fn: Optional[DATA_FN_TYPE] = None,
    default_params_fn: Optional[Callable[[dict], dict]] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """Backward compatible entry point to running pytorch models"""
    parent = inspect.getouterframes(inspect.currentframe())[1]
    run_dir = os.path.dirname(os.path.abspath(parent.filename))
    params = get_params_from_args(run_dir, extra_args_parser_fn)

    if default_params_fn:
        params = default_params_fn(params) or params

    main(params, model_fn, train_data_fn, eval_data_fn, script=parent.filename)


def main(
    params: Dict[str, Any],
    model_fn: Callable[[dict], torch.nn.Module],
    train_data_fn: Optional[DATA_FN_TYPE] = None,
    eval_data_fn: Optional[DATA_FN_TYPE] = None,
    script: Optional[str] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """Entry point to running pytorch models"""
    if not script:
        parent = inspect.getouterframes(inspect.currentframe())[1]
        script = parent.filename

    if params["runconfig"]["mode"] == modes.EVAL_ALL:
        sideband_eval_all(script, sys.argv[1:], params)
        return None
        # TODO ambiguity on what to return, possibly just run the final checkpoint in
        # the main process below
    # TODO enable existing train_and_eval functionality to work with cs
    if (
        params["runconfig"]["mode"] == modes.TRAIN_AND_EVAL
        and params["runconfig"]["target_device"] == DeviceType.CSX
    ):
        sideband_train_eval_all(script, sys.argv[1:], params)
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
    model_fn: Callable[[dict], torch.nn.Module],
    train_data_fn: Optional[DATA_FN_TYPE] = None,
    eval_data_fn: Optional[DATA_FN_TYPE] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """
    Runs a full end-to-end CS/non-CS workflow for a given model

    Args:
        model_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a torch.nn.Module
        train_data_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a PyTorch dataloader
            corresponding to the training dataset
        eval_data_fn: A callable that takes in a 'params' argument
            which it uses to configure and return a PyTorch dataloader
            corresponding to the evaluation dataset
        default_params_fn: An optional callable that takes in the params
            dictionary and updates any missing params
            with default values
        extra_args_parser_fn: An optional callable that adds any
            extra parser args not covered in `get_parser` fn.
    """
    runconfig_params = params["runconfig"]
    RunConfigParamsValidator(extra_args_parser_fn).validate(runconfig_params)

    if (
        params["runconfig"]["target_device"] in (DeviceType.CSX, DeviceType.CPU)
        and (
            params["runconfig"].get("experimental_api", False)
            or is_ltc_mlir_mode_enabled()
        )
        # TODO: Remove this check once we add a no dependency flow
        and CSOFT_PACKAGE != CSoftPackage.NONE
    ):
        params["runconfig"]["experimental_api"] = True

        # pylint: disable=import-outside-toplevel
        from modelzoo.common.pytorch.run_cstorch_flow import run_cstorch_flow

        # Use the new experimental API flow
        return run_cstorch_flow(params, model_fn, train_data_fn, eval_data_fn)
    else:
        params["runconfig"]["experimental_api"] = False

        # Default to the previous PyTorchBaseModel/PyTorchBaseRunner flow
        return run_base_model_flow(
            params, model_fn, train_data_fn, eval_data_fn,
        )


def run_base_model_flow(params, model_fn, train_data_fn, eval_data_fn):
    """Runs PytorchBaseModel and Runner flow"""
    runconfig_params = params["runconfig"]

    # Set up logging level as soon as possible. Ideally, we could even move this
    # to the pytorch_cli entry point to set up logging during import of the
    # modelzoo model...
    setup_logging(
        runconfig_params.get("logging"),
        runconfig_params.get("streamer_logging"),
        logging_dir=runconfig_params.get("model_dir"),
    )

    if "seed" in runconfig_params:
        torch.manual_seed(runconfig_params["seed"])

    runner = PyTorchBaseRunner.create(model_fn, params)

    # Save params.yaml only in master task
    mode = runconfig_params["mode"]
    # using this dir structure to keep in sync with runners
    summary_dir = os.path.join(params["runconfig"]["model_dir"], f"{mode}")
    os.makedirs(summary_dir, exist_ok=True)
    if cm.is_master_ordinal():
        with open(
            os.path.join(summary_dir, f"params_{mode}.yaml"), "w+",
        ) as _fout:
            yaml.dump(params, _fout, default_flow_style=False)

    # Initialize the dataloaders depending on the mode
    if mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
        assert train_data_fn, "Train dataloader function has not been provided"
        train_loader = train_data_fn(params)
        runner.train_data_fn = train_data_fn
    if mode in (modes.EVAL, modes.TRAIN_AND_EVAL):
        assert eval_data_fn, "Eval dataloader function has not been provided"
        eval_loader = eval_data_fn(params)
        runner.eval_data_fn = eval_data_fn

    if mode == modes.TRAIN:
        runner.train(train_loader)
    elif mode == modes.EVAL:
        runner.evaluate(eval_loader)
    elif mode == modes.TRAIN_AND_EVAL:
        runner.train_and_eval(train_loader, eval_loader)
    else:
        raise ValueError(f"Mode {mode} is not supported.")
