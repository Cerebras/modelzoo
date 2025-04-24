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
import os
import subprocess
import sys
from shutil import which
from typing import Any, Callable, Dict, List, Optional

from cerebras.modelzoo.common.utils.run.cli_pytorch import get_params_from_args
from cerebras.modelzoo.common.utils.run.utils import DeviceType


def torchrun(filename: str, arguments: List[str]):
    """Starts a distributed GPU run using torchrun."""
    import torch

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
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """Entry point to running pytorch models including CLI argument parsing.

    Args:
        extra_args_parser_fn: An optional callable that adds any
            extra parser args not covered in `get_parser` fn.
    """
    parent = inspect.getouterframes(inspect.currentframe())[1]
    params = get_params_from_args(extra_args_parser_fn)
    main(
        params,
        script=parent.filename,
        extra_args_parser_fn=extra_args_parser_fn,
    )


def main(
    params: Dict[str, Any],
    script: Optional[str] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """Runs a full end-to-end CS/non-CS workflow for a PyTorch model.

    Args:
        params: The parsed YAML config dictionary.
        script: The script to run in subprocesses for distributed GPU runs.
        extra_args_parser_fn: An optional callable that adds any
            extra parser args not covered in `get_parser` fn.
    """
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

    from cerebras.modelzoo.common.pytorch_utils import RunConfigParamsValidator
    from cerebras.modelzoo.trainer.restartable_trainer import RestartableTrainer
    from cerebras.modelzoo.trainer.utils import (
        inject_cli_args_to_trainer_params,
        run_trainer,
    )

    runconfig_params = params["runconfig"]
    RunConfigParamsValidator(extra_args_parser_fn).validate(runconfig_params)
    mode = runconfig_params["mode"]

    # Recursively update the params with the runconfig
    if "runconfig" in params and "trainer" in params:
        params = inject_cli_args_to_trainer_params(
            params.pop("runconfig"), params
        )
        if RestartableTrainer.is_restart_config(params):
            return RestartableTrainer(params).run_trainer(mode)

    return run_trainer(mode, params)
