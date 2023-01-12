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

#!/usr/bin/env python3

"""
Helper utilities for running on cerebras appliance cluster
"""

import argparse
import inspect
import logging
import os
from typing import Callable, List, Optional

import tensorflow as tf
import yaml
from cerebras_tensorflow.cs_estimator_app import CerebrasAppEstimator

from cerebras_appliance.cs_run_config import CSRunConfig
from cerebras_appliance.CSConfig import CSConfig
from cerebras_appliance.pb.workflow.appliance.common.common_config_pb2 import (
    DebugArgs,
)
from cerebras_appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ExecutionStrategy as ApplianceExecutionStrategy,
)
from cerebras_appliance.run_utils import get_debug_args as parse_debug_args
from modelzoo.common.tf.run_utils import (
    get_csrunconfig_dict,
    get_params,
    save_params,
    update_params_from_args,
)


class ExecutionStrategy:
    pipeline = "pipeline"
    weight_streaming = "weight_streaming"

    @classmethod
    def strategies(cls):
        return [cls.pipeline, cls.weight_streaming]

    @classmethod
    def as_appliance_key(cls, key: str):
        if key == cls.pipeline:
            return ApplianceExecutionStrategy.ES_PIPELINE
        elif key == cls.weight_streaming:
            return ApplianceExecutionStrategy.ES_WEIGHT_STREAMING
        else:
            raise ValueError(f"Unhandled execution strategy: {key}")


def get_debug_mgr_args(debug_ini_fp, debug_args):
    """Appliance mode get debug_mgr related handling"""
    if os.path.exists(debug_ini_fp):
        with open(debug_ini_fp, "r") as fp:
            ini_content_dict = yaml.safe_load(fp)
            if not ini_content_dict:
                return debug_args
        clean_pod_policy = ini_content_dict.get('clean_pod_policy', True)
        cbcore_task_spec = ini_content_dict.get('cbcore_task_spec', None)
        scheduler_hint_allow_systems = ini_content_dict.get(
            'scheduler_hint_allow_systems', None
        )
        allow_systems = ini_content_dict.get('allow-systems', None)
        if clean_pod_policy != True:
            if clean_pod_policy.lower() == 'none':
                debug_args.debug_mgr.clean_pod_policy = (
                    DebugArgs.DebugMGR.CleanPodPolicy.NONE
                )
        if cbcore_task_spec:
            for task_type in [
                "coordinator",
                "chief",
                "worker",
                "activation",
                "command",
                "weight",
                "broadcastreduce",
            ]:
                debug_args.debug_mgr.task_spec_hints[
                    task_type
                ].container_image = cbcore_task_spec
        if scheduler_hint_allow_systems or allow_systems:
            try:
                debug_args.debug_mgr.scheduler_hints[
                    "allow-systems"
                ] = allow_systems
            except:
                debug_args.debug_mgr.labels[
                    "!scheduler-hint-allow-systems"
                ] = scheduler_hint_allow_systems

    return debug_args


def get_debug_args(debug_args_path, debug_ini_path):
    """Appliance mode DebugArgs."""
    if debug_args_path:
        debug_args = parse_debug_args(debug_args_path)
    else:
        debug_args = DebugArgs()

    debug_ini_fp = os.path.join(debug_ini_path, "debug.ini")
    if os.path.exists(debug_ini_fp):
        with open(debug_ini_fp, "r") as fp:
            ini_content = fp.read()
            debug_args.debug_ini.frozen_content = ini_content
            debug_args = get_debug_mgr_args(debug_ini_fp, debug_args)
    return debug_args


def create_arg_parser(run_dir: str) -> argparse.ArgumentParser:
    """Create a commandline argument parser.

    Args:
        run_dir: The root directory where to create the model_dir in.
    Returns:
        An ArgumentParser object.
    """
    default_model_dir = os.path.join(run_dir, "model_dir")

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--credentials_path",
        default=None,
        help="Credentials for cluster access.",
    )
    parser.add_argument(
        "--mgmt_address",
        default="cluster-server.cerebras.com:443",
        help="<host>:<port> for cluster management.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "eval"],
        help="Whether to train or evaluate the model.",
    )
    parser.add_argument(
        "--params",
        required=True,
        help="Path to the model parameters YAML file.",
    )
    parser.add_argument(
        "--num_csx",
        default=1,
        type=int,
        help="Number of CS-X nodes to use in weight streaming mode. Pipeline "
        "mode only supports execution on one CS-X.",
    )
    parser.add_argument(
        "--steps",
        default=None,
        type=int,
        help="Number of steps to train, regardless of the global step loaded "
        "from the checkpoint, if any.",
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="Maximum global step (possibly loaded from checkpoint) to train "
        "up to.",
    )
    parser.add_argument(
        "--num_wgt_servers",
        default=None,
        type=int,
        help="Maximum number of weight servers to use in weight streaming "
        "execution strategy.",
    )
    parser.add_argument(
        "--num_workers_per_csx",
        default=0,
        type=int,
        help="Number of workers to use for streaming inputs per CS node. If "
        "0, a default value based on the model will be chosen. Defaults "
        "to 0.",
    )
    group.add_argument(
        "--compile_only",
        action="store_true",
        help="Compile model completely, generating compiled executables.",
    )
    group.add_argument(
        "--validate_only",
        action='store_true',
        help="Compile model up to kernel matching only.",
    )
    parser.add_argument(
        "--execution_strategy",
        choices=ExecutionStrategy.strategies(),
        type=str,
        default=None,
        help="Execution strategy for running the model.",
    )
    parser.add_argument(
        "--multireplica",
        action="store_true",
        help="Run multiple copies of the model data-parallel on the wafer at "
        "the same time. This option only takes effect when running in "
        "pipeline execution strategy.",
    )
    parser.add_argument(
        "--model_dir",
        default=default_model_dir,
        help="Model directory where checkpoints will be written to. If "
        "directory exists, weights are loaded from the checkpoint file.",
    )
    parser.add_argument(
        "--debug_args_path", default=None, help="Path to debug args file.",
    )
    parser.add_argument(
        "--mount_dirs",
        nargs="+",
        help="A list of paths to be mounted to the appliance containers.",
    )
    parser.add_argument(
        "--python_paths",
        nargs="+",
        help="A list of paths to be exported into PYTHONPATH for worker "
        "containers.",
    )
    parser.add_argument(
        "--transfer_processes",
        type=int,
        default=None,
        help="Number of processes to use when transferring weights.",
    )
    parser.add_argument(
        "--compile_dir", default=None, help="Compile directory."
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to initialize weights from.",
    )
    parser.add_argument(
        "--logging",
        default=None,
        help="Specifies the logging level. Defaults to INFO.",
    )

    return parser


def parse_args_and_params(
    run_dir: str, set_default_params: Optional[Callable] = None,
) -> dict:
    """Parses commandline arguments and returns the params.

    Args:
        run_dir: The root directory where to create the model_dir in.
        set_default_params: A callable that updates params with some defaults
            specific to this model. Defaults to None.
    Returns:
        Params parsed from cmdline arguments and the params file.
    """
    parser = create_arg_parser(run_dir)
    args = parser.parse_args()

    params = get_params(args.params)
    if set_default_params:
        set_default_params(params)

    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)

    return params


def update_debug_args_from_stack_params(
    debug_args: DebugArgs, stack_params_fn: Callable, params: dict
) -> None:
    """Gets stack params and encodes them in the give debug args.

    Args:
        debug_args: The debug args in which to inject the stack params.
        stack_params_fn: A callable that takes in params and returns a dict of
            stack params for the model.
        params: The parsed model params.
    """
    from google.protobuf import json_format

    from cerebras_appliance.pb.stack.full_pb2 import FullConfig

    # FullConfig is only set for CS-X runs, so here we set a dummy cs-ip and
    # revert later
    params["cs_ip"] = "localhost:1234"
    stack_params = stack_params_fn(params)
    params.pop("cs_ip")

    ir_mode = stack_params.get("ir_mode", "mlir-cirh")
    if ir_mode != "mlir-cirh":
        raise ValueError(
            f"Appliance mode only supports models with `mlir-cirh` backend, "
            f"but this model uses {ir_mode}. Please contact Cerebras support "
            f"for more details."
        )

    runtime_full_config = stack_params.get("config", None)
    if runtime_full_config:
        if debug_args.debug_crd.pipeline_options.full_config:
            full_config = json_format.Parse(
                debug_args.debug_crd.pipeline_options.full_config, FullConfig(),
            )
            full_config.MergeFrom(runtime_full_config)
        else:
            full_config = runtime_full_config

        debug_args.debug_crd.pipeline_options.full_config = json_format.MessageToJson(
            full_config,
            including_default_value_fields=False,
            preserving_proto_field_name=True,
            indent=0,
            sort_keys=True,
        )


def setup_logging(level: str):
    """Sets up the logging verbosity level.

    Args:
        level: The logging level string.
    """
    level = level or "INFO"
    level = level.upper()
    assert level in (
        "CRITICAL",
        "ERROR",
        "WARNING",
        "INFO",
        "DEBUG",
    ), f"Invalid logging level: {level}"
    level = logging.getLevelName(level)

    tf.compat.v1.logging.set_verbosity(level)
    logging.basicConfig(level=level)


def run_appliance(
    model_fn: Callable,
    train_input_fn: Callable,
    eval_input_fn: Callable,
    supported_strategies: List[str],
    default_params_fn: Optional[Callable] = None,
    stack_params_fn: Optional[Callable] = None,
):
    """Helper method for running models on Appliance.

    Args:
        model_fn: A callable for creating the model.
        train_input_fn: A callable for creating a data input pipeline for train.
        eval_input_fn: A callable for creating a data input pipeline for eval.
        supported_strategies: List of supported execution strategies. If a
            strategy is not explicitly selected in cmdline args, the default
            strategy chosen is the first item in this list.
        default_params_fn: A callable that takes in the parsed params and sets
            defaults for missing params.
        stack_params_fn: A callable that takes in the parsed params and sets
            Cerebras-specific config for stack compilation.
    """
    # Set run directory to the directory location of the file from which this
    # method was called.
    parent = inspect.getouterframes(inspect.currentframe())[1]
    run_dir = os.path.dirname(os.path.abspath(parent.filename))

    # Parse cmdline arguments and get params
    params = parse_args_and_params(run_dir, default_params_fn)
    runconfig_params = params["runconfig"]

    # Figure out the execution strategy
    if not runconfig_params.get("execution_strategy"):
        # If not provided, default to the first one in supported strategies
        runconfig_params["execution_strategy"] = supported_strategies[0]
    if runconfig_params["execution_strategy"] not in supported_strategies:
        raise ValueError(
            f"This model does not support "
            f"{runconfig_params['execution_strategy']} execution strategy."
        )

    # Save params to file for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])

    # Setup logging
    setup_logging(runconfig_params.get("logging"))

    # Create debug args
    debug_args = get_debug_args(runconfig_params["debug_args_path"], run_dir)
    if runconfig_params["execution_strategy"] == ExecutionStrategy.pipeline:
        update_debug_args_from_stack_params(debug_args, stack_params_fn, params)

    # Log some settings to console
    logging.info(f"Credentials path: {runconfig_params['credentials_path']}")
    logging.info(f"Debug args: {debug_args}")

    # Figure out the input_fn to use
    if runconfig_params["mode"] == "train":
        input_fn = train_input_fn
        mode = tf.estimator.ModeKeys.TRAIN
    elif runconfig_params["mode"] == "eval":
        input_fn = eval_input_fn
        mode = tf.estimator.ModeKeys.EVAL
    else:
        raise ValueError(f'Mode not supported: {runconfig_params["mode"]}')

    # Create the run config
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    cs_run_config = CSRunConfig(
        cs_config=CSConfig(
            num_csx=runconfig_params["num_csx"],
            max_wgt_servers=runconfig_params["num_wgt_servers"],
            mgmt_address=runconfig_params["mgmt_address"],
            credentials_path=runconfig_params["credentials_path"],
            debug_args=debug_args,
            mount_dirs=runconfig_params["mount_dirs"],
            python_paths=runconfig_params["python_paths"],
            transfer_processes=runconfig_params["transfer_processes"],
            num_workers_per_csx=runconfig_params["num_workers_per_csx"],
            execution_strategy=ExecutionStrategy.as_appliance_key(
                runconfig_params["execution_strategy"]
            ),
        ),
        **csrunconfig_dict,
    )

    # Create estimator
    cs_estimator = CerebrasAppEstimator(
        model_fn,
        params=params,
        config=cs_run_config,
        model_dir=runconfig_params["model_dir"],
        compile_dir=runconfig_params["compile_dir"],
        warm_start_from=runconfig_params["checkpoint_path"],
    )

    # Run the requested mode
    if runconfig_params["validate_only"] or runconfig_params["compile_only"]:
        cs_estimator.compile(
            input_fn, validate_only=runconfig_params["validate_only"], mode=mode
        )
    elif runconfig_params["mode"] == 'eval':
        cs_estimator.evaluate(
            input_fn,
            steps=runconfig_params["eval_steps"],
            checkpoint_path=runconfig_params["checkpoint_path"],
            use_cs=True,
        )
    elif runconfig_params["mode"] == "train":
        cs_estimator.train(
            input_fn,
            steps=runconfig_params["steps"],
            max_steps=runconfig_params["max_steps"],
            use_cs=True,
        )
    else:
        raise ValueError(f'Mode not supported: {runconfig_params["mode"]}')
