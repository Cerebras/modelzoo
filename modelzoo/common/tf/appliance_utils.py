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
import sys
import time
import traceback
from typing import Callable, List, Optional

import tensorflow as tf
import yaml

from modelzoo import CSOFT_PACKAGE, CSoftPackage
from modelzoo.common.run_utils.cli_parser import get_params_from_args
from modelzoo.common.run_utils.utils import DeviceType
from modelzoo.common.tf.estimator.cs_estimator import CerebrasEstimator
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.estimator.utils import (
    cs_disable_summaries,
    cs_enable_summaries,
)
from modelzoo.common.tf.run_utils import get_csrunconfig_dict, save_params

if CSOFT_PACKAGE == CSoftPackage.WHEEL or CSOFT_PACKAGE == CSoftPackage.SRC:
    from cerebras_tensorflow.saver.checkpoint_reader import CheckpointReader

    from cerebras_appliance.CSConfig import CSConfig
    # pylint: disable=import-error
    from cerebras_appliance.pb.workflow.appliance.common.common_config_pb2 import (
        DebugArgs,
    )
    from cerebras_appliance.pb.workflow.appliance.common.common_config_pb2 import (
        ExecutionStrategy as ApplianceExecutionStrategy,
    )
    from cerebras_appliance.run_utils import get_debug_args as parse_debug_args


class ExecutionStrategy:
    """Represent Cerebras Execution Strategies"""

    pipeline = "pipeline"
    weight_streaming = "weight_streaming"

    @classmethod
    def strategies(cls):
        """Returns all available strategies."""
        return [cls.pipeline, cls.weight_streaming]

    @classmethod
    def as_appliance_key(cls, key: str):
        """Transform strategy string key to a typed enum."""
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
            debug_args.debug_mgr.scheduler_hints[
                "allow-systems"
            ] = allow_systems

    return debug_args


def get_debug_args(debug_args_path, debug_ini_path):
    """Appliance mode DebugArgs."""
    debug_args = DebugArgs()
    if debug_args_path:
        debug_args = parse_debug_args(debug_args_path)

    debug_ini_fp = os.path.join(debug_ini_path, "debug.ini")
    if os.path.exists(debug_ini_fp):
        with open(debug_ini_fp, "r") as fp:
            ini_content = fp.read()
            debug_args.debug_ini.frozen_content = ini_content
            debug_args = get_debug_mgr_args(debug_ini_fp, debug_args)
    return debug_args


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

    def tf_specific_args_parser():
        tf_parser = argparse.ArgumentParser(add_help=False)
        tf_parser_opt = tf_parser.add_argument_group(
            "Optional Arguments, Tensorflow Specific"
        )
        tf_parser_opt.add_argument(
            "--steps",
            type=int,
            default=None,
            help="Specifies the number of steps to run",
        )
        tf_gpu_parser = argparse.ArgumentParser(add_help=False)
        tf_parser_gpu_opt = tf_gpu_parser.add_argument_group(
            "Optional Arguments, Tensorflow GPU Runs"
        )
        tf_parser_gpu_opt.add_argument(
            "--device",
            default=None,
            help="Force model to run on a specific device (e.g., --device /gpu:0)",
        )
        return {
            DeviceType.ANY: [tf_parser],
            DeviceType.GPU: [tf_gpu_parser],
        }

    params = get_params_from_args(
        run_dir, extra_args_parser_fn=tf_specific_args_parser
    )

    if set_default_params:
        set_default_params(params)

    return params


def update_debug_args_from_stack_params(
    debug_args, stack_params_fn: Callable[[dict], dict], params: dict
) -> None:
    """Gets stack params and encodes them in the give debug args.

    Args:
        debug_args: The debug args in which to inject the stack params.
        stack_params_fn: A callable that takes in params and returns a dict of
            stack params for the model.
        params: The parsed model params.
    """
    # pylint: disable=import-error
    from google.protobuf import json_format

    from cerebras_appliance.pb.stack.full_pb2 import FullConfig

    # FullConfig is only set for CS-X runs, so
    # here we set a dummy cs-ip and revert later
    params["runconfig"]["cs_ip"] = "localhost:1234"
    stack_params = stack_params_fn(params) or {}
    params["runconfig"].pop("cs_ip")

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


def setup_logging(level: str, logging_dir: Optional[str] = None):
    """Sets up the logging verbosity level.

    Args:
        level: The logging level string.
        logging_dir: Where to store logs for archival purposes.
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
    handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handlers.append(handler)
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        logging_file = os.path.join(logging_dir, f"run_{time_stamp}.log")
        handler = logging.FileHandler(logging_file)
        handlers.append(handler)
        tf_logger = logging.getLogger("tensorflow")
        tf_logger.addHandler(handler)
    # Remove any handlers that may have been inadvertently set before
    logging.getLogger().handlers.clear()

    logging.basicConfig(level=level, handlers=handlers)

    original_hook = sys.excepthook

    def cerebras_logging_hook(exc_type, exc_value, exc_traceback):
        """Pipe uncaught exceptions through logger"""
        msg = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        logging.error(f"Uncaught exception:\n{msg}")
        if (
            original_hook != sys.__excepthook__
            and original_hook != cerebras_logging_hook
        ):
            original_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = cerebras_logging_hook


def run_appliance(
    model_fn: Callable,
    train_input_fn: Callable,
    eval_input_fn: Callable,
    supported_strategies: List[str],
    default_params_fn: Optional[Callable] = None,
    stack_params_fn: Optional[Callable] = None,
    enable_cs_summaries: bool = False,
):
    """Helper method for running models locally or on CS-X Systems.

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
        enable_summaries: Enable summaries when running on CS-X hardware.
    """
    # Set run directory to the directory location of the file from which this
    # method was called.
    parent = inspect.getouterframes(inspect.currentframe())[1]
    run_dir = os.path.dirname(os.path.abspath(parent.filename))

    # Parse cmdline arguments and get params
    params = parse_args_and_params(run_dir, default_params_fn)
    runconfig_params = params["runconfig"]

    use_cs = runconfig_params["target_device"] == DeviceType.CSX
    if not use_cs:
        assert not runconfig_params[
            "multireplica"
        ], "Multi-replica training is only possible on the Cerebras System."

    # Figure out the execution strategy
    if not runconfig_params.get("execution_strategy"):
        # If not provided, default to the first one in supported strategies
        runconfig_params["execution_strategy"] = supported_strategies[0]
    if runconfig_params["execution_strategy"] not in supported_strategies:
        raise ValueError(
            f"This model does not support "
            f"{runconfig_params['execution_strategy']} execution strategy."
        )
    if (
        runconfig_params["execution_strategy"]
        == ExecutionStrategy.weight_streaming
    ):
        assert not runconfig_params["multireplica"], (
            f"Multi-replica training is not possible in "
            f"{runconfig_params['execution_strategy']} execution strategy."
        )

    # Save params to file for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])

    # Setup logging
    setup_logging(
        runconfig_params.get("logging"), runconfig_params.get("model_dir")
    )

    # Figure out the input_fn to use
    if runconfig_params["mode"] == "train":
        input_fn = train_input_fn
        mode = tf.estimator.ModeKeys.TRAIN
    elif (
        runconfig_params["mode"] == "eval"
        or runconfig_params["mode"] == "eval_all"
    ):
        input_fn = eval_input_fn
        mode = tf.estimator.ModeKeys.EVAL
    else:
        raise ValueError(f'Mode not supported: {runconfig_params["mode"]}')

    # Create debug args and CSConfig
    cs_config = None
    if CSOFT_PACKAGE is not CSoftPackage.NONE:
        debug_args = get_debug_args(
            runconfig_params["debug_args_path"], run_dir
        )
        if (
            runconfig_params["execution_strategy"] == ExecutionStrategy.pipeline
            and stack_params_fn is not None
        ):
            update_debug_args_from_stack_params(
                debug_args, stack_params_fn, params
            )
        cs_config = CSConfig(
            num_csx=runconfig_params.get("num_csx", 1),
            max_wgt_servers=runconfig_params["num_wgt_servers"],
            max_act_per_csx=runconfig_params["num_act_servers"],
            mgmt_address=runconfig_params.get("mgmt_address"),
            mgmt_namespace=runconfig_params.get("mgmt_namespace"),
            credentials_path=runconfig_params.get("credentials_path"),
            debug_args=debug_args,
            mount_dirs=runconfig_params["mount_dirs"],
            python_paths=runconfig_params["python_paths"],
            transfer_processes=runconfig_params.get("transfer_processes"),
            num_workers_per_csx=runconfig_params["num_workers_per_csx"],
            execution_strategy=ExecutionStrategy.as_appliance_key(
                runconfig_params["execution_strategy"]
            ),
            job_labels=runconfig_params["job_labels"],
            job_time_sec=runconfig_params["job_time_sec"],
            disable_version_check=runconfig_params["disable_version_check"],
        )
    # Create the run config if running within some sort of Cerebras environment
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)

    cs_run_config = CSRunConfig(cs_config=cs_config, **csrunconfig_dict,)

    # Create context for capturing summaries on CS-X. Note that summaries in
    # multi-replica training are not supported.
    summary_ctx = (
        cs_enable_summaries()
        if enable_cs_summaries and not runconfig_params["multireplica"]
        else cs_disable_summaries()
    )

    with summary_ctx:
        # Create estimator
        cs_estimator = CerebrasEstimator(
            model_fn,
            params=params,
            config=cs_run_config,
            model_dir=runconfig_params["model_dir"],
            compile_dir=runconfig_params["compile_dir"],
            warm_start_from=runconfig_params["checkpoint_path"],
        )

        # Run the requested mode
        if (
            runconfig_params["validate_only"]
            or runconfig_params["compile_only"]
        ):
            cs_estimator.compile(
                input_fn,
                validate_only=runconfig_params["validate_only"],
                mode=mode,
            )
        elif runconfig_params["mode"] == 'eval':
            cs_estimator.evaluate(
                input_fn,
                steps=runconfig_params["eval_steps"],
                checkpoint_path=runconfig_params["checkpoint_path"],
                use_cs=use_cs,
            )
        elif runconfig_params["mode"] == 'eval_all':
            # Each individual run is a single eval
            params["runconfig"]['mode'] = "eval"

            model_dir = runconfig_params["model_dir"]

            if CSOFT_PACKAGE is not CSoftPackage.NONE:
                ckpts = CheckpointReader.all_checkpoints(model_dir)
            else:
                ckpts = tf.train.get_checkpoint_state(
                    model_dir
                ).all_model_checkpoint_paths

            if not ckpts:
                raise ValueError(
                    f"model_dir {model_dir} does not contain any checkpoints. "
                    f"Please double check your directory and associated "
                    f"metadata files."
                )

            for ckpt in ckpts:
                if CSOFT_PACKAGE is not CSoftPackage.NONE:
                    if not CheckpointReader.is_a_checkpoint(ckpt):
                        logging.warning(
                            f"Checkpoint {ckpt} is in the checkpoint metadata file "
                            f"in the model dir {model_dir}, but does not exist. "
                            f"Skipping eval for this checkpoint."
                        )
                        continue

                logging.info(f"Running evaluate on checkpoint: {ckpt}")
                cs_estimator.evaluate(
                    input_fn,
                    steps=runconfig_params["eval_steps"],
                    checkpoint_path=ckpt,
                    use_cs=use_cs,
                )
        elif runconfig_params["mode"] == "train":
            cs_estimator.train(
                input_fn,
                steps=runconfig_params["steps"],
                max_steps=runconfig_params["max_steps"],
                use_cs=use_cs,
            )
        else:
            raise ValueError(f'Mode not supported: {runconfig_params["mode"]}')
