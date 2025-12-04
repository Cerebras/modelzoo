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

"""GPT Inference script built using the cstorch API"""

import argparse
import logging
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import yaml

# isort: off
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
# isort: on

import cerebras.pytorch as cstorch
import cerebras.pytorch.distributed as dist
from cerebras.appliance.utils.debug_args import (
    get_debug_args,
    update_debug_args_from_keys,
)
from cerebras.appliance.utils.file import create_symlink
from cerebras.appliance.utils.ini import set_ini
from cerebras.modelzoo.common.pytorch_utils import RunConfigParamsValidator
from cerebras.pytorch.utils.call_once import call_once


def format_rate(rate):
    return f"{rate:.3g}" if rate < 1.0 else f"{rate:.2f}"


def get_parser():
    parser = argparse.ArgumentParser(
        "Script for running inference for GPT style models", add_help=False
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=None,
        help="Specifies the number of steps to run for inference.",
    )
    return parser


def _get_cluster_config(params):
    runconfig = params["runconfig"]

    debug_args = get_debug_args(runconfig.get("debug_args_path"))
    if extra_debug_args := runconfig.get("debug_args"):
        update_debug_args_from_keys(debug_args, extra_debug_args)
    if ini := runconfig.get("ini"):
        set_ini(debug_args, **ini)

    cluster_config = cstorch.distributed.ClusterConfig(
        mgmt_address=runconfig.get("mgmt_address"),
        mgmt_namespace=runconfig.get("mgmt_namespace"),
        credentials_path=runconfig.get("credentials_path"),
        num_csx=runconfig.get("num_csx"),
        max_wgt_servers=runconfig.get("num_wgt_servers"),
        max_act_per_csx=runconfig.get("num_act_servers"),
        num_workers_per_csx=runconfig.get("num_workers_per_csx"),
        cbcore_image=runconfig.get("cbcore_image"),
        job_labels=runconfig.get("job_labels"),
        job_time_sec=runconfig.get("job_time_sec"),
        mount_dirs=runconfig.get("mount_dirs"),
        python_paths=runconfig.get("python_paths"),
        disable_version_check=runconfig.get("disable_version_check"),
    )

    job_priority = runconfig.get("job_priority")
    if job_priority:
        cluster_config.job_priority = job_priority

    transfer_processes = runconfig.get("transfer_processes")
    if transfer_processes:
        cstorch.backends.csx.performance.transfer_processes = transfer_processes

    fabric_type_blacklist = runconfig.get("fabric_type_blacklist")
    if fabric_type_blacklist:
        cstorch.backends.csx.debug.fabric_type_blacklist = fabric_type_blacklist

    cstorch.backends.csx.debug.debug_args = debug_args

    if "precision_opt_level" in params["model"]:
        raise ValueError(
            "Passing `precision_opt_level` via `model` params is no longer supported. "
            "Please use `params[\"runconfig\"][\"precision_opt_level\"]` instead."
        )
    precision_opt_level = runconfig.get("precision_opt_level")
    if precision_opt_level is None:
        precision_opt_level = 1

    cstorch.backends.csx.precision.optimization_level = precision_opt_level

    return cluster_config


def get_cluster_config(params):
    cluster_config = _get_cluster_config(params)

    if (
        cluster_config.max_act_per_csx is not None
        and cluster_config.max_act_per_csx > 1
    ):
        warn("max_act_per_csx is forced to 1 for inference")

    cluster_config.max_act_per_csx = 1

    if cluster_config.num_workers_per_csx is None:
        cluster_config.num_workers_per_csx = 1

    return cluster_config


def get_all_checkpoints(model_dir: str) -> List[str]:
    """Return the path to all available checkpoints"""
    ckpts = []
    for checkpoint in Path(model_dir).glob("checkpoint_*.mdl"):
        match = re.match(
            r"checkpoint_(?P<step>\d+)(?:_(?P<timestamp>\d{8}_\d{6}))?.mdl",
            checkpoint.name,
        )
        if not match:
            continue

        step = int(match.group("step"))
        timestamp = match.group("timestamp")
        if timestamp is not None:
            try:
                date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
        else:
            date = datetime.min

        ckpts.append((checkpoint, step, date))

    # sort by step and then by timestamp
    return (
        [ckpt[0] for ckpt in sorted(ckpts, key=lambda x: (x[1], x[2]))]
        if ckpts
        else []
    )


def get_latest_checkpoint(model_dir: str) -> Union[str, None]:
    """Get the path to the checkpoint with the highest global step"""
    ckpts = get_all_checkpoints(model_dir)
    return ckpts[-1] if ckpts else None


def get_model_checkpoint(runconfig: Dict[str, Any]) -> Union[str, None]:
    """Get the path to the model checkpoint, if any."""
    model_dir = runconfig["model_dir"]
    ckpt_path = None

    # if a checkpoint path is provided, use that
    if runconfig.get("checkpoint_path"):
        ckpt_path = runconfig["checkpoint_path"]
    elif runconfig.get("autoload_last_checkpoint", True):
        logging.info(
            f"Checkpoint autoloading is enabled. Looking for latest checkpoint "
            f"in \"{model_dir}\" directory with the following naming "
            f"convention: `checkpoint_(step)(_timestamp)?.mdl`."
        )
        ckpt_path = get_latest_checkpoint(model_dir)
        if ckpt_path:
            logging.info(f"Found latest checkpoint at \"{ckpt_path}\".")
        else:
            logging.info(f"No checkpoints were found in \"{model_dir}\".")

    if not ckpt_path:
        logging.info(
            f"No checkpoint was provided. Using randomly initialized model "
            f"parameters."
        )

    return ckpt_path


def load_from_checkpoint_file(checkpoint_path: str) -> dict:
    """Loads state dict from checkpoint path and checks for version compatibilty."""
    logging.info(f"Loading weights from checkpoint {checkpoint_path}")
    state_dict = cstorch.load(checkpoint_path)
    return state_dict


def setup_logging(
    chief_logging_level: str,
    streamer_logging_level: str,
    logging_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
):
    """Configure default logging format."""

    class CustomFormatter(logging.Formatter):
        """Cerebras Preferred Log Formatting."""

        def __init__(self):
            ordinal = dist.get_ordinal()
            num_tasks = dist.num_tasks() - 1

            if num_tasks > 1 and dist.is_streamer():
                ordinal_msg = f"[{ordinal}/{num_tasks}]"
            else:
                ordinal_msg = ""

            fmt = f"%(asctime)s %(levelname)s: {ordinal_msg}  %(message)s"
            super().__init__(fmt=fmt)

            self.info_formatter = None
            # Only enable shorter info logging depending on environment variable
            # This is so that we have the option to experiment with this in the future
            if "USE_SHORT_INFO_LOGGING" in os.environ:
                fmt = "{}%(message)s".format(
                    f"{ordinal_msg}:  " if ordinal > 0 else ""
                )
                self.info_formatter = logging.Formatter(fmt)

        def format(self, record):
            if self.info_formatter and record.levelno == logging.INFO:
                return logging.Formatter.format(self.info_formatter, record)

            return super().format(record)

    def build_block_filter(handler_type: str):
        """Build a filter to block records from a specific handler."""

        def block_filter(record):
            if hasattr(record, "block"):
                return record.block != handler_type
            return True

        return block_filter

    handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    handler.addFilter(build_block_filter("console"))
    handlers.append(handler)
    if logging_dir:
        logging_file = os.path.join(logging_dir, f"run.log")
        handler = logging.FileHandler(logging_file)
        handler.setFormatter(CustomFormatter())
        handler.addFilter(build_block_filter("file"))
        handlers.append(handler)
        # set up run log symlink
        symlink_dir = Path(model_dir) if model_dir else Path(logging_dir)
        run_log_symlink = symlink_dir / "latest_run.log"
        create_symlink(
            run_log_symlink, Path(logging_file).relative_to(symlink_dir)
        )

    def get_level_name(level):
        if not isinstance(level, str):
            raise ValueError(
                f"Invalid logging level: `{level}`. "
                f"Expected a string or int level."
            )

        try:
            level = int(level)
        except ValueError:
            level = level.upper()

        # Custom levels defined by cerebras.appliance
        if level == "TRACE":
            level = logging.DEBUG - 5
        elif level == "VERBOSE":
            level = logging.INFO - 5
        else:
            if (
                isinstance(level, str)
                and level not in logging._nameToLevel  # pylint: disable=W0212
            ):
                # pylint: disable=protected-access
                raise ValueError(
                    f"Invalid logging level: `{level}`. Expected one of "
                    f"{list(logging._nameToLevel.keys())}."
                )

            level = logging.getLevelName(level)

        return level

    if dist.is_master_ordinal():
        level = get_level_name(chief_logging_level or "info")
    else:
        level = get_level_name(streamer_logging_level or "error")

    # Remove any handlers that may have been inadvertently set before
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=level, handlers=handlers)

    setup_logging_excepthook()


@call_once()
def setup_logging_excepthook():
    """Setup a logging hook that runs whenever an exception is raised that
    catches and logs the exception to ensure that the full traceback is printed
    in the log file.
    """
    original_hook = sys.excepthook

    def cerebras_logging_hook(exc_type, exc_value, exc_traceback):
        """Pipe uncaught exceptions through logger."""
        msg = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        # Block console logging to avoid duplicate messages since exceptions
        # are logged by python interpreter by default anyways.
        logging.error(f"Uncaught exception:\n{msg}", extra={"block": "console"})

        # Run the original except hook which prints the exception to stderr
        original_hook(exc_type, exc_value, exc_traceback)

    sys.excepthook = cerebras_logging_hook


def setup_artifact_dir(model_dir: str, mode: str):
    """
    Create a unique subdirectory for this run by generating a time stamp so
    that parallel runs using the same model_dir don't overwrite common files.
    """

    def _create():
        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        artifact_dir = cerebras_logs_path / mode / time_stamp
        artifact_dir.mkdir(parents=True)
        return artifact_dir

    cerebras_logs_path = Path(model_dir) / "cerebras_logs"

    # CPU runs could potentially finish very fast, so back-to-back runs
    # may end up getting the same timestamp and we'd fail in creating
    # the duplicate directory. In case of directory already existing,
    # sleep for more than 1 second and try again. If we fail again,
    # then throw.
    try:
        artifact_dir = _create()
    except FileExistsError:
        time.sleep(1.5)
        try:
            artifact_dir = _create()
        except Exception as e:
            raise e from None

    # Create a symlink to the artifact_dir so that it's easy to find the latest run.
    # The symlink needs to be at the same level as the subdirectories.
    latest = cerebras_logs_path.joinpath("latest")
    # symlink to relative path
    create_symlink(
        latest,
        artifact_dir.relative_to(cerebras_logs_path),
        target_is_directory=True,
    )
    return str(artifact_dir)


def inference_input_dataloader(params):
    from copy import deepcopy

    params = deepcopy(params)
    inference_input = params["inference_input"]
    data_processor = inference_input["data_processor"]
    if data_processor == "GptHDF5MapDataProcessor":
        from cerebras.modelzoo.data.nlp.gpt.GptHDF5MapDataProcessor import (  # noqa
            GptHDF5MapDataProcessor,
        )
        from cerebras.modelzoo.common.input_utils import PaddingSample

        def map_fn(x):
            # If the input is undefined then it is padding the last batch
            # We fill it with the first start token here so that it can be special cased
            if isinstance(x, PaddingSample):
                start_token = params["model"]["start_token"]
                if isinstance(start_token, list):
                    start_token = start_token[0]
                x.fill_(start_token)
            # In the inference case, we only have a single input_ids feature
            return {"input_ids": x}

        return GptHDF5MapDataProcessor(
            # Only provide keys needed for inference
            {
                "batch_size": inference_input["batch_size"],
                "data_dir": inference_input["data_dir"],
                "max_sequence_length": inference_input.get(
                    "max_sequence_length"
                ),
                "drop_last": inference_input.get("drop_last", False),
                "pad_last": True,
                "dataset_map_fn": map_fn,
            }
        ).create_dataloader()
    else:
        raise ValueError(
            f"Invalid data processsor. Expected one of "
            f"'GptHDF5MapDataProcessor' or 'Gpt2SyntheticDataProcessor'. "
            f"Got: {data_processor}"
        )


def set_attention_params(params):
    '''
    Set attention-related parameters.
    :param params: An object containing model, runconfig attributes
    :return: None
    '''
    # Attention softmax is fp32 by default.
    params["model"]["attention_softmax_fp32"] = True

    if params["runconfig"].get("precision_opt_level", 1) == 2:
        params["model"]["attention_softmax_fp32"] = False

    if (
        params["model"].get("fp16_type", "bfloat16") == "cbfloat16"
        and params["runconfig"].get("precision_opt_level", 1) == 1
    ):
        params["model"]["attention_softmax_fp32"] = False


def set_defaults(params):
    """
    Update any missing parameters in the params dictionary with default values

    Args:
        params/object: The dictionary containing the params
    """
    if (
        params.get("train_input", {}).get("data_processor")
        == "Gpt2SyntheticDataProcessor"
    ):
        if "train_input" in params:
            params["train_input"]["vocab_size"] = params["train_input"].get(
                "vocab_size", params["model"]["vocab_size"]
            )
            assert (
                params["train_input"]["vocab_size"]
                == params["model"]["vocab_size"]
            ), f"Found different vocab_size in train_input ({params['train_input']['vocab_size']}) vs. model ({params['model']['vocab_size']})"
            params["train_input"]["max_sequence_length"] = params[
                "train_input"
            ].get(
                "max_sequence_length",
                params["model"]["max_position_embeddings"],
            )

        if "eval_input" in params:
            params["eval_input"]["vocab_size"] = params["eval_input"].get(
                "vocab_size", params["model"]["vocab_size"]
            )
            assert (
                params["eval_input"]["vocab_size"]
                == params["model"]["vocab_size"]
            ), f"Found different vocab_size in eval_input ({params['eval_input']['vocab_size']}) vs. model ({params['model']['vocab_size']})"
            params["eval_input"]["max_sequence_length"] = params[
                "eval_input"
            ].get(
                "max_sequence_length",
                params["model"]["max_position_embeddings"],
            )

    params["model"]["fp16_type"] = params["model"].get("fp16_type", "bfloat16")
    params["optimizer"]["loss_scaling_factor"] = params["optimizer"].get(
        "loss_scaling_factor", 1.0
    )
    params["optimizer"]["log_summaries"] = params["optimizer"].get(
        "log_summaries", False
    )
    params["runconfig"]["precision_opt_level"] = params["runconfig"].get(
        "precision_opt_level", 1
    )

    set_attention_params(params)

    return params


def main():
    from cerebras.modelzoo.common.utils.run.cli_parser import (
        get_params_from_args,
    )
    from cerebras.modelzoo.common.utils.run.utils import DeviceType

    # Parse args
    parser_fn = lambda: [get_parser()]
    parser_args = {
        "parser_epilog": (
            "Please run 'python run_gpt_inference.py CSX -h'. \n \n"
            "Here is an example command for running on CSX: \n \n"
            "    python run_gpt_inference.py CSX --params /path/to/params --checkpoint_path "
            "/path/to/checkpoint \n \n"
            "Note that inference is currently only supported for device CSX"
        ),
        "csx_parser_epilog": (
            "To see a complete list of all available arguments, \n"
            "please run 'python run_gpt_inference.py CSX -h'. \n\n"
            "Here is an example command for running with CSX: \n \n"
            "    python run_gpt_inference.py CSX --params /path/to/params --checkpoint_path "
            "/path/to/checkpoint \n \n"
            "Inference flow resides in the Cerebras Model Zoo. Please specify --python_paths and \n"
            "--mount_dirs here or in your params.yaml under the 'runconfig' section with \n"
            "the path to the directory in which the Cerebras Model Zoo resides. \n"
        ),
        "modes": ["inference"],
    }

    params = get_params_from_args(
        argv=sys.argv[1:],
        extra_args_parser_fn=parser_fn,
        device_type=DeviceType.CSX,
        **parser_args,
    )

    set_defaults(params)

    # Validate runconfig
    runconfig = params["runconfig"]
    RunConfigParamsValidator(parser_fn).validate(runconfig)

    log_steps = runconfig.get("log_steps")

    # Set up logging level and env vars
    artifact_dir = Path(
        setup_artifact_dir(runconfig["model_dir"], mode="inference")
    )
    setup_logging(
        runconfig.get("logging"),
        runconfig.get("streamer_logging"),
        logging_dir=artifact_dir,
        model_dir=runconfig["model_dir"],
    )

    # Save the params.yaml that is being used in this run to the artifact dir
    with open(os.path.join(artifact_dir, f"params_inference.yaml"), "w") as f:
        yaml.dump(params, f, default_flow_style=False)

    cluster_config = get_cluster_config(params)

    from torch.utils._pytree import tree_map

    import cerebras.pytorch as cstorch
    from cerebras.modelzoo.common.input_utils import (
        validate_streaming_and_micro_batch_size,
    )
    from cerebras.modelzoo.models.nlp.gpt2.model import (
        Gpt2Model,
        GPT2ModelConfig,
    )

    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)

    input_params = params.get("inference_input", {})
    micro_batch_size = input_params.get("micro_batch_size", "auto")
    if "batch_size" in input_params:
        # Checks for invalid setting of num_csx, micro_batch_size and batch_size
        validate_streaming_and_micro_batch_size(
            input_params["batch_size"],
            micro_batch_size,
            cluster_config.num_csx,
        )

    cstorch.backends.csx.performance.micro_batch_size = micro_batch_size
    cstorch.backends.csx.debug.retrace_every_iteration = runconfig.get(
        "retrace_every_iteration", False
    )
    cstorch.backends.csx.debug.lazy_initialization = runconfig.get(
        "lazy_initialization", True
    )

    # Initialize the backend
    backend = cstorch.backend(
        "CSX",
        artifact_dir=artifact_dir,
        compile_dir=runconfig.get("compile_dir"),
        compile_only=compile_only,
        validate_only=validate_only,
        cluster_config=cluster_config,
    )

    # Set the 16 bit dtype we want the automatic mixed precision module to use
    cstorch.amp.set_half_dtype(params["model"].get("fp16_type", "float16"))

    # Initialize model (config_validation returns the Config Class if it finds one)
    with backend.device:
        model = Gpt2Model(GPT2ModelConfig(**params["model"]))

    compiled_model = cstorch.compile(model, backend)
    compiled_model.eval()

    # Load weights
    checkpoint_path = get_model_checkpoint(runconfig)
    if checkpoint_path:
        state_dict = load_from_checkpoint_file(checkpoint_path)
        model.load_state_dict(state_dict["model"], strict=True)
    else:
        raise RuntimeError(
            "Expected a checkpoint to load for inference but got none."
        )

    predictions_dir = artifact_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    dataloader = cstorch.utils.data.DataLoader(
        inference_input_dataloader, params
    )

    executor = cstorch.utils.data.DataExecutor(
        dataloader,
        num_steps=runconfig.get("inference_steps"),
    )

    @cstorch.trace
    def inference_step(batch):
        return compiled_model(batch, autoregressive=True)

    @cstorch.step_closure
    def post_inference_step(predictions):
        is_log_step = executor.on_final_iteration or (
            log_steps and executor.user_iteration % log_steps == 0
        )

        if is_log_step:
            rate = executor.profiler.rate_tracker.rate
            global_rate = executor.profiler.rate_tracker.global_rate

            logging.info(
                f"| Inference Device={backend.device}, "
                f"Step={executor.user_iteration}, "
                f"Rate={format_rate(rate)} samples/sec, "
                f"GlobalRate={format_rate(global_rate)} samples/sec"
            )

        # Save the predictions to a file
        np.savez(
            predictions_dir / f"prediction_{executor.user_iteration}.npz",
            predictions=tree_map(cstorch.to_numpy, predictions),
            global_step=executor.user_iteration,
        )

    try:
        for batch in executor:
            predictions = inference_step(batch)
            post_inference_step(predictions)
    finally:
        if not (compile_only or validate_only) and executor.profiler:
            # compute the total samples processed based on the number of steps
            # and the number of Cerebras systems in the cluster
            total_samples = int(executor.profiler.rate_tracker.total_samples)
            total_time = executor.profiler.rate_tracker.total_time

            logging.info(
                f"Processed {total_samples} sample(s) "
                f"in {total_time} seconds."
            )


if __name__ == '__main__':
    main()
