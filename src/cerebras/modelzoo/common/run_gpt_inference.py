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
import inspect
import logging
import os
import sys

# isort: off
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
# isort: on
from pathlib import Path
from warnings import warn

import numpy as np
import yaml

from cerebras.modelzoo.common.utils.utils import format_rate
from cerebras.modelzoo.config_manager.config_loader import (
    validate_config_params,
)

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


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


def get_cluster_config(params):
    from cerebras.modelzoo.common.run_cstorch_flow import (
        get_cluster_config as _get_cluster_config,
    )

    cs_config = _get_cluster_config(params)

    if cs_config.max_act_per_csx is not None and cs_config.max_act_per_csx > 1:
        warn("max_act_per_csx is forced to 1 for inference")

    cs_config.max_act_per_csx = 1

    if cs_config.num_workers_per_csx is None:
        cs_config.num_workers_per_csx = 1

    return cs_config


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
                "drop_last": False,
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


def main():
    from cerebras.modelzoo.common.utils.run.cli_parser import (
        get_params_from_args,
    )
    from cerebras.modelzoo.common.utils.run.utils import DeviceType

    # Parse args
    parent = inspect.getouterframes(inspect.currentframe())[1]
    run_dir = os.path.dirname(os.path.abspath(parent.filename))
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
        run_dir,
        argv=sys.argv[1:],
        extra_args_parser_fn=parser_fn,
        device_type=DeviceType.CSX,
        **parser_args,
    )

    from cerebras.modelzoo.common.pytorch_utils import (
        RunConfigParamsValidator,
        load_from_checkpoint_file,
        setup_artifact_dir,
        setup_logging,
    )

    # Set default model parameters
    from cerebras.modelzoo.models.nlp.gpt2.utils import set_defaults

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

    cs_config = get_cluster_config(params)

    from torch.utils._pytree import tree_map

    import cerebras.pytorch as cstorch
    from cerebras.modelzoo.common.half_dtype import set_half_dtype_from_params
    from cerebras.modelzoo.common.input_utils import (
        validate_streaming_and_micro_batch_size,
    )
    from cerebras.modelzoo.common.run_cstorch_flow import get_model_checkpoint
    from cerebras.modelzoo.models.nlp.gpt2.model import GptInferenceModel

    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)

    input_params = params.get("inference_input", {})
    micro_batch_size = input_params.get("micro_batch_size", "auto")
    if "batch_size" in input_params:
        # Checks for invalid setting of num_csx, micro_batch_size and batch_size
        validate_streaming_and_micro_batch_size(
            input_params["batch_size"],
            micro_batch_size,
            cs_config.num_csx,
        )

    # Initialize the backend
    backend = cstorch.backend(
        "CSX",
        artifact_dir=artifact_dir,
        compile_dir=runconfig.get("compile_dir"),
        compile_only=compile_only,
        validate_only=validate_only,
        retrace_every_iteration=runconfig.get("retrace_every_iteration", False),
    )
    backend.device.config.lazy_initialization = runconfig.get(
        "lazy_initialization", True
    )

    # Set the 16 bit dtype we want the automatic mixed precision module to use
    set_half_dtype_from_params(params["model"])

    def config_validation(params, model_key):
        # EEH-specific params added to the runconfig are not supported by our config class validation.
        # We remove EEH args from the runconfig, perform config validation and then re-add the args
        extra_parser_param_keys = []
        parser = parser_fn()[0]
        if parser:
            if parser and isinstance(parser, argparse.ArgumentParser):
                extra_parser_param_keys.extend(
                    [
                        action.dest
                        for action in parser._actions
                        if not isinstance(action, argparse._HelpAction)
                    ]
                )
        run_params = params["runconfig"]
        extra_parser_params = {}
        for eeh_arg in extra_parser_param_keys:
            if eeh_arg in run_params:
                extra_parser_params[eeh_arg] = run_params.pop(eeh_arg, None)
        # validate the params with config class
        validate_config_params(params, model_key)
        # Re-add extra inference args to the runconfig after config validation
        run_params.update(extra_parser_params)

    # Initialize model
    with backend.device:
        config_validation(params, "gpt_inference")
        model = GptInferenceModel(params)

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

    summary_dir = (
        runconfig["summary_dir"]
        if ("summary_dir" in runconfig and runconfig["summary_dir"] is not None)
        else artifact_dir / "inference"
    )

    writer = cstorch.utils.tensorboard.SummaryWriter(log_dir=summary_dir)

    executor = cstorch.utils.data.DataExecutor(
        dataloader,
        num_steps=runconfig.get("inference_steps"),
        cs_config=cs_config,
        writer=writer,
        micro_batch_size=micro_batch_size,
    )

    @cstorch.trace
    def inference_step(batch):
        return compiled_model(batch)

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
