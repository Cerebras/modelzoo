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
from copy import deepcopy
from pathlib import Path
from warnings import warn

import numpy as np
import torch

# isort: off
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
# isort: on


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
    from modelzoo.common.pytorch.run_cstorch_flow import (
        get_cluster_config as _get_cluster_config,
    )

    cs_config = _get_cluster_config(params)

    if cs_config.num_csx is not None and cs_config.num_csx != 1:
        raise ValueError(
            "Multi-box inference is not yet supported. "
            "Please specify num_csx=1 in the runconfig"
        )

    if cs_config.max_act_per_csx is not None and cs_config.max_act_per_csx > 1:
        warn("max_act_per_csx is forced to 1 for inference")

    cs_config.max_act_per_csx = 1

    if (
        cs_config.num_workers_per_csx is not None
        and cs_config.num_workers_per_csx > 1
    ):
        warn("num_workers_per_csx is forced to 1 for inference")

    cs_config.num_workers_per_csx = 1

    return cs_config


def inference_input_dataloader(params):
    from copy import deepcopy

    params = deepcopy(params)
    inference_input = params["inference_input"]
    data_processor = inference_input["data_processor"]
    if data_processor == "Gpt2SyntheticDataProcessor":
        from modelzoo.transformers.pytorch.gpt2.input.Gpt2SyntheticDataset import (  # noqa
            Gpt2SyntheticDataProcessor,
        )

        return Gpt2SyntheticDataProcessor(inference_input).create_dataloader()

    elif data_processor == "GptHDF5MapDataProcessor":
        from modelzoo.transformers.pytorch.gpt2.input.GptHDF5MapDataProcessor import (  # noqa
            GptHDF5MapDataProcessor,
        )
        from modelzoo.common.pytorch.input_utils import PaddingSample

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
    from modelzoo.common.pytorch.utils import (
        RunConfigParamsValidator,
        setup_artifact_dir,
        setup_logging,
    )
    from modelzoo.common.run_utils.cli_parser import get_params_from_args
    from modelzoo.common.run_utils.utils import DeviceType

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

    # Set default model parameters
    from modelzoo.transformers.pytorch.gpt2.utils import set_defaults

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
    )

    # Initialize the backend
    cs_config = get_cluster_config(params)

    from torch.utils._pytree import tree_map

    import cerebras_pytorch as cstorch
    from modelzoo.common.pytorch.half_dtype import set_half_dtype_from_params
    from modelzoo.common.pytorch.run_cstorch_flow import get_model_checkpoint
    from modelzoo.transformers.pytorch.gpt2.model import Gpt2Model

    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)

    backend = cstorch.backend(
        "CSX",
        artifact_dir=artifact_dir,
        compile_dir=runconfig.get("compile_dir"),
        compile_only=compile_only,
        validate_only=validate_only,
        use_cs_grad_accum=False,  # grad acc not supported in inference
    )
    backend.device.config.lazy_initialization = runconfig.get(
        "lazy_initialization", True
    )

    # Set the 16 bit dtype we want the automatic mixed precision module to use
    set_half_dtype_from_params(params["model"])

    class GptInferenceModel(Gpt2Model):
        def __init__(self, params):
            params = deepcopy(params)

            if "start_token" not in params["model"]:
                raise KeyError(
                    "Inference requires a start token. "
                    "Please provide `start_token` in the model params."
                )
            if "end_token" not in params["model"]:
                raise KeyError(
                    "Inference requires a end token. "
                    "Please provide `end_token` in the model params."
                )

            self.loop_dim = params["model"].pop("loop_dim", 1)
            self.start_token = params["model"].pop("start_token")
            self.end_token = params["model"].pop("end_token")
            self.max_tokens = params["model"].pop("max_tokens", None)

            super().__init__(params)

        def forward(self, data):
            """The forward pass on the input data. This method
            returns the predictions of the network as tokens.
            """
            if "input_ids" not in data:
                raise KeyError(
                    "GPT-2 model expects these data fields: input_ids"
                )
            elif data["input_ids"].dtype != torch.int32:
                raise TypeError(
                    "The dtype for all inputs should be torch.int32"
                )

            input_ids = data["input_ids"]
            # Note: attention_mask is a misnomer in this model; its contents are
            # ignored and only its shape is used.
            lm_logits = self.model(
                input_ids=input_ids,
                attention_mask=input_ids,  # doesn't actually mask anything
            )

            predictions = torch.argmax(lm_logits, dim=-1).int()
            cstorch.experimental.run_implicit_autoregressive_loop(
                input_tensor=input_ids,
                output_tensor=predictions,
                loop_dim=self.loop_dim,
                start_token=self.start_token,
                stop_token=self.end_token,
                max_tokens=self.max_tokens,
            )
            return predictions

    # Iniialize model
    with backend.device:
        model = GptInferenceModel(params)

    compiled_model = cstorch.compile(model, backend)
    compiled_model.eval()

    # Load weights
    checkpoint_path = get_model_checkpoint(runconfig)
    if checkpoint_path:
        logging.info(f"Loading weights from checkpoint {checkpoint_path}")
        state_dict = cstorch.load(checkpoint_path)
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

    writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=runconfig.get("summary_dir", artifact_dir / "inference")
    )

    executor = cstorch.utils.data.DataExecutor(
        dataloader,
        num_steps=runconfig.get("inference_steps"),
        cs_config=cs_config,
        writer=writer,
    )

    global_step = 0

    @cstorch.trace
    def inference_step(batch):
        return compiled_model(batch)

    @cstorch.step_closure
    def post_inference_step(predictions, step):
        is_log_step = executor.on_final_iteration or (
            log_steps and global_step % log_steps == 0
        )

        if is_log_step:
            rate = executor.profiler.rate_tracker.rate
            global_rate = executor.profiler.rate_tracker.global_rate

            logging.info(
                f"| Inference Device={backend.device}, "
                f"Step={step}, "
                f"Rate={rate:.2f} samples/sec, "
                f"GlobalRate={global_rate:.2f} samples/sec"
            )

        # Save the predictions to a file
        np.savez(
            predictions_dir / f"prediction_{global_step}.npz",
            predictions=tree_map(cstorch.to_numpy, predictions),
            global_step=global_step,
        )

    try:
        for step, batch in enumerate(executor, start=1):
            predictions = inference_step(batch)
            global_step += 1
            post_inference_step(predictions, step)
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
