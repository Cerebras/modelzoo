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

"""Eval Harness run script."""

import argparse
import logging
import sys
from copy import deepcopy
from warnings import warn

# isort: off
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
# isort: on
from cerebras.modelzoo.common.utils.run.cli_parser import get_params_from_args
from cerebras.modelzoo.common.utils.run.utils import DeviceType
from cerebras.modelzoo.trainer.extensions.eleuther.eval_harness_utils import (
    SUPPORTED_MODELS,
)
from cerebras.modelzoo.trainer.utils import (
    configure_trainer_from_params,
    convert_legacy_params_to_trainer_params,
    is_legacy_params,
    merge_trainer_params,
)


def eeh_parser():
    parser = argparse.ArgumentParser(
        "Script for running Eleuther Eval Harness for GPT style models",
        add_help=False,
    )
    optional_arguments = parser.add_argument_group(
        "Eleuther Eval Harness Arguments"
    )
    # EEH-SPECIFIC ARGS
    # Ref: https://github.com/EleutherAI/lm-evaluation-harness/blob/c9bbec6e7de418b9082379da82797522eb173054/lm_eval/__main__.py#L26
    optional_arguments.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    optional_arguments.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    optional_arguments.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    optional_arguments.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    optional_arguments.add_argument(
        "--use_cache",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    optional_arguments.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    optional_arguments.add_argument(
        "--check_integrity",
        action="store_true",
        default=None,
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    optional_arguments.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=None,
        help="Prints the prompt for the first few documents.",
    )
    optional_arguments.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=None,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    optional_arguments.add_argument(
        "--show_config",
        action="store_true",
        default=None,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    optional_arguments.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    optional_arguments.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=None,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    optional_arguments.add_argument(
        "--seed",
        default=None,
        help=(
            "Set seed for python's random, numpy and torch.\n"
            "Accepts a comma-separated list of 3 values for python's random, numpy, and torch seeds, respectively, "
            "or a single integer to set the same seed for all three.\n"
            "The values are either an integer or 'None' to not set the seed. Default is `0,1234,1234` (for backward compatibility).\n"
            "E.g. `--seed 0,None,8` sets `random.seed(0)` and `torch.manual_seed(8)`. Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all three seeds to 42."
        ),
    )
    optional_arguments.add_argument(
        "--trust_remote_code",
        default=None,
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    # NONGREEDY SAMPLING ARGS FOR GENERATIVE TASKS
    optional_arguments.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature used for generation.",
    )
    optional_arguments.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p parameter used for nucleus sampling.",
    )
    optional_arguments.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k parameter used for generation.",
    )
    # CEREBRAS-SPECIFIC ARGS
    optional_arguments.add_argument(
        "--keep_data_dir",
        action="store_true",
        default=False,
        help=(
            "Specifies whether dumped data samples should be kept for reuse. "
            "Defaults to False, i.e. data samples are deleted after the run."
        ),
    )

    return parser


def run_eval_harness():
    """Main run script."""
    parser_fn = lambda: [eeh_parser()]
    parser_args = {
        "parser_epilog": (
            "Please run 'python run_eleuther_eval_harness.py CSX -h'. \n \n"
            "Here is an example command for running on CSX: \n \n"
            "    python run_eleuther_eval_harness.py CSX --params /path/to/params --checkpoint_path "
            "/path/to/checkpoint --tasks 'hellaswag,winogrande' --num_fewshot 0 \n \n"
            "Note that Eval Harness is currently only supported for device CSX"
        ),
        "csx_parser_epilog": (
            "To see a complete list of all available arguments, \n"
            "please run 'python run_eleuther_eval_harness.py CSX -h'. \n\n"
            "Here is an example command for running with CSX: \n \n"
            "    python run_eleuther_eval_harness.py CSX --params /path/to/params "
            "--checkpoint_path /path/to/checkpoint --tasks 'hellaswag,winogrande' --num_fewshot 0 "
            "\n \nEval Harness resides in the Cerebras Model Zoo. Please specify --python_paths and "
            "\n --mount_dirs here or in your params.yaml under the 'runconfig' section with \n"
            "the path to the directory in which the Cerebras Model Zoo resides. \n"
        ),
        "modes": ["eval"],
    }

    # Parse args
    params = get_params_from_args(
        argv=sys.argv[1:],
        extra_args_parser_fn=parser_fn,
        device_type=DeviceType.CSX,
        **parser_args,
    )
    runconfig_params = params["runconfig"]

    parser = parser_fn()[0]
    eeh_args = {}
    other_eeh_args = {}
    for arg in parser._action_groups[0]._actions:
        arg_name = arg.dest
        # Exclude Cerebras-specific args
        if arg_name in {"keep_data_dir"}:
            other_eeh_args[arg_name] = runconfig_params.pop(arg_name, None)
        elif arg_name in runconfig_params:
            arg_val = runconfig_params.pop(arg_name, None)
            if arg_val is not None:  # Only consider specified CLI args
                eeh_args[arg_name] = arg_val

    if is_legacy_params(params):
        warn(
            f"Detected that legacy params are being used. "
            f"Automatically converting params to new format."
        )
        params = convert_legacy_params_to_trainer_params(
            params,
            # Allow None objects inside the params
            obj_filter=lambda obj: obj is None,
        )

        # Convert ScopedValidateFlags to ScopedEleutherEvalHarnessFlags
        for callback in params["trainer"]["init"].get("callbacks", []):
            if "ScopedValidateFlags" in callback:
                callback["ScopedEleutherEvalHarnessFlags"] = callback.pop(
                    "ScopedValidateFlags"
                )

        # Add EleutherEvalHarness callback to the list of callbacks
        dataloader_args = (
            params["trainer"].get("validate_all", {}).pop("val_dataloaders", {})
        )
        dataloader_args["data_processor"] = "InferenceDataProcessor"
        params["trainer"]["init"]["callbacks"].append(
            {
                "EleutherEvalHarness": {
                    "eeh_args": eeh_args,
                    **deepcopy(other_eeh_args),
                    **deepcopy(dataloader_args),
                }
            }
        )

        # Remove fit/validate keys that are not used in the standalone flow
        for key in ("fit", "validate"):
            params["trainer"].pop(key, None)

    elif "runconfig" in params:
        converted = convert_legacy_params_to_trainer_params(
            {"runconfig": params.pop("runconfig")}
        )
        trainers = params["trainer"]
        if isinstance(trainers, (list, tuple)):
            params["trainer"] = [
                merge_trainer_params(trainer, converted) for trainer in trainers
            ]
        else:
            params = merge_trainer_params(params, converted)

    if "trainer" not in params:
        raise KeyError(
            "Trainer configuration not found in params. "
            "Please ensure that the params contain a 'trainer' key."
        )

    if isinstance(params["trainer"], (list, tuple)):
        raise ValueError(
            "Standalone Eleuther evaluation harness script only supports "
            "a single trainer instance, but found a list of trainers."
        )

    if "model_name" in params["trainer"]["init"]["model"]:
        model_name = params["trainer"]["init"]["model"].pop("model_name")
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Invalid model_name specified. Please choose a "
                f"valid model name from: {SUPPORTED_MODELS}"
            )
    else:
        raise RuntimeError(
            f"No model_name specified under config trainer.init.model. Please "
            f"choose a valid model name from: {SUPPORTED_MODELS}"
        )

    # Extract EleutherEvalHarness callbacks from the list of callbacks
    eeh_callbacks = [
        callback["EleutherEvalHarness"]
        for callback in params["trainer"]["init"].get("callbacks", [])
        if "EleutherEvalHarness" in callback
    ]
    if not eeh_callbacks:
        raise RuntimeError(f"Found no EleutherEvalHarness callback")

    for eeh_callback in eeh_callbacks:
        eeh_callback.setdefault("eeh_args", {}).update(
            (key, value) for key, value in deepcopy(eeh_args).items()
        )

    trainer = configure_trainer_from_params(params, model_name)

    if "val_dataloaders" in params["trainer"].get("validate_all") is not None:
        logging.warning(
            f"Found `validate_all.val_dataloaders` specified in the yaml, "
            f"but no upstream validation will be run for the standalone "
            f"Eleuther Eval Harness script."
        )
        params["trainer"]["validate_all"]["val_dataloaders"] = None

    trainer.validate_all(**params["trainer"].get("validate_all", {}))


if __name__ == "__main__":
    run_eval_harness()
