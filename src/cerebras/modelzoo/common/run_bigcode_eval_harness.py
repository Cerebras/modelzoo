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

"""Single flow BigCode Eval Harness run script."""

import argparse
import fnmatch
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


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def add_bigcode_args(parser):
    optional_arguments = parser.add_argument_group(
        "Big Code Eval Harness Arguments"
    )
    optional_arguments.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'",
    )
    optional_arguments.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate.",
    )
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
    optional_arguments.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of completions to generate for each sample.",
    )
    optional_arguments.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used for evaluation.",
    )
    optional_arguments.add_argument(
        "--tasks",
        default=None,
        type=str,
        metavar="task1,task2",
        help=(
            "To get full list of tasks, go to the bigcode_eval repository: "
            "https://github.com/bigcode-project/bigcode-evaluation-harness"
        ),
    )
    optional_arguments.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    optional_arguments.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    optional_arguments.add_argument(
        "--limit_start",
        type=int,
        default=None,
        help="Optional offset to start from when limiting the number of samples",
    )
    optional_arguments.add_argument(
        "--save_every_k_tasks",
        type=int,
        default=None,
        help="Optional saving after every k tasks",
    )
    optional_arguments.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    optional_arguments.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    optional_arguments.add_argument(
        "--metric_output_path",
        type=str,
        default=None,
        help="Path to save the results",
    )
    optional_arguments.add_argument(
        "--load_generations_intermediate_paths",
        type=str,
        nargs="*",
        default=None,
        help="List of paths for saving the intermediate code generations",
    )
    optional_arguments.add_argument(
        "--save_generations_path",
        type=str,
        default=None,
        help="Path for saving the code generations",
    )
    optional_arguments.add_argument(
        "--save_references_path",
        type=str,
        default=None,
        help="Path for saving the references solutions/tests",
    )
    optional_arguments.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    optional_arguments.add_argument(
        "--check_references",
        action="store_true",
        default=None,
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
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


def bigcode_parser():
    parser = argparse.ArgumentParser(
        "Script for running Big Code Eval Harness for GPT style models",
        add_help=False,
    )

    add_bigcode_args(parser)

    return parser


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns."""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def run_bigcode_eval(params, parser):
    from cerebras.modelzoo.trainer.extensions.bigcode.bigcode_eval_harness import (
        BigCodeEvalHarness,
    )
    from cerebras.modelzoo.trainer.extensions.eleuther.eval_harness_utils import (
        SUPPORTED_MODELS,
    )
    from cerebras.modelzoo.trainer.utils import (
        configure_trainer_from_config,
        convert_legacy_params_to_trainer_params,
        inject_cli_args_to_trainer_params,
        is_legacy_params,
    )
    from cerebras.modelzoo.trainer.validate import validate_trainer_params

    warn(
        "Running eval harness using a standalone script is deprecated. Please switch to using the ModelZoo CLI. "
        "See https://training-docs.cerebras.ai/model-zoo/cli-overview for more details."
    )

    runconfig_params = params.setdefault("runconfig", {})

    bigcode_args = {}
    other_bigcode_args = {}
    for arg in parser._action_groups[0]._actions:
        arg_name = arg.dest
        # Exclude Cerebras-specific args
        if arg_name in {"keep_data_dir"}:
            other_bigcode_args[arg_name] = runconfig_params.pop(
                arg_name, arg.default
            )
        elif arg_name in runconfig_params:
            arg_val = runconfig_params.pop(arg_name, None)
            if arg_val is not None:  # Only consider specified CLI args
                bigcode_args[arg_name] = arg_val

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

        # Convert ScopedValidateFlags to ScopedBigCodeEvalHarnessFlags
        for callback in params["trainer"]["init"].get("callbacks", []):
            if "ScopedValidateFlags" in callback:
                callback["ScopedBigCodeEvalHarnessFlags"] = callback.pop(
                    "ScopedValidateFlags"
                )

        # Add BigCodeEvalHarness callback to the list of callbacks
        dataloader_args = (
            params["trainer"].get("validate_all", {}).pop("val_dataloaders", {})
        )
        dataloader_args["data_processor"] = "InferenceDataProcessor"
        params["trainer"]["init"]["callbacks"].append(
            {
                "BigCodeEvalHarness": {
                    "bigcode_args": bigcode_args,
                    **deepcopy(other_bigcode_args),
                    **deepcopy(dataloader_args),
                }
            }
        )

        # Remove fit/validate keys that are not used in the standalone flow
        for key in ("fit", "validate"):
            params["trainer"].pop(key, None)
    elif "runconfig" in params:
        params = inject_cli_args_to_trainer_params(
            params.pop("runconfig"), params
        )

    if "trainer" not in params:
        raise KeyError(
            "Trainer configuration not found in params. "
            "Please ensure that the params contain a 'trainer' key."
        )

    if isinstance(params["trainer"], (list, tuple)):
        raise ValueError(
            "Standalone BigCode evaluation harness script only supports "
            "a single trainer instance, but found a list of trainers."
        )

    # Inject CLI overrides to the BigCode callbacks
    for callback in params["trainer"]["init"].get("callbacks", []):
        if "BigCodeEvalHarness" in callback:
            callback["BigCodeEvalHarness"].setdefault(
                "bigcode_args", {}
            ).update(
                (key, value) for key, value in deepcopy(bigcode_args).items()
            )

    # Create a trainer config
    config = validate_trainer_params(params)[0]

    if config.init.model.config.name not in SUPPORTED_MODELS:
        raise ValueError(
            f"BigCode evaluation is not supported for model {config.init.model.config.name}. "
            f"Please choose a valid model name from: {SUPPORTED_MODELS}"
        )
    if config.validate_all is None:
        raise ValueError(
            "To run BigCode Eval Harness, `validate_all` section of the "
            "trainer config must be provided. It may be left as an empty "
            "dictionary or have a `ckpt_paths` key to load from one or "
            "more checkpoints."
        )

    # Construct the trainer object
    trainer = configure_trainer_from_config(config, mode="eval_all")

    # Check that only validation callbacks are of BigCode type
    has_bigcode_callback = False
    for callback in trainer.validation_callbacks:
        if isinstance(callback, BigCodeEvalHarness):
            has_bigcode_callback = True
        else:
            raise ValueError(
                f"Standalone BigCode Eval Harness script does not support "
                f"other evaluation harnesses, but found a callback of type "
                f"{type(callback)} that is a validation callback. Please "
                f"remove this callback from the list of callbacks or run "
                f"outside the standalone flow."
            )
    if not has_bigcode_callback:
        raise ValueError("No BigCodeEvalHarness callback found.")

    if config.validate_all.val_dataloaders:
        logging.warning(
            f"Found `validate_all.val_dataloaders` specified in the yaml, "
            f"but no upstream validation will be run for the standalone "
            f"BigCode Eval Harness script."
        )

    # Run validate_all which runs the eval harness callbacks
    trainer.validate_all(ckpt_paths=config.validate_all.ckpt_paths)


def main():
    parser_fn = lambda: [bigcode_parser()]
    parser_args = {
        "parser_epilog": (
            "Please run 'python run_bigcode_eval_harness.py CSX -h'. \n \n"
            "Here is an example command for running on CSX: \n \n"
            "    python run_bigcode_eval_harness.py CSX --params /path/to/params --checkpoint_path "
            "/path/to/checkpoint --tasks 'mbpp' \n \n"
            "Note that BigCode Eval Harness is currently only supported for device CSX"
        ),
        "csx_parser_epilog": (
            "To see a complete list of all available arguments, \n"
            "please run 'python run_bigcode_eval_harness.py CSX -h'. \n\n"
            "Here is an example command for running with CSX: \n \n"
            "    python run_bigcode_eval_harness.py CSX --params /path/to/params "
            "--checkpoint_path /path/to/checkpoint --tasks 'mbpp' "
        ),
        "modes": ["eval"],
    }

    params = get_params_from_args(
        argv=sys.argv[1:],
        extra_args_parser_fn=parser_fn,
        device_type=DeviceType.CSX,
        **parser_args,
    )

    run_bigcode_eval(params, parser_fn()[0])


if __name__ == "__main__":
    main()
