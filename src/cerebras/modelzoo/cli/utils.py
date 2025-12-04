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

import argparse
import os
from typing import Union

MZ_CLI_NAME = "cszoo"

# We don't want the cszoo assistant to try to use the cszoo assistant command
# as it would lead to an infinite loop. As a result, we have two separate
# variables:
#   1. EPILOG_WITHOUT_ASSISTANT which contains the epilog without information
#      about the cszoo assistant (used in the assistant's system message)
#   2. EPILOG which contains the full epilog (with info on cszoo assistant).
#      This is used in cszoo -h.
EPILOG_WITHOUT_ASSISTANT = (
    f"Use `{MZ_CLI_NAME} <cmd> -h` to learn how to use individual sub-commands. "
    f"See below for some basic examples.\n\n"
    f"List all models:\n"
    f"  $ {MZ_CLI_NAME} model list\n\n"
    f"Get additional information on gpt2:\n"
    f"  $ {MZ_CLI_NAME} model info gpt2\n\n"
    f"List all data processors:\n"
    f"  $ {MZ_CLI_NAME} data_processor list\n\n"
    f"Get additional information on GptHDF5DataProcessor:\n"
    f"  $ {MZ_CLI_NAME} data_processor info GptHDF5DataProcessor\n\n"
    f"Copy config file to local workdir:\n"
    f"  $ {MZ_CLI_NAME} config pull gpt2_tiny -o workdir\n\n"
    f"Validate config file:\n"
    f"  $ {MZ_CLI_NAME} config validate workdir/params_gpt_tiny.yaml\n\n"
    f"List all data preprocessing configurations:\n"
    f"  $ {MZ_CLI_NAME} data_preprocess list\n\n"
    f"Copy a data configuration file to local workdir:\n"
    f"  $ {MZ_CLI_NAME} data_preprocess pull summarization_preprocessing -o workdir\n\n"
    f"Run data preprocessing using given configuration:\n"
    f"  $ {MZ_CLI_NAME} data_preprocess run --config workdir/summarization_preprocessing.yaml\n\n"
    f"Train a gpt2 model:\n"
    f"  $ {MZ_CLI_NAME} fit workdir/params_gpt_tiny.yaml\n\n"
    f"Validate a gpt2 model:\n"
    f"  $ {MZ_CLI_NAME} validate workdir/params_gpt_tiny.yaml\n\n"
    f"Run Eleuther Eval Harness:\n"
    f"  $ {MZ_CLI_NAME} lm_eval workdir/params_gpt_tiny.yaml --tasks=winogrande --checkpoint_path=workdir/my_ckpt.mdl\n\n"
    f"Run BigCode Eval Harness:\n"
    f"  $ {MZ_CLI_NAME} bigcode_eval workdir/params_gpt_tiny.yaml --tasks=mbpp --checkpoint_path=workdir/my_ckpt.mdl\n\n"
    f"Convert a checkpoint to Huggingface format\n"
    f"  $ {MZ_CLI_NAME} checkpoint convert --model gpt2 --src-fmt "
    f"cs-auto --tgt-fmt hf --config workdir/params_gpt_tiny.yaml "
    f"model_dir/checkpoint.mdl\n\n"
)

EPILOG = (
    f"{EPILOG_WITHOUT_ASSISTANT}"
    f"Ask {MZ_CLI_NAME} LLM assistant a query in natural language:\n"
    f"  $ {MZ_CLI_NAME} assistant \"Is llama 3.1 supported in the "
    f"checkpoint converter?\""
)


def copy_config(cfg, additions: dict = None):
    from cerebras.modelzoo.common.utils.utils import merge_recursively
    from cerebras.modelzoo.trainer.validate import validate_trainer_params

    params = cfg.model_dump()
    if additions:
        params = merge_recursively(params, additions)
    if "trainer" not in params:
        params = {"trainer": params}
    return validate_trainer_params(params)


def get_table_parser():
    table_parser = argparse.ArgumentParser(description="The parent parser")
    table_parser.add_argument(
        "-P",
        "--no-pager",
        action="store_true",
        help="Disable pager and display output as plain text.",
    )
    table_parser.add_argument(
        "--json",
        action="store_true",
        help="Display information in JSON format.",
    )

    return table_parser


def _args_to_params(args, validate=True, extra_legacy_mapping_fn=None):
    from cerebras.modelzoo.common.pytorch_utils import RunConfigParamsValidator
    from cerebras.modelzoo.common.utils.run.cli_parser import (
        get_params,
        update_params_from_args,
    )
    from cerebras.modelzoo.trainer.utils import (
        inject_cli_args_to_trainer_params,
    )

    specified_args = set(
        filter(lambda a: a in args.seen_args, vars(args).keys())
    )

    params = get_params(
        args.params,
    )

    runconfig_params = params.setdefault("runconfig", {})
    update_params_from_args(args, specified_args, runconfig_params)
    runconfig_params.pop("config", None)
    runconfig_params.pop("params", None)

    if validate:
        runconfig_params = params["runconfig"]
        RunConfigParamsValidator().validate(runconfig_params)

    # Recursively update the params with the runconfig
    if "runconfig" in params and "trainer" in params:
        params = inject_cli_args_to_trainer_params(
            params.pop("runconfig"),
            params,
            extra_legacy_mapping_fn=extra_legacy_mapping_fn,
        )

    return params


def add_run_args(parser, devices=["CSX", "CPU", "GPU"]):
    from cerebras.modelzoo.common.utils.run.cli_parser import (
        patch_to_collect_specified_args,
    )

    parser.add_argument(
        "params",
        help="Path to .yaml file with model parameters.",
    )

    parser.add_argument(
        "--target_device",
        choices=devices,
        help=f"Target device to run on. Can be one of {', '.join(devices)}.",
    )

    parser.add_argument(
        "-o",
        "--model_dir",
        default=os.path.abspath("./model_dir"),
        help="Model directory where checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to initialize weights from.",
    )
    parser.add_argument(
        "--load_checkpoint_states",
        default="all",
        help=(
            "Comma-separated string of keys to explicitly specify the components "
            "whose state should be loaded if present in a checkpoint. If this flag is "
            "used, then all component states that exist in a checkpoint, but are not "
            "specified to load via the flag will be ignored. For example, for fine-tuning "
            "runs on a different dataset, setting `--load_checkpoint_states=\"model\" will only "
            "load the model state; any `optimizer` or `dataloader` state present in the "
            "checkpoint will not be loaded. By default, the config is `all`, i.e. "
            "everything present in the checkpoint is loaded."
        ),
    )
    parser.add_argument(
        "--logging",
        default="INFO",
        help="Specifies the default logging level. Defaults to INFO.",
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        help="Enables compile only workflow.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Enables validate only workflow"
        "validate_only stops the compilation at ws_km stage for weight streaming mode.",
    )
    parser.add_argument(
        "--job_labels",
        nargs="+",
        default=list(),
        help="A list of equal-sign-separated key value pairs served as job labels.",
    )
    parser.add_argument(
        "--job_priority",
        choices=["p1", "p2", "p3"],
        default="p2",
        help="Priority of the job. When launching jobs, valid priority should be between "
        "p1 and p3, where p1 is highest priority.",
    )
    parser.add_argument(
        "--mount_dirs",
        nargs="+",
        default=list(),
        help="A list of paths to be mounted to the appliance containers. "
        "It should generally contain path to the directory containing the "
        "Cerebras modelzoo.",
    )
    parser.add_argument(
        "--python_paths",
        nargs="+",
        default=list(),
        help="A list of paths to be exported into PYTHONPATH for worker containers. "
        "It should generally contain path to the directory containing the "
        "Cerebras modelzoo, as well as any external python packages needed.",
    )
    parser.add_argument(
        "--credentials_path",
        help="Credentials for cluster access. Defaults to None. If None, the value from "
        "a pre-configured location will be used if available.",
    )
    parser.add_argument(
        "--mgmt_address",
        help="<host>:<port> for cluster management. If None, the value from "
        "a pre-configured location will be used if available. Defaults to None.",
    )
    parser.add_argument(
        "--disable_version_check",
        action="store_true",
        help="Disable version check for local experimentation and debugging",
    )
    parser.add_argument(
        "--num_csx",
        default=1,
        type=int,
        help="Number of CS nodes. Defaults to 1",
    )

    def parse_value(value: str) -> Union[bool, int, float, str]:
        """
        Parses an value from the commandline into its most restricted primitive
        type.

        Args:
            value: The string from the commandline

        Returns:
            The parsed primitive, or the original string.

        """
        # Try bool, int, float, string in that order.
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value

    class ParseKV(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            dest = getattr(namespace, self.dest, None) or {}
            for value in values:
                key, value = value.split('=', 1)
                dest[key] = parse_value(value)
            setattr(namespace, self.dest, dest)

    parser.add_argument(
        "--debug_args",
        nargs="*",
        action=ParseKV,
        help="DebugArgs to pass to the Cerebras compile and execution, "
        "pass as --debug_args sub.object.key=value, where value can be bool "
        "int, float or str",
    )

    parser.add_argument(
        "--debug_args_path",
        help="Path to debugs args file. Defaults to None.",
    )

    parser.add_argument(
        "--ini",
        nargs="*",
        action=ParseKV,
        help="Debug INI settings to pass to the Cerebras compile and "
        "execution, pass as --ini key=value, where value can be bool, int, "
        "float or str",
    )
    parser.add_argument(
        "--mgmt_namespace",
        help=argparse.SUPPRESS,
    )
    return patch_to_collect_specified_args(parser)


def is_dir(path: str):
    # NOTE: Specifically use os.path here instead of pathlib.Path to be able
    # to support S3 paths
    return os.path.basename(path) == "" or os.path.isdir(path)


def append_source_basename(source_path: str, dest_path: str) -> str:
    """
    Append the source path's basename if the dest path is a
    directory
    """
    if is_dir(dest_path):
        # NOTE: Specifically use os.path here instead of pathlib.Path to be able
        # to support S3 paths
        return os.path.join(dest_path, os.path.basename(source_path))
    return dest_path
