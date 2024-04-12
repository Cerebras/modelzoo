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

""" CLI Utilities"""
import argparse
import collections
import logging
import os
import sys
from typing import Callable, List, Optional, Union

import yaml

from cerebras.modelzoo.common.utils.run.utils import DeviceType
from cerebras.modelzoo.config_manager.config_loader import (
    validate_config_params,
)


def read_params_file(params_file: str) -> dict:
    """Helper for loading params file."""
    with open(params_file, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def get_params(
    params_file: str,
) -> dict:
    """Reads params from file and returns them as a dict.

    Args:
        params_file: The YAML file holding the params.
        config: Optional config to load from the params. If None, the default
            config is returned. Defaults to None.
    Returns:
        A dict containing the params.
    """
    params = read_params_file(params_file)

    return params


def get_all_args(
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """Helper for returning all valid params for each device."""
    parser = get_parser(
        first_parse=False, extra_args_parser_fn=extra_args_parser_fn
    )

    # Only the top level parser is exposed when get_parser() is called.
    # All subparsers will inherit the extra_arg_parser, so we can use the CPU
    # parser to get the required args to grab the required args and make
    # un-required.
    # parser._action_groups[0]._actions[1].choices["CPU"] accesses the CPU subparser
    # doing (CPU subparser)._action_groups[0]._actions gets a list of all args
    for arg in (
        parser._action_groups[0]
        ._actions[1]
        .choices["CPU"]
        ._action_groups[0]
        ._actions
    ):
        if arg.required:
            arg.required = False

    cpu_args = parser.parse_args([DeviceType.CPU])
    gpu_args = parser.parse_args([DeviceType.GPU])
    csx_args = parser.parse_args([DeviceType.CSX])

    return cpu_args, gpu_args, csx_args


def discard_params(
    device: str,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
):
    """External utility for determining
    invalid parameters for the current device type."""

    cpu_args, gpu_args, csx_args = get_all_args(extra_args_parser_fn)

    if device == DeviceType.CPU:
        curr_device_args = vars(cpu_args).keys()
    elif device == DeviceType.GPU:
        curr_device_args = vars(gpu_args).keys()
    elif device == DeviceType.CSX:
        curr_device_args = vars(csx_args).keys()
    else:
        raise ValueError(f"Invalid entry for device {device}.")

    all_params = (
        vars(cpu_args).keys() | vars(gpu_args).keys() | vars(csx_args).keys()
    )
    discard_params_list = set(all_params) - (set(curr_device_args))

    return discard_params_list


def assemble_disallowlist(
    params: dict,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
) -> list:
    """Determine invalid parameters for the current device type."""
    # cpu parser does not currently contain any additional information
    # or parameters but will parse through its elements as well just
    # in case that changes in the future.
    cpu_args, gpu_args, csx_args = get_all_args(extra_args_parser_fn)

    all_params_template = {
        **vars(cpu_args),
        **vars(gpu_args),
        **vars(csx_args),
    }
    unacceptable_params = set(all_params_template.keys()).difference(
        set(params.keys())
    )

    return all_params_template, list(unacceptable_params)


def add_general_arguments(
    parser: argparse.ArgumentParser,
    default_model_dir: str,
    first_parse: bool = True,
    modes: List[str] = ["train", "eval", "train_and_eval", "eval_all"],
):
    """Injects general parser arguments.

    Args:
        parser: Parser into which the arguments are being added.
        default_model_dir: String containing model directory path.
        first_parse: Boolean indicating whether this is the first
          time processing the arguments. If True, the "params" arg
          is required to get additional parameters, if False then
          the params file has already been read and is not required.
        modes: Optional list of valid modes to be passed under the `--mode`
            argument of the parser. We default to choices
            ["train", "eval", "train_and_eval", "eval_all"].
            If an empty list if provided, then `--mode` isn't added as a
            parser arg.
    """
    required_arguments = parser.add_argument_group("Required Arguments")
    required_arguments.add_argument(
        "-p",
        "--params",
        required=first_parse,
        help="Path to .yaml file with model parameters",
    )
    if len(modes) > 1:
        required_arguments.add_argument(
            "-m",
            "--mode",
            required=True,
            choices=modes,
            help=(
                "Select mode of execution for the run. Can choose among "
                f"the following: {', '.join(modes)}"
            ),
        )
    optional_arguments = parser.add_argument_group(
        "Optional Arguments, All Devices"
    )
    if len(modes) == 1:
        optional_arguments.add_argument(
            "-m",
            "--mode",
            default=modes[0],
            choices=modes,
            help=(
                f"The mode of execution for the run. Defaults to {modes[0]}."
            ),
        )
    optional_arguments.add_argument(
        "-o",
        "--model_dir",
        default=default_model_dir,
        help="Model directory where checkpoints will be written.",
    )
    optional_arguments.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to initialize weights from.",
    )
    optional_arguments.add_argument(
        "--disable_strict_checkpoint_loading",
        action="store_true",
        default=None,
        help=(
            "Disabled strict loading of the model state from the checkpoint."
        ),
    )
    optional_arguments.add_argument(
        "--load_checkpoint_states",
        default=None,
        help=(
            "Comma-separated string of keys to explicitly specify the components "
            "whose state should be loaded if present in a checkpoint. If this flag is "
            "used, then all component states that exist in a checkpoint, but are not "
            "specified to load via the flag will be ignored. For example, for fine-tuning "
            "runs on a different dataset, setting `--load_checkpoint_states=\"model\" will only "
            "load the model state; any `optimizer` or `dataloader` state present in the "
            "checkpoint will not be loaded. By default, the config is `None`, i.e. "
            "everything present in the checkpoint is loaded."
        ),
    )
    optional_arguments.add_argument(
        "--logging",
        default=None,
        help="Specifies the default logging level. Defaults to INFO.",
    )

    class ParseToDict(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if getattr(namespace, self.dest, None) is None:
                setattr(namespace, self.dest, dict())
            for kv_pair in values:
                tokens = kv_pair.strip().split("=")
                if len(tokens) == 1:
                    key = ''
                    value = tokens[0]
                elif len(tokens) == 2:
                    key = tokens[0]
                    value = tokens[1]
                else:
                    raise ValueError(
                        f"'{kv_pair}' is an invalid label. Expecting a single value or the label key and "
                        f"the label value to be separated by a single equal sign(=) character."
                    )
                getattr(namespace, self.dest)[key] = value

    optional_arguments.add_argument(
        "--wsc_log_level",
        action=ParseToDict,
        nargs="+",
        help=(
            "Specifies the log level for particular Wafer-Scale Cluster servers."
            "A list of equal-sign-separated key value pairs (or just a value for global setting) of a "
            "server task and a log level which can be either an integer or a string (e.g. INFO, DEBUG, 10)."
        ),
    )
    optional_arguments.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Specifies the maximum number of steps to run.",
    )
    optional_arguments.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Specifies the number of steps to run for eval.",
    )

    return


def add_csx_arguments(
    parser: argparse.ArgumentParser,
    first_parse: bool = True,
):
    """Injects Cerebras System specific parser arguments.

    Args:
        parser: Parser into which the arguments are being added.
        first_parse: Boolean indicating whether this is the first
          time processing the arguments. If True, default args
          are ignored in case the other has them stored elsewhere.
          If False, default args are set.
    """
    optional_arguments = parser.add_argument_group(
        "Optional Arguments, CSX Specific"
    )
    group = optional_arguments.add_mutually_exclusive_group()
    group.add_argument(
        "--compile_only",
        action="store_true",
        default=None,
        help="Enables compile only workflow. Defaults to None.",
    )
    group.add_argument(
        "--validate_only",
        action="store_true",
        default=None,
        help="Enables validate only workflow"
        "validate_only stops the compilation at ws_km stage for weight streaming mode."
        "for pipeline mode, the compilation is stopped at the optimize_graph stage."
        "Defaults to None.",
    )
    optional_arguments.add_argument(
        "--num_workers_per_csx",
        default=None if first_parse else 0,
        type=int,
        help="Number of workers to use for streaming inputs per CS node. If "
        "0, a default value based on the model will be chosen. Defaults "
        "to 0.",
    )
    optional_arguments.add_argument(
        "-c",
        "--compile_dir",
        default=None,
        help="Remote compile directory where compile artifacts will be written."
        " This path is appended to a base root directory common for all"
        " compiles on the Wafer-Scale Cluster and is written in the remote"
        " filesystem. Defaults to None.",
    )
    optional_arguments.add_argument(
        "--job_labels",
        nargs="+",
        help="A list of equal-sign-separated key value pairs served as job labels.",
    )
    optional_arguments.add_argument(
        "--job_priority",
        choices=["p1", "p2", "p3"],
        default="p2",
        help="Priority of the job. When launching jobs, valid priority should be between "
        "p1 and p3, where p1 is highest priority.",
    )
    optional_arguments.add_argument(
        "--debug_args_path",
        default=None,
        help="Path to debugs args file. Defaults to None.",
    )
    optional_arguments.add_argument(
        "--mount_dirs",
        nargs="+",
        help="A list of paths to be mounted to the appliance containers. "
        "It should generally contain path to the directory containing the "
        "Cerebras modelzoo.",
    )
    optional_arguments.add_argument(
        "--python_paths",
        nargs="+",
        help="A list of paths to be exported into PYTHONPATH for worker containers. "
        "It should generally contain path to the directory containing the "
        "Cerebras modelzoo, as well as any external python packages needed.",
    )
    optional_arguments.add_argument(
        "--credentials_path",
        default=None,
        help="Credentials for cluster access. Defaults to None. If None, the value from "
        "a pre-configured location will be used if available.",
    )
    optional_arguments.add_argument(
        "--mgmt_address",
        default=None,
        help="<host>:<port> for cluster management. Defaults to None. If None, the value from "
        "a pre-configured location will be used if available.",
    )
    optional_arguments.add_argument(
        "--job_time_sec",
        type=int,
        default=None,
        help="time limit in seconds for the appliance jobs. When the time limit "
        "is hit, the appliance jobs will be cancelled and the run will be terminated",
    )
    optional_arguments.add_argument(
        "--disable_version_check",
        action="store_true",
        default=None,
        help="Disable version check for local experimentation and debugging",
    )
    optional_arguments.add_argument(
        "--num_csx",
        default=None if first_parse else 1,
        type=int,
        help="Number of CS nodes. Defaults to 1",
    )
    optional_arguments.add_argument(
        "--num_wgt_servers",
        default=None,
        type=int,
        help="Maximum number of weight servers to use in weight streaming "
        "execution strategy. Defaults to None.",
    )
    optional_arguments.add_argument(
        "--num_act_servers",
        default=None,
        type=int,
        help="Maximum number of activation servers to use per device. "
        "Defaults to None",
    )

    def parse_value(value: str) -> Union[bool, int, float, str]:
        """
        Parses an value from the commandline into its most restricted primitive
        type

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
            setattr(namespace, self.dest, dict())
            for value in values:
                key, value = value.split('=')
                getattr(namespace, self.dest)[key] = parse_value(value)

    optional_arguments.add_argument(
        "--debug_args",
        nargs="*",
        action=ParseKV,
        help="DebugArgs to pass to the Cerebras compile and execution, "
        "pass as --debug_args sub.object.key=value, where value can be bool "
        "int, float or str",
    )

    optional_arguments.add_argument(
        "--ini",
        nargs="*",
        action=ParseKV,
        help="Debug INI settings to pass to the Cerebras compile and "
        "execution, pass as --ini key=value, where value can be bool, int, "
        "float or str",
    )
    optional_arguments.add_argument(
        "--mgmt_namespace",
        default=None,
        help=argparse.SUPPRESS,
    )
    return


def add_gpu_arguments(
    gpu_parser: argparse.ArgumentParser, first_parse: bool = True
):
    """Injects GPU specific parser arguments.

    Args:
        gpu_parser: Parser into which the arguments are being added.
    """
    optional_arguments = gpu_parser.add_argument_group(
        "Optional Arguments, GPU Specific"
    )
    optional_arguments.add_argument(
        "-dist_addr",
        "--dist_addr",
        default=None if first_parse else "localhost:8888",
        help="To init master_addr and master_port of distributed. Defaults to localhost:8888.",
    )
    optional_arguments.add_argument(
        "-dist_backend",
        "--dist_backend",
        choices=["nccl", "mpi", "gloo"],
        default=None if first_parse else "nccl",
        help="Distributed backend engine. Defaults to nccl.",
    )
    optional_arguments.add_argument(
        "-init_method",
        "--init_method",
        default=None if first_parse else "env://",
        help="URL specifying how to initialize the process group. Defaults to env://",
    )

    return


def get_parser(
    run_dir: Optional[str] = None,
    first_parse: bool = True,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
    device_type: Optional[DeviceType] = None,
    parser_epilog: Optional[str] = None,
    csx_parser_epilog: Optional[str] = None,
    modes: List[str] = ["train", "eval", "train_and_eval", "eval_all"],
) -> argparse.ArgumentParser:
    """Returns an ArgumentParser for parsing commandline options.

    Args:
        run_dir: String to be used to determine model directory.
        first_parse: Boolean indicating whether this is the first
          time processing the arguments. If True, the parser is
          being used to collect commandline inputs. If False, it
          is only being used for verification on existing params.
        extra_args_parser_fn: Parent parser passed in by models with
          unique specific arguments.
        device_type: The device type for which to fetch to add the args
            to the new parser. If None, all device type (CPU, GPU, CSX)
            args are added.
        parser_epilog: Optional helpful text to add to the end of the
            main parser's help message
        csx_parser_epilog: Optional helpful text to add to the end of the
            CSX subparser's help message
        modes: Optional list of valid modes to be passed under the `--mode`
            argument of the parser. We default to choices
            ["train", "eval", "train_and_eval", "eval_all"].
            If an empty list if provided, then `--mode` isn't added as a
            parser arg.

    Returns:
        A parser instance.
    """
    default_model_dir = None

    if first_parse:
        # Set default model dir to be inside same directory
        # as the top level run.py
        if run_dir:
            default_model_dir = os.path.join(run_dir, "model_dir")

        if not default_model_dir:
            raise ValueError("Could not get default model directory")

    parents = []
    extra_args = {}
    if extra_args_parser_fn:
        extra_args = extra_args_parser_fn()

        if isinstance(extra_args, argparse.ArgumentParser):
            # pylint: disable=protected-access
            extra_args._action_groups[
                1
            ].title = "User-Defined and/or Model Specific Arguments"
            parents.append(extra_args)
        if not isinstance(extra_args, dict):
            # Rename the action groups of each parser passed into this parser generator
            if isinstance(extra_args, list):
                for item in extra_args:
                    # pylint: disable=protected-access
                    item._action_groups[
                        1
                    ].title = "User-Defined and/or Model Specific Arguments"
            extra_args = {DeviceType.ANY: extra_args}

        parents.extend(extra_args.get(DeviceType.ANY, []))

    parent = argparse.ArgumentParser(parents=parents, add_help=False)
    add_general_arguments(parent, default_model_dir, first_parse, modes)

    parser = argparse.ArgumentParser(
        epilog=(
            "Please run 'python run.py {CPU,GPU,CSX} -h'. \n \n"
            "Here are some example commands for running on different devices: \n \n"
            "    python run.py CSX --params /path/to/params --mode train --num_csx 1 \n \n"
            "    python run.py CPU --params /path/to/params --mode eval --checkpoint_path /path/to/checkpoint \n \n"
        )
        if parser_epilog is None
        else parser_epilog,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sp = parser.add_subparsers(title="Target Device", dest="target_device")
    sp.required = True

    if device_type == None or device_type == DeviceType.CPU:
        sp.add_parser(
            DeviceType.CPU,
            parents=[parent] + extra_args.get(DeviceType.CPU, []),
            help="Run on CPU",
        )

    if device_type == None or device_type == DeviceType.GPU:
        gpu_parser = sp.add_parser(
            DeviceType.GPU,
            parents=[parent] + extra_args.get(DeviceType.GPU, []),
            help="Run on GPU",
        )
        add_gpu_arguments(gpu_parser, first_parse)

    if device_type == None or device_type == DeviceType.CSX:
        csx_parser = sp.add_parser(
            DeviceType.CSX,
            help="Run on Cerebras System",
            parents=[parent] + extra_args.get(DeviceType.CSX, []),
            epilog=(
                "To see a complete list of all available arguments for \n"
                "please run 'python run.py CSX -h'. \n\n"
                "Here is an example command for running with CSX: \n \n"
                "    python run.py CSX --params /path/to/params --mode train --num_csx 1 \n \n"
                "When running from the Cerebras Model Zoo, you generally specify --python_paths and \n"
                "--mount_dirs. This can be done here or in your params.yaml under the 'runconfig' section. \n"
                "Both should at least include a path \n"
                "to the directory in which the Cerebras Model Zoo resides. \n"
            )
            if csx_parser_epilog is None
            else csx_parser_epilog,
            formatter_class=argparse.RawTextHelpFormatter,
        )
    add_csx_arguments(csx_parser, first_parse)

    return parser


def update_defaults(params: dict, default_params: dict) -> dict:
    """Updates the params dict with global default for a key
    if a key is not present.
    Works on nested dictionaries and recursively updates defaults
    for nested dictionaries.
    All other types, apart from dict are considered as base type
    and aren't updated recursively.
    Args:
        params: dict holding the params.
        default_params: dict holding the default params.
    Returns:
        A dict containing the params, with the defaults updated
    """
    for k, v in default_params.items():
        if isinstance(v, collections.abc.Mapping):
            params[k] = update_defaults(params.get(k, {}), v)
        elif k not in params:
            params[k] = v
    return params


def update_params_from_file(params, params_file):
    """Update provided params from provided file"""
    if os.path.exists(params_file):
        default_params = read_params_file(params_file)
        update_defaults(params, default_params)


def update_params_from_args(
    args: argparse.Namespace, params: dict, sysadmin_params: dict
):
    """Update params in-place with the arguments from args.

    Args:
        args: The namespace containing args from the commandline.
        params: The params to be updated.
        first_parse: Indicates whether to keep in the params path.
    """
    # First checking to see if wsc_log_level was given as a string.
    # If so, convert to dict so that we can merge with the dict
    # generated by cli.
    wsc_log_level = params.get("wsc_log_level", None)
    if wsc_log_level:
        if isinstance(wsc_log_level, str):
            wsc_log_level = {'': wsc_log_level}
            params["wsc_log_level"] = wsc_log_level
        else:
            assert isinstance(
                wsc_log_level, dict
            ), "wsc_log_level must be a string or dict."
    if args:
        for k, v in list(vars(args).items()):
            if k in ["config", "params"]:
                continue
            elif k in ["debug_args", "ini", "wsc_log_level"]:
                # merge dict settings recursively
                if v:
                    if k in params:
                        params[k].update(v)
                    else:
                        params[k] = v
                    continue
            elif k in ["python_paths", "mount_dirs"]:
                append_args = []
                if v is not None:
                    logging.info(v)
                    append_args.extend([v] if isinstance(v, str) else v)
                if params.get(k) is not None:
                    logging.info(params[k])
                    append_args.extend(
                        [params[k]] if isinstance(params[k], str) else params[k]
                    )
                if sysadmin_params.get(k) is not None:
                    logging.info(sysadmin_params[k])
                    append_args.extend(
                        [sysadmin_params[k]]
                        if isinstance(sysadmin_params[k], str)
                        else sysadmin_params[k]
                    )
                if append_args:
                    params[k] = append_args
                continue

            params[k] = (
                v
                if v is not None
                else (params.get(k) or sysadmin_params.get(k))
            )

    mode = params.get("mode")

    # Nice to have warning for users to understand behavior for
    # --load_checkpoint_states and --checkpoint_path
    if mode == "train" and params.get("checkpoint_path"):
        if params.get("load_checkpoint_states"):
            checkpoint_keys_to_load = set(
                params["load_checkpoint_states"].split(",")
            )

            if (
                len(checkpoint_keys_to_load) == 1
                and "model" in checkpoint_keys_to_load
            ):  # Load model state only
                logging.info(
                    "A checkpoint path is provided, and `--load_checkpoint_states=\"model\"` is "
                    "set which is loading state for the model only. This will load the model "
                    "model weights from the checkpoint, reset the optimizer states and start "
                    "training from step 0."
                )
            else:
                states = (
                    "All"
                    if len(checkpoint_keys_to_load) == 0
                    else (", ".join(checkpoint_keys_to_load))
                )
                logging.info(
                    f"{states} states from the checkpoint will be loaded (if present). "
                    "To only include a subset of components to load, specify "
                    "`--load_checkpoint_states` with the components to include."
                )

    model_dir = params["model_dir"]
    os.makedirs(model_dir, exist_ok=True)


def update_params_from_args_and_env(args: argparse.Namespace, params: dict):
    """Update params in-place with the arguments from args and the cerebras
    sysadmin overrides..

    Args:
        args: The namespace containing args from the commandline.
        params: The params to be updated.
    """
    sysadmin_file = os.getenv('CEREBRAS_WAFER_SCALE_CLUSTER_DEFAULTS')
    sysadmin_params = (
        get_params(sysadmin_file) if sysadmin_file is not None else {}
    )
    return update_params_from_args(args, params, sysadmin_params)


def post_process_params(
    params: dict, valid_arguments: list, invalid_arguments: list
) -> list:
    """Removes arguments that are not used by this target device."""
    target_device = params["runconfig"].pop("target_device", None)
    assert target_device is not None

    new_command, invalid_params = [], []
    new_command.append(target_device)

    for k, v in params["runconfig"].copy().items():
        if v is None or v is False:
            continue

        if k in ["debug_args", "ini", "wsc_log_level"]:
            # Undo the ArgParse action on dicts...
            v = [f"{ik}={iv}" for ik, iv in v.items()]

        # Ignore arguments from params.yaml that apply to different devices
        if k in invalid_arguments:
            invalid_params.append(k)
            params["runconfig"].pop(k)
        # Construct new parser input ignoring extra args that the parser
        # does not handle, such as num_epochs
        elif k in valid_arguments:
            new_command.append(f"--{k}")
            if isinstance(v, list):
                new_command.extend(map(str, v))
            elif not isinstance(v, bool):
                new_command.append(f"{v}")

    if invalid_params:
        logging.info(
            f"User specified a {target_device} run, but the following "
            f"non-{target_device} configurations were found in params file: "
            f"{str(invalid_params)}. Ignoring these arguments and continuing."
        )

    return new_command


def get_params_from_args(
    run_dir: Optional[str] = None,
    argv: Optional[List] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
    device_type: Optional[DeviceType] = None,
    **parser_args,
) -> dict:
    """
    Parse the arguments and get the params dict from the resulting args

    Args:
        run_dir: The path to the `run.py` file
        argv: The args to be parse. Defaults to sys.argv if not provided
        extra_args_parser_fn: An optional callable that adds any
            extra parser args in the parser.
        device_type: The device type for which to fetch to add the args
            to the new parser. If None, all device type (CPU, GPU, CSX)
            args are added.
        parser_args: Any extra keyword arguments to be passed to the `get_parser`
            method for constructing the parser
    """
    parser = get_parser(
        run_dir,
        extra_args_parser_fn=extra_args_parser_fn,
        device_type=device_type,
        **parser_args,
    )

    args = parser.parse_args(argv if argv else sys.argv[1:])

    params_template, invalid_params = assemble_disallowlist(
        vars(args), extra_args_parser_fn=extra_args_parser_fn
    )

    params = get_params(args.params)

    update_params_from_args_and_env(args, params["runconfig"])

    rerun_command = post_process_params(
        params, params_template.keys(), invalid_params
    )
    parser = get_parser(
        run_dir,
        first_parse=False,
        extra_args_parser_fn=extra_args_parser_fn,
        device_type=device_type,
        **parser_args,
    )
    try:
        logging.info(rerun_command)
        params_final = parser.parse_args(rerun_command)
    except SystemExit:
        logging.error(
            f"A mismatch was detected between your params.yaml file "
            f"and specified command-line arguments. "
            f"Please correct the error and run again."
        )
        raise

    params["runconfig"] = {**params["runconfig"], **vars(params_final)}
    params["runconfig"] = {**params_template, **params["runconfig"]}
    params["runconfig"].pop("config", None)
    params["runconfig"].pop("params", None)
    # Validate config using config class
    model_key = os.path.basename(run_dir)

    # If we got additional parser, there may be new params added to runconfig which are not supported by config class.
    # We remove them before passing for validation and then add them back
    extra_parser_param_keys = []
    if extra_args_parser_fn:
        parser_fn = extra_args_parser_fn()
        for parser_fn_instance in parser_fn:
            if parser_fn_instance and isinstance(
                parser_fn_instance, argparse.ArgumentParser
            ):
                extra_parser_param_keys.extend(
                    [
                        action.dest
                        for action in parser_fn_instance._actions
                        if not isinstance(action, argparse._HelpAction)
                    ]
                )
    runconfig_params = params["runconfig"]
    extra_parser_params = {}
    for key in extra_parser_param_keys:
        if key in runconfig_params:
            extra_parser_params[key] = runconfig_params.pop(key, None)
    curr_log_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.INFO)
    params = validate_config_params(params, model_key)
    logging.getLogger().setLevel(curr_log_level)
    runconfig_params.update(extra_parser_params)
    return params
