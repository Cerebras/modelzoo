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

""" CLI Utilities."""

import argparse
import collections
import logging
import os
import sys
from typing import Callable, List, Optional, Set, Union

import yaml

from cerebras.modelzoo.common.utils.run.utils import DeviceType
from cerebras.modelzoo.common.utils.utils import UniqueKeyLoader


def read_params_file(params_file: str) -> dict:
    """Helper for loading params file."""
    with open(params_file, 'r') as stream:
        params = yaml.load(stream, Loader=UniqueKeyLoader)
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
    parser = get_parser(extra_args_parser_fn=extra_args_parser_fn)

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


def add_general_arguments(
    parser: argparse.ArgumentParser,
    modes: List[str] = ["train", "eval", "train_and_eval", "eval_all"],
):
    """Injects general parser arguments.

    Args:
        parser: Parser into which the arguments are being added.
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
        required=True,
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
        default=os.path.abspath("./model_dir"),
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
        help=(
            "Disabled strict loading of the model state from the checkpoint."
        ),
    )
    optional_arguments.add_argument(
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
    optional_arguments.add_argument(
        "--logging",
        default="INFO",
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
):
    """Injects Cerebras System specific parser arguments.

    Args:
        parser: Parser into which the arguments are being added.
    """
    optional_arguments = parser.add_argument_group(
        "Optional Arguments, CSX Specific"
    )
    group = optional_arguments.add_mutually_exclusive_group()
    group.add_argument(
        "--compile_only",
        action="store_true",
        help="Enables compile only workflow.",
    )
    group.add_argument(
        "--validate_only",
        action="store_true",
        help="Enables validate only workflow"
        "validate_only stops the compilation at ws_km stage for weight streaming mode.",
    )
    optional_arguments.add_argument(
        "--num_workers_per_csx",
        default=1,
        type=int,
        help="Number of workers to use for streaming inputs per CS node. If "
        "0, a default value based on the model will be chosen. Defaults "
        "to 1.",
    )
    optional_arguments.add_argument(
        "-c",
        "--compile_dir",
        help="Remote compile directory where compile artifacts will be written."
        " This path is appended to a base root directory common for all"
        " compiles on the Wafer-Scale Cluster and is written in the remote"
        " filesystem. Defaults to None.",
    )
    optional_arguments.add_argument(
        "--job_labels",
        nargs="+",
        default=list(),
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
        help="Path to debugs args file. Defaults to None.",
    )
    optional_arguments.add_argument(
        "--mount_dirs",
        nargs="+",
        default=list(),
        help="A list of paths to be mounted to the appliance containers. "
        "It should generally contain path to the directory containing the "
        "Cerebras modelzoo.",
    )
    optional_arguments.add_argument(
        "--python_paths",
        nargs="+",
        default=list(),
        help="A list of paths to be exported into PYTHONPATH for worker containers. "
        "It should generally contain path to the directory containing the "
        "Cerebras modelzoo, as well as any external python packages needed.",
    )
    optional_arguments.add_argument(
        "--credentials_path",
        help="Credentials for cluster access. Defaults to None. If None, the value from "
        "a pre-configured location will be used if available.",
    )
    optional_arguments.add_argument(
        "--mgmt_address",
        help="<host>:<port> for cluster management. If None, the value from "
        "a pre-configured location will be used if available. Defaults to None.",
    )
    optional_arguments.add_argument(
        "--cbcore_image",
        default=None,
        help="Image to use for the appliance job. If None, a default image "
        "will be used if available. Defaults to None.",
    )
    optional_arguments.add_argument(
        "--job_time_sec",
        type=int,
        help="time limit in seconds for the appliance jobs. When the time limit "
        "is hit, the appliance jobs will be cancelled and the run will be terminated",
    )
    optional_arguments.add_argument(
        "--disable_version_check",
        action="store_true",
        help="Disable version check for local experimentation and debugging",
    )
    optional_arguments.add_argument(
        "--num_csx",
        default=1,
        type=int,
        help="Number of CSX nodes to use. Defaults to 1",
    )
    optional_arguments.add_argument(
        "--num_wgt_servers",
        default=0,
        type=int,
        help="Maximum number of weight servers to use in weight streaming "
        "execution strategy. If not specified, an appropriate default is chosen "
        "based on cluster architecture.",
    )
    optional_arguments.add_argument(
        "--num_act_servers",
        default=0,
        type=int,
        help="Maximum number of activation servers to use per device. "
        "If not specified, an appropriate default is chosen "
        "based on cluster architecture.",
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
        help=argparse.SUPPRESS,
    )
    return


def add_gpu_arguments(gpu_parser: argparse.ArgumentParser):
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
        default="localhost:8888",
        help="To init master_addr and master_port of distributed. Defaults to localhost:8888.",
    )
    optional_arguments.add_argument(
        "-dist_backend",
        "--dist_backend",
        choices=["nccl", "mpi", "gloo"],
        default="nccl",
        help="Distributed backend engine. Defaults to nccl.",
    )
    optional_arguments.add_argument(
        "-init_method",
        "--init_method",
        default="env://",
        help="URL specifying how to initialize the process group. Defaults to env://",
    )

    return


def get_parser(
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
    parents = []
    extra_args = {}
    if extra_args_parser_fn:
        extra_args = extra_args_parser_fn()

        if isinstance(extra_args, argparse.ArgumentParser):
            # pylint: disable=protected-access
            extra_args._action_groups[1].title = (
                "User-Defined and/or Model Specific Arguments"
            )
            parents.append(extra_args)
        if not isinstance(extra_args, dict):
            # Rename the action groups of each parser passed into this parser generator
            if isinstance(extra_args, list):
                for item in extra_args:
                    # pylint: disable=protected-access
                    item._action_groups[1].title = (
                        "User-Defined and/or Model Specific Arguments"
                    )
            extra_args = {DeviceType.ANY: extra_args}

        parents.extend(extra_args.get(DeviceType.ANY, []))

    parent = argparse.ArgumentParser(parents=parents, add_help=False)
    add_general_arguments(parent, modes)

    parser = argparse.ArgumentParser(
        epilog=(
            (
                "Please run 'python run.py {CPU,GPU,CSX} -h'. \n \n"
                "Here are some example commands for running on different devices: \n \n"
                "    python run.py CSX --params /path/to/params --mode train --num_csx 1 \n \n"
                "    python run.py CPU --params /path/to/params --mode eval --checkpoint_path /path/to/checkpoint \n \n"
            )
            if parser_epilog is None
            else parser_epilog
        ),
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
        add_gpu_arguments(gpu_parser)

    if device_type == None or device_type == DeviceType.CSX:
        csx_parser = sp.add_parser(
            DeviceType.CSX,
            help="Run on Cerebras System",
            parents=[parent] + extra_args.get(DeviceType.CSX, []),
            epilog=(
                (
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
                else csx_parser_epilog
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
    add_csx_arguments(csx_parser)

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
        A dict containing the params, with the defaults updated.
    """
    for k, v in default_params.items():
        if isinstance(v, collections.abc.Mapping):
            params[k] = update_defaults(params.get(k, {}), v)
        elif k not in params:
            params[k] = v
    return params


def update_params_from_file(params, params_file):
    """Update provided params from provided file."""
    if os.path.exists(params_file):
        default_params = read_params_file(params_file)
        update_defaults(params, default_params)


def update_params_from_args(args: dict, specified_args: Set[str], params: dict):
    """Update params in-place with the arguments from args.

    Args:
        args: The namespace containing args from the commandline.
        params: The params to be updated.
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
        for k, v in vars(args).items():
            if k in ["config", "params"]:
                continue
            if k not in specified_args:
                # If a CLI arg was not specified but it's in the params
                # already, don't overwrite it.
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
                    append_args.extend([v] if isinstance(v, str) else v)
                if params.get(k) is not None:
                    append_args.extend(
                        [params[k]] if isinstance(params[k], str) else params[k]
                    )
                if append_args:
                    params[k] = append_args
                continue

            params[k] = v if v is not None else params.get(k)

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

    if "model_dir" in params:
        os.makedirs(params["model_dir"], exist_ok=True)


def patch_to_collect_specified_args(
    parser: argparse.ArgumentParser,
) -> Set[str]:
    """Patch all actions in the parser to record specified CLI arguments.

    argparse doesn't provide a mechanism to know which CLI args were specified
    on cmdline and which come from default values. This method patches the
    parser actions to record what was specified in CLI and adds it to `args_seen`
    that is returned from this method.
    """
    args_seen = set()
    actions_seen = set()

    def patch_call(instance):
        class _(type(instance)):
            def __call__(self, *args, **kwargs):
                args_seen.add(self.dest)
                return super().__call__(*args, **kwargs)

        instance.__class__ = _

    def override_actions(parser: argparse.ArgumentParser):
        for action in parser._actions:
            if action in actions_seen:
                continue
            actions_seen.add(action)

            if isinstance(action, argparse._SubParsersAction):
                for a in action.choices.values():
                    override_actions(a)

            call_method = getattr(action, "__call__", None)
            if call_method is None:
                continue

            patch_call(action)

        for group in parser._action_groups:
            if group in actions_seen:
                continue
            actions_seen.add(group)
            override_actions(group)

        for group in parser._mutually_exclusive_groups:
            if group in actions_seen:
                continue
            actions_seen.add(group)
            override_actions(group)

    override_actions(parser)

    return args_seen


def get_params_from_args(
    argv: Optional[List] = None,
    extra_args_parser_fn: Optional[
        Callable[[], List[argparse.ArgumentParser]]
    ] = None,
    device_type: Optional[DeviceType] = None,
    **parser_args,
) -> dict:
    """
    Parse the arguments and get the params dict from the resulting args.

    Args:
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
        extra_args_parser_fn=extra_args_parser_fn,
        device_type=device_type,
        **parser_args,
    )
    seen_args = patch_to_collect_specified_args(parser)
    args = parser.parse_args(argv if argv else sys.argv[1:])
    specified_args = set(filter(lambda a: a in seen_args, vars(args).keys()))

    params = get_params(
        args.params,
    )

    runconfig_params = params.setdefault("runconfig", {})
    update_params_from_args(args, specified_args, runconfig_params)
    runconfig_params.pop("config", None)
    runconfig_params.pop("params", None)

    return params
