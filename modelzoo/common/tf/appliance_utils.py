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
import os

from cerebras_appliance.pb.client.common_config_pb2 import DebugArgs
from modelzoo.common.tf.run_utils import (
    get_params,
    save_params,
    update_params_from_args,
)


def get_debug_args(debug_ini_path):
    """Appliance mode debug.ini"""
    debug_ini_fp = os.path.join(debug_ini_path, "debug.ini")
    debug_args = None
    if os.path.exists(debug_ini_fp):
        with open(debug_ini_fp, "r") as fp:
            ini_content = fp.read()
            debug_ini = DebugArgs.DebugINI(frozen_content=ini_content)
            debug_args = DebugArgs(debug_ini=debug_ini)
    return debug_args


def create_arg_parser(run_dir):
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
        required=True,
        help="credentials for cluster access",
    )
    parser.add_argument(
        "--mgmt_address",
        default="cluster-server.cerebras.com:443",
        help="<host>:<port> for cluster management",
    )
    parser.add_argument(
        "--mode", required=True, choices=["train", "eval"], help="train or eval"
    )
    parser.add_argument(
        "--params", required=True, help="path to model parameters YAML file",
    )
    parser.add_argument(
        "--num_csx", default=1, type=int, help="number of CS nodes",
    )
    parser.add_argument(
        "--steps", default=None, type=int, help="number of steps to train",
    )
    parser.add_argument(
        "--max_steps",
        default=None,
        type=int,
        help="number of total steps to train",
    )
    parser.add_argument(
        "--num_wgt_servers", type=int, help="number of WGT server",
    )
    group.add_argument(
        "--compile_only",
        action="store_true",
        help="compile model completely, generating compiled executables",
    )
    group.add_argument(
        "--validate_only",
        action='store_true',
        help="compile model up to kernel matching",
    )
    parser.add_argument(
        "--model_dir",
        default=default_model_dir,
        help="model directory where checkpoints will be written model_dir. "
        + "If directory exists, weights are loaded from the checkpoint file.",
    )
    parser.add_argument(
        "--mount_dirs",
        nargs="+",
        help="a list of paths to be mounted to the appliance containers",
    )
    parser.add_argument(
        "--python_paths",
        nargs="+",
        help="a list of paths to be exported into PYTHONPATH for worker containers",
    )
    parser.add_argument("--compile_dir", default=None, help="compile directory")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="checkpoint to initialize weights from.",
    )

    # The following arguments are deprecated.
    # We should use "--mount_dirs" instead.
    parser.add_argument(
        "--data_mount_path",
        required=True,
        help="volume mount path to training data",
    )
    # We should use "--mount_dirs" and "--python_paths" instead.
    parser.add_argument(
        "--modelzoo_mount_path",
        required=True,
        help="volume mount path containing modelzoo module",
    )
    return parser


def parse_args_and_params(
    run_dir, set_default_params=None,
):
    """Parses commandline arguments and returns the params.

    Args:
        run_dir: The root directory where to create the model_dir in.
        set_default_params: A callable that updates params with some defaults
            specific to this model. Defaults to None.
    """
    parser = create_arg_parser(run_dir)
    args = parser.parse_args()

    params = get_params(args.params)
    if set_default_params:
        set_default_params(params)

    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    save_params(params, model_dir=runconfig_params["model_dir"])

    return params
