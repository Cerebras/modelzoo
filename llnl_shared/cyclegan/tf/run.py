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

"""
Script to run CycleGAN model
"""
import argparse
import os
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from llnl_shared.cyclegan.tf.data import input_fn
from llnl_shared.cyclegan.tf.model import model_fn
from llnl_shared.cyclegan.tf.utils import get_params

from modelzoo.common.tf.estimator.cs_estimator import CerebrasEstimator
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.run_utils import get_csrunconfig_dict, save_params


def parse_args():
    parser = argparse.ArgumentParser(
        description="run predict or train on subgraph "
        + "example: python run.py -p params.yaml -s forward -m train"
    )
    parser.add_argument(
        "-p",
        "--params",
        help="path to params yaml file",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--subgraph_type",
        help="specifies which subgraph to run",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["validate_only", "compile_only", "train", "eval",],
        help=(
            "Can choose from validate_only, compile_only, train "
            + "or eval. Defaults to validate_only."
            + "  Validate only will only go up to kernel matching."
            + "  Compile only continues through and generate compiled"
            + "  executables."
            + "  Train will compile and train if on CS-1,"
            + "  and just train locally (CPU/GPU) if not on CS-1."
            + "  Eval will run eval locally."
        ),
        required=True,
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        type=str,
        help="Save compilation and non-simfab outputs",
        required=False,
    )
    args = vars(parser.parse_args())
    params = get_params(args["subgraph_type"], param_path=args["param_path"])
    params["runconfig"]["mode"] = args["mode"]
    params["input"]["mode"] = args["mode"]
    if args["model_dir"]:
        params["runconfig"]["model_dir"] = args["model_dir"]

    return params


def check_env(params):
    """
    Perform basic checks for parameters and env
    """
    if (
        params["runconfig"]["cs_ip"] is not None
        and params["mode"] != tf.estimator.ModeKeys.TRAIN
    ):
        tf.compat.v1.logging.warn("No need to specify CS-1 IP if not training")


def main(params):
    save_params(params, params["runconfig"]["model_dir"])
    use_cs1 = (
        params["runconfig"]["mode"] == tf.estimator.ModeKeys.TRAIN
        and params["runconfig"]["cs_ip"] is not None
    )
    params["runconfig"]["cs_ip"] = (
        params["runconfig"]["cs_ip"] + ":9000" if use_cs1 else None
    )
    check_env(params)
    runconfig_dict = get_csrunconfig_dict(params)
    config = CSRunConfig(**runconfig_dict,)
    est = CerebrasEstimator(
        model_fn,
        params=params,
        model_dir=params["runconfig"]["model_dir"],
        use_cs=use_cs1,
        config=config,
    )
    if params["runconfig"]["mode"] == tf.estimator.ModeKeys.TRAIN:
        est.train(input_fn=input_fn, max_steps=params["runconfig"]["max_steps"])
    elif params["runconfig"]["mode"] == tf.estimator.ModeKeys.EVAL:
        est.evaluate(input_fn=input_fn)
    else:
        est.compile(
            input_fn,
            validate_only=(params["runconfig"]["mode"] == "validate_only"),
        )


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    params = parse_args()
    main(params)
