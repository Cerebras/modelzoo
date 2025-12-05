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
Script to run Cretin model
"""
import argparse
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from llnl_shared.cretin.tf.data import input_fn
from llnl_shared.cretin.tf.model import model_fn
from llnl_shared.cretin.tf.utils import get_params

from modelzoo.common.tf.estimator.cs_estimator import CerebrasEstimator
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.run_utils import (
    check_env,
    get_csrunconfig_dict,
    is_cs,
    save_params,
    update_params_from_args,
)


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="run model in either train, eval, compile_only, or \
                validate_only mode. example: python run.py -p params.yaml \
                -m train"
    )
    parser.add_argument(
        "--cs_ip", help="CS-1 IP address, defaults to None", default=None
    )
    parser.add_argument(
        "-p",
        "--params",
        help="path to params yaml file",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=[
            "validate_only",
            "compile_only",
            "train",
            "eval",
            "infer",
            "infer_cpp",
        ],
        help=(
            "Can choose from validate_only, compile_only, train, infer "
            + "or eval. Defaults to validate_only."
            + "  Validate only will only go up to kernel matching."
            + "  Compile only continues through and generate compiled"
            + "  executables."
            + "  Train will compile and train if on CS-1,"
            + "  and just train locally (CPU/GPU) if not on CS-1."
            + "  Eval will run eval locally."
            "  Predict will generate predictions and will skip loss calculation.\
               The number of generated inferences is given by \
               params['inference']['infer_params']['max_steps']"
        ),
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        type=str,
        help="Save compilation and non-simfab outputs",
        default="./model_dir",
    )
    return parser


def main():
    # SET UP
    parser = create_arg_parser()
    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params)
    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    # save params for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])
    # get runtime configurations
    use_cs = is_cs(runconfig_params)
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    check_env(runconfig_params)
    if params["runconfig"]["mode"] == "infer_cpp":
        # This import will result in global imports of modules that are built
        # and thus not accessible on a gpu run (will result in import error).
        # So moving the import to the context it is needed.
        from cerebras.tf.utils import prep_orchestrator

        prep_orchestrator()
    est_config = CSRunConfig(
        cs_ip=runconfig_params["cs_ip"], **csrunconfig_dict,
    )
    est = CerebrasEstimator(
        model_fn=model_fn,
        model_dir=runconfig_params["model_dir"],
        config=est_config,
        params=params,
    )
    output = None

    if params["runconfig"]["mode"] == tf.estimator.ModeKeys.TRAIN:
        est.train(
            input_fn=input_fn,
            max_steps=runconfig_params["max_steps"],
            use_cs=use_cs,
        )
    elif params["runconfig"]["mode"] == tf.estimator.ModeKeys.EVAL:
        output = est.evaluate(
            input_fn=input_fn, steps=runconfig_params["eval_steps"],
        )
    elif params["runconfig"]["mode"] == tf.estimator.ModeKeys.PREDICT:
        pred_dir = os.path.join(runconfig_params["model_dir"], "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        sys_name = "cs" if use_cs else "tf"
        file_to_save = f"predictions_{sys_name}_{est_config.task_id}.npz"

        output = []
        num_samples = runconfig_params["infer_steps"]
        preds = est.predict(
            input_fn=input_fn, num_samples=num_samples, use_cs=use_cs
        )
        for pred in preds:
            output.append(pred)
        if len(output) > 0:
            np.savez(os.path.join(pred_dir, file_to_save), output)
    elif params["runconfig"]["mode"] == "infer_cpp":
        preds = est.predict(input_fn=input_fn, num_samples=1, use_cs=True)
    else:
        est.compile(
            input_fn,
            validate_only=(params["runconfig"]["mode"] == "validate_only"),
        )
    return output


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    output = main()
