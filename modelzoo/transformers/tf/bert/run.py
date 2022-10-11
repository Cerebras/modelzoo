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
import sys

import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from modelzoo.common.tf.estimator.cs_estimator import CerebrasEstimator
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.run_utils import (
    check_env,
    create_warm_start_settings,
    get_csconfig,
    get_csrunconfig_dict,
    is_cs,
    save_params,
    save_predictions,
    update_params_from_args,
)
from modelzoo.transformers.tf.bert.data import eval_input_fn, train_input_fn
from modelzoo.transformers.tf.bert.model import model_fn
from modelzoo.transformers.tf.bert.utils import (
    get_custom_stack_params,
    get_params,
)

CS1_MODES = ["train", "eval"]


def create_arg_parser(default_model_dir, include_multireplica=False):
    """
    Create parser for command line args.

    :param str default_model_dir: default value for the model_dir
    :returns: ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        default=default_model_dir,
        help="Model directory where checkpoints will be written. "
        + "If directory exists, weights are loaded from the checkpoint file.",
    )
    parser.add_argument(
        "--cs_ip",
        default=None,
        help="IP address of the Cerebras System, defaults to None. Ignored on GPU.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of steps to run mode train."
            + " Runs repeatedly for the specified number."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Number of total steps to run mode train or for defining training"
            + " configuration for train_and_eval. Runs incrementally till"
            + " the specified number."
        ),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help=(
            "Number of total steps to run mode eval, eval_all or for defining"
            + " eval configuration for train_and_eval. Runs once for"
            + " the specified number."
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["train", "eval", "eval_all", "train_and_eval", "predict",],
        help=(
            "Can train, eval, eval_all, train_and_eval, or predict."
            + "  Train, eval, and predict will compile and train if on the Cerebras System,"
            + "  and just run locally (CPU/GPU) if not on the Cerebras System."
            + "  train_and_eval will run locally."
            + "  Eval_all will run eval locally for all available checkpoints."
        ),
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Compile model up to kernel matching.",
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        help="Compile model completely, generating compiled executables.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force model to run on a specific device (e.g., --device /gpu:0)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Checkpoint to initialize weights from.",
    )
    if include_multireplica:
        parser.add_argument(
            "--multireplica",
            action="store_true",
            help="run multiple copies of the model data-parallel"
            + " on the wafer at the same time.",
        )

    return parser


def validate_params(params, cs1_modes):
    # check validate_only/compile_only
    runconfig_params = params["runconfig"]
    assert not (
        runconfig_params["validate_only"] and runconfig_params["compile_only"]
    ), "Please only use one of validate_only and compile_only."

    # check for gpu optimization flags
    if (
        runconfig_params["mode"] not in ["compile_only", "validate_only"]
        and not is_cs(runconfig_params)
        and not params["model"]["enable_gpu_optimizations"]
    ):
        tf.compat.v1.logging.warn(
            "Set enable_gpu_optimizations to True in model params "
            "to improve GPU performance."
        )

    # ensure runconfig is compatible with the Cerebras System
    if (
        is_cs(runconfig_params)
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        assert runconfig_params["mode"] in cs1_modes, (
            "To run this model on the Cerebras System, please use one of the following modes: "
            ", ".join(cs1_modes)
        )
        assert not (
            runconfig_params.get("multireplica")
            and runconfig_params["mode"] != "train"
        ), "--multireplica can only be used in train mode."
    else:
        assert not runconfig_params.get(
            "multireplica"
        ), "--multireplica can only be used on the Cerebras System!"


def run(
    args,
    params,
    model_fn,
    train_input_fn=None,
    eval_input_fn=None,
    predict_input_fn=None,
    output_layer_name=None,
    cs1_modes=CS1_MODES,
):
    """
    Set up estimator and run based on mode

    :params dict params: dict to handle all parameters
    :params tf.estimator.EstimatorSpec model_fn: Model function to run with
    :params tf.data.Dataset train_input_fn: Dataset to train with
    :params tf.data.Dataset eval_input_fn: Dataset to validate against
    :params tf.data.Dataset predict_input_fn: Dataset to run inference on
    :params str output_layer_name: name of the output layer to be excluded
        from weight initialization when performing fine-tuning.
    """
    # update and validate runtime params
    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    validate_params(params, cs1_modes)
    # save params for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])

    # get cs-specific configs
    cs_config = get_csconfig(params.get("csconfig", dict()))
    # get runtime configurations
    use_cs = is_cs(runconfig_params)
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    stack_params = get_custom_stack_params(params)

    # prep cs1 run environment, run config and estimator
    check_env(runconfig_params)
    est_config = CSRunConfig(
        cs_ip=runconfig_params["cs_ip"],
        cs_config=cs_config,
        stack_params=stack_params,
        **csrunconfig_dict,
    )
    warm_start_settings = create_warm_start_settings(
        runconfig_params, exclude_string=output_layer_name
    )
    est = CerebrasEstimator(
        model_fn=model_fn,
        model_dir=runconfig_params["model_dir"],
        config=est_config,
        params=params,
        warm_start_from=warm_start_settings,
    )

    # execute based on mode
    if runconfig_params["validate_only"] or runconfig_params["compile_only"]:
        if runconfig_params["mode"] == "train":
            input_fn = train_input_fn
            mode = tf.estimator.ModeKeys.TRAIN
        elif runconfig_params["mode"] == "eval":
            input_fn = eval_input_fn
            mode = tf.estimator.ModeKeys.EVAL
        else:
            input_fn = predict_input_fn
            mode = tf.estimator.ModeKeys.PREDICT
        est.compile(
            input_fn, validate_only=runconfig_params["validate_only"], mode=mode
        )
    elif runconfig_params["mode"] == "train":
        est.train(
            input_fn=train_input_fn,
            steps=runconfig_params["steps"],
            max_steps=runconfig_params["max_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval":
        est.evaluate(
            input_fn=eval_input_fn,
            checkpoint_path=runconfig_params["checkpoint_path"],
            steps=runconfig_params["eval_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval_all":
        ckpt_list = tf.train.get_checkpoint_state(
            runconfig_params["model_dir"]
        ).all_model_checkpoint_paths
        for ckpt in ckpt_list:
            est.evaluate(
                eval_input_fn,
                checkpoint_path=ckpt,
                steps=runconfig_params["eval_steps"],
                use_cs=use_cs,
            )
    elif runconfig_params["mode"] == "train_and_eval":
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=runconfig_params["max_steps"]
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            steps=runconfig_params["eval_steps"],
            throttle_secs=runconfig_params["throttle_secs"],
        )
        tf.estimator.train_and_evaluate(est, train_spec, eval_spec)
    elif runconfig_params["mode"] == "predict":
        sys_name = "cs" if use_cs else "tf"
        file_to_save = f"predictions_{sys_name}_{est_config.task_id}.npz"
        predictions = est.predict(
            input_fn=predict_input_fn,
            checkpoint_path=runconfig_params["checkpoint_path"],
            num_samples=runconfig_params["predict_steps"],
            use_cs=use_cs,
        )
        save_predictions(
            model_dir=runconfig_params["model_dir"],
            outputs=predictions,
            name=file_to_save,
        )


def main():
    """
    Main function
    """
    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_dir"
    )
    parser = create_arg_parser(default_model_dir, include_multireplica=True)
    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params, mode=args.mode)
    run(
        args=args,
        params=params,
        model_fn=model_fn,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
    )


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()
