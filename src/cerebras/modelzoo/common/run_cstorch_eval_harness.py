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

"""Eval Harness run script"""
import argparse
import inspect
import logging
import os
import sys

# isort: off
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
# isort: on
from cerebras.modelzoo.common.utils.run.cli_parser import get_params_from_args
from cerebras.modelzoo.common.utils.run.utils import DeviceType
from cerebras.modelzoo.config_manager.config_loader import (
    validate_config_params,
)


def setup_hf_env_vars(hf_cache_dir=None):
    from cerebras.appliance.environment import appliance_environ

    # Removes annoying logs relating to process forking
    appliance_environ["TOKENIZERS_PARALLELISM"] = "false"

    if hf_cache_dir is not None:
        appliance_environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        appliance_environ["HF_HOME"] = hf_cache_dir
        appliance_environ["HF_DATASETS_CACHE"] = hf_cache_dir


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
        default=None,
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    optional_arguments.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    optional_arguments.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    optional_arguments.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    optional_arguments.add_argument(
        "--use_cache",
        type=str,
        default=None,
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    optional_arguments.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    optional_arguments.add_argument(
        "--write_out",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents",
    )
    optional_arguments.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    optional_arguments.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    optional_arguments.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    # CEREBRAS-SPECIFIC ARGS
    optional_arguments.add_argument(
        "--hf_cache_dir",
        default=None,
        help=("Path to directory for caching Hugging Face downloaded data."),
    )
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
    parent = inspect.getouterframes(inspect.currentframe())[1]
    run_dir = os.path.dirname(os.path.abspath(parent.filename))
    parser_fn = lambda: [eeh_parser()]
    parser_args = {
        "parser_epilog": (
            "Please run 'python run_cstorch_eval_harness.py CSX -h'. \n \n"
            "Here is an example command for running on CSX: \n \n"
            "    python run_cstorch_eval_harness.py CSX --params /path/to/params --checkpoint_path "
            "/path/to/checkpoint --tasks 'hellaswag,winogrande' --num_fewshot 0 \n \n"
            "Note that Eval Harness is currently only supported for device CSX"
        ),
        "csx_parser_epilog": (
            "To see a complete list of all available arguments, \n"
            "please run 'python run_cstorch_eval_harness.py CSX -h'. \n\n"
            "Here is an example command for running with CSX: \n \n"
            "    python run_cstorch_eval_harness.py CSX --params /path/to/params "
            "--checkpoint_path /path/to/checkpoint --tasks 'hellaswag,winogrande' --num_fewshot 0 "
            "\n \nEval Harness resides in the Cerebras Model Zoo. Please specify --python_paths and "
            "\n --mount_dirs here or in your params.yaml under the 'runconfig' section with \n"
            "the path to the directory in which the Cerebras Model Zoo resides. \n"
        ),
        "modes": ["eval"],
    }

    # Parse args
    params = get_params_from_args(
        run_dir,
        argv=sys.argv[1:],
        extra_args_parser_fn=parser_fn,
        device_type=DeviceType.CSX,
        **parser_args,
    )
    runconfig_params = params["runconfig"]
    from lm_eval.api.registry import get_model

    from cerebras.modelzoo.common.eval_harness_impl import CS_LLM
    from cerebras.modelzoo.common.pytorch_utils import (
        RunConfigParamsValidator,
        setup_artifact_dir,
        setup_logging,
    )
    from cerebras.modelzoo.data.nlp.gpt.InferenceDataProcessor import (
        RequestType,
    )

    # Set default model parameters
    from cerebras.modelzoo.models.nlp.gpt2.utils import set_defaults

    set_defaults(params)

    # Validate runconfig
    RunConfigParamsValidator(parser_fn).validate(runconfig_params)

    # Validate input params
    if params.get("eval_input") is not None:
        num_pt_workers = params["eval_input"].get("num_workers")
        if num_pt_workers is not None and num_pt_workers > 1:
            raise ValueError(
                "Eval harness does not support multiple process data "
                "loading for `eval_input.num_workers` > 1, but specified "
                f"{num_pt_workers} worker processes.\nPlease ensure that "
                "`eval_input.num_workers` is either 0 (default) or 1."
            )
    else:
        raise RuntimeError(
            "No `eval_input` section specified in the .yaml config."
        )

    # Set up logging level and env vars
    artifact_dir = setup_artifact_dir(runconfig_params["model_dir"], "eval")
    setup_logging(
        runconfig_params.get("logging"),
        runconfig_params.get("streamer_logging"),
        logging_dir=artifact_dir,
        model_dir=runconfig_params.get("model_dir"),
    )
    setup_hf_env_vars(hf_cache_dir=runconfig_params.get("hf_cache_dir"))

    # Debug logs
    logging.debug(f"CMD: {sys.argv}")
    logging.debug(f"Runconfig: {runconfig_params}")

    # Construct args namespace object for EEH's main script
    parser = parser_fn()[0]
    args = {}
    for arg in parser._action_groups[0]._actions:
        arg_name = arg.dest
        # Exclude Cerebras-specific args
        if arg_name in {"hf_cache_dir", "keep_data_dir"}:
            continue
        else:
            arg_val = runconfig_params.get(arg_name)
        args[arg_name] = arg_val

    from cerebras.modelzoo.data.nlp.gpt.InferenceDataProcessor import (
        InferenceDataProcessor,
    )
    from cerebras.modelzoo.models.nlp.gpt2.model import (
        Gpt2Model,
        GptInferenceModel,
    )

    def config_validation(params, model_key):
        # EEH-specific params added to the runconfig are not supported by our config class validation.
        # We remove EEH args from the runconfig, perform config validation and then re-add the args
        extra_parser_param_keys = []
        if parser:
            if parser and isinstance(parser, argparse.ArgumentParser):
                extra_parser_param_keys.extend(
                    [
                        action.dest
                        for action in parser._actions
                        if not isinstance(action, argparse._HelpAction)
                    ]
                )
        run_params = params["runconfig"]
        extra_parser_params = {}
        for eeh_arg in extra_parser_param_keys:
            if eeh_arg in run_params:
                extra_parser_params[eeh_arg] = run_params.pop(eeh_arg, None)
        # validate the params with config class
        validate_config_params(params, model_key)
        # Re-add extra EEH args to the runconfig after config validation
        run_params.update(extra_parser_params)

    def model_fn(request_type, params):
        if request_type == RequestType.loglikelihood:
            # TODO : params here contain start_token etc which are only part of inference model.
            # If we use gpt2 model, validation will fail. We need to clean up params to
            # contain start_token etc only when inference is used
            config_validation(params, "gpt_inference")
            return Gpt2Model(params)
        elif request_type == RequestType.generate_until:
            config_validation(params, "gpt_inference")
            return GptInferenceModel(params)
        else:
            raise TypeError(
                f"Invalid request type: {request_type}. At present, only "
                "`RequestType.loglikelihood` and `RequestType.generate_until` "
                "request types are supported."
            )

    def eval_input_fn(params, samples_file_list, dataset_size, request_type):
        return InferenceDataProcessor.from_request_type(
            request_type,
            params["eval_input"],
            samples_file_list,
            dataset_size,
        ).create_dataloader()

    lm = get_model(CS_LLM).create_from_arg_string(
        arg_string="",
        additional_config={
            "params": params,
            "model_fn": model_fn,
            "input_fn": eval_input_fn,
            "data_fn": InferenceDataProcessor.gen_data_samples,
            "artifact_dir": artifact_dir,
        },
    )

    # These are additional EEH args that we don't expose in our parser
    additional_args = {
        "model": lm,
        "verbosity": "INFO",  # EEH logging level
        "model_args": None,
        "batch_size": None,
        "max_batch_size": None,
        "device": None,
        "decontamination_ngrams_path": None,
        "gen_kwargs": None,
    }

    final_args = {**args, **additional_args}
    args_namespace = argparse.Namespace(**final_args)

    # Invoke EEH script
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate(args=args_namespace)


if __name__ == "__main__":
    run_eval_harness()
