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

"""Eval Harness run script built using the cstorch API"""
import argparse
import atexit
import fnmatch
import inspect
import json
import logging
import os
import sys
from typing import List, Tuple
from warnings import warn

import torch
from lm_eval import base, evaluator, tasks

import cerebras_pytorch as cstorch
from cerebras_pytorch.nn.functional import one_hot

# isort: off
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
# isort: on
from modelzoo.common.pytorch.half_dtype import set_half_dtype_from_params
from modelzoo.common.pytorch.input.utils import SamplesSaver
from modelzoo.common.pytorch.run_cstorch_flow import (
    get_cluster_config,
    get_model_checkpoint,
)
from modelzoo.common.pytorch.utils import (
    RunConfigParamsValidator,
    setup_logging,
)
from modelzoo.common.run_utils.cli_parser import get_params_from_args
from modelzoo.common.run_utils.utils import DeviceType


class CSEvalHarnessAdapter(base.LM):
    """Initializes cstorch components required for executing eval harness
    on appliance, overriding the `loglikelihood` method that performs this
    execution. Subclasses the base `LM` class that is accepted by the main
    `evaluator.evaluate` method of the EEH script.
    """

    def __init__(self, params, model_fn, input_fn, data_fn):
        data_dir = params["eval_input"].get("data_dir")

        if data_dir is None:
            raise RuntimeError(
                f"No data directory specified in params. "
                "Please specify a valid path to mounted "
                "dir under config `eval_input.data_dir` "
                "where data samples will be saved."
            )
        if not os.path.isdir(data_dir):
            raise RuntimeError(
                "Invalid path to mounted directory specified "
                f"under param config `eval_input.data_dir`: {data_dir} "
                "Please ensure that the path dir is valid dir visible "
                "to the appliance nodes."
            )

        self.eh_tasks_dir = os.path.join(data_dir, "eval_harness_tasks_data")

        self.global_step = 0
        self.msl = params["eval_input"].get("max_sequence_length", 2048)
        self.batch_size = params["eval_input"]["batch_size"]

        self.data_fn = data_fn
        self.input_fn = input_fn
        self.params = params

        runconfig = params["runconfig"]
        self.log_steps = runconfig.get("log_steps")
        self.keep_data_dir = runconfig.get("keep_data_dir", False)

        # Configure cluster
        self.cs_config = self.configure_cluster(params)

        # Initialize the backend
        self.model_dir = runconfig["model_dir"]
        compile_dir = runconfig.get("compile_dir")

        self.compile_only = runconfig.get("compile_only", False)
        self.validate_only = runconfig.get("validate_only", False)
        drop_data = runconfig.get(
            "drop_data", self.compile_only or self.validate_only
        )

        self.backend = cstorch.backend(
            "CSX",
            artifact_dir=os.path.join(self.model_dir, "cerebras_logs"),
            compile_dir=compile_dir,
            compile_only=self.compile_only,
            validate_only=self.validate_only,
            drop_data=drop_data,
            # Turning on grad accumulation in order to pick whatever sub-batch size is optimal
            use_cs_grad_accum=True,
        )
        self.backend.device.config.lazy_initialization = runconfig.get(
            "lazy_initialization", True
        )

        # Initialize the model
        self.model = self.init_model(params, model_fn)

    def configure_cluster(self, params):
        """Sets up CS cluster config for the run."""
        runconfig = params["runconfig"]
        cs_config = get_cluster_config(params)

        if cs_config.num_csx is not None and cs_config.num_csx != 1:
            raise ValueError(
                "Multi-box Eval Harness is not yet supported. "
                "Please specify num_csx=1 in the runconfig"
            )

        if (
            cs_config.num_workers_per_csx is not None
            and cs_config.num_workers_per_csx > 1
        ):
            raise ValueError(
                "Eval Harness does not support multiple workers. "
                "Please specify num_workers_per_csx=1 in the runconfig"
            )

        cs_config.num_workers_per_csx = 1

        if "precision_opt_level" in params["model"]:
            raise ValueError(
                "Passing `precision_opt_level` via `model` params is not longer supported. "
                "Please use `params[\"runconfig\"][\"precision_opt_level\"]` instead."
            )
        precision_opt_level = runconfig.get("precision_opt_level", None)
        if precision_opt_level is None:
            precision_opt_level = 1

        cs_config.precision_opt_level = precision_opt_level

        return cs_config

    def init_model(self, params, model_fn):
        """Initializes the model for the cstorch API."""

        set_half_dtype_from_params(params["model"])

        with self.backend.device:
            model = model_fn(params)

        compiled_model = cstorch.compile(model, self.backend)
        compiled_model.eval()

        def load_checkpoint(checkpoint_path):
            logging.info(f"Loading weights from checkpoint {checkpoint_path}")
            state_dict = cstorch.load(checkpoint_path)

            if "model" not in state_dict:
                warn(
                    f"Checkpoint does not contain a model state dict. "
                    f"Model state was not loaded"
                )

            # This check is required for backward compatibility with checkpoints
            # saved with older versions of ModelZoo (pre rel-2.0.0)
            # We check that the model state dict keys start with "model."
            # and if they don't, we load the state dict into the model's model
            elif hasattr(model, "model") and not any(
                k.startswith("model.") for k in state_dict["model"].keys()
            ):
                model.model.load_state_dict(state_dict["model"])

            # This should be the case that is used for all checkpoints saved
            # post rel-2.0.0
            else:
                model.load_state_dict(state_dict["model"])

            self.global_step = state_dict.get("global_step", 0)

        # Raise error here if no checkpoint_path is provided
        if not self.compile_only and not self.validate_only:
            checkpoint_path = get_model_checkpoint(params["runconfig"])
            if checkpoint_path is not None:
                load_checkpoint(checkpoint_path)
            else:
                raise RuntimeError(
                    "No checkpoint loaded. Please provide a valid "
                    "checkpoint file path under `runconfig.checkpoint_path` "
                    "or command line arg `--checkpoint_path`, or if autoloading "
                    f"from `model_dir` then please ensure this dir `{self.model_dir}` "
                    "comprises valid checkpoint files."
                )

        return compiled_model

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """This is the overriden method of the LM base class in EEH script.
        This method preprocesses the raw text requests, generates the data
        samples to be consumed by the GPT2 model, and executes the data on
        the appliance.

        Args:
            requests: list of raw text context, continuation pair of tuples

        Returns:
            list of size `len(requests)` comprising post-processed results
            tuple as expected by the EEH script

        """
        tok_file_path = self.params["eval_input"].get("tokenizer_file_path")
        eos_id = self.params["eval_input"].get("eos_id")

        if tok_file_path is not None and eos_id is None:
            raise RuntimeError(
                f"No end of sentence token id specified under "
                "`eval_input.eos_id` for custom tokenizer: "
                f"{tok_file_path}\n Please specify an end "
                "of sentence token id under `eval_input.eos_id`"
            )
        elif eos_id is None:
            logging.warning(
                f"End of sentence token id not specified under "
                "`eval_input.eos_id`. Defaulting to GPT2 tokenizer "
                "with eos_id 50256"
            )
            eos_id = 50256

        # Create data samples and dump these to file
        MAX_FILE_SIZE = 1024 * 1024 * 500  # 500 MB

        samples_saver = SamplesSaver(
            data_dir=self.eh_tasks_dir,
            max_file_size=MAX_FILE_SIZE,
            filename_prefix=f"requests_{self.msl}_msl",
        )
        samples_file_list, dataset_size, token_lengths = self.data_fn(
            requests,
            self.batch_size,
            self.msl,
            eos_id,
            samples_saver,
            tok_file_path,
        )
        # Register clean-up method to remove data dumps
        if not self.keep_data_dir:
            atexit.register(samples_saver.delete_data_dumps)

        sample_idx = 0
        results = []

        @cstorch.trace
        @torch.no_grad()
        def eval_step(batch):
            _, lm_logits = self.model(batch, output_logits=True)  # forward pass

            # Calculate softmax of logits
            lm_logits = torch.nn.functional.log_softmax(
                lm_logits.float(), dim=-1, dtype=torch.float32
            )

            # Post processing of output logits to produce
            # predictions and logits for continuation tokens
            attn_mask = batch["attention_mask"].to(torch.float32)
            cont_tokens = batch["continuation"].to(torch.long)

            # Only keep logits corresponding to the continuation token positions
            cont_logits = lm_logits.clone()
            # Step 1: repeat attn_mask vocab_size times along the 2nd dim
            # [bs, msl] -> [bs, msl, vocab_size]
            attn_mask = attn_mask.unsqueeze(2).repeat(
                1, 1, cont_logits.shape[-1]
            )

            # Step 2: zero out the logits except the ones corresponding to continuation
            # token positions
            cont_logits = cont_logits * attn_mask

            # Step 3: gather probs corresponding to the tokens in continuation
            cont_toks_one_hot = one_hot(
                cont_tokens, num_classes=lm_logits.shape[-1]
            ).to(cont_logits.dtype)

            cont_logits = cont_logits * cont_toks_one_hot
            cont_log_probs = cont_logits.sum(-1)

            predictions = lm_logits.argmax(-1).int()
            # Subtract `cont_tokens` from `predictions` and output
            # comparisons tensor to check if the continuation token
            # predictions match the input
            cont_comparisons = torch.add(predictions * -1, cont_tokens)

            return cont_comparisons, cont_log_probs

        @cstorch.step_closure
        def post_eval_step(cont_comparisons, log_probs, step):
            nonlocal token_lengths
            nonlocal sample_idx
            nonlocal results

            # Post processing of model output to produce results
            for comparison, cont_logits in zip(cont_comparisons, log_probs):
                tok_lengths = token_lengths[sample_idx]
                ctx_len, cont_len = tok_lengths
                if not ctx_len or not cont_len:
                    # Skip post processing for padded 0 tensors
                    continue

                # Since we subtracted the model's predictions from the input
                # tokens, predictions exactly match the continuation tokens
                # where the `comparison` tensor has 0s
                cont_comparison = comparison[
                    ctx_len - 1 : ctx_len + cont_len - 1
                ]
                max_equal = (cont_comparison == 0).all()

                # Answer: (log prob, is-exact-match)
                answer = (float(cont_logits.sum()), bool(max_equal))

                results.append(answer)
                sample_idx += 1

            # Provide useful logs
            is_log_step = executor.on_final_iteration or (
                self.log_steps and step % self.log_steps == 0
            )

            rate = executor.profiler.rate_tracker.rate
            global_rate = executor.profiler.rate_tracker.global_rate

            if is_log_step:
                logging.info(
                    f"| Eval Device=CSX, "
                    f"Step={step}, "
                    f"Rate={rate:.2f} samples/sec, "
                    f"GlobalRate={global_rate:.2f} samples/sec"
                )
                logging.debug(
                    f"Continuation Comparisons={cont_comparisons}, "
                    f"Logits={log_probs}, "
                )

        dataloader = cstorch.utils.data.DataLoader(
            self.input_fn, self.params, samples_file_list, dataset_size
        )
        if self.compile_only or self.validate_only:
            steps = None
        else:
            steps = len(dataloader)
        logging.info(f"Running for steps: {steps}")

        executor = cstorch.utils.data.DataExecutor(
            dataloader, num_steps=steps, cs_config=self.cs_config,
        )

        for step, batch in enumerate(executor, start=1):
            cont_comparisons, log_probs = eval_step(batch)

            post_eval_step(cont_comparisons, log_probs, step)

        logging.debug(f"Output results: {results}")
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "Loglikelihood rolling is currently not supported"
        )

    def greedy_until(self, requests):
        raise NotImplementedError(
            "Greedy generation is currently not supported"
        )


def setup_hf_env_vars(hf_cache_dir=None):
    from cerebras_appliance.environment import appliance_environ

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
    optional_arguments.add_argument(
        "--tasks",
        default=None,
        help="Comma separated string specifying Eleuther Eval Harness tasks.",
    )
    optional_arguments.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of examples to be added to the fewshot context string. Defaults to 0.",
    )
    optional_arguments.add_argument(
        "--output_path",
        required=False,
        default="./eval_harness_output",
        help="Path to directory where eval harness output results will be written.",
    )
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


# Util method from EEH `main.py` script:
# https://github.com/EleutherAI/lm-evaluation-harness/blob/62ca18400ebe0fc0dbb14274b27170d2d5ae9e3d/main.py#L47-L54
# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


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

    params = get_params_from_args(
        run_dir,
        argv=sys.argv[1:],
        extra_args_parser_fn=parser_fn,
        device_type=DeviceType.CSX,
        **parser_args,
    )
    runconfig_params = params["runconfig"]

    # Set default model parameters
    from modelzoo.transformers.pytorch.gpt2.utils import set_defaults

    set_defaults(params)

    # Validate runconfig
    RunConfigParamsValidator(parser_fn).validate(runconfig_params)

    # Set up logging level and env vars
    setup_logging(
        runconfig_params.get("logging"),
        runconfig_params.get("streamer_logging"),
        logging_dir=runconfig_params.get("model_dir"),
    )
    setup_hf_env_vars(hf_cache_dir=runconfig_params.get("hf_cache_dir"))

    # Debug logs
    logging.debug(f"CMD: {sys.argv}")
    logging.debug(f"Runconfig: {runconfig_params}")

    specified_tasks = runconfig_params.get("tasks")
    if specified_tasks is not None:
        eh_tasks = pattern_match(specified_tasks.split(","), tasks.ALL_TASKS)
    else:
        eh_tasks = tasks.ALL_TASKS

    from modelzoo.transformers.pytorch.gpt2.input.InferenceDataProcessor import (
        InferenceDataProcessor,
    )
    from modelzoo.transformers.pytorch.gpt2.model import Gpt2Model

    def eval_input_fn(params, samples_file_list, dataset_size):
        return InferenceDataProcessor(
            params["eval_input"], samples_file_list, dataset_size
        ).create_dataloader()

    cs_llm = CSEvalHarnessAdapter(
        params,
        Gpt2Model,
        eval_input_fn,
        data_fn=InferenceDataProcessor.gen_data_samples,
    )

    results = evaluator.evaluate(
        lm=cs_llm,
        task_dict=tasks.get_task_dict(eh_tasks),
        description_dict=None,
        num_fewshot=runconfig_params.get("num_fewshot"),
        limit=None,
        bootstrap_iters=100000,
    )

    # Print results
    dumped = json.dumps(results, indent=2)
    logging.info(dumped)

    output_path = runconfig_params.get("output_path")
    if output_path is not None:
        dirname = os.path.dirname(output_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(dumped)

    logging.info(f"\n{evaluator.make_table(results)}")


if __name__ == "__main__":
    run_eval_harness()
