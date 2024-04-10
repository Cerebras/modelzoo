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

"""Implements `CSEvalHarnessAdapter` class for running Eval Harness on CSX via the cstorch API"""
import atexit
import logging
import os
from typing import List, Tuple, Union
from warnings import warn

import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.half_dtype import set_half_dtype_from_params
from cerebras.modelzoo.common.input_utils import (
    validate_streaming_and_micro_batch_size,
)
from cerebras.modelzoo.common.pytorch_utils import load_from_checkpoint_file
from cerebras.modelzoo.common.run_cstorch_flow import (
    get_cluster_config,
    get_model_checkpoint,
)
from cerebras.modelzoo.common.utils.input.utils import SamplesSaver
from cerebras.modelzoo.common.utils.utils import format_rate
from cerebras.modelzoo.data.nlp.gpt.InferenceDataProcessor import RequestType
from cerebras.pytorch.nn.functional import one_hot

CS_LLM = "cs-llm"


@register_model(CS_LLM)
class CSEvalHarnessAdapter(LM):
    """Initializes cstorch components required for executing eval harness
    on appliance, overriding the `loglikelihood` method that performs this
    execution. Subclasses the base `LM` class that is accepted by the main
    `evaluator.evaluate` method of the EEH script.
    """

    def __init__(self, params, model_fn, input_fn, data_fn, artifact_dir=None):
        super().__init__()
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

        self.msl = params["eval_input"].get("max_sequence_length")
        if self.msl is None:
            info_msg = (
                "No maximum sequence length specified under `eval_input.max_sequence_length`. "
                "This setting is required for preprocessing input data samples from the specified "
                "eval harness tasks.\n{0} Note that input sequences will be truncated to fit "
                "within this length."
            )

            max_position_embeddings = params["model"].get(
                "max_position_embeddings"
            )
            if max_position_embeddings is None:
                raise RuntimeError(
                    info_msg.format(
                        "Please specify the maximum sequence (or context) length."
                    )
                )

            self.msl = max_position_embeddings
            logging.info(
                info_msg.format(
                    f"Defaulting to max sequence length of {max_position_embeddings}, as "
                    "specified under setting `model.max_position_embeddings`."
                )
            )

        self.global_step = 0
        self.batch_size = params["eval_input"]["batch_size"]
        self.micro_batch_size = params["eval_input"].get(
            "micro_batch_size", "auto"
        )
        # Checks for invalid setting of num_csx, micro_batch_size and batch_size
        validate_streaming_and_micro_batch_size(
            self.batch_size, self.micro_batch_size, num_csx=1
        )

        self.model_fn = model_fn
        self.data_fn = data_fn
        self.input_fn = input_fn
        self.params = params

        runconfig = params["runconfig"]
        self.log_steps = runconfig.get("log_steps")
        self.keep_data_dir = runconfig.get("keep_data_dir", False)

        # Configure cluster
        self.cs_config = self.configure_cluster(params)

        self.model_dir = runconfig["model_dir"]
        self.compile_dir = runconfig.get("compile_dir")

        self.compile_only = runconfig.get("compile_only", False)
        self.validate_only = runconfig.get("validate_only", False)
        self.drop_data = runconfig.get(
            "drop_data", self.compile_only or self.validate_only
        )

        # Init backend
        lazy_initialization = runconfig.get("lazy_initialization", True)
        retrace_every_iteration = runconfig.get(
            "retrace_every_iteration", False
        )
        if artifact_dir is None:
            artifact_dir = os.path.join(self.model_dir, "cerebras_logs")
        self.backend = cstorch.backend(
            "CSX",
            artifact_dir=artifact_dir,
            compile_dir=self.compile_dir,
            compile_only=self.compile_only,
            validate_only=self.validate_only,
            drop_data=self.drop_data,
            retrace_every_iteration=retrace_every_iteration,
        )
        self.backend.device.config.lazy_initialization = lazy_initialization

        # Dummy model attr needed for EEH script
        # Ref: https://github.com/EleutherAI/lm-evaluation-harness/blob/c9bbec6e7de418b9082379da82797522eb173054/lm_eval/evaluator.py#L165-L167
        self.model = lambda: None
        self.model.config = lambda: None
        self.model.config._name_or_path = CS_LLM

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

    def init_model(self, request_type, params):
        """Initializes the model for the cstorch API."""

        set_half_dtype_from_params(params["model"])

        with self.backend.device:
            model = self.model_fn(request_type, params)

        compiled_model = cstorch.compile(model, self.backend)
        compiled_model.eval()

        def load_checkpoint(checkpoint_path):
            state_dict = load_from_checkpoint_file(checkpoint_path)

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

    def preprocess_requests(
        self, requests: List[Instance], request_type: RequestType
    ) -> Tuple[
        Union[Tokenizer, PreTrainedTokenizerBase],
        List[str],
        int,
        Tuple[int, int],
    ]:
        """Tokenize EEH raw text requests and returns metadata associated with the
        request type

        Args:
            requests: List of EEH's Instance dataclass objects
                holding raw text data
            request_type: The type of request

        Returns:
            tuple of
            - tokenizer used to tokenize the raw text data;
            - list of file paths where the samples are dumped;
            - int representing the size of the dataset (total no. of samples);
            - tuple of request metadata needed for postprocessing.
        """
        tokenizer_file_path = self.params["eval_input"].get(
            "tokenizer_file_path"
        )
        eos_token_id = self.params["eval_input"].get("eos_id")

        if tokenizer_file_path is not None:
            if eos_token_id is None:
                raise RuntimeError(
                    f"No end of sentence token id specified under "
                    "`eval_input.eos_id` for custom tokenizer: "
                    f"{tokenizer_file_path}\n Please specify an end "
                    "of sentence token id under `eval_input.eos_id`"
                )
            tokenizer = Tokenizer.from_file(tokenizer_file_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            eos_token_id = tokenizer.eos_token_id
            logging.warning(
                "No tokenizer file path specified under `eval_input.tokenizer_file_path`. "
                f"Defaulting to GPT2 tokenizer with end of sentence token id {eos_token_id}"
            )

        # Create data samples and dump these to file
        MAX_FILE_SIZE = 1024 * 1024 * 500  # 500 MB

        samples_saver = SamplesSaver(
            data_dir=self.eh_tasks_dir,
            max_file_size=MAX_FILE_SIZE,
            filename_prefix=f"requests_{self.msl}_msl",
        )

        samples_file_list, dataset_size, requests_metadata = self.data_fn(
            requests,
            self.batch_size,
            self.msl,
            tokenizer,
            eos_token_id,
            samples_saver,
            request_type=request_type,
            inf_start_token=self.params["model"].get("start_token"),
            max_gen_tokens=self.params["model"].get("max_tokens"),
        )

        # Register clean-up method to remove data dumps
        if not self.keep_data_dir:
            atexit.register(samples_saver.delete_data_dumps)

        return tokenizer, samples_file_list, dataset_size, requests_metadata

    def loglikelihood(
        self, requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """This method provides an implementation for the abstract method of EEH's LM interface
        class: https://github.com/EleutherAI/lm-evaluation-harness/blob/c9bbec6e7de418b9082379da82797522eb173054/lm_eval/api/model.py#L34

        This method preprocesses the raw text requests, generates the data
        samples to be consumed by the GPT2 model, and executes the data on
        the appliance.

        Args:
            requests: A list of EEH's Instance objects, with property `args` which returns a tuple
            of (context, continuation) strings.

        Returns:
            list of size `len(requests)` comprising tuple of (float, bool) representing
            - logprob of generating the continuation string
            - whether `continuation` is generated by greedy sampling from `context`
        """
        (
            _,
            samples_file_list,
            dataset_size,
            token_lengths,
        ) = self.preprocess_requests(requests, RequestType.loglikelihood)

        # Init model
        model = self.init_model(RequestType.loglikelihood, self.params)

        sample_idx = 0
        results = []

        @cstorch.trace
        @torch.no_grad()
        def eval_step(batch):
            _, lm_logits = model(batch, output_logits=True)  # forward pass

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
        def post_eval_step(cont_comparisons, log_probs):
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
                self.log_steps and executor.user_iteration % self.log_steps == 0
            )

            if is_log_step:
                rate = executor.profiler.rate_tracker.rate
                global_rate = executor.profiler.rate_tracker.global_rate

                logging.info(
                    f"| Eval Device=CSX, "
                    f"Step={executor.user_iteration}, "
                    f"Rate={format_rate(rate)} samples/sec, "
                    f"GlobalRate={format_rate(global_rate)} samples/sec"
                )
                logging.debug(
                    f"Continuation Comparisons={cont_comparisons}, "
                    f"Logits={log_probs}, "
                )

        dataloader = cstorch.utils.data.DataLoader(
            self.input_fn,
            self.params,
            samples_file_list,
            dataset_size,
            RequestType.loglikelihood.value,
        )
        if self.compile_only or self.validate_only:
            steps = None
            results = [(-0.0, False)] * dataset_size
        else:
            steps = len(dataloader)
        logging.info(f"Running for steps: {steps}")

        executor = cstorch.utils.data.DataExecutor(
            dataloader,
            num_steps=steps,
            cs_config=self.cs_config,
            micro_batch_size=self.micro_batch_size,
        )

        for batch in executor:
            cont_comparisons, log_probs = eval_step(batch)
            post_eval_step(cont_comparisons, log_probs)

        logging.debug(f"Output results: {results}")
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "Loglikelihood rolling is currently not supported"
        )

    def generate_until(self, requests: List[Instance]) -> List[str]:
        """This method provides an implementation for the abstract method of EEH's LM interface
        class: https://github.com/EleutherAI/lm-evaluation-harness/blob/c9bbec6e7de418b9082379da82797522eb173054/lm_eval/api/model.py#L102

        Args:
            requests: A list of EEH Instance objects with property `args` which returns a tuple
            of (context, until) strings

        Returns:
            list of size `len(requests)` comprising generated continuation strings
        """
        (
            tokenizer,
            samples_file_list,
            dataset_size,
            metadata,
        ) = self.preprocess_requests(requests, RequestType.generate_until)

        # Initialize generative inference model
        gen_model = self.init_model(RequestType.generate_until, self.params)

        # `until_token_seqs` are guaranteed to be fixed for all generative requests for now
        until_token_seqs, _ = metadata[0]

        start_token = self.params["model"].get("start_token")
        sample_idx = 0
        results = []

        @cstorch.trace
        @torch.no_grad()
        def inference_step(batch):
            predictions = gen_model(
                batch, stop_sequences=until_token_seqs
            )  # forward pass
            return predictions

        @cstorch.step_closure
        def post_inf_step(predictions):
            nonlocal metadata, sample_idx, results, start_token

            # Post processing of model output to produce results
            for pred in predictions:
                if not metadata[sample_idx]:
                    # Skip post processing for padded 0 tensors
                    continue
                _, ctx_len = metadata[sample_idx]

                # Get tokens for the generated continuation string
                gen_continuation = pred[ctx_len:].tolist()
                try:
                    start_token_idx = gen_continuation.index(start_token)
                    gen_continuation = gen_continuation[:start_token_idx]
                except ValueError:  # Generated string spans msl
                    pass

                gen_continuation_str = tokenizer.decode(gen_continuation)
                results.append(gen_continuation_str)
                sample_idx += 1

            # Provide useful logs
            is_log_step = executor.on_final_iteration or (
                self.log_steps and executor.user_iteration % self.log_steps == 0
            )

            if is_log_step:
                rate = executor.profiler.rate_tracker.rate
                global_rate = executor.profiler.rate_tracker.global_rate

                logging.info(
                    f"| Generative Eval Device=CSX, "
                    f"Step={executor.user_iteration}, "
                    f"Rate={format_rate(rate)} samples/sec, "
                    f"GlobalRate={format_rate(global_rate)} samples/sec"
                )

        dataloader = cstorch.utils.data.DataLoader(
            self.input_fn,
            self.params,
            samples_file_list,
            dataset_size,
            RequestType.generate_until.value,
        )
        if self.compile_only or self.validate_only:
            steps = None
        else:
            steps = len(dataloader)
        logging.info(f"Running for steps: {steps}")

        # No activation sharding for generative inference
        if (
            self.cs_config.max_act_per_csx is not None
            and self.cs_config.max_act_per_csx > 1
        ):
            warn(
                "Activation server sharding is not supported for generative eval tasks. "
                "Forcing `max_act_per_csx` to 1."
            )
        self.cs_config.max_act_per_csx = 1

        executor = cstorch.utils.data.DataExecutor(
            dataloader,
            num_steps=steps,
            cs_config=self.cs_config,
            micro_batch_size=self.micro_batch_size,
        )
        for batch in executor:
            predictions = inference_step(batch)

            post_inf_step(predictions)

        logging.debug(f"Output results: {results}")
        return results
