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
This module provides a callback class to run BigCode's Evaluation Harness.
"""

import inspect
import json
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import cached_property
from math import ceil
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn

from bigcode_eval import tasks as bigcode_tasks
from bigcode_eval.base import Task
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.utils import update_code_gens
from lm_eval.utils import pattern_match

import cerebras.pytorch as cstorch
from cerebras.appliance.environment import appliance_environ
from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.modelzoo.data.nlp.gpt.InferenceDataProcessor import RequestType
from cerebras.modelzoo.trainer import Trainer
from cerebras.modelzoo.trainer.callbacks import (
    Callback,
    ValidationCallback,
    ValidationLoop,
)
from cerebras.modelzoo.trainer.callbacks.flags import _ScopedFlags
from cerebras.modelzoo.trainer.extensions.eval_harness_adapter import (
    CSEvalHarnessAdapter,
)
from cerebras.modelzoo.trainer.loggers import ProgressLogger


@dataclass
class BigCodeCLIArgs:
    r"""Captures BigCode EH's CLI arguments with defaults.

    Fields:
        prefix: Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'
        do_sample: Sample from the language model's output distribution.
        temperature: Sampling temperature used for generation.
        top_k: Top-k parameter used for generation.
        top_p: Top-p parameter used for nucleus sampling.
        n_samples: Number of completions to generate for each sample.
        seed: Random seed used for evaluation.
        tasks: List of tasks to evaluate code evals
        instruction_tokens: A series of instruction tokens used for instruction-tuning
            benchamrks separated by comma e.g.
            <user_message>,<end_user_message>,<assistant_message>
        max_tokens: Maximum number of tokens to generate.
        limit: Number of samples to solve and evaluate from the benchmark
        limit_start: Optional offset to start from when limiting the number of samples
        save_every_k_tasks: Optional saving after every k tasks
        postprocess: Postprocess model outputs before execution, always on except
            during generation tests
        allow_code_execution: Allow code evaluation to execute external/untrusted Python
            code on your machine
        generation_only: Do code generation but no evaluation
        load_generations_path: Path of file with previously generated solutions, if
            provided generation is skipped and only evaluation is done
        load_data_path: Path of additional data to load for the tasks
        metric_output_path: Path to save the results
        save_generations: Whether to save code generations
        load_generations_intermediate_paths: List of paths for saving the
            intermediate code generations
        save_generations_path: Path for saving the code generations
        save_references: Whether to save reference solutions/tests
        save_references_path: Path for saving the references solutions/tests
        prompt: Prompt type to use for generation in HumanEvalPack tasks
        check_references: Don't run generation but benchmark groundtruth (useful for debugging)
    """

    # BigCode `EvalArguments` dataclass injected as into CLI
    prefix: str = ""
    do_sample: bool = True
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n_samples: int = 1
    seed: int = 0
    # Other BigCode CLI arguments
    tasks: Optional[Union[str, List[str]]] = None
    instruction_tokens: Optional[str] = None
    max_tokens: Optional[int] = None
    limit: Optional[int] = None
    limit_start: int = 0
    save_every_k_tasks: int = -1
    postprocess: bool = True
    allow_code_execution: bool = False
    generation_only: bool = True  # We only run this flow by default
    load_generations_path: Optional[str] = None
    load_data_path: Optional[str] = None
    metric_output_path: str = "evaluation_results.json"
    save_generations: bool = (
        True  # We always save for the separate code execution flow
    )
    load_generations_intermediate_paths: Optional[List[str]] = None
    save_generations_path: str = "generations.json"
    save_references: bool = True
    save_references_path: str = "references.json"
    prompt: str = "prompt"
    check_references: bool = False


@named_class_logger("BigCodeEvalHarnessRunner")
class BigCodeEvalHarnessRunner(ClassLogger):
    """Util class for invoking BigCode's run script with CSX-specific components."""

    def __init__(
        self,
        bigcode_args: BigCodeCLIArgs,
    ):
        """Constructs a `BigCodeEvalHarnessRunner` instance.

        Args:
            bigcode_args: `BigCodeCLIArgs` dataclass object capturing BCEH's CLI args
        """
        super().__init__()

        self.args = deepcopy(bigcode_args)

        # Validate user-specified tasks
        if not self.task_names:
            raise ValueError(
                f"Task not found: {self.args.tasks}.\n"
                f"Available tasks: {','.join(bigcode_tasks.TASK_REGISTRY.keys())}"
            )

    @cached_property
    def task_names(self) -> List[str]:
        """Returns the task names list for the specified tasks."""
        if self.args.tasks is None:
            raise ValueError(
                "Need to specify a bigcode task to evaluate.\n"
                f"Available tasks: {','.join(bigcode_tasks.TASK_REGISTRY.keys())}"
            )
        else:
            return pattern_match(
                self.args.tasks.split(","), bigcode_tasks.ALL_TASKS
            )

    def evaluate(self, trainer: Trainer, evaluator: Evaluator) -> None:
        # pylint: disable=line-too-long
        """Invoke's logic from BigCode's run script on the `bigcode evaluator <bigcode_evaluator>`_.

        .. bigcode_evaluator: https://github.com/bigcode-project/bigcode-evaluation-harness/blob/a1b4a7949a24c8e3ef0d05a01097b2d14ffba56e/main.py#L372

        Args:
            trainer: Trainer object
            evaluator: The evaluator object (subclass of BigCode's Evaluator class)
        """
        self.logger.info(
            f"Starting BigCode evaluation harness on selected tasks: {self.task_names}"
        )

        load_generations_intermediate_paths = (
            self.args.load_generations_intermediate_paths
        )
        if load_generations_intermediate_paths and len(
            load_generations_intermediate_paths
        ) != len(self.task_names):
            raise ValueError(
                "If passing --load_generations_intermediate_paths, "
                "must pass equal number of files as number of tasks"
            )

        results = {}
        for idx, task in enumerate(self.task_names):
            if self.args.load_generations_path:
                raise RuntimeError(
                    "Code evaluation mode is not yet supported. "
                    "Please specify `--generation_only` flag to run "
                    "bigcode's generation flow on CSX."
                )
            elif self.args.generation_only:
                self.logger.info("Running with generation-only mode")
                intermediate_generations = None
                if load_generations_intermediate_paths:
                    with open(
                        load_generations_intermediate_paths[idx], "r"
                    ) as f_in:
                        # intermediate_generations: list[list[str | None]] of len n_tasks
                        # where list[i] = generated codes or empty
                        intermediate_generations = json.load(f_in)

                generations, references = evaluator.generate_text(
                    task, intermediate_generations=intermediate_generations
                )

                save_generations_path = os.path.splitext(
                    self.args.save_generations_path
                )[0]
                save_generations_path = (
                    f"{save_generations_path}_{task}_{trainer.global_step}.json"
                )
                save_references_path = os.path.splitext(
                    self.args.save_references_path
                )[0]
                save_references_path = (
                    f"{save_references_path}_{task}_{trainer.global_step}.json"
                )
                evaluator.save_json_files(
                    generations,
                    references,
                    save_generations_path,
                    save_references_path,
                )
            else:
                raise RuntimeError(
                    f"Code evaluation mode is not yet supported. "
                    "Please specify `--generation_only` flag to run "
                    "bigcode's generation flow on CSX."
                )

        # Save all args to config
        results["config"] = asdict(self.args)
        if not self.args.generation_only:
            dumped = json.dumps(results, indent=2)
            self.logger.info(dumped)

            with open(self.args.metric_output_path, "w") as f:
                f.write(dumped)


class BigCodeEvaluator(CSEvalHarnessAdapter, Evaluator):
    """
    Subclasses BigCode's `Evaluator` base class, overriding the
    `generate_text` method.
    """

    def __init__(
        self,
        trainer,
        bigcode_args: BigCodeCLIArgs,
        dataloader_args: Dict[str, Any],
    ):
        """
        Args:
            trainer: Trainer object
            bigcode_args: `BigCodeCLIArgs` dataclass object capturing BCEH's CLI args
            dataloader_args: Dict of dataloader args.
        """
        self.args: BigCodeCLIArgs
        self.dataloader_args: Dict[str, Any]

        Evaluator.__init__(self, None, None, None, args=bigcode_args)
        CSEvalHarnessAdapter.__init__(
            self, trainer=trainer, dataloader_args=dataloader_args
        )

    def evaluate(
        self,
        task_name: str,
        intermediate_generations: Optional[
            List[Optional[List[Optional[str]]]]
        ] = None,
    ):
        """Override of the BCEH's Evaluator class' method.

        Note: Code evaluation flow is not yet supported.
        """
        raise NotImplementedError("Code evaluation flow is not yet supported.")

    def _construct_prompts(
        self,
        task: Any,
        dataset: Any,
        n_tasks: int,
        limit_start: int = 0,
        n_copies: int = 1,
        instruction_tokens: Optional[List[str]] = None,
    ) -> List[str]:
        """Helper from BigCode's implementaion to preprocess task dataset into a list of
        raw text samples.
        """

        def _make_infill_prompt(self, prefix, suffix, preprefix=""):
            """Make a prompt for infilling.
            Currently supported only for official InCoder and SantaCoder implementations.
            """
            model_id = self.tokenizer.name_or_path
            if model_id in ["facebook/incoder-1B", "facebook/incoder-6B"]:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                return f"{preprefix}{prefix}<|mask:0|>{suffix}<|mask:0|>"
            elif model_id in ["bigcode/santacoder"]:
                return f"<fim-prefix>{preprefix}{prefix}<fim-suffix>{suffix}<fim-middle>"
            elif model_id in ["bigcode/starcoder", "bigcode/starcoderbase"]:
                return f"<fim_prefix>{preprefix}{prefix}<fim_suffix>{suffix}<fim_middle>"
            else:
                raise ValueError(f"Infilling not yet supported for: {model_id}")

        def _make_instruction_prompt(self, instruction, context, prefix=""):
            """Make a prompt for instruction-tuning. Delimit instruction and
            context with specific tokens if provided.
            """
            if not instruction_tokens:
                warn(
                    "Instruction-tuning tokens are not provided for an "
                    "instruction-tuning task, we will leave them empty."
                )
                user_token, end_token, assistant_token = "", "", "\n"
            else:
                user_token, end_token, assistant_token = instruction_tokens
                if not user_token or not assistant_token or not end_token:
                    warn(
                        "Instruction-tuning tokens provided but one or more are empty. "
                        "Ignore warning if this was intended"
                    )
            return (
                prefix
                + user_token
                + instruction
                + end_token
                + assistant_token
                + context
            )

        # Extract stop words
        stopping_criteria = []
        if task.stop_words:
            for stop_word in task.stop_words:
                stopping_criteria.append(stop_word)

        prompts = []
        infill = False
        instruction = False
        mixed_error_log = (
            "Mixing tasks with infill/instruction "
            "and completion prompts is not supported."
        )
        for sample in range(limit_start, limit_start + n_tasks):
            prompt_contents = task.get_prompt(dataset[sample])
            if isinstance(prompt_contents, str):
                # Normal code completion mode
                if infill:
                    raise ValueError(mixed_error_log)
                instruction = True
                prompt = self.args.prefix + prompt_contents
            elif isinstance(prompt_contents, dict):
                if instruction:
                    raise ValueError(mixed_error_log)
                infill = True
                if set(prompt_contents.keys()) == {"prefix", "suffix"}:
                    # Infilling mode (Currently supported only for official InCoder and SantaCoder
                    # implementations.)
                    prompt = _make_infill_prompt(
                        **prompt_contents, preprefix=self.args.prefix
                    )
                elif set(prompt_contents.keys()) == {"instruction", "context"}:
                    # Instruction-tuning mode
                    prompt = _make_instruction_prompt(
                        **prompt_contents, prefix=self.args.prefix
                    )
            else:
                raise ValueError(
                    f"Unsupported prompt format: {type(prompt_contents)}"
                )
            prompts.append((prompt, deepcopy(stopping_criteria)))

        return prompts

    def generate_on_csx(
        self,
        task: Any,
        prompts: List[str],
        gen_kwargs: Dict[str, Any],
        n_tasks: int,
        limit_start: int = 0,
        intermediate_generations: Optional[
            List[Optional[List[Optional[str]]]]
        ] = None,
        instruction_tokens: Optional[str] = None,
    ) -> List[List[Optional[str]]]:
        """Generate code samples on CSX from the given prompts.

        Args:
            task: Code evaluation task object
            prompts: List of raw text prompts as processed by BCEH's script
            gen_kwargs: Dict specifying settings for generative inference
            n_tasks: Number of data samples
            limit_start: Offset to limit the number of samples. Defaults to 0.
            intermediate_generations: List of previously loaded generations. Defaults to None.
            instruction_tokens: List of instruction tokens used for instruction-tuning benchamrks.

        Returns:
           List of generated code samples
        """
        (
            samples_file_list,
            dataset_size,
            metadata,
        ) = self.preprocess_dataset(
            prompts,
            request_type=RequestType.bigcode_eh,
            max_tokens=gen_kwargs.get("max_tokens"),
        )

        # keep track of the list of generated codes
        # where len(code_gens) = n_tasks and len(code_gens[0]) = number of generated code samples
        code_gens: List[List[Optional[str]]] = [[] for _ in range(n_tasks)]
        generations = (
            [] if not intermediate_generations else intermediate_generations
        )

        # Generate tokens on appliance
        with GenerateTokens(metadata["requests"], gen_kwargs) as gen:
            self.trainer.validate(
                val_dataloader=cstorch.utils.data.DataLoader(
                    self.input_fn,
                    self.dataloader_args,
                    samples_file_list,
                    dataset_size,
                    RequestType.bigcode_eh.value,
                    **metadata["dataset_kwargs"],
                ),
                loop=BigCodeEvalHarnessLoop(),
                ckpt_path=None,
            )

        self.logger.debug(f"Output results: {gen.gen_token_dict}")

        code_gens = update_code_gens(
            task,
            self.tokenizer,
            limit_start,
            self.args.prefix,
            instruction_tokens,
            self.args.postprocess,
            code_gens,
            gen.gen_token_dict,
        )

        generations.extend(code_gens)

        return generations

    def generate_text(
        self,
        task_name: str,
        intermediate_generations: Optional[
            List[Optional[List[Optional[str]]]]
        ] = None,
    ) -> Tuple[List[List[str]], List[str]]:
        """Override of the BCEH's Evaluator class' method.

        Args:
            task_name: Name of the BigCode task to evaluate
            intermediate_generations: List of intermediate generations, if loaded

        Returns: Tuple of list of generated code samples and list of references
        """
        task: Task = bigcode_tasks.get_task(task_name, self.args)

        if (
            hasattr(task, "max_length_multiplier")
            and task.max_length_multiplier
        ):
            raise RuntimeError(
                f"BigCode task {task_name} specifies a max_length_multipler "
                f"stopping criterion, which is currently not supported. Please "
                f"choose a different task."
            )

        dataset = task.get_dataset()
        # if args.limit is None, use all samples
        # if args.limit is used, make sure args.limit_start + args.limit <= len(dataset)
        n_tasks = (
            min(self.args.limit, len(dataset) - self.args.limit_start)
            if self.args.limit
            else len(dataset)
        )
        # when args.limit is None
        # adjust n_tasks by args.limit_start to prevent out of bounds issues
        if not self.args.limit:
            n_tasks -= self.args.limit_start
        references = [
            task.get_reference(dataset[i])
            for i in range(
                self.args.limit_start, self.args.limit_start + n_tasks
            )
        ]

        if self.args.check_references:
            if (
                "get_solution"
                in inspect.signature(task.get_reference).parameters
            ):
                solutions = [
                    [task.get_reference(dataset[i], get_solution=True)]
                    for i in range(
                        self.args.limit_start, self.args.limit_start + n_tasks
                    )
                ]
            else:
                solutions = [[ref] for ref in references]
            return solutions, references

        curr_generations = []  # list[list[str | None] | None]
        if intermediate_generations:
            curr_generations = [gen for gen in intermediate_generations if gen]
            n_tasks -= len(curr_generations)

        curr_sample_idx = len(curr_generations)

        self.logger.info(f"Number of problems for this task is {n_tasks}")
        n_copies = ceil(self.args.n_samples / self.batch_size)
        limit_start = self.args.limit_start + curr_sample_idx

        if self.args.instruction_tokens:
            instruction_tokens = self.args.instruction_tokens.split(",")
            if len(instruction_tokens) != 3:
                raise ValueError(
                    "Instruction tokens should contain exactly 3 tokens "
                    "separated by a comma. If a token is empty, represent it as ''"
                )
            for token in instruction_tokens:
                if token.strip() != "":
                    task.stop_words.append(token)
        else:
            instruction_tokens = None

        # Set up generation settings
        gen_kwargs = {
            "do_sample": self.args.do_sample,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "max_tokens": self.args.max_tokens,
        }

        stopping_criteria = []
        if task.stop_words:
            for stop_word in task.stop_words:
                stopping_criteria.append(stop_word)

        if stopping_criteria:
            gen_kwargs["stopping_criteria"] = stopping_criteria

        # Fetch list of prompts
        prompts = self._construct_prompts(
            task,
            dataset,
            n_tasks=n_tasks,
            limit_start=limit_start,
            n_copies=n_copies,
            instruction_tokens=instruction_tokens,
        )

        # Generate tokens on CSX for the given prompts data
        generations = self.generate_on_csx(
            task,
            prompts,
            gen_kwargs=gen_kwargs,
            intermediate_generations=curr_generations,
            n_tasks=n_tasks,
            limit_start=limit_start,
            instruction_tokens=instruction_tokens,
        )

        if len(generations[0]) > self.args.n_samples:
            generations = [g[: self.args.n_samples] for g in generations]
            warn(
                f"Number of tasks wasn't proportional to number of devices, we "
                f"removed extra predictions to only keep nsamples={self.args.n_samples}"
            )
        return generations, references


class BigCodeEvalHarnessLoop(ValidationLoop):
    """Subclass of `ValidationLoop` to run BigCode's Evaluation Harness."""

    def __init__(self):
        """Initializes the BigCodeEvalHarnessLoop object."""
        super().__init__(hook="bigcode_eval_harness")

    def on_bigcode_eval_harness_start(
        self, trainer, model, val_dataloader, loop
    ):
        """
        Run ValidationLoop's `on_validate_start` method to ensure that
        eval_steps is being computed correctly.
        """
        model.eval()
        self.on_validate_start(trainer, model, val_dataloader, loop)


class GenerateTokens(Callback):
    """
    Callback class to post-process model output tokens.
    """

    def __init__(
        self,
        metadata: List[Tuple[int, int]],
        gen_kwargs: Dict[str, Any],
    ):
        """
        Args:
            metadata: List of tuples of (sample idx, prompt encoding length)
                for each sample in the batch
            gen_kwargs: Dict specifying settings for generative inference.
        """
        self.metadata = metadata
        self.start_token = None
        self.sample_idx = 0
        self.gen_token_dict = defaultdict(
            list
        )  # dict of list of generated tokens

        # Generation settings
        self.temperature = gen_kwargs.get("temperature")
        self.top_p = gen_kwargs.get("top_p")
        self.top_k = gen_kwargs.get("top_k")
        self.max_tokens = gen_kwargs.get("max_tokens")

    def on_bigcode_eval_harness_start(
        self, trainer, model, val_dataloader, loop
    ):
        """Runs before the BigCode Evaluation Harness starts."""
        self.start_token = getattr(model, "start_token", None)

        if self.start_token is None:
            raise RuntimeError(
                "No start token specified under `model.start_token`. "
                "Please specify a start token for generative tasks."
            )

        if self.max_tokens is not None:
            model.max_tokens = self.max_tokens

        if self.temperature is not None:
            model.temperature = self.temperature

        if self.top_p is not None:
            model.top_p = self.top_p

        if self.top_k is not None:
            model.top_k = self.top_k

    def on_bigcode_eval_harness_batch_end(
        self, trainer, model, outputs, batch, batch_idx
    ):  # pylint: disable=no-self-use
        """Runs after every batch is processed."""
        if progress := trainer.get_callback(ProgressLogger):
            progress.print_validation_progress(
                trainer, batch_idx, "BigCode Generative Eval"
            )

    def on_before_forward(self, trainer, model, batch, args, kwargs):
        kwargs["autoregressive"] = True

    def on_after_forward(self, trainer, model, outputs, batch):
        self.post_process(predictions=outputs["output"])

    @cstorch.step_closure
    def post_process(self, predictions):
        """
        Post-processes the model generated output tokens.

        Args:
            predictions: Tensor of shape (batch_size, max_seq_len)
                containing the model's predictions
        """
        for gen_tokens in predictions:
            if not self.metadata[self.sample_idx]:
                continue

            sample_idx, _ = self.metadata[self.sample_idx]
            assert sample_idx == self.sample_idx, "Mismatching sample indices"

            # Grab generation tokens
            try:
                start_token_idx = gen_tokens.tolist().index(self.start_token)
                gen_tokens = gen_tokens[:start_token_idx].numpy()
            except ValueError:  # Generated string spans msl
                pass

            self.gen_token_dict[sample_idx].append(gen_tokens)
            self.sample_idx += 1

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class BigCodeEvalHarness(ValidationCallback):
    """
    Callback class to run BigCode's Evaluation Harness.
    """

    id = 0

    def __init__(
        self,
        # BigCode Args
        bigcode_args: Union[BigCodeCLIArgs, Dict[str, Any]],
        # Cerebras specific args
        keep_data_dir: bool = False,
        every_n_vals: int = 1,
        flags: Optional[dict] = None,
        name_scope: Optional[str] = None,
        # Data Args
        batch_size: Optional[int] = None,
        data_dir: Optional[str] = None,
        max_sequence_length: Optional[int] = None,
        tokenizer_file_path: Optional[str] = None,
        eos_id: Optional[int] = None,
        **dataloader_args,
    ):
        """
        Args:
            bigcode_args: `BigCodeCLIArgs` dataclass or dict capturing BCEH's CLI args
            keep_data_dir: Specifies whether dumped data samples should be kept for reuse.
                Defaults to False, i.e. data samples are deleted after the run.
            every_n_vals: Run the BigCode eval harness script every N validations.
                e.g. If the eval_frequency is set to 200 and N=2,
                     then BigCode eval harness runs every 400 training steps.
                The BigCode eval harness script will also always run after the
                final training iteration.
            flags: A optional dictionary of scoped global flags to set
                during the BigCode eval harness run.
            name_scope: An optional string that gets added to the trainer's name scope.
            batch_size: Batch size to BigCodeEvalHarness to preprocess
                input data samples from the specified eval harness tasks.
            data_dir: Path to data directory
            max_sequence_length: Maximum sequence length
            tokenizer_file_path: Path to tokenizer file
            eos_id: End of sentence token id
            dataloader_args: Any additional dataloader args, e.g. num_workers.
        """
        # Handling parsing for creating trainer from yaml
        if isinstance(bigcode_args, dict):
            self.bigcode_args = BigCodeCLIArgs(**bigcode_args)
        else:
            self.bigcode_args = bigcode_args

        self.bceh_runner = BigCodeEvalHarnessRunner(
            bigcode_args=self.bigcode_args
        )

        self.dataloader_args = dict(
            batch_size=batch_size,
            data_dir=os.path.realpath(data_dir),
            keep_data_dir=keep_data_dir,
            max_sequence_length=max_sequence_length,
            tokenizer_file_path=tokenizer_file_path,
            eos_id=eos_id,
            **dataloader_args,
        )

        # Removes annoying logs relating to process forking
        appliance_environ["TOKENIZERS_PARALLELISM"] = "false"

        self._every_n_vals = every_n_vals

        self.scoped_flags = ScopedBigCodeEvalHarnessFlags(**(flags or {}))

        self._id = BigCodeEvalHarness.id
        BigCodeEvalHarness.id += 1

        if name_scope is None:
            name_scope = f"bigcode_{self._id}"

        self._name_scope = name_scope

    def run(self, trainer):
        """Run BigCode Eval Harness.

        Args:
            trainer: the Trainer object
        """
        trainer.logger.info("Running BigCode Eval Harness")

        # If no absolute file paths for output dumps are provided, dump inside model_dir
        if not os.path.isabs(self.bceh_runner.args.save_generations_path):
            self.bceh_runner.args.save_generations_path = os.path.join(
                trainer.summary_dir,
                trainer.name_scope_path,
                self.bceh_runner.args.save_generations_path,
            )
            os.makedirs(
                os.path.dirname(self.bceh_runner.args.save_generations_path),
                exist_ok=True,
            )
        if not os.path.isabs(self.bceh_runner.args.save_references_path):
            self.bceh_runner.args.save_references_path = os.path.join(
                trainer.summary_dir,
                trainer.name_scope_path,
                self.bceh_runner.args.save_references_path,
            )
            os.makedirs(
                os.path.dirname(self.bceh_runner.args.save_references_path),
                exist_ok=True,
            )

        bc_evaluator = BigCodeEvaluator(
            trainer,
            deepcopy(self.bigcode_args),
            deepcopy(self.dataloader_args),
        )

        with self.scoped_flags:
            self.bceh_runner.evaluate(trainer=trainer, evaluator=bc_evaluator)

    @property
    def name_scope(self):
        return self._name_scope

    @property
    def every_n_vals(self):
        return self._every_n_vals

    @property
    def num_validate_loops(self):
        return len(self.bceh_runner.task_names)

    def run_validation(self, trainer, loop_idx, is_last):
        if not is_last and (loop_idx + 1) % self.every_n_vals != 0:
            return

        with trainer.name_scope(self.name_scope):
            self.run(trainer)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class ScopedBigCodeEvalHarnessFlags(_ScopedFlags):
    """
    Class to set and restore global flags during the BigCode Evaluation
    Harness run.
    """

    def on_bigcode_eval_harness_start(
        self, trainer, model, val_dataloader, loop
    ):
        """Sets the global flags before the BigCode Evaluation Harness run."""
        self._set_all_flags()

    def on_bigcode_eval_harness_end(self, trainer, model, loop):
        """Restores the global flags after the BigCode Evaluation Harness run."""
        self._restore_all_flags()
