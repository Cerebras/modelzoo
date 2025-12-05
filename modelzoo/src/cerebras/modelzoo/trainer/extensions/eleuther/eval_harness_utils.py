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

"""Defines utils for running Eval Harness on CSX."""

import glob
import json
import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from warnings import warn

import numpy as np
from lm_eval import evaluator, utils
from lm_eval.__main__ import _int_or_none_list_arg_type
from lm_eval.api.model import LM
from lm_eval.api.task import Task
from lm_eval.evaluator_utils import get_task_list
from lm_eval.tasks import TaskManager, get_task_dict
from lm_eval.utils import handle_non_serializable, make_table

from cerebras.appliance.log import ClassLogger, named_class_logger

DEFAULT_RESULTS_FILE = "results.json"

SUPPORTED_MODELS = {
    "btlm",
    "bloom",
    "gpt2",
    "gptj",
    "falcon",
    "gpt3",
    "gpt-neox",
    "llama",
    "mistral",
    "mixtral",
    "mpt",
    "jais",
    "santacoder",
    "starcoder",
}


@dataclass
class EleutherCLIArgs:
    """Captures EEH's CLI arguments with defaults.

    Fields:
        tasks: List of tasks to evaluate
            To get full list of tasks, use the command ``lm-eval --tasks list``
        num_fewshot: Number of examples in few-shot context
        output_path: The path to the output file where the result metrics
            will be saved. If the path is a directory and log_samples is true,
            the results will be saved in the directory. Else the parent
            directory will be used.
        limit: Limit the number of examples per task.
            If <1, limit is a percentage of the total number of examples.
        use_cache: A path to a sqlite db file for caching model responses.
            `None` if not caching.
        cache_requests: Speed up evaluation by caching the building of
            dataset requests. `None` if not caching.
        check_integrity: Whether to run the relevant part of the test suite
            for the tasks.
        write_out: Prints the prompt for the first few documents.
        log_samples: If True, write out all model outputs and documents for
            per-sample measurement and post-hoc analysis. Use with
            --output_path.
        system_instruction: System instruction to be used in the prompt
        apply_chat_template: If True, applies the chat template to the prompt
        fewshot_as_multiturn: If True, uses the fewshot as a multi-turn conversation
        show_config: If True, shows the the full config of all tasks at the
            end of the evaluation.
        include_path: Additional path to include if there are external tasks
            to include.
        predict_only: Use with --log_samples. Only model outputs will be
            saved and metrics will not be evaluated.
        seed: Set seed for python's random, numpy and torch.
            Accepts a comma-separated list of 3 values for python's random,
            numpy, and torch seeds, respectively, or a single integer to set
            the same seed for all three. The values are either an integer
            or ``None`` to not set the seed. Default is ``0,1234,1234`` (for
            backward compatibility). E.g. ``--seed 0,None,8`` sets
            ``random.seed(0)`` and ``torch.manual_seed(8)``. Here numpy's seed
            is not set since the second value is ``None``.  E.g, ``--seed 42``
            sets all three seeds to 42.
        trust_remote_code: Sets trust_remote_code to True to execute code to
            create HF Datasets from the Hub
        verbosity: EEH logging level
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature used for generation.
        top_k: Top-k parameter used for generation.
        top_p: Top-p parameter used for nucleus sampling.
    """

    tasks: Union[str, List[str]]
    num_fewshot: Optional[int] = None
    output_path: Optional[str] = None
    limit: Optional[float] = None
    use_cache: Optional[str] = None
    cache_requests: Optional[Literal["true", "refresh", "delete"]] = None
    check_integrity: bool = False
    write_out: bool = False
    log_samples: bool = False
    system_instruction: Optional[str] = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    show_config: bool = False
    include_path: Optional[str] = None
    predict_only: bool = False
    seed: Union[int, str] = "0,1234,1234,1234"
    trust_remote_code: bool = False
    verbosity: str = "INFO"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

    def __post_init__(self):
        """Specially handle the seed."""
        # Special handling of `seed` arg
        self.seed = _int_or_none_list_arg_type(
            min_len=3,
            max_len=4,
            defaults="0,1234,1234,1234",
            value=str(self.seed),
        )


@named_class_logger("EvalHarnessRunner")
class EvalHarnessRunner(ClassLogger):
    """Util class for invoking EEH's run script with CSX-specific components."""

    def __init__(self, eeh_args: EleutherCLIArgs):
        """
        Args:
            eeh_args: Eval Harness CLI args.
        """
        super().__init__()

        self.args = deepcopy(eeh_args)
        self.task_manager: TaskManager = None
        self.task_names: Union[str, List[Union[str, Dict, Task]]] = []

        self.init_tasks()

    def init_tasks(self):
        # pylint: disable=line-too-long
        """Captures the task initialization logic from `Eleuther's run script <lm_eval_main>`_.

        .. _lm_eval_main: https://github.com/EleutherAI/lm-evaluation-harness/blob/4600d6bf73ba2cf7037ae7feada03315839ef185/lm_eval/__main__.py#L271-L307

        Includes CSX-specific validation for the user-specified eval harness tasks.
        """
        if self.args.include_path is not None:
            self.logger.info(
                f"Including path: {self.args.include_path} for externally created tasks."
            )
        task_manager = TaskManager(
            self.args.verbosity, include_path=self.args.include_path
        )

        if self.args.limit:
            self.logger.warning(
                " --limit should only be used for testing. "
                "Real metrics should not be computed using limit."
            )

        if self.args.tasks is None:
            raise ValueError("Need to specify task to evaluate.")
        else:
            if os.path.isdir(self.args.tasks):
                task_names = []
                yaml_path = os.path.join(self.args.tasks, "*.yaml")
                for yaml_file in glob.glob(yaml_path):
                    self.logger.info(f"Loading task from file: {yaml_file}")
                    config = utils.load_yaml_config(yaml_file)
                    task_names.append(config)
            else:
                task_list = self.args.tasks.split(",")
                task_names = task_manager.match_tasks(task_list)
                for task in [
                    task for task in task_list if task not in task_names
                ]:
                    if os.path.isfile(task):
                        config = utils.load_yaml_config(task)
                        task_names.append(config)
                task_missing = [
                    task
                    for task in task_list
                    if task not in task_names and "*" not in task
                ]  # we don't want errors if a wildcard ("*") task name was used

                if task_missing:
                    missing = ", ".join(task_missing)
                    raise ValueError(
                        f"Tasks not found: {missing}.\n"
                        f"{utils.SPACING} Try `lm-eval --tasks "
                        f"{{list_groups,list_subtasks,list_tags,list}}` to "
                        f"list out all available names for task groupings; "
                        f"only (sub)tasks; tags; or all of the above, or pass "
                        f"'--verbosity DEBUG' to troubleshoot task registration issues."
                    )

        # Validate tasks and cache task related properties.
        self.task_names = EvalHarnessRunner.validate_and_sort_tasks(
            task_names, task_manager
        )
        self.task_manager = task_manager

    @cached_property
    def task_dict(self) -> Dict[str, Any]:
        """Returns the task dictionary for the specified tasks."""
        return get_task_dict(self.task_names, self.task_manager)

    @staticmethod
    def validate_and_sort_tasks(
        task_names: Union[str, List[Union[str, Dict, Task]]],
        task_manager: Optional[TaskManager] = None,
    ) -> List[str]:
        """Validate and sort user specification of eval harness tasks on CSX.

        We currently do not support tasks with `loglikelihood_rolling` output types.

        Args:
            task_names: List of task names or config dicts
            task_manager: TaskManager object that stores indexed tasks

        Returns:
            List of sorted task names.
        """
        # Get the nested hierarchy of tasks and groups
        task_dict = get_task_dict(task_names, task_manager)
        # Flatten the task list
        task_list = get_task_list(task_dict)

        # Separate out generative and non-generative tasks
        gen_tasks, non_gen_tasks = [], []
        for task_output in task_list:
            task: Task = task_output.task
            if task.OUTPUT_TYPE == "loglikelihood_rolling":
                raise RuntimeError(
                    f"Tasks with `loglikelihood_rolling` output types are not yet supported."
                    f"Please unspecify task {task_output.task_name} from the specified tasks list "
                    f"or the task group that it belongs to."
                )
            elif task.OUTPUT_TYPE == "generate_until":
                gen_tasks.append(task_output.task_name)
            else:
                non_gen_tasks.append(task_output.task_name)

        # Put non generative task names after generative so EH will execute them first.
        # This is needed minimize the amount of appliance restarts, so train->non_generative
        # will use the same appliance.
        return gen_tasks + non_gen_tasks

    def evaluate(self, trainer, model: LM) -> dict:
        # pylint: disable=line-too-long
        """Invoke's evaluation logic from `EEH's run script <lm_eval_main>`_ on the given model.

        .. _lm_eval_main: https://github.com/EleutherAI/lm-evaluation-harness/blob/4600d6bf73ba2cf7037ae7feada03315839ef185/lm_eval/__main__.py#L240

        Args:
            trainer: The Trainer object to log to.
            model: The language model object (subclass of EEH's LM abstract base class)
        """
        if self.args.predict_only:
            self.args.log_samples = True
        if (
            self.args.log_samples or self.args.predict_only
        ) and not self.args.output_path:
            self.args.output_path = (
                trainer.summary_dir / trainer.name_scope_path
            )

        if self.args.output_path:
            path = Path(self.args.output_path)
            if not path.is_absolute():
                path = trainer.summary_dir / trainer.name_scope_path / path
                if path.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)

            # check if file or 'dir/results.json' exists
            if path.is_file():
                raise FileExistsError(f"File already exists at {path}")
            output_path_file = path.joinpath(DEFAULT_RESULTS_FILE)
            output_path_file = (
                output_path_file.parent
                / f"{output_path_file.stem}_{trainer.global_step}{output_path_file.suffix}"
            )
            if output_path_file.is_file():
                self.logger.warning(
                    f"File {output_path_file} already exists. Results will be overwritten."
                )
            # if path json then get parent dir
            elif path.suffix in (".json", ".jsonl"):
                output_path_file = path
                path.parent.mkdir(parents=True, exist_ok=True)
                path = path.parent
            else:
                output_path_file = output_path_file.resolve()
                path = path.resolve()
                path.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Starting Eleuther evaluation harness on selected tasks: {self.task_names}"
        )

        request_caching_args = evaluator.request_caching_arg_to_dict(
            cache_requests=self.args.cache_requests
        )

        # Set generative inference settings
        gen_kwargs = {
            "temperature": self.args.temperature,
            "top_k": self.args.top_k,
            "top_p": self.args.top_p,
            "max_tokens": self.args.max_tokens,
        }
        model.gen_kwargs = gen_kwargs

        results = evaluator.simple_evaluate(
            model=model,
            tasks=self.task_names,
            num_fewshot=self.args.num_fewshot,
            use_cache=self.args.use_cache,
            limit=self.args.limit,
            check_integrity=self.args.check_integrity,
            write_out=self.args.write_out,
            log_samples=self.args.log_samples,
            system_instruction=self.args.system_instruction,
            apply_chat_template=self.args.apply_chat_template,
            fewshot_as_multiturn=self.args.fewshot_as_multiturn,
            task_manager=self.task_manager,
            verbosity=self.args.verbosity,
            predict_only=self.args.predict_only,
            random_seed=self.args.seed[0],
            numpy_random_seed=self.args.seed[1],
            torch_random_seed=self.args.seed[2],
            fewshot_random_seed=self.args.seed[3],
            **request_caching_args,
        )

        if results is not None:
            if self.args.log_samples:
                samples = results.pop("samples")
            dumped = json.dumps(
                results,
                indent=2,
                default=handle_non_serializable,
                ensure_ascii=False,
            )
            if self.args.show_config:
                self.logger.info(dumped)

            batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
            batch_size = None
            model_args = None

            try:
                self.log_eval_results(trainer, results)
                if self.args.log_samples:
                    self.log_eval_samples(trainer, samples, results)
            except Exception as e:  # pylint: disable=broad-except
                self.logger.error(
                    f"Logging eval results/samples failed due to: {e}"
                )

            if self.args.output_path is not None:
                self.logger.info(
                    f"Saving Eleuther Eval Harness results to {output_path_file}"
                )
                with output_path_file.open("w", encoding="utf-8") as f:
                    f.write(dumped)

                if self.args.log_samples:
                    for task_name, _ in results["configs"].items():
                        filename = path.joinpath(
                            f"{task_name}_{trainer.global_step}.json"
                        )
                        samples_dumped = json.dumps(
                            samples[task_name],
                            indent=2,
                            default=handle_non_serializable,
                            ensure_ascii=False,
                        )
                        filename.write_text(samples_dumped, encoding="utf-8")

            self.logger.info(
                f"{model} ({model_args}), gen_kwargs: ({gen_kwargs}), "
                f"limit: {self.args.limit}, num_fewshot: {self.args.num_fewshot}, "
                f"batch_size: {batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
            )
            self.logger.info("\n" + make_table(results))
            if "groups" in results:
                self.logger.info("\n" + make_table(results, "groups"))

    def log_eval_results(self, trainer, results: Dict[str, Any]) -> None:
        """Logs the evaluation results to the trainer."""

        results = deepcopy(results)

        # TODO: Do we need to update the wandb config?
        # configs = {
        #     "task_configs": results.get("configs", {}),
        #     "cli_configs": results.get("config", {}),
        # }
        # wandb.run.config.update(configs)

        pattern = re.compile(r",none$")

        # Log the evaluation metrics
        trainer.log_metrics(
            **{
                # Remove None from the metric string name
                pattern.sub("", f"{task_name}/{metric_name}"): metric_value
                for task_name, task_value in results.get("results", {}).items()
                for metric_name, metric_value in task_value.items()
            }
        )

        self.log_eval_results_as_table(trainer, results)

        # Log the results dict as json
        self.log_as_json(trainer, "eval_results", results)

    def log_eval_results_as_table(  # pylint: disable=line-too-long
        self, trainer, results: Dict[str, Any]
    ) -> None:
        """Logs the eval results as a table to the trainer's loggers.

        Note, this method is adapted to construct a pandas DataFrame off the
        `original WandB specific implementation <log_table>`_ in EEH.

        .. _log_table: https://github.com/EleutherAI/lm-evaluation-harness/blob/3fa4fd725c8a428710109f1d6c14eda37e95baea/lm_eval/loggers/wandb_logger.py#L112-L160
        """
        try:
            import pandas as pd
        except ImportError:
            warn("Pandas not installed. Skipping logging of results as table.")
            return

        group_names = list(results.get("groups", {}))

        def make_dataframe(column1: str, key: str = "results"):
            data = []

            for k, dic in results.get(key).items():
                if k in group_names and key != "groups":
                    continue

                version = results.get("versions").get(k)
                if version == "N/A":
                    version = None

                num_fewshot = results.get("n-shot").get(k)

                for metric_filter, value in dic.items():
                    # pylint: disable=redefined-builtin
                    metric, _, filter = metric_filter.partition(",")
                    if metric.endswith("_stderr") or metric == "alias":
                        continue

                    if f"{metric}_stderr,{filter}" in dic:
                        stderr = dic[f"{metric}_stderr,{filter}"]
                        if stderr != "N/A":
                            stderr = f"{stderr:.4f}"
                    else:
                        stderr = ""

                    data.append(
                        {
                            column1: k,
                            "Version": version,
                            "Filter": filter,
                            "num_fewshot": num_fewshot,
                            "Metric": metric,
                            "Value": str(value),
                            "Stderr": str(stderr),
                        }
                    )

            return pd.DataFrame(data=data)

        if "results" in results:
            trainer.log_metrics(
                **{
                    "evaluation/eval_results": make_dataframe(
                        "Tasks", "results"
                    )
                }
            )

        if "groups" in results:
            trainer.log_metrics(
                **{
                    "evaluation/group_eval_results": make_dataframe(
                        "Groups", "groups"
                    )
                }
            )

    def log_as_json(self, trainer, key, results: Dict[str, Any]):
        """Serializes the results dict as json and logs it to the trainer."""

        def _handle_non_serializable(o: Any) -> Union[int, str, list]:
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        trainer.log_metrics(
            **{
                key: json.dumps(
                    results,
                    indent=4,
                    default=_handle_non_serializable,
                    ensure_ascii=False,
                )
            }
        )

    def log_eval_samples(
        self, trainer, samples: Dict[str, Any], results: Dict[str, Any]
    ) -> None:
        """Logs the evaluation samples to the trainer."""
        try:
            import pandas as pd
        except ImportError:
            warn("Pandas not installed. Skipping logging of eval samples")
            return

        samples = deepcopy(samples)

        def generate_dataset(*args, **kwargs) -> pd.DataFrame:
            from lm_eval.loggers import WandbLogger

            # Its okay to pass in `None` as self as this method
            # has no self uses
            # pylint: disable=protected-access
            return WandbLogger._generate_dataset(None, *args, **kwargs)

        group_names = list(results.get("groups", {}))
        task_names = [
            x for x in results.get("results", {}) if x not in group_names
        ]

        ungrouped_tasks = []
        tasks_by_groups = defaultdict(list)

        task_configs = results.get("configs", {})

        for task_name in task_names:
            group_names = task_configs[task_name].get("group", None)
            if group_names:
                if isinstance(group_names, str):
                    group_names = [group_names]

                for group_name in group_names:
                    tasks_by_groups[group_name].append(task_name)
            else:
                ungrouped_tasks.append(task_name)

        for task_name in ungrouped_tasks:
            eval_preds = samples[task_name]

            trainer.log_metrics(
                **{
                    # log the samples as a table
                    f"{task_name}_eval_results": generate_dataset(
                        eval_preds,
                        task_configs.get(task_name),
                    ),
                }
            )
            # Log the samples dict as json
            self.log_as_json(trainer, f"{task_name}_eval_samples", eval_preds)

        for group, grouped_tasks in tasks_by_groups.items():
            grouped_df = pd.DataFrame()
            for task_name in grouped_tasks:
                eval_preds = samples[task_name]
                df = generate_dataset(eval_preds, task_configs.get(task_name))
                df["group"] = group
                df["task"] = task_name
                grouped_df = pd.concat([grouped_df, df], ignore_index=True)

                # Log the samples dict as json
                self.log_as_json(
                    trainer, f"{task_name}_eval_samples", eval_preds
                )

            trainer.log_metrics(**{f"{group}_eval_results": grouped_df})
