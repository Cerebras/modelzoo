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

""""Implementation of the RestartableTrainer wrapper process class."""

import atexit
import json
import logging
import multiprocessing as mp
import signal
import sys
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Set, Union

import yaml
from pydantic import BaseModel, Field, computed_field, field_serializer

import cerebras.pytorch as cstorch
from cerebras.appliance.cluster.client import TelemetryClient
from cerebras.appliance.errors import (
    ApplianceCompilationError,
    ApplianceDeadlockError,
    ApplianceNanError,
    ApplianceUnknownError,
    ClusterJobCancelledByCsctl,
)
from cerebras.appliance.log import ClassLogger, named_class_logger
from cerebras.modelzoo.config import BaseConfig
from cerebras.modelzoo.trainer.callbacks.artifact_dir import (
    create_timestamped_dir,
)
from cerebras.modelzoo.trainer.callbacks.autorestart import (
    TRAINER_STATE_FILENAME,
)
from cerebras.modelzoo.trainer.callbacks.logging import _CustomFormatter
from cerebras.modelzoo.trainer.utils import (
    ModeT,
    create_backend_from_config,
    is_legacy_params,
    mode_to_cmd,
    run_trainer,
)
from cerebras.modelzoo.trainer.validate import validate_trainer_params


def find_and_replace(node: Union[dict, list, tuple], key: Any, val: Any):
    """Utility to find all occurences of a given key and replace the associated value."""
    if isinstance(node, (list, tuple)):
        for i in node:
            find_and_replace(i, key, val)
    elif isinstance(node, dict):
        if key in node:
            node[key] = val
        for j in node.values():
            find_and_replace(j, key, val)


def _run_trainer(
    mode: ModeT,
    params: Dict[str, Any],
    pipe: mp.Pipe,
    logfile: Optional[Path] = None,
    run_number: Optional[int] = None,
    stdout_pipe: Optional[Path] = None,
):
    """Wrapper around the `run_trainer` util function to catch and return exceptions.

    Args:
        mode: The mode to run the Trainer in. Can be one of:
            - "train": Train the model.
            - "eval": Evaluate the model.
            - "train_and_eval": Train the model and then evaluate it.
            - "eval_all": Evaluate the model on all available checkpoints and dataloaders.
        params: A dictionary/object containing the configuration for the Trainer.
        pipe: A multiprocessing Pipe object to send the exception and traceback info back to
            the parent process in the event of any failure.
        logfile: (Optional) Path of consolidated auto-restart run log file.
        run_number: Current run attempt. This is used to create a unique prefix for the log file.
        stdout_pipe: (Optional) File path to pipe stdout and stderr to a file.
    """
    cstorch.backends.csx.cluster.auto_taint = True

    try:
        # add global handler to add logs to restartable logs
        if logfile is not None:
            logger = logging.getLogger()
            run_idx = 0 if run_number is None else run_number
            _setup_restartable_logging(logger, logfile, f"restart_{run_idx}")

        if stdout_pipe is not None:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = open(stdout_pipe, "a")
            sys.stderr = sys.stdout  # Redirect stderr to the same file

        run_trainer(mode, params)
    except Exception as e:  # pylint: disable=broad-except
        traceback_info = traceback.format_exc()
        pipe.send((e, traceback_info))
        raise
    finally:
        if stdout_pipe is not None:
            sys.stdout.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _setup_restartable_logging(
    logger: logging.Logger, logfile: Path, prefix: str
):
    handler = logging.FileHandler(logfile)
    # pylint: disable=protected-access
    fmt = f"[{prefix}] " + _CustomFormatter()._fmt
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    handler.is_sticky = True
    logger.addHandler(handler)


def _get_global_step(ckpt_path: str) -> Optional[int]:
    state_dict = cstorch.load(ckpt_path)
    return state_dict.get("global_step")


@dataclass
class RestartConfig:
    """Data structure representing the configuration for auto-restarting the trainer."""

    max_num_restarts: int
    "The maximum number of restarts allowed before the run is considered as failed."

    min_num_csx: int = 1
    "The minimum number of CSX systems with which to perform a restart."

    trainer_state_file: Optional[str] = None
    "The path to the trainer state checkpoint file to restart from."

    num_csx: Optional[int] = None
    "Current num_csx setting used. Subject to change based on available resources."

    failure_count: int = 0
    "The number of times the run has failed without progress."

    nan_checker_enabled: bool = False
    "Whether the NaN checker is enabled in the trainer configuration."

    @property
    def exceeded_failure_threshold(self) -> bool:
        """Determine whether the maximum failure threshold was exceeded."""
        return self.failure_count > self.max_num_restarts


class RunInstance(BaseModel):
    """Data structure respresenting information pertaining to an individual restart run."""

    class Config:
        arbitrary_types_allowed = True

    artifact_dir: str = ""
    """Directory containing artifacts for the run."""

    start_time: datetime = datetime.min
    """Start time of the run."""

    end_time: datetime = datetime.min
    """End time of the run."""

    status: Literal["success", "failed", "non_recoverable_failure"] = (
        "non_recoverable_failure"
    )
    """Result of the run. Note that 'failed' means the job is recoverable."""

    failure_reason: str = ""
    """Error message of the failure. Empty on 'success'."""

    failure_exception: Optional[Exception] = Field(exclude=True, default=None)
    """Exception that cause the failure."""

    starting_step: Optional[int] = None
    """Global step the run started at."""

    ending_step: Optional[int] = None
    """Global step the run ended at."""

    compile_job_ids: List[str] = []
    """Compile job ids of the run."""

    execute_job_ids: List[str] = []
    """Execute job ids of the run."""

    restart_config: Optional[RestartConfig] = Field(exclude=True, default=None)
    """A copy of the RestartConfig used in the run."""

    @computed_field
    @property
    def steps_progressed(self) -> Optional[int]:
        """Calculate the number of steps progressed since the last restart."""
        if self.starting_step is None or self.ending_step is None:
            return None
        return max(0, self.ending_step - self.starting_step)

    @computed_field
    @property
    def loaded_trainer_state_file(self) -> Optional[str]:
        """Trainer state file the run started from. Empty if it is the first run."""
        return (
            None
            if self.restart_config is None
            else self.restart_config.trainer_state_file
        )

    @computed_field
    @property
    def num_csx(self) -> Optional[int]:
        """Number of CSX systems specified for the run."""
        return (
            None if self.restart_config is None else self.restart_config.num_csx
        )

    @computed_field
    @property
    def nan_checker_enabled(self) -> bool:
        """Whether or not the NaN checker was enabled for the run."""
        return (
            self.restart_config is not None
            and self.restart_config.nan_checker_enabled
        )

    @field_serializer("start_time", "end_time")
    def serialize_datetime(self, dt: datetime, _info):
        return str(dt)

    def get_start_step(self) -> Optional[int]:
        """Get the starting step of the run. Used to compute steps_progressed."""
        if self.loaded_trainer_state_file:
            return _get_global_step(self.loaded_trainer_state_file)
        return None


class RunSummary(BaseModel):
    """A summary of a full auto-restart job."""

    class Config:
        arbitrary_types_allowed = True

    workflow_id: str = ""
    """Workflow ID for the auto-restart job."""

    max_num_restarts: int
    """The maximum number of restarts allowed before the run is considered as failed."""

    min_num_csx: int
    """The minimum number of CSX systems with which to perform a restart."""

    runs: List[RunInstance] = []
    """A list of all individual runs performed in the auto-restart job."""

    def save(self, json_file: Path):
        """Serializes and saves the sumary to a JSON file."""
        with json_file.open("w") as f:
            json.dump(self.model_dump(), f, sort_keys=False, indent=4)

    @property
    def previous_failure_exc(self) -> Optional[Exception]:
        """Return the last encountered failure exception (if any)."""
        return self.runs[-1].failure_exception if self.runs else None


class CompilePrefetcher:
    """Class to prefetch compile-only jobs in subprocesses."""

    class _JobStatus(Enum):
        """Enum to represent the status of a prefetch compile job."""

        IN_PROGRESS = auto()
        SUCCESS = auto()
        FAILED = auto()  # Compile failure
        IGNORED_FAILURE = auto()  # Other failures

    class _PrefetchJob:
        """Tracks a single prefetch compile process and its state for a num_csx config."""

        def __init__(
            self,
            num_csx: int,
            mode: ModeT,
            params: Dict[str, Any],
            model_dir: Path,
            mp_ctx: mp.context.BaseContext,
        ):
            self.num_csx = num_csx

            self._from_child, self._to_self = mp_ctx.Pipe()
            logfile = model_dir / f"prefetch_num_csx_{num_csx}.log"
            self._process = mp_ctx.Process(
                target=_run_trainer,
                args=(
                    mode,
                    params,
                    self._to_self,
                    None,
                    None,
                    logfile,  # stdout_pipe
                ),
            )
            self._process.daemon = True
            self._process.start()

            self.status = CompilePrefetcher._JobStatus.IN_PROGRESS
            logging.info(
                f"Spawned process {self._process.pid} to prefetch compile for num_csx={num_csx}"
            )

        @property
        def is_running(self) -> bool:
            """Check if the compile process is still alive."""
            return self._process.is_alive()

        def finalize(self) -> None:
            """Join process and check for errors."""
            self._process.join()  # Ensure proper completion

            if self._process.exitcode == 0:
                logging.info(
                    f"Prefetching compile for num_csx={self.num_csx} completed successfully."
                )
                self.status = CompilePrefetcher._JobStatus.SUCCESS
            elif abs(self._process.exitcode) == signal.SIGTERM:
                logging.info(
                    f"Prefetching compile for num_csx={self.num_csx} was terminated."
                )
                self.status = CompilePrefetcher._JobStatus.IGNORED_FAILURE
            else:
                logging.error(
                    f"Prefetching compile for num_csx={self.num_csx} failed with exit code {self._process.exitcode}"
                )
                self.status = CompilePrefetcher._JobStatus.IGNORED_FAILURE
                if self._from_child.poll(10):
                    try:
                        exc_obj, _ = self._from_child.recv()
                        if isinstance(exc_obj, ApplianceCompilationError):
                            self.status = CompilePrefetcher._JobStatus.FAILED
                    except Exception as e:
                        logging.error(
                            f"Failed to retrieve error from compile process: {e}"
                        )

        def terminate(self):
            """Terminates the compile process and closes the pipe."""
            if self.is_running:
                self._process.terminate()
                self._process.join()
            self._from_child.close()
            self._to_self.close()

    def __init__(
        self,
        mode,
        params: Dict[str, Any],
        model_dir: Path,
        min_num_csx: int = 1,
        mp_ctx: Optional[mp.context.BaseContext] = None,
    ):
        """
        Args:
            mode: The mode to run the Trainer in.
            params: The configuration for the Trainer.
            model_dir: Path to the model directory for the prefetch compile jobs.
            min_num_csx: The minimum num_csx config for the prefetch compile job. Defaults to 1.
            mp_ctx: (Optional) Multiprocessing context to create pipes and processes.
        """
        # Modify params for compile-only mode
        params["trainer"]["init"]["model_dir"] = str(model_dir)
        params["trainer"]["init"]["backend"]["compile_only"] = True
        find_and_replace(params, "csx.performance.micro_batch_size", "auto")

        self._params = params
        self._mode = mode
        self._model_dir = model_dir
        self._min_num_csx = min_num_csx
        self._mp_ctx = mp_ctx if mp_ctx is not None else mp.get_context('spawn')

        self.prefetched_compiles: Dict[int, CompilePrefetcher._PrefetchJob] = {}

        # Set up background monitor thread
        self._stop_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor_compile_jobs, daemon=True
        )
        self._monitoring_started = False

        # Register the cleanup method to be called on program exit
        atexit.register(self.cleanup)

    @property
    def num_csx_blacklist(self) -> Set[int]:
        """Get the set of num_csx configs that have failed to compile."""
        return {
            num_csx
            for num_csx, job in self.prefetched_compiles.items()
            if job.status == CompilePrefetcher._JobStatus.FAILED
        }

    @property
    def active_compile_jobs(self) -> Iterator[_PrefetchJob]:
        """Returns iterator over active prefetch compile jobs."""
        return filter(
            lambda job: job.status == CompilePrefetcher._JobStatus.IN_PROGRESS,
            self.prefetched_compiles.values(),
        )

    @property
    def has_active_compile_jobs(self) -> bool:
        """Check if there are any active compile jobs."""
        return any(
            job.status == CompilePrefetcher._JobStatus.IN_PROGRESS
            for job in self.prefetched_compiles.values()
        )

    def prefetch_compiles(
        self,
        num_csx: int,
    ) -> None:
        """Prefetch compile jobs in child processes for the next two `num_csx` configs.

        Args:
            num_csx: The number of CSX systems for the current run. Compiles will be prefetched for (num_csx - 1) and (num_csx - 2).
        """
        if (
            num_csx == self._min_num_csx
            or (num_csx - 1) in self.prefetched_compiles
            or (num_csx - 2) in self.prefetched_compiles
        ):
            return

        self._model_dir.mkdir(exist_ok=True)

        # Prefetch compile jobs for num_csx - 1 and num_csx - 2
        for prefetch_num_csx in range(
            max(num_csx - 1, self._min_num_csx),
            max(num_csx - 2, self._min_num_csx) - 1,
            -1,
        ):
            self._params["trainer"]["init"]["backend"]["cluster_config"][
                "num_csx"
            ] = prefetch_num_csx

            self.prefetched_compiles[prefetch_num_csx] = (
                CompilePrefetcher._PrefetchJob(
                    prefetch_num_csx,
                    self._mode,
                    self._params,
                    self._model_dir,
                    self._mp_ctx,
                )
            )

        if not self._monitoring_started:
            self._monitor_thread.start()
            self._monitoring_started = True

    def _monitor_compile_jobs(self):
        """Background thread that monitors running compile jobs and updates their status."""
        while not self._stop_event.is_set():
            for job in self.active_compile_jobs:
                if not job.is_running:
                    job.finalize()

            time.sleep(5)  # Check every 5 seconds

    def cleanup(self) -> None:
        """Terminate all running compile processes and clean up resources."""
        if self._monitoring_started:
            self._stop_event.set()
            self._monitor_thread.join()

        for job in self.prefetched_compiles.values():
            job.terminate()
        self.prefetched_compiles.clear()

        logging.debug("All prefetch compile processes have been cleaned up.")


@named_class_logger("RestartableTrainer")
class RestartableTrainer(ClassLogger):
    """Wrapper around Trainer to support auto-restarting jobs."""

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self.validated_configs = validate_trainer_params(params)

        if isinstance(params.get("trainer"), (tuple, list)):
            self.trainer_configs = params["trainer"]
        else:
            self.trainer_configs = [params]

        self._mp_ctx = mp.get_context('spawn')
        self._telemetry_client = None

    @staticmethod
    def is_restart_config(params: Dict[str, Any]) -> bool:
        """Check if autorestart is enabled for the given params configuration."""
        if is_legacy_params(params):
            return False
        if isinstance(params.get("trainer"), (tuple, list)):
            restart_configs = [
                RestartableTrainer.is_restart_config(trainer_params)
                for trainer_params in params["trainer"]
            ]
            if any(restart_configs):
                if not all(restart_configs):
                    raise ValueError(
                        "Either all or none of the trainer configurations should specify "
                        "an autorestart callback."
                    )
                return True
            return False
        else:
            return params["trainer"]["init"].get("autorestart") is not None

    def handle_exception(
        self,
        exc_obj: Exception,
        params: Dict[str, Any],
        restart_config: RestartConfig,
        run_summary: RunSummary,
        available_systems: Optional[int] = None,
    ) -> bool:
        """Handle the exception raised by the child process.

        Args:
            exc_obj: The exception object raised by the child process.
            params: The configuration for the Trainer to update if needed depending on the exception.
            restart_config: The current restart configuration.
            run_summary: A summary of the whole autorestart job. Used to get recent exception information.
            available_systems: Current number of available systems in the CSX cluster (if applicable).

        Returns:
            A boolean indicating whether the exception failure is non-recoverable.
        """
        non_recoverable_failure = False
        if isinstance(exc_obj, ApplianceCompilationError):
            self.logger.error(
                "Ran into a compilation error during the training run. "
                "Exiting the autorestart loop ..."
            )
            non_recoverable_failure = True
        elif isinstance(exc_obj, ClusterJobCancelledByCsctl):
            self.logger.error(
                "Cluster job was manually cancelled by `csctl cancel job`. "
                "Exiting the autorestart loop ..."
            )
            non_recoverable_failure = True
        elif isinstance(exc_obj, ApplianceUnknownError):
            self.logger.error(
                "An unknown error occurred during the training run. "
                "Attempting to restart the run ..."
            )
        elif isinstance(exc_obj, ApplianceNanError):
            self.logger.error(
                "The training run failed due to encountering NaN values."
            )
            if not restart_config.nan_checker_enabled:
                self.logger.info(
                    "Attempting to restart the run with the NaN checker enabled ..."
                )
                self.enable_nan_checker(
                    params,
                    num_steps=params["trainer"]["init"]
                    .get("checkpoint", {})
                    .get("steps"),
                    restart_config=restart_config,
                )
        elif isinstance(exc_obj, ApplianceDeadlockError):
            if (
                isinstance(run_summary.previous_failure_exc, ApplianceNanError)
                and restart_config.nan_checker_enabled
                and restart_config.num_csx is not None
                and available_systems is not None
            ):
                # Check whether a faulty system was found (and subsequently tainted).
                # If a system was tainted, the current number of available systems
                # is strictly less than the number of systems for the previous run
                if available_systems < restart_config.num_csx:
                    self.logger.info(
                        "The NaN Checker flagged potentially faulty system(s) as being "
                        "the likely cause of NaN values and moved them out of the usable "
                        "pool. Attempting to restart the run without these systems and "
                        "with the NaN Checker disabled ..."
                    )
                    self.disable_nan_checker(params, restart_config)
                else:
                    self.logger.info(
                        "The NaN Checker couldn't automatically detect any specific faulty "
                        "system. Attempting another restart with NaN Checker enabled ..."
                    )
            elif (
                isinstance(
                    run_summary.previous_failure_exc, ApplianceDeadlockError
                )
                and restart_config.nan_checker_enabled
            ):
                self.logger.info(
                    "The NaN Checker did not flag any particular system as being the "
                    "cause of NaNs. The NaNs may be due to a software or an ML-related "
                    "issue. Exiting the autorestart loop ..."
                )
                non_recoverable_failure = True
            else:
                self.logger.error(
                    "The training run failed due to an Appliance deadlock. "
                    "Attempting to restart the run ..."
                )
        elif isinstance(exc_obj, AssertionError):
            self.logger.error(
                "Encountered a failed assertion due to an invalid configuration. "
                "Exiting the autorestart loop ..."
            )
            non_recoverable_failure = True
        else:
            self.logger.error(
                f"Run failed due to an unknown exception: {exc_obj}\n"
                f"Attempting to restart the run ..."
            )

        return non_recoverable_failure

    @staticmethod
    def enable_nan_checker(
        params: Dict[str, Any],
        num_steps: Optional[int] = None,
        restart_config: Optional[RestartConfig] = None,
    ) -> None:
        """Enable the NaN checker in the trainer configuration.

        Args:
            params: The configuration for the Trainer to update.
            num_steps: (Optional) The number of steps for which to enable the NaN Checker.
            restart_config: (Optional) The current restart configuration.
        """
        params["trainer"]["init"].setdefault("callbacks", []).append(
            {"NaNChecker": {"num_steps": num_steps}}
        )
        if restart_config is not None:
            restart_config.nan_checker_enabled = True

    @staticmethod
    def disable_nan_checker(
        params: Dict[str, Any],
        restart_config: Optional[RestartConfig] = None,
    ) -> None:
        """Disable the NaN checker in the trainer configuration.

        Args:
            params: The configuration for the Trainer to update.
            restart_config: (Optional) The current restart configuration.
        """
        params["trainer"]["init"]["callbacks"] = [
            callback
            for callback in params["trainer"]["init"].setdefault(
                "callbacks", []
            )
            if next(iter(callback)) != "NaNChecker"
        ]
        if restart_config is not None:
            restart_config.nan_checker_enabled = False

    @staticmethod
    def get_trainer_state_file(model_dir: Path) -> Optional[str]:
        """Fetch the path of the trainer state file (if it exists) from the given model dir."""
        artifact_dir = (model_dir / "cerebras_logs" / "latest").resolve()
        if artifact_dir.exists():
            return next(
                map(
                    str,
                    artifact_dir.glob(TRAINER_STATE_FILENAME.format(step="*")),
                ),
                None,
            )
        return None

    def log_telemetry(
        self,
        workflow_id: str,
        run_instance: RunInstance,
        run_num: int = 0,
    ) -> None:
        """Push the run information to Grafana via the telemetry client.

        Args:
            workflow_id: The workflow ID of the current autorestart job.
            run_instance: The RunInstance object containing information about the current run.
            run_num: The current auto-restart attempt number, incrementing with each restart.
        """
        restart_run_info = {
            "run_number": str(run_num),
            "workflow_id": workflow_id,
        }
        # Job type mapping
        job_types = [
            ("compile", run_instance.compile_job_ids),
            ("execute", run_instance.execute_job_ids),
        ]

        # Codify failure category
        if (exc := run_instance.failure_exception) is not None:
            failure_category = type(exc).__name__
        else:
            failure_category = ""

        restart_failure_info = {
            **restart_run_info,
            "failure_reason": run_instance.failure_reason,
            "status": run_instance.status,
            "failure_category": failure_category,
        }
        with self._telemetry_client as client:
            # Push job info
            for job_type, job_ids in job_types:
                for job_id in job_ids:
                    client.push(
                        metrics={"auto_restart_job": run_num},
                        labels={
                            **restart_run_info,
                            "job_id": job_id,
                            "job_type": job_type,
                        },
                    )
            # Push failure info
            client.push(
                metrics={"auto_restart_reason": run_num},
                labels=restart_failure_info,
            )

    def _run_trainer_config(
        self,
        mode: ModeT,
        params: Dict[str, Any],
        config: BaseConfig,
    ):
        """Run the trainer configuration in a child process."""
        backend = create_backend_from_config(config.init)

        # Initialize cluster workflow on CSX backend
        if backend.is_csx:
            backend.cluster.start_workflow()
            # Use the same workflow id for jobs running in the subprocess
            params["trainer"]["init"].setdefault("backend", {}).setdefault(
                "cluster_config", {}
            )["workflow_id"] = backend.cluster.workflow_id

        try:
            model_dir = Path(config.init.model_dir).resolve()
            compile_prefetcher = None

            restartable_artifact_dir = create_timestamped_dir(
                model_dir, "_restartable"
            )
            logfile = restartable_artifact_dir / "run.log"
            params_file = restartable_artifact_dir / "params.yaml"
            with params_file.open("w") as f:
                yaml.dump(params, f, sort_keys=False)

            _setup_restartable_logging(self.logger, logfile, "main")

            ckpting_configured = (
                config.init.checkpoint is not None
                and config.init.checkpoint.steps
            )
            if mode in ("train", "train_and_eval") and not ckpting_configured:
                self.logger.warning(
                    f"The trainer configuration for {mode} does not configure "
                    f"checkpointing. In the event of a failure, the run will be "
                    f"restarted from the beginning. If this is not intentional, "
                    f"please specifiy a checkpoint callback with a checkpointing "
                    f"frequency to enable auto-restarting from previously saved "
                    f"checkpoints."
                )

            restart_config = RestartConfig(
                **params["trainer"]["init"]["autorestart"]
            )

            summary = RunSummary(
                max_num_restarts=restart_config.max_num_restarts,
                min_num_csx=restart_config.min_num_csx,
            )
            summary_file = restartable_artifact_dir / "summary.json"
            summary.workflow_id = (
                backend.cluster.workflow_id if backend.is_csx else ""
            )

            original_num_csx = None
            compile_job_ids = set()
            execute_job_ids = set()

            if backend.is_csx:
                original_num_csx = backend.cluster.config.num_csx
                if restart_config.min_num_csx > original_num_csx:
                    raise ValueError(
                        f"The minimum number of CSX systems specified in the auto-restart "
                        f"configuration `{restart_config.min_num_csx}` cannot be greater "
                        f"than the specified num_csx `{original_num_csx}` for the run."
                    )
                prefetch_compile_model_dir = (
                    restartable_artifact_dir / "prefetched_compiles"
                )
                compile_prefetcher = CompilePrefetcher(
                    mode,
                    deepcopy(params),
                    model_dir=prefetch_compile_model_dir,
                    min_num_csx=restart_config.min_num_csx,
                    mp_ctx=self._mp_ctx,
                )

                # Initialize telemetry client to propagate restart runs' info to Grafana dashboards
                self._telemetry_client = TelemetryClient(
                    mgmt_client=backend.cluster.client,
                    max_buffer_size=5,
                )

                for callback in getattr(config.init, "callbacks", []):
                    if callback.__class__.__name__ == "GlobalFlags":
                        # Set any global flags in the parent to be consistent with those set
                        # in the run in the child processes.
                        callback().pre_setup(trainer=None)

            def _fetch_current_jobs():
                """Helper to fetch the compile and execute job ids for the current run instance.

                NOTE: We fetch the current run's job ids by comparing against `compile_job_ids` and
                `execute_job_ids` specified above that hold job ids of all the runs so far in the workflow.
                """
                wf_compile_jobs, wf_execute_jobs = (
                    backend.cluster.client.list_workflow_jobs()
                )
                compile_jobs = [
                    job_id
                    for job_id in wf_compile_jobs
                    if job_id not in compile_job_ids
                ]
                execute_jobs = [
                    job_id
                    for job_id in wf_execute_jobs
                    if job_id not in execute_job_ids
                ]
                return compile_jobs, execute_jobs

            restart_config.num_csx = original_num_csx
            num_failures = 0
            while True:
                self.logger.info(
                    f"Spawning child process for the {mode_to_cmd(mode)} run ..."
                )

                # save existing summary
                summary.save(summary_file)

                run_instance = RunInstance(
                    start_time=datetime.now(),
                    restart_config=deepcopy(restart_config),
                )

                from_child, to_self = self._mp_ctx.Pipe()
                process = self._mp_ctx.Process(
                    target=_run_trainer,
                    args=(
                        mode,
                        params,
                        to_self,
                        logfile,
                        len(summary.runs) - 1,
                    ),
                )
                process.start()
                self.logger.info(
                    f"Child process started with pid: {process.pid}"
                )

                if backend.is_csx:
                    while process.is_alive():
                        _, execute_jobs = _fetch_current_jobs()
                        # Kick off prefetch compile jobs once the original compile has finished;
                        # i.e. when an execute job has been initialized for the current run
                        if execute_jobs:
                            compile_prefetcher.prefetch_compiles(
                                num_csx=restart_config.num_csx
                            )
                            break

                        time.sleep(5)  # Wait before checking again

                process.join()
                run_instance.end_time = datetime.now()

                artifact_dir = (
                    model_dir / "cerebras_logs" / "latest"
                ).resolve()
                run_instance.artifact_dir = str(artifact_dir)

                # Fetch current run's job ids
                if backend.is_csx:
                    (
                        run_instance.compile_job_ids,
                        run_instance.execute_job_ids,
                    ) = _fetch_current_jobs()

                    # Update job ids for all runs so far in the workflow
                    compile_job_ids.update(run_instance.compile_job_ids)
                    execute_job_ids.update(run_instance.execute_job_ids)

                # Compute starting step
                start_step = run_instance.get_start_step()
                if start_step is None and (
                    ckpt_path := params["trainer"]["fit"].get("ckpt_path")
                ):
                    start_step = _get_global_step(ckpt_path)
                run_instance.starting_step = (
                    start_step if start_step is not None else 0
                )

                # Find the last saved trainer state file in the model dir
                trainer_state_file = self.get_trainer_state_file(model_dir)

                # Compute ending step
                run_instance.ending_step = (
                    _get_global_step(trainer_state_file)
                    if trainer_state_file is not None
                    else run_instance.starting_step
                )

                try:
                    if process.exitcode == 0:
                        self.logger.debug("Child process succeeded!")
                        run_instance.status = "success"

                        if not restart_config.nan_checker_enabled:
                            break

                        # Find the last saved trainer state file in the artifact dir
                        if ckpting_configured:
                            if trainer_state_file is None:
                                raise RuntimeError(
                                    "No trainer state file found in the artifact dir. "
                                    "Exiting the autorestart loop ..."
                                )
                            # EDGE CASE: If the run completed successfully while NaN Checker was enabled,
                            # we should not restart the run.
                            is_final_step = (
                                (
                                    (num_steps := config.init.loop.num_steps)
                                    is not None
                                    and num_steps == run_instance.ending_step
                                )
                                or config.init.loop.max_steps
                                == run_instance.ending_step
                            )
                            if is_final_step:
                                self.logger.info(
                                    "Run completed successfully with NaN Checker enabled. "
                                    "No further restarts required."
                                )
                                break

                        # If the NaN Checker was enabled, disable it and restart the run from
                        # the new trainer state file
                        self.disable_nan_checker(params, restart_config)

                        restart_config.trainer_state_file = trainer_state_file

                        num_failures = 0  # Since we are restarting from a new checkpoint (progress)

                        self.logger.info(
                            f"Run with the NaN Checker enabled progressed beyond the step "
                            f"at which it had previously NaN'd, i.e. a new checkpoint was "
                            f"saved. Restarting from trainer state file with NaN Checker "
                            f"disabled: {trainer_state_file}"
                        )

                        params["trainer"]["fit"][
                            "ckpt_path"
                        ] = restart_config.trainer_state_file

                        continue
                    else:
                        self.logger.error(
                            f"Child process failed with with code: {process.exitcode}\n"
                        )

                        non_recoverable_failure = False
                        run_instance.status = "failed"
                        exc_obj = None
                        if from_child.poll(10):
                            try:
                                exc_obj, tb = from_child.recv()
                                self.logger.error(
                                    f"Run failed due to:\n{tb}\n\n"
                                )
                                run_instance.failure_reason = str(tb)
                            except (
                                Exception
                            ) as e:  # pylint: disable=broad-except
                                self.logger.error(
                                    f"Failed to extract exception and traceback info "
                                    f"from child process due to:\n{e}\n\n"
                                )

                        self.logger.error("Checking for restartability ...")

                        cluster_resources = (
                            backend.cluster.active_resources()
                            if backend.is_csx
                            else None
                        )

                        non_recoverable_failure = self.handle_exception(
                            exc_obj,
                            params,
                            restart_config,
                            summary,
                            cluster_resources,
                        )
                        if non_recoverable_failure:
                            raise RuntimeError(
                                f"{mode_to_cmd(mode)} run failed due to encountering a non-recoverable failure."
                            )

                        # Add the failure exception to the run instance
                        run_instance.failure_exception = exc_obj

                        # Check if there are sufficient cluster resources for restarting
                        if cluster_resources is not None:
                            self.logger.info(
                                f"Current number of available systems in the cluster: {cluster_resources}"
                            )
                            if cluster_resources < restart_config.min_num_csx:
                                raise RuntimeError(
                                    f"{mode_to_cmd(mode)} cannot restart due to an insufficient "
                                    f"number of resources. Current available system count is "
                                    f"{cluster_resources}, which is lower than the specified "
                                    f"min_num_csx threshold {restart_config.min_num_csx}."
                                )
                            if cluster_resources < original_num_csx:
                                self.logger.info(
                                    f"Requested num_csx {original_num_csx} but only "
                                    f"{cluster_resources} systems available. Attempting to restart "
                                    f"with the number of currently available systems ..."
                                )

                                # Set micro_batch_size to "auto" for perf reasons
                                find_and_replace(
                                    params,
                                    "csx.performance.micro_batch_size",
                                    "auto",
                                )

                            # Terminate any active compile jobs if there are insufficient systems for restart
                            for job in compile_prefetcher.active_compile_jobs:
                                if job.num_csx > cluster_resources:
                                    self.logger.info(
                                        f"Terminating active compile job for num_csx={job.num_csx} "
                                        f"because the current available cluster resources are {cluster_resources}"
                                    )
                                    job.terminate()

                            # Wait for all active compile jobs to finish before attempting to restart
                            while compile_prefetcher.has_active_compile_jobs:
                                time.sleep(5)

                            restart_num_csx = min(
                                original_num_csx, cluster_resources
                            )
                            num_csx_blacklist = (
                                compile_prefetcher.num_csx_blacklist
                            )
                            while restart_num_csx in num_csx_blacklist:
                                self.logger.info(
                                    f"num_csx {restart_num_csx} was blacklisted due to a failed prefetch compile-only job."
                                )
                                restart_num_csx -= 1

                            self.logger.info(
                                f"Attempting to restart with {restart_num_csx} systems ..."
                            )
                            if restart_num_csx < restart_config.min_num_csx:
                                raise RuntimeError(
                                    f"{mode_to_cmd(mode)} cannot restart with num_csx {restart_num_csx} because "
                                    f"it is lower than the specified min_num_csx threshold {restart_config.min_num_csx}"
                                )

                            # Update the system count on the cluster config in the params
                            params["trainer"]["init"]["backend"][
                                "cluster_config"
                            ]["num_csx"] = restart_num_csx

                            # Cache the num_csx used on the restart config
                            restart_config.num_csx = restart_num_csx

                        if mode in ("train", "train_and_eval"):
                            # Increment failure count for the first encountered train failure
                            if (
                                restart_config.trainer_state_file is None
                            ):  # First failure
                                if trainer_state_file is None:
                                    artifact_dir = (
                                        model_dir / "cerebras_logs" / "latest"
                                    ).resolve()
                                    raise RuntimeError(
                                        f"No trainer state file found in the artifact dir {artifact_dir} "
                                        f"This could be due to an early failure in the trainer before "
                                        f"any checkpoint could be saved, or due to the artifact directory "
                                        f"being cleaned up between runs. Exiting the autorestart loop ..."
                                    )
                                elif not num_failures:
                                    # Increment failure count for the first encountered train failure
                                    num_failures += 1

                            # Determine if the run progressed since the last restart
                            if (
                                restart_config.trainer_state_file is not None
                            ):  # Restarted run
                                if run_instance.steps_progressed:
                                    self.logger.info(
                                        "Run progressed from previous point of failure. "
                                        "Resetting the failure count."
                                    )
                                    num_failures = 0
                                else:
                                    self.logger.warning(
                                        f"No progress detected since last restart; i.e, a "
                                        f"new checkpoint was not saved."
                                    )
                                    num_failures += 1

                            restart_config.failure_count = num_failures
                            if restart_config.exceeded_failure_threshold:
                                raise RuntimeError(
                                    f"{mode_to_cmd(mode)} run failed due to there being no progress "
                                    f"after {restart_config.max_num_restarts} restart attempts."
                                )

                            # Update trainer state file on the restart config
                            if trainer_state_file is not None:
                                restart_config.trainer_state_file = (
                                    trainer_state_file
                                )

                            self.logger.info(
                                f"Auto-restarting from last saved trainer state file: "
                                f"{restart_config.trainer_state_file}"
                            )
                            params["trainer"]["fit"][
                                "ckpt_path"
                            ] = restart_config.trainer_state_file
                        else:
                            num_failures += 1

                            restart_config.failure_count = num_failures
                            if restart_config.exceeded_failure_threshold:
                                raise RuntimeError(
                                    f"{mode_to_cmd(mode)} run failed due to there being no progress "
                                    f"after {restart_config.max_num_restarts} restart attempts."
                                )

                            self.logger.info(
                                f"Auto-restarting {mode_to_cmd(mode)} run ..."
                            )
                except RuntimeError as e:
                    run_instance.failure_reason = str(e)
                    run_instance.status = "non_recoverable_failure"
                    raise e
                finally:
                    summary.runs.append(run_instance)

                    if self._telemetry_client is not None:
                        self.log_telemetry(
                            backend.cluster.workflow_id,
                            run_instance,
                            len(summary.runs),
                        )
        finally:
            if backend.is_csx:
                if compile_prefetcher is not None:
                    compile_prefetcher.cleanup()
                backend.cluster.stop_workflow()

            summary.save(summary_file)

    def run_trainer(self, mode: ModeT):
        """Run the trainer."""

        if len(self.trainer_configs) > 1:
            self.logger.info(
                "Multiple trainer configurations specified. Each "
                "config will run sequentially with auto-restart; "
                "i.e. the second one will only run if the first "
                "is successful, and so on."
            )

        for params, config in zip(self.trainer_configs, self.validated_configs):
            self._run_trainer_config(mode, deepcopy(params), config)
