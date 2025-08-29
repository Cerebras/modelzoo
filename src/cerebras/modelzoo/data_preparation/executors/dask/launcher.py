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
A Dask-based executor for running pipeline steps on a SLURM cluster.

This is meant to run as an alternative to Datatrove's SLURM executor
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Callable, List

from dask_jobqueue import SLURMCluster
from datatrove.executor.base import PipelineExecutor
from datatrove.io import DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from distributed import Client


class TaskExecutor(PipelineExecutor):
    """Single task executor for running pipeline steps on a worker node."""

    def __init__(
        self,
        pipeline: List[PipelineStep | Callable],
        logging_dir: DataFolderLike,
        world_size: int,
    ):
        super().__init__(pipeline, logging_dir)
        self._world_size = world_size

    @property
    def world_size(self) -> int:
        return self._world_size

    def run(self):
        """Not implemented as this executor only runs single tasks."""


def run_pipeline_task(
    pipeline_steps: List[PipelineStep | Callable],
    logging_dir: DataFolderLike,
    rank: int,
    world_size: int,
) -> any:
    """Execute a single pipeline task on a worker node."""

    executor = TaskExecutor(pipeline_steps, logging_dir, world_size)
    return executor._run_for_rank(rank)


class DaskPipelineExecutor(PipelineExecutor):
    """Distributed pipeline executor using Dask on a SLURM cluster.

    Handles distribution of pipeline tasks across SLURM workers using Dask,
    with support for job dependencies, retries, and robust error handling.

    Args:
        pipeline: List of pipeline steps to execute
        tasks: Total number of parallel tasks to run
        time: SLURM job time limit (e.g. "24:00:00")
        partition: SLURM partition to use
        cpus_per_task: CPUs allocated per task
        mem_per_cpu_gb: Memory in GB per CPU
        workers: Number of concurrent workers (-1 for unlimited, defaults to num of tasks)
        processes: Cut the job up into this many processes
        job_name: Name for SLURM job
        qos: SLURM QOS level
        env_command: Command to activate Python environment
        logging_dir: Directory for logs and output
        skip_completed: Skip already completed tasks
        depends: Dependency pipeline that must complete first
        timeout: Maximum runtime per task
        retry_attempts: Number of retry attempts on failure
        heartbeat_interval: Worker heartbeat check interval
        wait_for_workers: Add a blocking call to wait for atleast n number of workers before continuing
    """

    def __init__(
        self,
        pipeline: List[PipelineStep | Callable],
        tasks: int,
        time: str,
        partition: str,
        cpus_per_task: int = 1,
        mem_per_cpu_gb: int = 2,
        workers: int = -1,
        processes: int = 1,
        job_name: str = "dask_processing",
        qos: str = "normal",
        env_command: str | None = None,
        logging_dir: DataFolderLike | None = None,
        skip_completed: bool = True,
        depends: DaskPipelineExecutor | None = None,
        timeout: str = "24h",
        retry_attempts: int = 3,
        heartbeat_interval: str = "30m",
        wait_for_workers: int = -1,
    ):
        super().__init__(pipeline, logging_dir, skip_completed)
        self.tasks = tasks
        self.workers = workers if workers > 0 else tasks
        self.processes = processes
        self.partition = partition
        self.cpus_per_task = cpus_per_task
        self.mem_per_cpu_gb = mem_per_cpu_gb
        self.time = time
        self.job_name = job_name
        self.qos = qos
        self.env_command = env_command
        self.cluster = None
        self.client = None
        self.depends = depends
        self._future = None
        self.client_timeout = timeout
        self.retry_attempts = retry_attempts
        self.heartbeat_interval = heartbeat_interval
        self.wait_for_workers = (
            wait_for_workers if wait_for_workers >= 0 else self.workers
        )

    def setup_cluster(self):
        """Initialize SLURM cluster and Dask client."""
        if self.cluster is None:
            slurm_logs_dir = self.logging_dir.resolve_paths("slurm_logs")
            os.makedirs(slurm_logs_dir, exist_ok=True)

            job_extra_directives = [
                f"--qos={self.qos}",
                f"--output={slurm_logs_dir}/slurm-%j.out",
                f"--error={slurm_logs_dir}/slurm-%j.out",
                "--requeue",
                "--open-mode=append",
                f"--time={self.time}",
            ]

            self.cluster = SLURMCluster(
                queue=self.partition,
                cores=self.cpus_per_task,
                memory=f"{self.mem_per_cpu_gb}GB",
                walltime=self.time,
                name=self.job_name,
                job_extra_directives=job_extra_directives,
                python=self.env_command if self.env_command else "python3",
                local_directory=self.logging_dir.resolve_paths(
                    "dask-worker-space"
                ),
                death_timeout=self.heartbeat_interval,
                processes=self.processes,
            )

            logger.info(f"Scaling cluster to {self.workers} workers")
            self.cluster.scale(jobs=self.workers)

            self.client = Client(
                self.cluster,
                timeout=self.client_timeout,
                asynchronous=False,
                direct_to_workers=True,
                heartbeat_interval=self.heartbeat_interval,
            )
            logger.info(
                f"Dask dashboard available at: {self.client.dashboard_link}"
            )

            if self.wait_for_workers > 0:
                logger.info(
                    f"Waiting for {self.wait_for_workers} workers to be available..."
                )
                self.cluster.wait_for_workers(self.workers)

    def run(self):
        """Execute the pipeline with dependency handling"""
        try:
            if self.depends:
                logger.info(f"Running dependency job: {self.depends.job_name}")
                self.depends.run()
                if (
                    not hasattr(self.depends, '_future')
                    or not self.depends._future
                ):
                    raise RuntimeError(
                        f"Dependency job {self.depends.job_name} failed"
                    )

            self.setup_cluster()

            ranks_to_run = self.get_incomplete_ranks()
            if len(ranks_to_run) == 0:
                logger.info(f"All {self.tasks} tasks already completed")
                self._future = []
                return

            logger.info(
                f"Processing {len(ranks_to_run)} tasks with {self.workers} concurrent workers"
            )
            serializable_pipeline = deepcopy(self.pipeline)

            futures = []
            for rank in ranks_to_run:
                # Pin task to specific worker using resources
                future = self.client.submit(
                    run_pipeline_task,
                    serializable_pipeline,
                    self.logging_dir,
                    rank,
                    self.tasks,
                    pure=False,
                    retries=3,
                )
                futures.append(future)

            # Wait for all tasks to complete
            try:
                results = self.client.gather(futures)
                self._future = results
                logger.info("All tasks completed successfully")
            except Exception as e:
                logger.error(f"Task execution failed: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up Dask resources."""
        try:
            if self.client:
                logger.info("Shutting down Dask client...")
                self.client.close()
                self.client = None
            if self.cluster:
                logger.info("Shutting down SLURM cluster...")
                self.cluster.close()
                self.cluster = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    @property
    def world_size(self) -> int:
        """Total number of tasks in the pipeline."""
        return self.tasks
