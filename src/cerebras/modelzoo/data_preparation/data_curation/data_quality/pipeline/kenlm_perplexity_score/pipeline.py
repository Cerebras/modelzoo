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

from typing import Any, Dict, List, Type

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import (
    CSVReader,
    HuggingFaceDatasetReader,
    JsonlReader,
    ParquetReader,
)
from datatrove.pipeline.writers import (
    HuggingFaceDatasetWriter,
    JsonlWriter,
    ParquetWriter,
)
from datatrove.utils.logging import logger
from pydantic import BaseModel, Field, PrivateAttr, field_validator

from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.reader import (
    KenLMReader,
    TextReader,
)
from cerebras.modelzoo.data_preparation.data_curation.data_quality.pipeline.kenlm_perplexity_score.writer import (
    TextWriter,
)

reader_map = {
    'jsonl': JsonlReader,
    'csv': CSVReader,
    'parquet': ParquetReader,
    'txt': TextReader,
    'text': TextReader,
    'huggingface': HuggingFaceDatasetReader,
}

writer_map = {
    'jsonl': JsonlWriter,
    'parquet': ParquetWriter,
    'huggingface': HuggingFaceDatasetWriter,
    'txt': TextWriter,
    'text': TextWriter,
}


class BasePipeline(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    input_folder: str
    output_folder: str
    input_format: str = Field(default='jsonl')
    output_format: str = Field(default='jsonl')
    num_tasks: int = 1
    num_workers: int = 1
    limit: int = -1
    start_method: str = Field(default='spawn')
    executor_type: str = Field(default='local')
    pipelines: Dict[str, Dict] = Field(default_factory=dict)
    exec_pipelines: List[Type] = []
    _input_reader: Type = PrivateAttr()
    _output_writer: Type = PrivateAttr()

    # SLURM specific parameters
    executor: SlurmPipelineExecutor = None
    job_name: str = Field(default='kenlm_pipeline')
    partition: str = Field(default='cpu-spot')
    time: str = Field(default='01:00:00')
    qos: str = Field(default='normal')
    mem_per_cpu_gb: int = Field(default=4)
    cpus_per_task: int = Field(default=1)
    logs_folder: str = Field(default='slurm_logs')
    depends: SlurmPipelineExecutor = None

    @field_validator('input_format')
    def validate_input_format(cls, v):
        if v not in reader_map:
            raise ValueError(f"Input format {v} not supported")
        return v

    @field_validator('output_format')
    def validate_output_format(cls, v):
        if v not in writer_map:
            raise ValueError(f"Output format {v} not supported")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        self._input_reader = lambda *args, **kwargs: KenLMReader(
            *args, input_format=self.input_format, **kwargs
        )
        self._output_writer = writer_map[self.output_format]

    def run_local(self):
        logger.info("Starting local pipeline execution")
        for pipeline, pipeline_name in self.exec_pipelines:
            logger.info(f"Running pipeline {pipeline_name}")

            # Use run_config parameters for perplexity pipeline
            if pipeline_name.endswith("_perplexity"):
                tasks = self.config.run_config.num_tasks
                workers = self.config.run_config.num_workers
            else:
                tasks = self.num_tasks
                workers = self.num_workers

            executor = LocalPipelineExecutor(
                pipeline=pipeline,
                tasks=tasks,
                workers=workers,
                start_method=self.start_method,
                logging_dir=f"{self.output_folder}/logs/{pipeline_name}",
            )
            executor.run()

    def run_slurm(self, depends=None):
        logger.info("Starting SLURM pipeline execution")
        previous_executor = depends

        for pipeline, pipeline_name in self.exec_pipelines:
            logger.info(f"Running pipeline {pipeline_name}")

            # Create SLURM executor for this pipeline
            self.executor = SlurmPipelineExecutor(
                job_name=pipeline_name,
                pipeline=pipeline,
                tasks=self.num_tasks,
                time=self.time,
                partition=self.partition,
                logging_dir=f"{self.output_folder}/logs/{pipeline_name}",
                slurm_logs_folder=f"{self.logs_folder}/{pipeline_name}",
                qos=self.qos,
                mem_per_cpu_gb=self.mem_per_cpu_gb,
                cpus_per_task=self.cpus_per_task,
                # Only use depends for subsequent jobs in the chain, and only if the previous job succeeded
                depends=(
                    previous_executor
                    if previous_executor and previous_executor.job_id != -1
                    else None
                ),
                # Allow jobs to run even if their dependency failed
            )

            self.executor.run()
            previous_executor = self.executor

    def run(self, depends=None):
        if self.executor_type == 'local':
            self.run_local()
        elif self.executor_type == 'slurm':
            self.run_slurm(depends)
        else:
            raise ValueError(
                f"Executor type {self.executor_type} not supported"
            )

    def get_pipeline(self):
        return self._pipelines

    def add_pipelines(self, pipelines: Any):
        self.exec_pipelines.append(pipelines)
