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

import logging
import os

import numpy as np
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup.url_dedup import (
    UrlDedupConfig,
    UrlDedupFilter,
    UrlDedupSignature,
    UrlFindDedups,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

logger = logging.getLogger(__name__)


class CustomURLDedupPipelineStep(PipelineStep):
    """
    A custom pipeline step for URL deduplication that can be plugged into an existing pipeline.
    This step will handle URL deduplication without requiring external arguments.
    The user only needs to provide the input folder.
    """

    def __init__(self, input_folder: str):
        """
        Initialize the URL Deduplication step.
        The only required argument is the input_folder containing the JSONL files.
        """
        # Internal configuration for URL deduplication
        self.url_dedup_config = UrlDedupConfig(
            document_priority=lambda doc: min(
                np.iinfo(np.uint16).max, len(doc.text) // 4
            ),
            url_normalizer=lambda url: url.lower(),
        )

        # Set up the directories (sigs_dup_folder and base_output_folder) based on the input folder
        self.input_folder = input_folder
        self.sigs_dup_folder = os.path.join(
            input_folder, "sigs_dup"
        )  # Default to inside input_folder
        self.base_output_folder = os.path.join(
            input_folder, "output"
        )  # Default to inside input_folder
        self.limit = -1  # Default to no limit
        self.finder_workers = 8  # Default number of workers

        # Ensure necessary directories exist
        os.makedirs(self.sigs_dup_folder, exist_ok=True)
        os.makedirs(self.base_output_folder, exist_ok=True)

    def run(self, *args, **kwargs):
        """
        Execute the URL deduplication process using LocalPipelineExecutor.
        This method will not take external arguments and uses internal configuration.
        """

        # Validate directories
        self._validate_directory(self.input_folder)
        self._validate_directory(self.sigs_dup_folder)
        self._validate_directory(self.base_output_folder)

        try:
            # Pipeline 1: Generate signatures
            pipeline_1 = [
                JsonlReader(
                    data_folder=self.input_folder,
                    limit=self.limit,
                    doc_progress=True,
                ),
                UrlDedupSignature(
                    output_folder=f"{self.sigs_dup_folder}/sigs",
                    config=self.url_dedup_config,
                    finder_workers=self.finder_workers,
                ),
            ]

            # Pipeline 2: Find duplicates
            pipeline_2 = [
                UrlFindDedups(
                    data_folder=f"{self.sigs_dup_folder}/sigs",
                    output_folder=f"{self.sigs_dup_folder}/dups",
                    config=self.url_dedup_config,
                )
            ]

            # Pipeline 3: Deduplicate and output final results
            pipeline_3 = [
                JsonlReader(
                    data_folder=self.input_folder,
                    limit=self.limit,
                    doc_progress=True,
                ),
                UrlDedupFilter(
                    data_folder=f"{self.sigs_dup_folder}/dups",
                    config=self.url_dedup_config,
                    exclusion_writer=JsonlWriter(
                        output_folder=f"{self.base_output_folder}/removed"
                    ),
                ),
                JsonlWriter(output_folder=f"{self.base_output_folder}/output"),
            ]

            # Create LocalPipelineExecutor to execute each pipeline
            executor_1 = LocalPipelineExecutor(pipeline=pipeline_1, tasks=4)
            executor_2 = LocalPipelineExecutor(
                pipeline=pipeline_2, tasks=self.finder_workers
            )
            executor_3 = LocalPipelineExecutor(pipeline=pipeline_3, tasks=4)

            logger.info("Starting Pipeline 1: Signature generation")
            result_1 = executor_1.run()
            logger.info(f"Pipeline 1 completed: {result_1}")

            logger.info("Starting Pipeline 2: Finding duplicates")
            result_2 = executor_2.run()
            logger.info(f"Pipeline 2 completed: {result_2}")

            logger.info("Starting Pipeline 3: Deduplication and output")
            result_3 = executor_3.run()
            logger.info(f"Pipeline 3 completed: {result_3}")

            final_output_path = os.path.join(self.base_output_folder, "output")
            return final_output_path

        except Exception as e:
            logger.error(
                f"An error occurred during the URL deduplication process: {str(e)}"
            )
            raise

    def _validate_directory(self, path: str):
        """Ensure that the directory exists."""
        if not os.path.isdir(path):
            logger.error(f"Directory does not exist: {path}")
            raise FileNotFoundError(f"Directory does not exist: {path}")
