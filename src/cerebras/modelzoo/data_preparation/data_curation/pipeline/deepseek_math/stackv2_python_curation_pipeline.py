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

#!/usr/bin/env python3
"""
Example usage of the Stack v2 Extraction Pipeline Step with datatrove.

This example shows how to set up and run a pipeline that processes
Stack v2 data (such as code and posts) and extracts structured data using
StackV2-specific extractors and filters.
"""
import glob
import os
import time

from data_curation.pipeline.deepseek_math.extractors import StackV2Extractor
from data_curation.pipeline.deepseek_math.filters import (
    StackV2_SmolIds_PythonFilter,
)
from datatrove.executor import SlurmPipelineExecutor

# Import your custom pipeline steps
from datatrove.pipeline.readers.jsonl import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the StackV2 python extraction pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from a YAML file."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def wait_for_completions(completions_dir, expected_tasks, timeout_seconds=3600):
    """Wait for all tasks to complete with timeout."""
    start_time = time.time()

    while True:
        if not os.path.exists(completions_dir):
            print(
                f"Completions directory {completions_dir} does not exist yet..."
            )
            time.sleep(10)
            continue

        num_completions = len(
            [
                f
                for f in os.listdir(completions_dir)
                if os.path.isfile(os.path.join(completions_dir, f))
            ]
        )
        print(f"Completions: {num_completions}/{expected_tasks}")

        if num_completions >= expected_tasks:
            return True

        # Check timeout
        if time.time() - start_time > timeout_seconds:
            print(
                f"Timeout reached after {timeout_seconds} seconds. Only {num_completions}/{expected_tasks} tasks completed."
            )
            return False

        time.sleep(10)  # Check every 10 seconds instead of every second


def main():
    """Example of how to run the StackV2 python code extraction pipeline."""
    # Parse arguments and load config once

    args = parse_args()
    config = load_config(args.config)

    # Get parameters from config with defaults
    start_chunk = config.get("pipeline", {}).get("start_chunk", 0)
    end_chunk = config.get("pipeline", {}).get("end_chunk", 63)
    timeout_seconds = config.get("pipeline", {}).get("timeout_seconds", 3600)

    for chunk_num in range(start_chunk, end_chunk):
        subdir_config = config.copy()
        # Define the pipeline

        if chunk_num < 10:
            subdir = f"0{chunk_num}"
        else:
            subdir = f"{chunk_num}"
        pipeline = [
            # Reader: finds PyPI project directories
            JsonlReader(
                data_folder=config["reader"]["input_folder"],
                text_key=config["reader"]["text_key"],
                glob_pattern=f"train-000{subdir}-of-00064-output_chunk_*.jsonl.zst",  # Assumes structure: project_name/version/
            ),
            StackV2_SmolIds_PythonFilter(),
            # Processor: extracts structured data from projects
            StackV2Extractor(),
            # Writer: saves the results to JSONL files with all fields preserved
            JsonlWriter(
                output_folder=os.path.join(
                    config["writer"]["output_folder"], f"subdir{subdir}"
                ),
                # filename="pypi_extracted_data.jsonl",
                # compression=True  # Creates .jsonl.gz files
            ),
        ]

        # Create subdirectory-specific paths
        logging_dir = os.path.join(
            config["writer"]["output_folder"], f"subdir{subdir}", "logs"
        )
        slurm_logs_folder = os.path.join(
            config["writer"]["output_folder"], f"subdir{subdir}", "slurm_logs"
        )

        # Create directories if they don't exist
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(slurm_logs_folder, exist_ok=True)

        # Update config for this subdirectory
        subdir_slurm_config = config['slurm_executor'].copy()
        subdir_slurm_config['logging_dir'] = logging_dir
        subdir_slurm_config['slurm_logs_folder'] = slurm_logs_folder
        subdir_slurm_config['tasks'] = len(
            glob.glob(
                os.path.join(
                    config["reader"]["input_folder"],
                    f"train-000{subdir}-of-00064-output_chunk_*.jsonl.zst",
                )
            )
        )

        executor = SlurmPipelineExecutor(
            pipeline=pipeline, **subdir_slurm_config
        )
        executor.run()
        # Wait till all tasks are done with timeout
        completions_dir = os.path.join(logging_dir, "completions")
        success = wait_for_completions(
            completions_dir,
            subdir_slurm_config['tasks'],
            timeout_seconds=timeout_seconds,
        )


if __name__ == "__main__":
    # Run the main example
    main()
