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
Example usage of the PyPI Extraction Pipeline Step with datatrove.

This example shows how to set up and run a pipeline that processes
PyPI project directories and extracts structured data.
"""
import os
import sys
import time

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")),
)
from data_curation.pipeline.deepseek_math.extractors import PyPIExtractor
from data_curation.pipeline.deepseek_math.filters import PyPIPackageFilter

# Import your custom pipeline steps
from data_curation.pipeline.deepseek_math.readers import PyPIReader
from data_curation.pipeline.deepseek_math.writers import PyPIJsonlWriter


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the PyPI extraction pipeline."
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
    """Example of how to run the PyPI extraction pipeline."""
    # Parse arguments and load config once
    args = parse_args()
    config = load_config(args.config)

    # Get parameters from config with defaults
    start_subdir = config.get("pipeline", {}).get("start_subdir", 2)
    end_subdir = config.get("pipeline", {}).get("end_subdir", 47)
    timeout_seconds = config.get("pipeline", {}).get("timeout_seconds", 3600)

    for subdir in range(start_subdir, end_subdir):
        print(f"Processing subdirectory {subdir}...")

        # Create subdirectory-specific config
        subdir_config = config.copy()

        # Define the pipeline
        pipeline = [
            # Reader: finds PyPI project directories
            PyPIReader(
                data_folder=os.path.join(
                    config["reader"]["input_folder"], f"subdir{subdir}"
                ),
                glob_pattern="*/*",  # Assumes structure: project_name/version/
            ),
            PyPIPackageFilter(),
            # Processor: extracts structured data from projects
            PyPIExtractor(),
            # Writer: saves the results to JSONL files with all fields preserved
            PyPIJsonlWriter(
                output_folder=os.path.join(
                    config["writer"]["output_folder"], f"subdir{subdir}"
                ),
                filename="pypi_extracted_data.jsonl",
                compression=True,  # Creates .jsonl.gz files
            ),
        ]

        from datatrove.executor.slurm import SlurmPipelineExecutor

        # Create subdirectory-specific paths
        logging_dir = os.path.join(
            config['slurm_executor']['logging_dir'], f"subdir{subdir}", "logs"
        )
        slurm_logs_folder = os.path.join(
            config['slurm_executor']['slurm_logs_folder'],
            f"subdir{subdir}",
            "slurm_logs",
        )

        # Create directories if they don't exist
        os.makedirs(logging_dir, exist_ok=True)
        os.makedirs(slurm_logs_folder, exist_ok=True)

        # Update config for this subdirectory
        subdir_slurm_config = config['slurm_executor'].copy()
        subdir_slurm_config['logging_dir'] = logging_dir
        subdir_slurm_config['slurm_logs_folder'] = slurm_logs_folder

        executor = SlurmPipelineExecutor(
            pipeline=pipeline, **subdir_slurm_config
        )

        # Run the pipeline
        executor.run()

        # Wait till all tasks are done with timeout
        completions_dir = os.path.join(logging_dir, "completions")
        success = wait_for_completions(
            completions_dir,
            config['slurm_executor']['tasks'],
            timeout_seconds=timeout_seconds,
        )

        if success:
            print(
                f"Pipeline completed successfully for subdirectory = {subdir}"
            )
        else:
            print(f"Pipeline timed out for subdirectory = {subdir}")


if __name__ == "__main__":
    # Run the main example
    main()
