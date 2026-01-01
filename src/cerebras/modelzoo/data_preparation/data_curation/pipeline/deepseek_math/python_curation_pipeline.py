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

# from executors.slurm.slurm import SlurmPipelineExecutor
import argparse

import yaml
from data_curation.pipeline.deepseek_math.filters.python_filters import (
    PythonLanguageCharacterFilter,
    PythonPylintScoreFilter,
    PythonSyntaxFilter,
)
from data_curation.pipeline.deepseek_math.readers.safe_jsonl_reader import (
    SafeJsonlReader,
)
from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.writers import JsonlWriter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text extraction pipeline configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file',
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    # Example of how to use these filters in a datatrove pipeline

    args = parse_args()
    config = load_config(args.config)
    pipeline = [
        SafeJsonlReader(
            data_folder=config["input_folder"],
            compression="zstd",
            text_key=config["text_key"],
        ),  # Read JSONL files with zstd compression
        PythonSyntaxFilter(
            exclusion_writer=JsonlWriter(
                config["syntax_failed_folder"],
            )
        ),  # Remove documents with syntax errors
        PythonPylintScoreFilter(
            min_score=7.0,
            exclusion_writer=JsonlWriter(
                config["pylint_failed_folder"],
            ),
        ),  # Keep only files with pylint score >= 5
        PythonLanguageCharacterFilter(
            exclusion_writer=JsonlWriter(
                config["python_language_character_failed_folder"],
            )
        ),  # Remove files with non-English/Japanese chars
        JsonlWriter(config["output_folder"]),
    ]

    executor_type = config["executor_type"]
    if executor_type == "local":
        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            tasks=config["local"][
                'tasks'
            ],  # More workers since Tree-sitter is fast
            workers=config['local']['workers'],
            logging_dir=config['logging_dir'],
        )
    elif executor_type == "slurm":
        executor = SlurmPipelineExecutor(pipeline=pipeline, **config['slurm'])
    else:
        raise ValueError(f"Unsupported executor type: {executor_type}")
    executor.run()
