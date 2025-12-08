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
Minimal decontamination pipeline using DataTrove with LightEval tasks
"""

import argparse

import yaml
from data_curation.pipeline.deepseek_math.readers import DecontJsonlReader
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.decont.n_grams import (
    NGramsDecontFilter,
    NGramsDecontIndexer,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

# LightEval task names from the registry
# Based on https://github.com/huggingface/lighteval/blob/main/src/lighteval/tasks/registry.py


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download and process Common Crawl WARC files in batches'
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the YAML configuration file',
    )

    return parser.parse_args()


def load_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main(config):

    lighteval_tasks = (
        []
    )  ## No need to add lighteval tasks as we already downloaded the benchmark datasets at location - config["benchmark_datasets"]
    # Create pipeline with decontamination
    index_creation_pipeline = [
        # Indexer - builds n-gram indices from LightEval tasks
        JsonlReader(
            data_folder=config["benchmark_datasets"], glob_pattern="*.jsonl.gz"
        ),
        NGramsDecontIndexer(
            output_folder=config["index_folder"],
            lighteval_tasks=lighteval_tasks,
        ),
    ]

    filtering_pipeline = [
        # Filter - removes contaminated documents
        # Reader
        DecontJsonlReader(
            data_folder=config["input_data"],
            text_key=config["text_key"],
            glob_pattern="*.jsonl.gz",
        ),
        NGramsDecontFilter(
            index_folder=config["index_folder"],
        ),
        # Writer
        JsonlWriter(
            output_folder=config["output_data"],
            compression="gzip",  # Optional compression
        ),
    ]

    # Execute pipeline
    if config.get("executor_type") == "slurm":
        index_creation_pipeline_executor = SlurmPipelineExecutor(
            pipeline=index_creation_pipeline,
            **config["slurm_index_creation_pipeline"],
        )
        filtering_pipeline_executor = SlurmPipelineExecutor(
            pipeline=filtering_pipeline,
            **config["slurm_filtering_pipeline"],
            depends=index_creation_pipeline_executor,
        )
    else:
        index_creation_pipeline_executor = LocalPipelineExecutor(
            pipeline=index_creation_pipeline,
            **config["local_index_creation_pipeline"],
        )
        filtering_pipeline_executor = LocalPipelineExecutor(
            pipeline=filtering_pipeline,
            **config["local_filtering_pipeline"],
            depends=index_creation_pipeline_executor,
        )

    filtering_pipeline_executor.run()


if __name__ == "__main__":

    args = parse_args()
    config = load_config(args)
    main(config)
