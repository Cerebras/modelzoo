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

import argparse

import yaml
from data_curation.pipeline.deepseek_math.filters import IDDedupFilter
from data_curation.pipeline.deepseek_math.readers import DedupJsonlReader
from data_curation.pipeline.deepseek_math.writers import DedupJsonlWriter
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages


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


def main():
    args = parse_args()
    config = load_config(args.config)

    print("Starting Minhash Deduplication pipeline ...")

    minhash_config = MinhashConfig(
        hash_config=HashConfig(precision=64),
        num_buckets=14,
        hashes_per_bucket=8,
    )  # better precision -> fewer false positives (collisions)

    INPUT_READER = DedupJsonlReader(
        data_folder=config["INPUT_FOLDER_PATH"], glob_pattern="*/*/*.jsonl.gz"
    )

    # stage 1 computes minhash signatures for each task (each task gets a set of files)
    stage1 = SlurmPipelineExecutor(
        job_name="mh1",
        pipeline=[
            INPUT_READER,
            IDDedupFilter(text_ids_path=config['deduped_input_path']),
            MinhashDedupSignature(
                output_folder=f"{config['S3_MINHASH_BASE_PATH']}/signatures",
                config=minhash_config,
                language=Languages.english,
            ),
        ],
        tasks=config["TOTAL_TASKS"],
        time="24:00:00",
        partition="cpu-spot",
        logging_dir=f"{config['S3_LOGS_FOLDER']}/signatures",
        slurm_logs_folder=f"{config['LOCAL_LOGS_FOLDER']}/signatures/slurm_logs",
        mem_per_cpu_gb=32,  ## increase memory if facing memory issues
        qos="high",
    )

    # stage 2 finds matches between signatures in each bucket
    stage2 = SlurmPipelineExecutor(
        job_name="mh2",
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{config['S3_MINHASH_BASE_PATH']}/signatures",
                output_folder=f"{config['S3_MINHASH_BASE_PATH']}/buckets",
                config=minhash_config,
            ),
        ],
        tasks=minhash_config.num_buckets,
        time="90:00:00",
        partition="cpu-spot",
        logging_dir=f"{config['S3_LOGS_FOLDER']}/buckets",
        depends=stage1,
        slurm_logs_folder=f"{config['LOCAL_LOGS_FOLDER']}/buckets/slurm_logs",
        qos="high",
    )

    # stage 3 creates clusters of duplicates using the results from all buckets
    stage3 = SlurmPipelineExecutor(
        job_name="mh3",
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{config['S3_MINHASH_BASE_PATH']}/buckets",
                output_folder=f"{config['S3_MINHASH_BASE_PATH']}/remove_ids",
                config=minhash_config,
            ),
        ],
        tasks=1,
        time="90:00:00",
        partition="cpu-spot",
        logging_dir=f"{config['S3_LOGS_FOLDER']}/clusters",
        mem_per_cpu_gb=70,
        cpus_per_task=2,
        depends=stage2,
        slurm_logs_folder=f"{config['LOCAL_LOGS_FOLDER']}/clusters/slurm_logs",
    )

    # stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
    # the data must match exactly stage 1, so number of tasks and the input source must be the same
    stage4 = SlurmPipelineExecutor(
        job_name="mh4",
        pipeline=[
            INPUT_READER,
            MinhashDedupFilter(
                input_folder=f"{config['S3_MINHASH_BASE_PATH']}/remove_ids",
                exclusion_writer=JsonlWriter(
                    f"{config['S3_MINHASH_BASE_PATH']}/removed"
                ),
            ),
            DedupJsonlWriter(
                output_folder=f"{config['S3_MINHASH_BASE_PATH']}/deduplicated_output"
            ),
        ],
        tasks=config['TOTAL_TASKS'],
        mem_per_cpu_gb=32,  # Should be sufficient with batched processing
        time="50:00:00",
        partition="cpu-spot",
        logging_dir=f"{config['S3_LOGS_FOLDER']}/filter",
        depends=stage3,
        slurm_logs_folder=f"{config['LOCAL_LOGS_FOLDER']}/filter/slurm_logs",
    )

    stage4.run()


if __name__ == "__main__":
    main()
