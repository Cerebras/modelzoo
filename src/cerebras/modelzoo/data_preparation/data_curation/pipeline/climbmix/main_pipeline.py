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
import os

import yaml
from datatrove.pipeline.readers import HuggingFaceDatasetReader, JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

from cerebras.modelzoo.data_preparation.data_curation.pipeline.climbmix.embeddings import (
    Embeddings,
)
from cerebras.modelzoo.data_preparation.data_curation.pipeline.climbmix.faiss_clustering import (
    FAISSClustering,
)
from cerebras.modelzoo.data_preparation.data_curation.pipeline.climbmix.llm_topic_labeler import (
    LLMTopicLabeler,
)
from cerebras.modelzoo.data_preparation.executors.slurm.slurm import (
    SlurmPipelineExecutor,
)


def main():
    parser = argparse.ArgumentParser(description="Run clustering pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]

    # All paths are relative to the base directory
    basedir = paths["basedir"]
    for key, value in paths.items():
        paths[key] = os.path.join(basedir, value)

    embeddings_config = config["embeddings"]
    clustering_config = config["clustering"]
    labeling_config = config["labeling"]

    if embeddings_config.get("enabled", True):
        # Generate embeddings
        embeddings_executor = SlurmPipelineExecutor(
            pipeline=[
                HuggingFaceDatasetReader(
                    embeddings_config["dataset"]["name"],
                    limit=embeddings_config["dataset"][
                        "limit"
                    ],  # Set limit for testing else -1
                    dataset_options={"split": "train"},
                ),
                Embeddings(
                    model_name_or_path=embeddings_config["model"],
                    model_kwargs=embeddings_config["model_kwargs"],
                    batch_size=embeddings_config["batch_size"],
                    output_folder=f"{paths['output_folder']}/embeddings",
                    use_gpu=embeddings_config["slurm"]["use_gpu"],
                ),
                JsonlWriter(
                    output_folder=f"{paths['output_folder']}/embeddings"
                ),
            ],
            tasks=embeddings_config["slurm"]["tasks"],
            randomize_start_duration=embeddings_config["slurm"][
                "randomize_start_duration"
            ],
            job_name=embeddings_config["slurm"]["job_name"],
            partition=embeddings_config["slurm"]["partition"],
            time=embeddings_config["slurm"]["time"],
            mem_per_cpu_gb=embeddings_config["slurm"]["mem_per_cpu_gb"],
            cpus_per_task=embeddings_config["slurm"]["cpus_per_task"],
            tasks_per_job=embeddings_config["slurm"]["tasks_per_job"],
            logging_dir=f"{paths['log_dir']}/embeddings",
            container_path=paths["container_path"],
            use_gpu=embeddings_config["slurm"]["use_gpu"],
        )

    # Perform clustering
    clustering_executor = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=f"{paths['output_folder']}/embeddings",
                glob_pattern="*.jsonl.gz",
            ),
            FAISSClustering(
                num_clusters=clustering_config["num_clusters"],
                niter=clustering_config["kmeans_iterations"],
                embeddings_folder=f"{paths['output_folder']}/embeddings",
                merge_clusters=clustering_config["merge_clusters"],
                merge_threshold=clustering_config["merge_threshold"],
                save_representatives_path=f"{paths['output_folder']}/clustering/representatives.jsonl",
            ),
            JsonlWriter(output_folder=f"{paths['output_folder']}/clustering"),
        ],
        depends=(
            embeddings_executor
            if embeddings_config.get("enabled", True)
            else None
        ),
        tasks=clustering_config["slurm"]["tasks"],
        job_name=clustering_config["slurm"]["job_name"],
        partition=clustering_config["slurm"]["partition"],
        time=clustering_config["slurm"]["time"],
        mem_per_cpu_gb=clustering_config["slurm"]["mem_per_cpu_gb"],
        cpus_per_task=clustering_config["slurm"]["cpus_per_task"],
        logging_dir=f"{paths['log_dir']}/clustering",
        container_path=paths["container_path"],
    )

    # Label the clusters
    labeling_executor = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=f"{paths['output_folder']}/clustering",
                glob_pattern="*.jsonl.gz",
            ),
            LLMTopicLabeler(
                model=labeling_config["model"],
                prompt_template=labeling_config["prompt_template"],
                api_key=labeling_config["api_key"],
                representatives_path=f"{paths['output_folder']}/clustering/representatives.jsonl",
            ),
            JsonlWriter(output_folder=f"{paths['output_folder']}/labeling"),
        ],
        depends=clustering_executor,
        tasks=labeling_config["slurm"]["tasks"],
        job_name=labeling_config["slurm"]["job_name"],
        partition=labeling_config["slurm"]["partition"],
        time=labeling_config["slurm"]["time"],
        mem_per_cpu_gb=labeling_config["slurm"]["mem_per_cpu_gb"],
        cpus_per_task=labeling_config["slurm"]["cpus_per_task"],
        logging_dir=f"{paths['log_dir']}/labeling",
        container_path=paths["container_path"],
    )

    labeling_executor.run()


if __name__ == "__main__":
    main()
