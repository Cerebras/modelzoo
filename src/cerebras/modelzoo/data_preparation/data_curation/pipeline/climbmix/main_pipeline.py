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
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter

from cerebras.modelzoo.data_preparation.data_curation.pipeline.climbmix import (
    Embeddings,
    FAISSClustering,
    JsonlWriterExt,
    LLMTopicLabeler,
    import_cls,
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

    # All non-absolute paths are relative to the base directory
    basedir = paths["basedir"]
    for key, value in paths.items():
        if not os.path.isabs(value):
            paths[key] = os.path.join(basedir, value)

    resharding_config = config["resharding"]
    embeddings_config = config["embeddings"]
    clustering_config = config["clustering"]
    labeling_config = config["labeling"]

    if resharding_config.get("enabled", True):
        # Reshard to have same number of docs per file
        reader = import_cls(resharding_config["reader"]["cls"])

        resharding_executor = SlurmPipelineExecutor(
            pipeline=[
                reader(**resharding_config["reader"]["params"]),
                JsonlWriterExt(
                    output_folder=f"{paths['output_folder']}/resharding",
                    **resharding_config["params"],
                ),
            ],
            **resharding_config["slurm"],
            logging_dir=f"{paths['log_dir']}/resharding",
            container_path=paths["container_path"],
        )

        resharding_executor.run()

        # If resharding is enabled, run resharding on cpu slurm cluster and exit
        return

    if embeddings_config.get("enabled", True):
        # Generate embeddings
        embeddings_executor = SlurmPipelineExecutor(
            pipeline=[
                JsonlReader(
                    data_folder=f"{paths['output_folder']}/resharding",
                    glob_pattern="[0-9]*.jsonl*",
                ),
                Embeddings(
                    **embeddings_config["params"],
                    output_folder=f"{paths['output_folder']}/embeddings",
                    use_gpu=embeddings_config["slurm"]["use_gpu"],
                ),
                JsonlWriter(
                    output_folder=f"{paths['output_folder']}/embeddings"
                ),
            ],
            **embeddings_config["slurm"],
            logging_dir=f"{paths['log_dir']}/embeddings",
            container_path=paths["container_path"],
        )
        embeddings_executor.run()

        # If embeddings are enabled, run embeddings on gpu slurm cluster and exit
        return

    # Perform clustering
    clustering_executor = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=f"{paths['output_folder']}/embeddings",
                glob_pattern="[0-9]*.jsonl*",
            ),
            FAISSClustering(
                **clustering_config["params"],
                embeddings_folder=f"{paths['output_folder']}/embeddings",
                save_representatives_path=f"{paths['output_folder']}/clustering/representatives.jsonl",
            ),
            JsonlWriter(
                output_folder=f"{paths['output_folder']}/clustering",
                compression=None,
            ),
        ],
        **clustering_config["slurm"],
        logging_dir=f"{paths['log_dir']}/clustering",
        container_path=paths["container_path"],
    )

    label_writer = import_cls(labeling_config["writer"]["cls"])

    # Label the clusters
    labeling_executor = SlurmPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=f"{paths['output_folder']}/clustering",
                glob_pattern="[0-9]*.jsonl*",
            ),
            LLMTopicLabeler(
                **labeling_config["params"],
                representatives_path=f"{paths['output_folder']}/clustering/representatives.jsonl",
            ),
            label_writer(
                output_folder=f"{paths['output_folder']}/labeling",
                **labeling_config["writer"]["params"],
            ),
        ],
        depends=clustering_executor,
        **labeling_config["slurm"],
        logging_dir=f"{paths['log_dir']}/labeling",
        container_path=paths["container_path"],
    )

    labeling_executor.run()


if __name__ == "__main__":
    main()
