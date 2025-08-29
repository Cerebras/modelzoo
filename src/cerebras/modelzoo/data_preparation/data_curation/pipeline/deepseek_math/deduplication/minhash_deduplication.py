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

import os

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages


class MinHashDeduplication(PipelineStep):
    def __init__(self, input_folder: str):
        """
        Initialize the MinHash deduplication process.
        """
        self.input_folder = input_folder
        self.local_base_path = "deduplicated_data"

        # Define subfolder paths
        self.signatures_folder = os.path.join(
            self.local_base_path, "signatures"
        )
        self.buckets_folder = os.path.join(self.local_base_path, "buckets")
        self.remove_ids_folder = os.path.join(
            self.local_base_path, "remove_ids"
        )
        self.removed_output = os.path.join(self.local_base_path, "removed")
        self.final_output = os.path.join(
            self.local_base_path, "deduplicated_output"
        )

        # Create required subfolders if they don't exist
        for folder in [
            self.input_folder,
            self.signatures_folder,
            self.buckets_folder,
            self.remove_ids_folder,
            self.removed_output,
            self.final_output,
        ]:
            os.makedirs(folder, exist_ok=True)

        # MinHash configuration
        self.minhash_config = MinhashConfig(
            hash_config=HashConfig(precision=64),
            num_buckets=14,
            hashes_per_bucket=7,
        )

    def run(self, *args, **kwargs):
        """
        Executes the entire MinHash-based deduplication process using LocalPipelineExecutor.
        This method will handle all the stages: signature generation, bucket creation, clustering,
        and filtering duplicates.
        """
        try:
            # Stage 1: Generate MinHash signatures
            signature_pipeline = [
                JsonlReader(self.input_folder),
                MinhashDedupSignature(
                    output_folder=self.signatures_folder,
                    config=self.minhash_config,
                    language=Languages.english,
                ),
            ]
            signature_executor = LocalPipelineExecutor(
                pipeline=signature_pipeline, tasks=4
            )
            signature_executor.run()

            # Stage 2: Create hash buckets
            bucket_pipeline = [
                MinhashDedupBuckets(
                    input_folder=self.signatures_folder,
                    output_folder=self.buckets_folder,
                    config=self.minhash_config,
                ),
            ]
            bucket_executor = LocalPipelineExecutor(
                pipeline=bucket_pipeline, tasks=self.minhash_config.num_buckets
            )
            bucket_executor.run()

            # Stage 3: Cluster similar documents
            cluster_pipeline = [
                MinhashDedupCluster(
                    input_folder=self.buckets_folder,
                    output_folder=self.remove_ids_folder,
                    config=self.minhash_config,
                ),
            ]
            cluster_executor = LocalPipelineExecutor(
                pipeline=cluster_pipeline, tasks=1
            )
            cluster_executor.run()

            # Stage 4: Filter duplicates from input
            filter_pipeline = [
                JsonlReader(self.input_folder),
                TokensCounter(),
                MinhashDedupFilter(
                    input_folder=self.remove_ids_folder,
                    exclusion_writer=JsonlWriter(self.removed_output),
                ),
                JsonlWriter(output_folder=self.final_output),
            ]
            filter_executor = LocalPipelineExecutor(
                pipeline=filter_pipeline, tasks=4
            )
            filter_executor.run()

            print("MinHash Deduplication process completed successfully.")
            return self.final_output

        except Exception as e:
            print(
                f"An error occurred during the MinHash deduplication process: {str(e)}"
            )
            raise
