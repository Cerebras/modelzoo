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

from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.batching import batched
from datatrove.utils.logging import logger


class Embeddings(PipelineStep):
    name = "🤖 SentenceTransformers"
    type = "🔢 - Embeddings"

    # TODO: Add "sentence_transformers" and "huggingface_hub" to _requires_dependencies
    # Currently importlib is unable to find these packages inside the container even though they are installed.
    _requires_dependencies = ["filelock"]

    def __init__(
        self,
        model_name_or_path: str,
        model_kwargs: dict | None = None,
        batch_size: int = 32,
        output_folder: str | None = None,
        embedding_key: str = "embeddings",
        show_encode_progress_bar: bool = True,
        use_gpu: bool = False,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.output_folder = output_folder
        self.embedding_key = embedding_key
        self.use_gpu = use_gpu
        self.show_encode_progress_bar = show_encode_progress_bar

        if self.output_folder:
            self._requires_dependencies.append("numpy")

    def _process_batch(self, batch: list[Document]):
        texts = [d.text for d in batch]
        embeddings_batch = self.model.encode(
            texts,
            show_progress_bar=self.show_encode_progress_bar,
            batch_size=self.batch_size,
        )

        if not self.output_folder:
            for doc_in_batch, embeddings in zip(batch, embeddings_batch):
                doc_in_batch.metadata[self.embedding_key] = embeddings.tolist()

        return batch, embeddings_batch if self.output_folder else None

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        """
        Generate embeddings for the documents in the pipeline using the specified model.

        Args:
            data (DocumentsPipeline): The input data containing documents.
            rank (int, optional): The rank of the current process in distributed training. Defaults to 0.
            world_size (int, optional): The total number of processes in distributed training. Defaults to 1.

        Returns:
            DocumentsPipeline: The input data with embeddings added to each document.
        """

        if self.model is None:
            logger.info(f"Loading model {self.model_name_or_path}")
            import os
            import shutil

            from filelock import FileLock
            from huggingface_hub import snapshot_download
            from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
            from sentence_transformers import SentenceTransformer

            # To prevent multiple downloads across nodes, we first download to a shared cache.
            # A lock file on the shared filesystem ensures only one process downloads the model.
            safe_model_name = self.model_name_or_path.replace("/", "_")
            shared_lock_path = os.path.join(
                HUGGINGFACE_HUB_CACHE, f".locks/.{safe_model_name}.lock"
            )
            os.makedirs(os.path.dirname(shared_lock_path), exist_ok=True)

            model_path_shared = None
            with FileLock(shared_lock_path):
                # This will download the model to the shared cache if not present.
                # Only one process across all nodes will do this.
                model_path_shared = snapshot_download(
                    self.model_name_or_path, cache_dir=HUGGINGFACE_HUB_CACHE
                )

            # Then, to avoid I/O bottlenecks from all processes reading from the shared
            # filesystem, we copy the model to a node-local cache in /tmp.
            local_cache_dir = f"/tmp/hf_cache_{safe_model_name}"
            local_model_path = os.path.join(
                local_cache_dir, os.path.basename(model_path_shared)
            )
            local_lock_path = os.path.join(local_cache_dir, ".lock")
            os.makedirs(local_cache_dir, exist_ok=True)

            with FileLock(local_lock_path):
                if not os.path.exists(local_model_path):
                    logger.info(
                        f"Copying model from shared cache {model_path_shared} to local cache {local_model_path}"
                    )
                    shutil.copytree(model_path_shared, local_model_path)
                else:
                    logger.info(
                        f"Model already in local cache at {local_model_path}"
                    )

            # SentenceTransformer is initialized from the node-local path.
            self.model = SentenceTransformer(
                local_model_path,
                device="cuda" if self.use_gpu else "cpu",
                **(self.model_kwargs or {}),
            )
        logger.info(f"Using model {self.model_name_or_path}")

        all_embeddings_list = []
        all_doc_ids = []
        if self.output_folder:
            import os

            import numpy as np

        for batch in batched(data, self.batch_size):
            if self.batch_size > 1:
                self.stat_update("batches")

            with self.track_time("batch" if self.batch_size > 1 else None):
                processed_batch, embeddings_batch = self._process_batch(batch)
                if self.output_folder:
                    all_embeddings_list.extend(embeddings_batch)
                    all_doc_ids.extend([d.id for d in processed_batch])

            yield from processed_batch

        if self.output_folder and all_embeddings_list:
            logger.info(
                f"Saving {len(all_doc_ids)} embeddings to disk for rank {rank}."
            )

            # Save embeddings and document IDs together
            embeddings_matrix = np.array(all_embeddings_list)
            doc_ids = np.array(all_doc_ids)

            output_path = os.path.join(
                self.output_folder, f"embeddings_{rank:05d}.npz"
            )
            np.savez_compressed(
                output_path, doc_ids=doc_ids, embeddings=embeddings_matrix
            )

            logger.info(
                f"Saved {embeddings_matrix.shape[0]} embeddings for rank {rank} to {output_path}"
            )
