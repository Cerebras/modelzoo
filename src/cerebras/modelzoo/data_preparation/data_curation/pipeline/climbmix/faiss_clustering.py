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

import glob
import json
import os

import numpy as np
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


class FAISSClustering(PipelineStep):
    name = "🎯 FAISS K-means Clustering"
    type = "🔢 - Clustering"

    # TODO: Add ("faiss", "faiss_cpu") to _requires_dependencies
    # Currently importlib is unable to find it inside the container even though it is installed
    _requires_dependencies = ["numpy"]

    def __init__(
        self,
        num_clusters: int,
        embeddings_folder: str,
        niter: int = 20,
        cluster_key: str = "cluster_id",
        num_cluster_representatives: int | None = 10,
        representative_rank_key: str = "representative_rank",
        save_centroids_path: str | None = None,
        save_representatives_path: str | None = None,
        merge_clusters: bool = False,
        merge_threshold: float = 0.3,
        verbose: bool = True,
        use_gpu: bool = False,
        seed: int = 123,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.embeddings_folder = embeddings_folder
        self.niter = niter
        self.cluster_key = cluster_key
        self.num_cluster_representatives = num_cluster_representatives
        self.representative_rank_key = representative_rank_key
        self.save_centroids_path = save_centroids_path
        self.save_representatives_path = save_representatives_path
        self.merge_clusters = merge_clusters
        self.merge_threshold = merge_threshold
        self.verbose = verbose
        self.use_gpu = use_gpu
        self.seed = seed
        self.kmeans = None
        self.centroids = None

    def _collect_embeddings(self):
        """Collect all embeddings for clustering from the embeddings folder."""
        logger.info(f"Loading embeddings from {self.embeddings_folder}")
        npz_files = sorted(
            glob.glob(os.path.join(self.embeddings_folder, "embeddings_*.npz"))
        )

        if not npz_files:
            logger.error(
                f"No .npz embedding files found in {self.embeddings_folder}. "
                f"Ensure `Embeddings` step ran with `output_folder` set."
            )
            return None, []

        embeddings_list = []
        all_doc_ids = []

        for f in npz_files:
            with np.load(f) as loaded_data:
                embeddings = loaded_data["embeddings"]
                doc_ids = loaded_data["doc_ids"]

                for i, doc_id in enumerate(doc_ids):
                    all_doc_ids.append(doc_id)
                    embeddings_list.append(embeddings[i].reshape(1, -1))

        embeddings_matrix = np.vstack(embeddings_list)

        # Check for duplicates
        num_duplicates = len(all_doc_ids) - len(set(all_doc_ids))
        if num_duplicates:
            logger.warning(
                f"Found {num_duplicates} duplicate document IDs in embeddings. "
                f"This may cause issues with clustering and representative selection."
            )

        logger.info(
            f"Loaded {len(all_doc_ids)} embeddings with shape {embeddings_matrix.shape}"
        )
        return embeddings_matrix, all_doc_ids

    def _fit_kmeans(self, embeddings: np.ndarray):
        """Fit FAISS K-means on the embeddings."""
        import faiss

        logger.info(f"Fitting FAISS K-means with {self.num_clusters} clusters")

        # Ensure embeddings are float32 (required by FAISS)
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Set random seed for reproducibility
        np.random.seed(self.seed)

        # Initialize FAISS K-means
        d = embeddings.shape[1]  # embedding dimension
        kmeans = faiss.Kmeans(
            d,
            self.num_clusters,
            niter=self.niter,
            verbose=self.verbose,
            seed=self.seed,
            gpu=self.use_gpu,
            spherical=True,
        )

        kmeans.train(embeddings)

        self.kmeans = kmeans
        self.centroids = kmeans.centroids

    def _predict_clusters(self, embeddings: np.ndarray):
        """Predict cluster assignments and distances for embeddings."""
        if self.kmeans is None:
            raise ValueError(
                "K-means model not fitted. Call _fit_kmeans first."
            )

        # Ensure embeddings are float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Get cluster assignments and distances (search returns distances, labels)
        distances, cluster_ids = self.kmeans.index.search(embeddings, 1)
        return cluster_ids.flatten(), distances.flatten()

    def _merge_clusters(self, centroids: np.ndarray):
        """Merge clusters based on centroid cosine distance using AgglomerativeClustering."""
        from sklearn.cluster import AgglomerativeClustering

        logger.info(
            f"Merging {self.num_clusters} clusters with a cosine distance threshold of {self.merge_threshold}"
        )

        # Initialize AgglomerativeClustering
        agg_clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.merge_threshold,
            metric="cosine",
            linkage="average",
        )

        # Fit the model on the centroids
        agg_clustering.fit(centroids)

        # Create a mapping from old cluster ID to new merged cluster ID
        old_to_new_cluster_map = {
            i: label for i, label in enumerate(agg_clustering.labels_)
        }
        num_merged_clusters = len(np.unique(agg_clustering.labels_))
        logger.info(
            f"Merged {self.num_clusters} clusters into {num_merged_clusters} new clusters."
        )

        return old_to_new_cluster_map

    def _rank_documents_in_clusters(
        self, document_ids, cluster_assignments, distances_to_centroid
    ):
        """Rank documents within each cluster based on their distance to the centroid."""
        logger.info("Ranking documents within each cluster.")
        # Group documents by cluster ID along with their distance
        cluster_data = {}
        for doc_id, cluster_id, distance in zip(
            document_ids, cluster_assignments, distances_to_centroid
        ):
            cluster_id = int(cluster_id)
            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = []
            cluster_data[cluster_id].append(
                {"doc_id": doc_id, "distance": distance}
            )

        # Rank documents within each cluster and create final mapping
        doc_id_to_cluster_info = {}
        for cluster_id, docs_in_cluster in cluster_data.items():
            # Sort documents by distance to find the closest ones
            docs_in_cluster.sort(key=lambda x: x["distance"])
            # Assign a rank based on proximity
            for position, doc_info in enumerate(docs_in_cluster, 1):
                # Assign a positive rank if within the limit, otherwise -1
                rank = (
                    position
                    if self.num_cluster_representatives is None
                    or position <= self.num_cluster_representatives
                    else -1
                )
                doc_id_to_cluster_info[doc_info["doc_id"]] = {
                    "cluster_id": cluster_id,
                    "rank": rank,
                }
        return doc_id_to_cluster_info

    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        # Load embeddings from folder
        embeddings_matrix, document_ids = self._collect_embeddings()

        if embeddings_matrix is None:
            logger.error("No embeddings found, returning documents unchanged.")
            yield from data
            return

        # Fit K-means clustering (only on rank 0 in distributed setting)
        if rank == 0 or world_size == 1:
            # Normalize embeddings to unit length for cosine similarity-based clustering
            import faiss

            logger.info(
                "Normalizing embeddings to unit length (L2 normalization)."
            )
            faiss.normalize_L2(embeddings_matrix)

            self._fit_kmeans(embeddings_matrix)

            # Save centroids if a path is provided
            if self.save_centroids_path:
                self._save_centroids(self.save_centroids_path)

            # Predict cluster assignments and distances to original centroids
            logger.info("Predicting cluster assignments for all embeddings.")
            cluster_assignments, distances_to_centroid = self._predict_clusters(
                embeddings_matrix
            )
            final_cluster_assignments = cluster_assignments

            # Merge clusters if enabled
            if self.merge_clusters:
                old_to_new_cluster_map = self._merge_clusters(self.centroids)
                # Update the cluster assignments to the new merged IDs
                final_cluster_assignments = np.array(
                    [
                        old_to_new_cluster_map[old_id]
                        for old_id in cluster_assignments
                    ]
                )

                # Recalculate centroids for the new merged clusters
                num_merged_clusters = len(
                    np.unique(list(old_to_new_cluster_map.values()))
                )
                new_centroids = np.zeros(
                    (num_merged_clusters, embeddings_matrix.shape[1]),
                    dtype=np.float32,
                )
                for new_cluster_id in range(num_merged_clusters):
                    # Find all documents belonging to this new cluster
                    mask = final_cluster_assignments == new_cluster_id
                    if np.any(mask):
                        # Calculate the new centroid by averaging the embeddings
                        new_centroids[new_cluster_id] = embeddings_matrix[
                            mask
                        ].mean(axis=0)

                # Normalize the new centroids before calculating distance
                faiss.normalize_L2(new_centroids)

                # Calculate distances to the new merged centroids
                cosine_similarities = np.sum(
                    embeddings_matrix
                    * new_centroids[final_cluster_assignments],
                    axis=1,
                )
                distances_to_centroid = 1 - cosine_similarities

            # Rank documents based on the final cluster assignments and distances
            doc_id_to_cluster_info = self._rank_documents_in_clusters(
                document_ids, final_cluster_assignments, distances_to_centroid
            )

            logger.info(
                f"Final cluster distribution: {np.bincount(final_cluster_assignments)}"
            )
        else:
            # In distributed setting, other ranks would need to receive the trained model
            doc_id_to_cluster_info = {}
            assert (
                False
            ), "Distributed processing not implemented for FAISS clustering."

        # Prepare for collecting representative texts if needed
        representative_texts = {} if self.save_representatives_path else None
        representative_doc_ids = set()

        if self.save_representatives_path:
            # Get list of representative document IDs
            representative_doc_ids = {
                doc_id
                for doc_id, info in doc_id_to_cluster_info.items()
                if info["rank"] > 0
            }
            logger.info(
                f"Collecting texts for {len(representative_doc_ids)} representative documents."
            )

        logger.info("Adding cluster information to documents.")

        for doc in data:

            if doc.id in doc_id_to_cluster_info:
                info = doc_id_to_cluster_info[doc.id]
                doc.metadata[self.cluster_key] = int(info["cluster_id"])
                doc.metadata[self.representative_rank_key] = int(info["rank"])
                self.stat_update("clustered_documents")

                # Collect text for representatives
                if (
                    representative_texts is not None
                    and doc.id in representative_doc_ids
                ):
                    representative_texts[doc.id] = doc.text
            else:
                logger.warning(
                    f"No cluster assignment found for document {doc.id}"
                )

            yield doc

        # Save representatives after processing all documents
        if self.save_representatives_path and representative_texts:
            self._save_representatives(
                doc_id_to_cluster_info,
                representative_texts,
                self.save_representatives_path,
            )

    def _save_representatives(
        self, doc_id_to_cluster_info: dict, doc_texts: dict, output_path: str
    ):
        """Save cluster representatives information with collected text content to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create a flattened list of representatives
        flattened_representatives = []
        missing_text_count = 0

        for id, info in doc_id_to_cluster_info.items():
            if info["rank"] > 0:
                cluster_id = info["cluster_id"]
                doc_text = doc_texts.get(id)

                if doc_text is not None:
                    # Add a flattened entry with cluster_id included
                    flattened_representatives.append(
                        {
                            "doc_id": id,
                            "cluster_id": cluster_id,
                            "rank": info["rank"],
                            "text": doc_text,
                        }
                    )
                else:
                    missing_text_count += 1

        if missing_text_count > 0:
            logger.warning(
                f"Text not found for {missing_text_count} representative documents"
            )

        # Sort the flattened list by cluster_id and then by rank
        flattened_representatives.sort(
            key=lambda x: (x["cluster_id"], x["rank"])
        )

        # Save to JSONL file
        with open(output_path, "w", encoding="utf-8") as f:
            for rep in flattened_representatives:
                f.write(json.dumps(rep, ensure_ascii=False) + "\n")

        # Count unique clusters
        unique_clusters = len(
            set(rep["cluster_id"] for rep in flattened_representatives)
        )

        logger.info(
            f"Saved {len(flattened_representatives)} representatives as JSONL "
            f"from {unique_clusters} clusters to {output_path}"
        )

    def _save_centroids(self, output_path: str):
        """Save cluster centroids to file."""
        if self.centroids is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, self.centroids)
            logger.info(f"Saved {self.num_clusters} centroids to {output_path}")
        else:
            logger.warning("No centroids to save. Fit the model first.")
