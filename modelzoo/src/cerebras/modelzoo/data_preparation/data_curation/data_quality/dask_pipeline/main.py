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
import glob
import gzip
import json
import logging
import os
import shutil
import tempfile

import dask
import dask.bag as db
import dask.dataframe as dd
import numpy as np
import pandas as pd
import yaml
from dask.distributed import Client, LocalCluster
from dask_jobqueue import SLURMCluster
from datasets import load_dataset
from sklearn.cluster import HDBSCAN
from transformers import pipeline
from utils import classify_partition, embed_partition, lang_partition


class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]


class LoggingSetup:
    def __init__(self, config_manager):
        self.config = config_manager
        self._setup_logging()

    def _setup_logging(self):
        log_file = self.config["logging"]["file"]
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)

        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"].upper()),
            format=self.config["logging"]["format"],
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        # Suppress bokeh/tornado websocket errors
        logging.getLogger('tornado.application').setLevel(logging.ERROR)
        logging.getLogger('bokeh.server.session').setLevel(logging.ERROR)
        logging.getLogger('asyncio').setLevel(logging.ERROR)


class FileProcessor:
    def __init__(self, config_manager):
        self.config = config_manager

    def decompress_all(self, gz_files, tmpdir):
        jsonl_paths = []
        total_records = 0
        for gz_path in gz_files:
            base = os.path.basename(gz_path).replace('.gz', '')
            out_path = os.path.join(tmpdir, base)
            with (
                gzip.open(gz_path, 'rt', encoding='utf-8') as f_in,
                open(out_path, 'w', encoding='utf-8') as f_out,
            ):
                line_count = 0
                for line in f_in:
                    f_out.write(line)
                    line_count += 1

                total_records += line_count
            jsonl_paths.append(out_path)
        return jsonl_paths, total_records

    def get_schema(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                return set(obj.keys())  # schema from first line

    def check_all_same_schema(self, jsonl_paths):
        base_schema = self.get_schema(jsonl_paths[0])
        for path in jsonl_paths[1:]:
            schema = self.get_schema(path)
            if schema != base_schema:
                raise ValueError(
                    f"Inconsistent schema in {path}: {schema} vs {base_schema}"
                )
        return base_schema

    def merge_jsonl_files(self, jsonl_paths, output_path):
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for path in jsonl_paths:
                with open(path, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)


class CheckpointManager:
    def __init__(self, config_manager):
        self.config = config_manager

    def read_offset(self):
        """Read last processed batch index, or return 0."""
        try:
            with open(self.config["checkpoint"]["offset_file"], "r") as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return 0

    def write_offset(self, idx):
        """Write current batch index to offset file."""
        with open(self.config["checkpoint"]["offset_file"], "w") as f:
            f.write(str(idx))


class DataProcessor:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)

    def row_generator(self, source, batch_index, batch_size, total_records):
        start = batch_index * batch_size
        end = min(start + batch_size, total_records)
        for i, row in enumerate(source):
            if i in range(start, end):
                yield {
                    "id": i,
                    "text": row["text"],
                    "url": row.get("metadata", {}).get("url"),
                }


class ClusteringManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)

    def perform_clustering(self, clustered_file):
        if not os.path.exists(clustered_file):
            emb_dir = self.config["output"]["embeddings_dir"]
            parquet_files = list(glob.glob(f"{emb_dir}/batch_*/*.parquet"))

            if parquet_files:
                embdf = dd.read_parquet(parquet_files)
            embedding_cols = [
                col for col in embdf.columns if col.startswith('embed_')
            ]
            emb_part = embdf[embedding_cols]

            X = emb_part
            ccfg = self.config["clustering"]
            model = HDBSCAN(min_cluster_size=ccfg["min_cluster_size"])
            labels = model.fit(X).labels_
            embdf = embdf.compute()
            embdf['cluster'] = labels
            embdf = dd.from_pandas(embdf, npartitions=32).persist()
            embdf.to_parquet(clustered_file, write_index=False)
            self.logger.info("Clustering checkpoint saved")


class DaskClusterManager:
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
        self.cluster = None
        self.client = None

    def setup_cluster(self):
        dcfg = self.config["dask_cluster"]
        cluster_type = dcfg.get("type", "slurm").lower()

        if cluster_type == "slurm":
            self.logger.info("Using SLURMCluster for distributed execution")
            self.cluster = SLURMCluster(
                queue=dcfg["queue"],
                cores=dcfg["cores"],
                memory=dcfg["memory"],
                walltime=dcfg["walltime"],
                processes=dcfg["processes"],
                interface=dcfg["interface"],
            )
            self.cluster.scale(jobs=dcfg["workers"])
        elif cluster_type == "local":
            self.logger.info("Using LocalCluster for execution")
            self.cluster = LocalCluster(
                n_workers=dcfg["workers"],
                threads_per_worker=dcfg["cores"],
                memory_limit=dcfg["memory"],
            )
        else:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")

        dask.config.set(self.config.get("dask_config", {}))
        self.client = Client(
            self.cluster, timeout="60s", direct_to_workers=True
        )
        self.logger.info(f"Dask dashboard: {self.client.dashboard_link}")

        # Only wait for workers in SLURM mode
        if cluster_type == "slurm":
            self.client.wait_for_workers(n_workers=dcfg["workers"], timeout=300)

        self.logger.info("Workers ready")
        return self.client

    def close_cluster(self):
        if self.client:
            self.client.close()
        if self.cluster:
            self.cluster.close()


class DataPipeline:
    def __init__(self, device='cpu'):
        self.device = device
        self.config_manager = ConfigManager()
        self.logging_setup = LoggingSetup(self.config_manager)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting pipeline with device: {self.device}")
        self.file_processor = FileProcessor(self.config_manager)
        self.checkpoint_manager = CheckpointManager(self.config_manager)
        self.data_processor = DataProcessor(self.config_manager)
        self.clustering_manager = ClusteringManager(self.config_manager)
        self.dask_cluster_manager = DaskClusterManager(self.config_manager)

    def run(self):
        # Setup Dask cluster
        client = self.dask_cluster_manager.setup_cluster()

        try:
            self._process_data()
        finally:
            self.dask_cluster_manager.close_cluster()

    def _process_data(self):
        # Prepare file list and batching
        files = sorted(
            glob.glob(self.config_manager["dataset"]["files_pattern"])
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_paths, total_records = self.file_processor.decompress_all(
                files, tmpdir
            )
            schema = self.file_processor.check_all_same_schema(jsonl_paths)
            combined_path = os.path.join("./combined.jsonl")
            self.file_processor.merge_jsonl_files(jsonl_paths, combined_path)

        batch_size = self.config_manager["dataset"]["batch_size"]
        total_batches = (total_records + batch_size - 1) // batch_size
        self.logger.info(f"{total_records} records → {total_batches} batches")

        offset = self.checkpoint_manager.read_offset()
        # Step 1: Load dataset
        dataset = load_dataset(
            "json",
            data_files={"train": combined_path},
            split="train",
            streaming=True,
        )

        self._process_batches(
            dataset, offset, total_batches, batch_size, total_records
        )
        self._perform_clustering()
        self._perform_classification()

    def _process_batches(
        self, dataset, offset, total_batches, batch_size, total_records
    ):
        dcfg = self.config_manager["dask_cluster"]

        for batch_index in range(offset, total_batches):
            self.logger.info(f"Loading batch {batch_index}")
            rows = list(
                self.data_processor.row_generator(
                    dataset, batch_index, batch_size, total_records
                )
            )
            bag = db.from_sequence(
                rows, npartitions=dcfg["workers"] * dcfg["processes"]
            )
            meta = {'id': int, 'text': str, 'url': str}
            df = bag.to_dataframe(meta=meta)

            if df.shape[0].compute() == 0:
                self.logger.warning(
                    "Dask DataFrame is empty. No data was loaded."
                )
                continue
            else:
                self.logger.info(
                    f"Data loaded successfully with {df.shape[0].compute()} rows."
                )

            # Step 2: Language detection before embedding
            self.logger.info("Running language detection before embedding...")
            meta_lang = pd.DataFrame(
                {
                    "id": pd.Series(dtype="int64"),
                    "text": pd.Series(dtype="string"),
                    "url": pd.Series(dtype="string"),
                    "language": pd.Series(dtype="string"),
                    "lang_confidence": pd.Series(dtype="float64"),
                }
            )

            fasttext_config = self.config_manager["fasttext"]
            logger = self.logger
            df = df.map_partitions(
                lang_partition, fasttext_config, logger, meta=meta_lang
            )

            # Step 3: Embedding
            self.logger.info("Starting embedding process...")
            embedding_config = self.config_manager["embedding_model"]
            EMBEDDING_DIM = self.config_manager["embedding_model"]["dim"]
            embedding_cols = [f"embed_{i}" for i in range(EMBEDDING_DIM)]
            embedding_meta = pd.DataFrame(
                columns=embedding_cols, dtype=np.float32
            )
            device = self.device
            embeddings = df["text"].map_partitions(
                embed_partition,
                embedding_config,
                logger,
                device,
                EMBEDDING_DIM,
                meta=embedding_meta,
                enforce_metadata=False,
            )

            embeddings_with_meta = dd.concat(
                [
                    df[["id", "text", "url", "language", "lang_confidence"]],
                    embeddings,
                ],
                axis=1,
            )

            self.logger.info("Saving embeddings with metadata to Parquet...")
            EMBEDDING_DIR = self.config_manager["output"]["embeddings_dir"]
            batch_output_path = f"{EMBEDDING_DIR}/batch_{batch_index}"
            if os.path.exists(batch_output_path):
                shutil.rmtree(batch_output_path)
            embeddings_with_meta.to_parquet(
                batch_output_path, write_index=False, append=True
            )
            self.logger.info(
                f"Embeddings with language info for batch {batch_index} saved successfully!"
            )

            self.checkpoint_manager.write_offset(batch_index + 1)

    def _perform_clustering(self):
        clustered_file = self.config_manager["output"]["clustered_file"]
        self.clustering_manager.perform_clustering(clustered_file)

    def _perform_classification(self):
        clustered_file = self.config_manager["output"]["clustered_file"]
        cdf = dd.read_parquet(clustered_file)
        final_file = self.config_manager["output"]["final_file"]

        if not os.path.exists(final_file):
            reps = cdf.groupby("cluster").first().reset_index()
            meta = cdf._meta.copy()
            meta["domain"], meta["domain_confidence"] = object, float
            expected = list(meta.columns)
            classifier_config = self.config_manager["classification"]
            zero_shot_config = self.config_manager["zero_shot_model"]
            device = self.device
            classified = reps.map_partitions(
                classify_partition,
                classifier_config,
                zero_shot_config,
                device,
                expected,
                meta=meta,
            ).compute()

            final = cdf.merge(
                classified[["cluster", "domain", "domain_confidence"]],
                on="cluster",
                how="left",
            )[
                [
                    "id",
                    "text",
                    "language",
                    "lang_confidence",
                    "cluster",
                    "domain",
                    "domain_confidence",
                ]
            ].compute()

            final.to_parquet(final_file, index=False)
            self.logger.info("Final classification saved")


if __name__ == "__main__":
    # --- Parse command line arguments ---
    parser = argparse.ArgumentParser(description='Run data processing pipeline')
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for GPU acceleration (e.g., "cuda", "cuda:0", "cpu")',
    )
    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = DataPipeline(device=args.device)
    pipeline.run()
