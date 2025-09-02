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

import hashlib
import logging
import os
import tempfile
import urllib.request
import zipfile

import dask.bag as db
import dask.dataframe as dd
import fasttext
import numpy as np
import pandas as pd
import yaml
from dask.distributed import Client, get_worker
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
from transformers import pipeline

with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

DATASET_PARQUET = "checkpoint_dataset"
EMBEDDINGS_PICKLE = "checkpoint_embeddings"
CLUSTERED_PARQUET = "checkpoint_clustered"
FINAL_PARQUET = "clustered_output"
CLASS_BATCH_SIZE = 16

# Get the logdir from SLURM environment
log_dir = os.environ.get("LOGDIR", ".")
os.makedirs(log_dir, exist_ok=True)

# Setup logging
log_path = os.path.join(log_dir, "distributed_clustering.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)
logger.info("Logging initialized")

# --------- Utility Functions ---------


def compute_data_hash(rows):
    hasher = hashlib.sha256()
    for row in rows:
        hasher.update(str(row).encode('utf-8'))
    return hasher.hexdigest()


def detect_lang(text):
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    if not hasattr(detect_lang, "model"):
        model_path = CONFIG["language_detection_model"]["path"]
        download_url = CONFIG["language_detection_model"]["download_url"]
        if not os.path.exists(model_path):
            logger.info(f"Downloading language model from {download_url}...")
            urllib.request.urlretrieve(download_url, model_path)
            logger.info("Download complete.")
        detect_lang.model = fasttext.load_model(model_path)

    clean_text = text.replace('\n', ' ').strip()
    label, prob = detect_lang.model.predict(clean_text, k=1)
    return label[0].replace("__label__", ""), prob[0]


def extract_language(series):
    return series.apply(lambda t: detect_lang(t)[0])


def extract_confidence(series):
    return series.apply(lambda t: detect_lang(t)[1])


def download_and_extract(url, dest_path):
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        logger.info(f"Downloading embedding model from {url}...")
        urllib.request.urlretrieve(url, tmp_file.name)
        logger.info("Download complete. Extracting...")
        with zipfile.ZipFile(tmp_file.name, "r") as zip_ref:
            zip_ref.extractall(dest_path)
        logger.info("Extraction complete.")


def get_embedder():
    worker = get_worker()
    if not hasattr(worker, "embedder"):
        model_config = CONFIG["embedding_model"]
        model_path = model_config["path"]
        trust_remote_code = model_config.get("trust_remote_code", False)

        if not os.path.exists(model_path):
            download_url = model_config.get("download_url")
            if not download_url:
                raise ValueError(
                    "Model path not found and no download_url provided."
                )
            # Create destination dir
            os.makedirs(model_path, exist_ok=True)
            download_and_extract(download_url, os.path.dirname(model_path))

        worker.embedder = SentenceTransformer(
            model_path, trust_remote_code=trust_remote_code
        )
    return worker.embedder


def embed_partition(texts):
    model = get_embedder()
    embeddings = model.encode(
        texts.tolist(), show_progress_bar=False, batch_size=64
    )
    return pd.DataFrame(embeddings)


def get_classifier():
    worker = get_worker()
    if not hasattr(worker, "classifier"):
        worker.classifier = pipeline(
            "zero-shot-classification",
            model=CONFIG["zero_shot_model"]["name"],
            device=CONFIG["zero_shot_model"]["device"],
        )
    return worker.classifier


def row_generator(source):
    for i, row in enumerate(source):
        if i < CONFIG["dataset"]["sample_size"]:
            yield {"text": row[CONFIG["dataset"]["key"]]}


def classify_batch_partition(df_partition, expected_columns):
    if df_partition.empty:
        return df_partition

    classifier = get_classifier()
    texts = df_partition['text'].fillna("").tolist()

    domains = []
    confidences = []

    num_batches = (len(texts) + CLASS_BATCH_SIZE - 1) // CLASS_BATCH_SIZE
    for i in tqdm(
        range(0, len(texts), CLASS_BATCH_SIZE),
        desc="Classifying",
        total=num_batches,
    ):
        batch = texts[i : i + CLASS_BATCH_SIZE]
        results = classifier(
            batch,
            candidate_labels=CONFIG['classification_attributes'][
                'candidate_labels'
            ],
        )

        if isinstance(results, dict):
            results = [results]

        domains.extend([r["labels"][0] for r in results])
        confidences.extend([r["scores"][0] for r in results])

    df_partition['domain'] = domains
    df_partition['domain_confidence'] = confidences

    return df_partition[expected_columns]


# --------- Main Pipeline ---------

if __name__ == "__main__":
    logger.info("Starting distributed pipeline...")

    scheduler_file = os.environ["SCHEDULER_FILE"]
    client = Client(scheduler_file=scheduler_file)
    logger.info(f"Dask dashboard: {client.dashboard_link}")

    # Step 1: Load dataset
    dataset = load_dataset(
        CONFIG["dataset"]["name"],
        split=CONFIG["dataset"]["split"],
        streaming=False,
    )

    rows = list(row_generator(dataset))
    current_hash = compute_data_hash(rows)
    dataset_file_name = f"{DATASET_PARQUET}_{current_hash}.parquet"
    if os.path.exists(dataset_file_name):
        df = dd.read_parquet(dataset_file_name)
    else:
        bag = db.from_sequence(rows, npartitions=4)
        meta = {'text': str}
        df = bag.to_dataframe(meta=meta)
    if df.shape[0].compute() == 0:
        logger.warning("Dask DataFrame is empty. No data was loaded.")
    else:
        logger.info(
            f"Data loaded successfully with {df.shape[0].compute()} rows."
        )

    # Language detection
    meta_str = pd.Series(dtype='object')
    meta_flt = pd.Series(dtype='float64')
    df['language'] = df['text'].map_partitions(extract_language, meta=meta_str)
    df['lang_confidence'] = df['text'].map_partitions(
        extract_confidence, meta=meta_flt
    )
    df = df.persist()
    df.to_parquet(dataset_file_name, write_index=False)
    logger.info("Language filtering complete and checkpoint saved.")

    # Step 2: Embedding
    embedding_file_name = f"{EMBEDDINGS_PICKLE}_{current_hash}.npy"
    if os.path.exists(embedding_file_name):
        embedding_matrix = np.load(embedding_file_name)
        logger.info("Loaded embeddings from checkpoint.")
    else:
        embeddings = df['text'].map_partitions(
            embed_partition, meta=pd.DataFrame(np.zeros((0, 1024)))
        )
        embedding_matrix = embeddings.compute().values
        np.save(embedding_file_name, embedding_matrix)
        logger.info("Embeddings computed and saved.")

    # Step 3: Clustering
    clustering_file_name = f"{CLUSTERED_PARQUET}_{current_hash}.parquet"
    if os.path.exists(clustering_file_name):
        df = dd.read_parquet(clustering_file_name)
        logger.info("Loaded clustered data from checkpoint.")
    else:
        clustering_model = HDBSCAN(
            CONFIG['clustering_attributes']['min_cluster_size']
        )
        cluster_labels = clustering_model.fit(embedding_matrix).labels_
        df = df.compute()
        df['cluster'] = cluster_labels
        logger.info(df.columns)
        df = dd.from_pandas(df, npartitions=32).persist()
        df.to_parquet(clustering_file_name, write_index=False)
        logger.info("Cluster sizes:\n", df['cluster'].value_counts())
        logger.info("Clustering done and checkpoint saved.")

    # Step 4: Zero-shot classification
    final_file_name = f"{FINAL_PARQUET}_{current_hash}.parquet"
    if os.path.exists(final_file_name):
        df_final = pd.read_parquet(final_file_name)
        logger.info("Final output loaded from checkpoint.")
    else:
        cluster_representatives = df.groupby('cluster').first().reset_index()

        meta = df._meta.copy()
        meta['domain'] = pd.Series(dtype='object')
        meta['domain_confidence'] = pd.Series(dtype='float64')
        expected_columns = meta.columns.tolist()
        classified_heads = cluster_representatives.map_partitions(
            classify_batch_partition, expected_columns, meta=meta
        ).compute()
        # Step 4: Merge predictions back into full dataframe
        df_final = df.merge(
            classified_heads[["cluster", "domain", "domain_confidence"]],
            on="cluster",
            how="left",
        )[
            [
                'text',
                'language',
                'lang_confidence',
                'cluster',
                'domain',
                'domain_confidence',
            ]
        ]
        df_final = df_final.compute()
        df_final.to_parquet(final_file_name, index=False)
        logger.info("Final classified output saved (cluster-level).")
    # Step 5: Show results
    logger.info(df_final.head(10))
    logger.info("Cluster sizes:\n", df_final['cluster'].value_counts())
