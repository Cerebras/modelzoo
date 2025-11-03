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
import subprocess
import tempfile
import threading
import urllib.request
import zipfile

import pandas as pd
from dask.distributed import get_worker
from sentence_transformers import SentenceTransformer
from transformers import pipeline


def get_embedder(model_cfg, logger, device):
    """Thread-safe per-worker SentenceTransformer loader."""
    worker = get_worker()
    if hasattr(worker, "embedder"):
        return worker.embedder
    # lock to avoid races
    if not hasattr(worker, "lock"):
        worker.lock = threading.Lock()
    with worker.lock:
        if hasattr(worker, "embedder"):
            return worker.embedder
        path = model_cfg["path"]
        # download if needed
        if model_cfg.get("download_url") and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            logger.info(
                f"Downloading embeddings from {model_cfg['download_url']}..."
            )
            urllib.request.urlretrieve(model_cfg["download_url"], tmp.name)
            with zipfile.ZipFile(tmp.name, "r") as zp:
                zp.extractall(os.path.dirname(path))
            logger.info("Embedding model ready")
        worker.embedder = SentenceTransformer(
            path,
            device,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        return worker.embedder


def get_classifier(device, zero_shot_model_config):
    """Per-worker zero-shot classification pipeline."""
    worker = get_worker()
    if hasattr(worker, "classifier"):
        return worker.classifier
    # Determine device for classifier - use GPU device if specified, otherwise fallback to config or CPU
    classifier_device = (
        device
        if device.startswith('cuda')
        else zero_shot_model_config.get("device", -1)
    )
    worker.classifier = pipeline(
        "zero-shot-classification",
        model=zero_shot_model_config.get("name", ""),
        device=classifier_device,
    )
    return worker.classifier


def embed_partition(texts, config, logger, device, dim):
    """Embed a pandas Series of texts into a DataFrame of embeddings."""
    model = get_embedder(config, logger, device)
    embs = model.encode(
        texts.tolist(), show_progress_bar=False, batch_size=config["batch_size"]
    )
    cols = [f"embed_{i}" for i in range(dim)]
    return pd.DataFrame(embs, index=texts.index, columns=cols)


def classify_partition(
    df, config, zero_shot_model_config, device, expected_cols
):
    """Apply zero-shot classification to a partition."""
    if df.empty:
        return df
    texts = df["text"].fillna("").tolist()
    domains, scores = [], []
    clf = get_classifier(device, zero_shot_model_config)
    batch = config["batch_size"]
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        res = clf(chunk, candidate_labels=config["labels"])
        if isinstance(res, dict):
            res = [res]
        domains += [r["labels"][0] for r in res]
        scores += [r["scores"][0] for r in res]
    df["domain"] = domains
    df["domain_confidence"] = scores
    return df[expected_cols]


def lang_partition(df_partition, config, logger):
    texts = df_partition["text"].fillna("").tolist()
    bin_path = config["binary_path"]
    model_path = config["model_path"]

    with tempfile.NamedTemporaryFile("w", delete=False) as tf:
        for line in texts:
            tf.write(line.replace("\n", " ") + "\n")
        tf_path = tf.name

    cmd = [bin_path, "predict", model_path, tf_path]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        raw_preds = result.stdout.strip().split("\n")
        langs = [x.replace("__label__", "") for x in raw_preds]
        confs = [1.0] * len(langs)  # fasttext binary doesn't give scores
    except subprocess.CalledProcessError as e:
        logger.error(f"FastText prediction failed: {e.stderr}")
        raise

    df_partition["language"] = langs
    df_partition["lang_confidence"] = confs
    return df_partition
