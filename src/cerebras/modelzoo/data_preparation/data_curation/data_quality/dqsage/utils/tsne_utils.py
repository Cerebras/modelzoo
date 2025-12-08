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

"""Core utilities for t-SNE visualizer (format & compute logic only).

Separated from the Streamlit UI so the logic can be unit tested and reused.

Responsibilities:
 - Metadata file discovery & streaming across .jsonl / .jsonl.gz / .jsonl.zst
 - Sampling (reservoir & fast skip sample)
 - Embedding shard gathering
 - t-SNE runner
 - Distribution / divergence metrics
 - Plot construction (Plotly figure) kept here for reuse (UI can call directly)

All functions are pure (side-effect free) except for reading files.  No Streamlit
imports: UI layer handles caching & messaging. Optional callbacks supplied for
status / progress reporting.
"""

from __future__ import annotations

import glob
import gzip
import io
import os
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import orjson
import pandas as pd
import plotly.graph_objects as go
import zstandard as zstd

SUPPORTED_META_EXTS = (".jsonl", ".jsonl.gz", ".jsonl.zst")


def iter_metadata_files(labelled_dir: str) -> List[str]:
    files: List[str] = []
    for ext in SUPPORTED_META_EXTS:
        files.extend(glob.glob(os.path.join(labelled_dir, f"*{ext}")))
    return sorted(files)


def open_metadata_file(path: str):
    if path.endswith(".jsonl.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    if path.endswith(".jsonl.zst"):
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(fh)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="ignore")
    return open(path, "rt", encoding="utf-8", errors="ignore")


def extract_path(obj: Dict[str, Any], path: str):
    if not path:
        return None
    if "." not in path:
        return obj.get(path)
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
        if cur is None:
            return None
    return cur


def reservoir_sample_metadata(
    labelled_dir: str,
    sample_size: int,
    id_field: str = "id",
    text_field: str = "text",
    cluster_field: str = "cluster_id",
    extra_field: str = "",
    seed: int = 42,
    status_cb: Optional[Callable[[str], None]] = None,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[np.ndarray, List[str], List[Any], int, List[Any]]:
    files = iter_metadata_files(labelled_dir)
    if not files:
        raise FileNotFoundError(
            f"No metadata files (.jsonl/.jsonl.gz/.jsonl.zst) found in {labelled_dir}"
        )
    rng = np.random.default_rng(seed)
    ids: List[str] = []
    texts: List[str] = []
    clusters: List[Any] = []
    extras: List[Any] = []
    total = 0
    PROGRESS_EVERY = 10_000
    last_prog = 0
    for fidx, fp in enumerate(files):
        try:
            with open_metadata_file(fp) as f:
                for line in f:
                    total += 1
                    if not line.strip():
                        continue
                    try:
                        obj = orjson.loads(line)
                    except Exception:
                        continue
                    _id = extract_path(obj, id_field)
                    if _id is None:
                        continue
                    _id = str(_id)
                    txt = extract_path(obj, text_field) if text_field else ""
                    cl = (
                        extract_path(obj, cluster_field)
                        if cluster_field
                        else None
                    )
                    ex = extract_path(obj, extra_field) if extra_field else None
                    if len(ids) < sample_size:
                        ids.append(_id)
                        texts.append(txt)
                        clusters.append(cl)
                        if extra_field:
                            extras.append(ex)
                    else:
                        j = rng.integers(0, total)
                        if j < sample_size:
                            ids[j] = _id
                            texts[j] = txt
                            clusters[j] = cl
                            if extra_field:
                                if len(extras) < sample_size:
                                    extras.extend(
                                        [None] * (sample_size - len(extras))
                                    )
                                extras[j] = ex
                    if progress_cb and (total - last_prog) >= PROGRESS_EVERY:
                        try:
                            progress_cb(fidx + 1, len(files), total)
                        except Exception:
                            pass
                        last_prog = total
            if status_cb and (fidx + 1) % 10 == 0:
                status_cb(
                    f"Metadata reservoir: {fidx+1}/{len(files)} files | Seen {total:,}"
                )
            if progress_cb:
                try:
                    progress_cb(fidx + 1, len(files), total)
                except Exception:
                    pass
        except Exception as e:  # pragma: no cover
            if status_cb:
                status_cb(f"Warn metadata sample {fp}: {e}")
            if progress_cb:
                try:
                    progress_cb(fidx + 1, len(files), total)
                except Exception:
                    pass
    if not ids:
        raise RuntimeError("No metadata sampled.")
    if extra_field and len(extras) < len(ids):
        extras.extend([None] * (len(ids) - len(extras)))
    return np.array(ids), texts, clusters, total, extras if extra_field else []


def gather_embeddings_for_ids(
    embedding_dir: str,
    target_ids: np.ndarray,
    embedding_id_key: str = "doc_ids",
    status_cb: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    wanted = set(target_ids.tolist())
    if not wanted:
        raise ValueError("No target ids provided.")
    id_to_row = {tid: i for i, tid in enumerate(target_ids)}
    emb_matrix: Optional[np.ndarray] = None
    collected = 0
    files = sorted(glob.glob(os.path.join(embedding_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {embedding_dir}")
    for fidx, fp in enumerate(files):
        try:
            with np.load(fp, allow_pickle=False, mmap_mode="r") as data:
                if embedding_id_key not in data:
                    raise KeyError(
                        f"Embedding id key '{embedding_id_key}' missing in shard {fp}"
                    )
                doc_ids = data[embedding_id_key]
                vecs = data["embeddings"]
                if emb_matrix is None:
                    emb_matrix = np.zeros(
                        (len(target_ids), vecs.shape[1]), dtype=np.float32
                    )
                for i, did in enumerate(doc_ids):
                    _id = str(did)
                    if _id in wanted:
                        emb_matrix[id_to_row[_id]] = vecs[i].astype(np.float32)
                        collected += 1
                        wanted.remove(_id)
                        if not wanted:
                            break
            if status_cb and (fidx + 1) % 5 == 0:
                status_cb(
                    f"Embeddings fetch: {fidx+1}/{len(files)} shards | Collected {collected}/{len(target_ids)}"
                )
            if not wanted:
                break
        except Exception as e:  # pragma: no cover
            if status_cb:
                status_cb(f"Warn embedding shard {fp}: {e}")
            continue
    if emb_matrix is None:
        raise RuntimeError("Failed to collect embeddings (matrix None).")
    mask = np.any(emb_matrix != 0, axis=1)
    return emb_matrix[mask]


def run_tsne(
    features: np.ndarray,
    dim: int,
    perplexity: int,
    learning_rate,
    max_iter: int,
) -> np.ndarray:
    import inspect

    from sklearn.manifold import TSNE

    sig = inspect.signature(TSNE)
    kwargs = dict(
        n_components=dim,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="pca",
        random_state=42,
    )
    if "max_iter" in sig.parameters:
        kwargs["max_iter"] = max_iter
    elif "n_iter" in sig.parameters:
        kwargs["n_iter"] = max_iter
    tsne = TSNE(**kwargs)
    return tsne.fit_transform(features)


def wrap_text_for_hover(s: str, width: int = 80, max_lines: int = 6) -> str:
    s = s or ""
    if len(s) > 500:
        s = s[:500] + "…"
    words = s.split()
    lines: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for w in words:
        add_len = len(w) + (1 if cur else 0)
        if cur_len + add_len > width:
            lines.append(" ".join(cur))
            if len(lines) >= max_lines:
                return "<br>".join(lines) + " …"
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += add_len
    if cur:
        lines.append(" ".join(cur))
    return "<br>".join(lines)


def plot_embedding(df_embed: pd.DataFrame, dim: int) -> go.Figure:
    cluster_col = (
        "cluster_value" if "cluster_value" in df_embed.columns else None
    )
    text_col = (
        "hover_text"
        if "hover_text" in df_embed.columns
        else ("text" if "text" in df_embed.columns else None)
    )
    traces: List[Any] = []
    if cluster_col:
        for cl, sub in df_embed.groupby(cluster_col):
            hover_text = sub[text_col] if text_col else None
            if dim == 2:
                traces.append(
                    go.Scatter(
                        name="<br>".join(textwrap.wrap(str(cl), width=20)),
                        x=sub["tsne_1"],
                        y=sub["tsne_2"],
                        mode="markers",
                        marker=dict(size=5, symbol="circle"),
                        text=hover_text,
                        hovertemplate=(
                            "%{text}<extra></extra>" if text_col else None
                        ),
                    )
                )
            else:
                traces.append(
                    go.Scatter3d(
                        name="<br>".join(textwrap.wrap(str(cl), width=20)),
                        x=sub["tsne_1"],
                        y=sub["tsne_2"],
                        z=sub["tsne_3"],
                        mode="markers",
                        marker=dict(size=3, symbol="circle"),
                        text=hover_text,
                        hovertemplate=(
                            "%{text}<extra></extra>" if text_col else None
                        ),
                    )
                )
    else:
        if dim == 2:
            traces.append(
                go.Scatter(
                    x=df_embed["tsne_1"],
                    y=df_embed["tsne_2"],
                    mode="markers",
                    marker=dict(size=5),
                    text=df_embed[text_col] if text_col else None,
                    hovertemplate=(
                        "%{text}<extra></extra>" if text_col else None
                    ),
                )
            )
        else:
            traces.append(
                go.Scatter3d(
                    x=df_embed["tsne_1"],
                    y=df_embed["tsne_2"],
                    z=df_embed["tsne_3"],
                    mode="markers",
                    marker=dict(size=3),
                    text=df_embed[text_col] if text_col else None,
                    hovertemplate=(
                        "%{text}<extra></extra>" if text_col else None
                    ),
                )
            )
    fig = go.Figure(data=traces)
    if dim == 2:
        fig.update_layout(
            height=650, margin=dict(l=10, r=10, b=10, t=30), uirevision="tsne"
        )
    else:
        fig.update_layout(
            height=750,
            scene=dict(
                xaxis_title="t-SNE 1",
                yaxis_title="t-SNE 2",
                zaxis_title="t-SNE 3",
            ),
            margin=dict(l=10, r=10, b=10, t=30),
            uirevision="tsne",
        )
    if traces:
        n = len(traces)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            label="Select All",
                            method="update",
                            args=[{"visible": [True] * n}],  # set all visible
                        ),
                        dict(
                            label="Deselect All",
                            method="update",
                            args=[
                                {"visible": ["legendonly"] * n}
                            ],  # hide but keep legend
                        ),
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0,
                    y=1.12,
                    xanchor="left",
                    yanchor="top",
                )
            ]
        )
    return fig


def normalize_cluster_label(raw):
    if raw is None:
        return "UNKNOWN"
    if isinstance(raw, str):
        s = raw.strip()
        if s == "" or s.lower() in {"none", "null"}:
            return "UNKNOWN"
        return s
    return str(raw)


def scan_label_distribution(
    labelled_dir: str,
    cluster_field: str,
    status_cb: Optional[Callable[[str], None]] = None,
):
    files = iter_metadata_files(labelled_dir)
    from collections import Counter

    counts: Counter = Counter()
    total = 0
    for fidx, fp in enumerate(files):
        try:
            with open_metadata_file(fp) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = orjson.loads(line)
                    except Exception:
                        continue
                    cl_raw = (
                        extract_path(obj, cluster_field)
                        if cluster_field
                        else None
                    )
                    counts[normalize_cluster_label(cl_raw)] += 1
                    total += 1
            if status_cb and (fidx + 1) % 10 == 0:
                status_cb(
                    f"Scanned {fidx+1}/{len(files)} files | {total:,} records"
                )
        except Exception as e:  # pragma: no cover
            if status_cb:
                status_cb(f"Warn distribution scan {fp}: {e}")
    return counts


def entropy_bits(percs: pd.Series) -> float:
    probs = percs / 100.0
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def jsd_bits(p: pd.Series, q: pd.Series) -> float:
    all_idx = sorted(set(p.index) | set(q.index))
    p_vec = p.reindex(all_idx, fill_value=0).to_numpy(dtype=float)
    q_vec = q.reindex(all_idx, fill_value=0).to_numpy(dtype=float)
    if p_vec.sum() == 0:
        p_vec = np.ones_like(p_vec)
    if q_vec.sum() == 0:
        q_vec = np.ones_like(q_vec)
    p_vec /= p_vec.sum()
    q_vec /= q_vec.sum()
    m = 0.5 * (p_vec + q_vec)

    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        return np.sum(a[mask] * (np.log2(a[mask]) - np.log2(b[mask])))

    return 0.5 * (_kl(p_vec, m) + _kl(q_vec, m))


__all__ = [
    "SUPPORTED_META_EXTS",
    "iter_metadata_files",
    "open_metadata_file",
    "reservoir_sample_metadata",
    "gather_embeddings_for_ids",
    "run_tsne",
    "plot_embedding",
    "wrap_text_for_hover",
    "extract_path",
    "scan_label_distribution",
    "entropy_bits",
    "jsd_bits",
    "normalize_cluster_label",
]
