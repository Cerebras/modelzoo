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

"""t-SNE Visualizer UI

UI layer only. All heavy lifting (I/O, sampling, t-SNE, plotting helpers,
distribution scanning) lives in ``tsne_utils.py``.

Workflow:
    1. Discover & sample metadata (multi-format .jsonl /.jsonl.gz /.jsonl.zst).
    2. Fetch corresponding embeddings (npz shards) lazily.
    3. Run t-SNE and display interactive Plotly scatter (2D/3D).
    4. Provide full label distribution (with optional sample comparison).

State Persistence:
    - Last computed embedding + figure retained in session_state so UI tweaks
        (e.g. sidebar changes) don't discard the plot until user re-runs.
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # needed for distribution bar charts
import streamlit as st
from dqsage.utils.tsne_utils import (
    entropy_bits,
    gather_embeddings_for_ids,
    iter_metadata_files,
    jsd_bits,
    open_metadata_file,
    plot_embedding,
    reservoir_sample_metadata,
    run_tsne,
    scan_label_distribution,
    wrap_text_for_hover,
)


@st.cache_data(show_spinner=False)
def _cached_label_distribution(
    labelled_dir: str,
    cluster_field: str,
    file_meta: Tuple[Tuple[str, float], ...],
) -> pd.DataFrame:
    counts = scan_label_distribution(labelled_dir, cluster_field)
    if not counts:
        return pd.DataFrame(
            columns=["cluster_value", "count", "percent", "cumulative_percent"]
        )
    total = sum(counts.values())
    rows = [(cl, ct, ct / total * 100.0) for cl, ct in counts.most_common()]
    df = pd.DataFrame(rows, columns=["cluster_value", "count", "percent"])
    df["cumulative_percent"] = df["percent"].cumsum()
    return df


def main():  # entry point required by launcher
    st.markdown(
        "<h2 style='text-align:center'>🌀 t-SNE Visualizer</h2>",
        unsafe_allow_html=True,
    )

    # Sidebar configuration -------------------------------------------------
    st.sidebar.title('t-SNE Parameters (Embeddings)')
    st.sidebar.markdown("**Input Directories**")
    labelled_dir = st.sidebar.text_input(
        'Labelled Data Directory',
        value='data_jsonl',
        help="Directory with metadata files (.jsonl / .jsonl.gz / .jsonl.zst) containing id, text, cluster_id",
    )
    embedding_dir = st.sidebar.text_input(
        'Embedding Directory (.npz)',
        value='embeddings',
        help="Directory with *.npz files (doc_ids, embeddings)",
    )
    sample_size = st.sidebar.number_input(
        'Sample Size',
        min_value=100,
        max_value=200000,
        step=100,
        value=5000,
        help="Uniform random sample size for t-SNE (reservoir sampling across shards).",
    )
    seed = st.sidebar.number_input(
        'Sample Seed',
        min_value=0,
        max_value=10_000_000,
        value=42,
        step=1,
        help="Random seed for reproducible sampling.",
    )

    dim = st.sidebar.selectbox('Embedded Space Dimension', [2, 3], index=0)
    perplexity = st.sidebar.slider(
        'Perplexity', min_value=5, max_value=100, step=1, value=30
    )
    max_iter = st.sidebar.slider(
        'Iterations (max_iter)',
        min_value=250,
        max_value=5000,
        step=50,
        value=750,
        help="Total optimization iterations (mapped to max_iter or n_iter depending on backend).",
    )
    st.sidebar.markdown("**Learning Rate**")
    lr_mode = st.sidebar.radio(
        'Mode',
        ['auto', 'manual'],
        index=0,
        help="Use 'auto' (sklearn adaptive heuristic) or specify a fixed numeric learning rate.",
    )
    if lr_mode == 'manual':
        learning_rate = st.sidebar.slider(
            'Learning Rate Value',
            min_value=10,
            max_value=1000,
            step=10,
            value=200,
            help="Numeric learning rate passed directly to TSNE.",
        )
    else:
        learning_rate = 'auto'

    # Dynamic field detection (after directories entered)
    @st.cache_data(show_spinner=False)
    def _peek_json_field_paths(labelled_dir: str):
        """Return top-level keys plus one-level nested keys as dot paths (k.sub)."""
        try:
            files = iter_metadata_files(labelled_dir)
            for fp in files:
                with open_metadata_file(fp) as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        roots = list(obj.keys())
                        nested: List[str] = []
                        for k, v in obj.items():
                            if isinstance(v, dict):
                                for sk in v.keys():
                                    nested.append(f"{k}.{sk}")
                        return roots + nested
            return []
        except Exception:
            return []

    @st.cache_data(show_spinner=False)
    def _peek_npz_fields(embedding_dir: str):
        try:
            files = sorted(glob.glob(os.path.join(embedding_dir, '*.npz')))
            for fp in files[:1]:
                with np.load(fp, allow_pickle=False) as data:
                    return list(data.keys())
            return []
        except Exception:
            return []

    json_fields = _peek_json_field_paths(labelled_dir) if labelled_dir else []
    npz_fields = _peek_npz_fields(embedding_dir) if embedding_dir else []

    # Suggested defaults
    # simple helper removed (not needed)

    id_field = st.sidebar.selectbox(
        'JSONL ID Field',
        options=json_fields or ['id'],
        index=(json_fields.index('id') if 'id' in json_fields else 0),
    )
    text_field = st.sidebar.selectbox(
        'JSONL Text Field',
        options=([''] + json_fields) if json_fields else ['text'],
        index=((json_fields.index('text') + 1) if 'text' in json_fields else 0),
    )
    cluster_field = st.sidebar.selectbox(
        'JSONL Cluster Field',
        options=([''] + json_fields) if json_fields else ['cluster_id'],
        index=(
            (json_fields.index('cluster_id') + 1)
            if 'cluster_id' in json_fields
            else 0
        ),
    )
    extra_hover_field = st.sidebar.selectbox(
        'JSONL Extra Hover Field',
        options=([''] + json_fields) if json_fields else [''],
        index=0,
        help='Optional additional field to append in hover tooltip.',
    )
    with st.sidebar.expander(
        'Manual Path Override (dot paths)', expanded=False
    ):
        manual_id = st.text_input('Manual ID Path', value='')
        manual_text = st.text_input('Manual Text Path', value='')
        manual_cluster = st.text_input('Manual Cluster Path', value='')
        manual_extra_hover_field = st.text_input(
            'Manual Extra Hover Field', value=''
        )
    if manual_id.strip():
        id_field = manual_id.strip()
    if manual_text.strip():
        text_field = manual_text.strip()
    if manual_cluster.strip():
        cluster_field = manual_cluster.strip()
    if manual_extra_hover_field.strip():
        extra_hover_field = manual_extra_hover_field.strip()
    embedding_id_key = st.sidebar.selectbox(
        'Embedding ID Key',
        options=npz_fields or ['doc_ids'],
        index=(npz_fields.index('doc_ids') if 'doc_ids' in npz_fields else 0),
    )

    tab_desc, tab_view, tab_dist = st.tabs(
        ['Description', 'Viewer', 'Distribution']
    )

    # --------------------------- Description Tab ---------------------------
    with tab_desc:
        st.title('t-SNE Embedding Visualization 😎')
        st.header('Explore High-Dimensional Embeddings in 2D / 3D')
        st.markdown(
            """
                        ### Quick Workflow
                        1. Select **Labelled Data Directory** containing `.jsonl`, `.jsonl.gz` or `.jsonl.zst` lines (each with an `id`, optional `text`, and optional `cluster_id`).
                        2. Select **Embedding Directory** of NPZ shards (`doc_ids`, `embeddings`).
                        3. Choose **sample size** (uniform reservoir sampling; unbiased across all files).
                        4. Tune t-SNE params (dimension, perplexity, iterations, learning rate).
                        5. Click **Run t-SNE** → interactive 2D / 3D Plotly scatter (color = cluster, hover = wrapped text + optional extra field).
                        6. Open the **Distribution** tab to scan the *full* dataset label frequencies & compare against the sample.

                        ### What This Gives You
                        - **Reservoir Sampling**: True single‑pass uniform sample across arbitrarily large corpora.
                        - **Multi-Format Metadata**: `.jsonl`, `.jsonl.gz`, `.jsonl.zst` (stream decompressed).
                        - **2D / 3D Projection**: Fast exploratory layout.
                        - **Cluster Diagnostics**: Legend bulk toggle (Select / Deselect All), hover tooltips, optional extra field.
                        - **Distribution Analytics** (full dataset, cached):
                            - *Entropy (bits)*: Diversity / balance of labels.
                            - *Jensen–Shannon Distance (bits)*: Bias of sample vs full distribution (0 = perfect match).

                        ### Parameter Tips
                        - **Sample Size**: Start 2k–5k; increase only if structure unclear (complexity ~ O(N²)).
                        - **Perplexity**: Keep `< sample_size / 3`; common range 20–40.
                        - **Dimension**: Start in 2D (faster), move to 3D once clusters validated.
                        - **Iterations**: 500–1000 usually enough; raise only for subtle separation.
                        - **Learning Rate**: Leave `auto` unless you see crowding (then try 200–400 manual).
                        - **Extra Hover Field**: Provide contextual label (e.g. topic, doc length bucket).

                        ### Interpreting Metrics
                        - **Entropy**: Low → dominated by few clusters; high → balanced. Max ≈ log2(k).
                        - **JSD**: < 0.05 very close; 0.05–0.12 mild drift; > 0.12 notable sampling bias.

                        ### Performance & Quality Hints
                        - Large dominant clusters? Consider smaller sample or stratified pre-filter outside the tool.
                        - Memory pressure? Reduce sample size or use 2D only.
                        - Re-run with same seed for deterministic comparison after parameter tweaks.
                        - If many labels show as UNKNOWN, adjust **Cluster Field** or manual dot‑path override.
            """
        )

    # --------------------------- Viewer Tab -------------------------------
    with tab_view:
        # Acquire Data -----------------------------------------------------
        st.subheader("Embeddings Dataset")

        run = st.button("Run t-SNE", type="primary")
        if run:
            if not labelled_dir or not embedding_dir:
                st.error("Provide both labelled and embedding directories.")
                return
            try:
                progress_box = st.empty()

                def status(msg: str):
                    progress_box.info(msg)

                with st.spinner(
                    "Reservoir sampling metadata (ids/text/cluster)..."
                ):
                    # progress_text = st.empty()

                    def progress(
                        files_done: int, total_files: int, lines_seen: int
                    ):
                        # progress_text.caption(
                        #     f"Lines read: {lines_seen:,} | Files: {files_done}/{total_files}"
                        # )
                        status(
                            f"Lines read: {lines_seen:,} | Files: {files_done}/{total_files}"
                        )

                    sampled_ids, texts, clusters, total_meta, extras = (
                        reservoir_sample_metadata(
                            labelled_dir,
                            int(sample_size or 0),
                            id_field=id_field,
                            text_field=text_field,
                            cluster_field=cluster_field or "cluster_id",
                            extra_field=extra_hover_field,
                            seed=seed,
                            status_cb=status,
                            progress_cb=progress,
                        )
                    )
                st.caption(
                    f"Metadata reservoir sampled {len(sampled_ids):,} / {total_meta:,} records. Fetching embeddings..."
                )
                with st.spinner("Gathering embeddings for sampled ids..."):
                    sampled_vecs = gather_embeddings_for_ids(
                        embedding_dir,
                        sampled_ids,
                        embedding_id_key=embedding_id_key,
                        status_cb=status,
                    )
                if sampled_vecs.shape[0] != len(
                    sampled_ids
                ):  # align w/ missing
                    valid_count = sampled_vecs.shape[0]
                    sampled_ids = sampled_ids[:valid_count]
                    texts = texts[:valid_count]
                    clusters = clusters[:valid_count]
                with st.spinner("Running t-SNE..."):
                    embedding = run_tsne(
                        sampled_vecs,
                        dim,
                        perplexity=perplexity,
                        learning_rate=learning_rate,
                        max_iter=max_iter,
                    )
                embed_df = pd.DataFrame(
                    embedding, columns=[f"tsne_{i+1}" for i in range(dim)]
                )
                cluster_values = [
                    (
                        'UNKNOWN'
                        if (
                            cluster_field == ''
                            or c is None
                            or (
                                isinstance(c, str)
                                and c.lower() in ('', 'none', 'null')
                            )
                        )
                        else str(c)
                    )
                    for c in clusters
                ]
                unique_vals = sorted(set(cluster_values))
                val_to_code = {v: i for i, v in enumerate(unique_vals)}
                embed_df['cluster_value'] = cluster_values
                embed_df['cluster_code'] = [
                    val_to_code[v] for v in cluster_values
                ]
                embed_df['text'] = texts

                extra_vals = extras[: len(embed_df)] if extras else []
                hv_texts: List[str] = []
                for i, row in embed_df.iterrows():
                    base = (
                        wrap_text_for_hover(str(row["text"]))
                        if "text" in embed_df.columns
                        else ""
                    )
                    parts = [
                        f"text: {base}",
                        f"cluster: {row['cluster_value']}",
                    ]
                    if extra_hover_field and extra_vals:
                        exv = extra_vals[i] if i < len(extra_vals) else None
                        if exv is not None:
                            parts.append(f"{extra_hover_field}: {exv}")
                    hv_texts.append('<br>'.join(parts))
                embed_df['hover_text'] = hv_texts
                embed_df['id'] = sampled_ids
                fig = plot_embedding(embed_df, dim)
                st.session_state['tsne_embedding'] = embed_df
                st.session_state['tsne_fig'] = fig
                st.session_state['tsne_cluster_map'] = {
                    code: val for val, code in val_to_code.items()
                }
                st.session_state['tsne_dim'] = dim
                st.success("t-SNE completed.")
            except Exception as e:
                st.error(f"Error processing embeddings dataset: {e}")
                return

        # Always display last successful plot (if any), regardless of sidebar tweaks
        if (
            'tsne_fig' in st.session_state
            and 'tsne_embedding' in st.session_state
        ):
            st.plotly_chart(
                st.session_state['tsne_fig'], use_container_width=True
            )
            # with st.expander("Embedding Data (head)", expanded=False):
            #     st.dataframe(st.session_state['tsne_embedding'].head())
            # with st.expander("Cluster Mapping", expanded=False):
            #     st.write(st.session_state.get('tsne_cluster_map', {}))
            if not run:
                st.caption(
                    "Showing last computed t-SNE. Press 'Run t-SNE' to recompute with current parameters."
                )
        else:
            st.info(
                "Configure parameters and click 'Run t-SNE' to generate embedding."
            )

    # --------------------------- Distribution Tab -------------------------------
    with tab_dist:
        st.subheader("Label Distribution (Full Dataset)")
        if not labelled_dir:
            st.info("Provide labelled data directory in sidebar.")
        else:
            compute_col, opts_col = st.columns([1, 3])
            with opts_col:
                top_k = st.number_input(
                    'Top K clusters',
                    min_value=5,
                    max_value=200,
                    value=30,
                    step=1,
                )
                log_scale = st.checkbox('Log scale (counts)', value=False)
                compare = st.checkbox(
                    'Compare with current sample',
                    value=False,
                    disabled='tsne_embedding' not in st.session_state,
                    help='Requires a t-SNE run (sampled embedding) in this session.',
                )
            with compute_col:
                run_dist = st.button('Compute Distribution', type='primary')

            if run_dist:
                try:
                    gz_files = iter_metadata_files(labelled_dir)
                    if not gz_files:
                        st.warning(
                            "No metadata files (.jsonl/.jsonl.gz/.jsonl.zst) found."
                        )
                    else:
                        file_meta = tuple(
                            (fp, os.path.getmtime(fp)) for fp in gz_files
                        )
                        progress_box = st.empty()

                        def stat(msg: str):
                            progress_box.info(msg)

                        with st.spinner('Scanning all label files...'):
                            # First call (uncached) uses scanning inside cached function; pass meta for invalidation.
                            dist_df = _cached_label_distribution(
                                labelled_dir,
                                cluster_field or 'cluster_id',
                                file_meta,
                            )
                        total_records = int(dist_df['count'].sum())
                        unique_clusters = int(dist_df.shape[0])
                        if total_records == 0:
                            st.warning('No records parsed from label files.')
                        else:
                            entropy = (
                                entropy_bits(dist_df['percent'])
                                if unique_clusters <= 50000
                                else float('nan')
                            )
                            top_row = (
                                dist_df.iloc[0] if not dist_df.empty else None
                            )
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric('Total Records', f"{total_records:,}")
                            m2.metric('Unique Clusters', f"{unique_clusters:,}")
                            if top_row is not None:
                                m3.metric(
                                    'Top Cluster',
                                    f"{top_row['cluster_value']} ({top_row['percent']:.2f}%)",
                                )
                            if not math.isnan(entropy):
                                m4.metric('Entropy (bits)', f"{entropy:.2f}")
                            else:
                                m4.metric('Entropy (bits)', '—')

                            # Prepare Top K + OTHERS table
                            top_slice = dist_df.head(top_k).copy()
                            if dist_df.shape[0] > top_k:
                                others_count = (
                                    dist_df['count'].iloc[top_k:].sum()
                                )
                                others_percent = (
                                    dist_df['percent'].iloc[top_k:].sum()
                                )
                                others_row = pd.DataFrame(
                                    [
                                        {
                                            'cluster_value': 'OTHERS',
                                            'count': others_count,
                                            'percent': others_percent,
                                            'cumulative_percent': 100.0,
                                        }
                                    ]
                                )
                                display_df = pd.concat(
                                    [top_slice, others_row], ignore_index=True
                                )
                            else:
                                display_df = top_slice

                            st.markdown("**Top Cluster Distribution**")
                            st.dataframe(display_df, use_container_width=True)

                            # Bar chart
                            bar_fig = go.Figure()
                            bar_fig.add_trace(
                                go.Bar(
                                    name='Full Dataset',
                                    x=top_slice['cluster_value'],
                                    y=top_slice['count'],
                                    marker_color='#1f77b4',
                                )
                            )
                            if compare and 'tsne_embedding' in st.session_state:
                                sample_df: pd.DataFrame = st.session_state[
                                    'tsne_embedding'
                                ]
                                sample_counts = sample_df[
                                    'cluster_value'
                                ].value_counts()
                                sample_counts = sample_counts.reindex(
                                    top_slice['cluster_value'], fill_value=0
                                )
                                bar_fig.add_trace(
                                    go.Bar(
                                        name='Sample',
                                        x=top_slice['cluster_value'],
                                        y=sample_counts.values,
                                        marker_color='#ff7f0e',
                                        opacity=0.6,
                                    )
                                )
                                # Divergence metrics
                                jsd = jsd_bits(
                                    top_slice.set_index('cluster_value')[
                                        'count'
                                    ],
                                    sample_counts,
                                )
                                abs_diff = sample_counts - top_slice.set_index(
                                    'cluster_value'
                                )['count'] * (
                                    sample_counts.sum()
                                    / top_slice['count'].sum()
                                )
                                st.caption(
                                    f"Jensen-Shannon Distance (bits, Top K only): {jsd:.4f}"
                                )
                            bar_fig.update_layout(
                                barmode='group',
                                xaxis_title='Cluster',
                                yaxis_title='Count',
                                height=500,
                                margin=dict(l=10, r=10, t=40, b=40),
                            )
                            if log_scale:
                                bar_fig.update_yaxes(type='log')
                            st.plotly_chart(bar_fig, use_container_width=True)

                            with st.expander(
                                'Raw Full Distribution (All Clusters)'
                            ):
                                st.dataframe(dist_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error computing distribution: {e}")


if __name__ == "__main__":  # pragma: no cover
    main()
