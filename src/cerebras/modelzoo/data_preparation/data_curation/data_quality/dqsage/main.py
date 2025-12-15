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

#!/usr/bin/env python3

"""Entry Point for DQSage Multi-App Suite

Adds a lightweight launcher allowing users to pick between:
1) Data Visualizer (existing functionality)
2) t-SNE Visualizer (new module for dimensionality reduction)

Design Goals:
- Extensible registry so future apps can be plugged in with minimal changes.
- Backwards compatible: default selection = Data Visualizer.
- Lazy import individual apps to avoid unnecessary startup overhead.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict

import streamlit as st

# Add parent directory to Python path so 'dqsage' package can be found
# This MUST be done before importing ui.app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ui.styles import configure_page


# --- Application Registry --------------------------------------------------
@dataclass
class RegisteredApp:
    name: str
    description: str
    entry: Callable[[], None]


def load_data_visualizer():
    from ui.app import main as data_main  # local import to keep startup light

    data_main()


def load_tsne_visualizer():
    from ui.tsne_app import main as tsne_main  # local import

    tsne_main()


def get_app_registry() -> Dict[str, RegisteredApp]:
    return {
        "Data Visualizer": RegisteredApp(
            name="Data Visualizer",
            description="Explore, batch load, index and SQL query large JSONL/Parquet datasets.",
            entry=load_data_visualizer,
        ),
        "t-SNE Visualizer": RegisteredApp(
            name="t-SNE Visualizer",
            description="Compute and explore 2D/3D t-SNE embeddings for selected fields.",
            entry=load_tsne_visualizer,
        ),
    }


def render_launcher(apps: Dict[str, RegisteredApp]):
    st.markdown(
        "<h1 style='text-align:center'>🧭 DQSage Application Launcher</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Select an application mode from side bar.")

    names = list(apps.keys())
    default_index = 0  # Data Visualizer first
    selected = st.sidebar.radio(
        "Application Module:",
        names,
        index=(
            default_index
            if 'selected_app' not in st.session_state
            else names.index(st.session_state['selected_app'])
        ),
        help="Choose which DQSage module to run.",
    )
    st.session_state['selected_app'] = selected

    # Summary panel
    with st.expander("ℹ️ Module Details", expanded=False):
        for name, app in apps.items():
            with st.container():
                highlight = "✅" if name == selected else "🔹"
                st.write(f"{highlight} **{app.name}** — {app.description}")

    return selected


if __name__ == "__main__":
    configure_page()
    registry = get_app_registry()
    chosen = render_launcher(registry)
    # Run chosen app below a divider to keep context clear
    st.markdown("---")
    registry[chosen].entry()
