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

"""
UI styles and configuration for the Streamlit application.
Contains CSS styles and page configuration settings.
"""

import streamlit as st


def configure_page():
    """Configure Streamlit page settings and load custom CSS"""
    # Set page config
    st.set_page_config(
        page_title="DQSage Data Visualizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load custom CSS
    load_custom_css()


def load_custom_css():
    """Load custom CSS styles for the application"""
    # Custom CSS for better styling and lazy loading interface
    st.markdown(
        """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .sidebar-header {
            font-size: 1.2rem;
            color: #1f77b4;
            font-weight: bold;
        }
        .lazy-loading-controls {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #1f77b4;
            margin: 1rem 0;
        }
        .batch-nav-button {
            background-color: #1f77b4;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 0.25rem;
            cursor: pointer;
        }
        .batch-nav-button:hover {
            background-color: #0d5a8a;
        }
        .batch-info {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border-left: 3px solid #28a745;
        }
        .memory-info {
            background-color: #fff3cd;
            padding: 0.5rem;
            border-radius: 0.25rem;
            border-left: 3px solid #ffc107;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
