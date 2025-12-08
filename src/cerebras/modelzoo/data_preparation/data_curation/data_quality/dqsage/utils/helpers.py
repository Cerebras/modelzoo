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
Helper utility functions for the data visualizer.
Contains common utility functions used across different modules.
"""

import streamlit as st


def get_nested_value(data, path, default=None):
    """Safely get nested dictionary values using dot notation - supports unlimited depth"""
    try:
        # Handle pandas Series (when used with df.apply)
        if hasattr(data, 'to_dict'):
            data = data.to_dict()
        elif not isinstance(data, dict):
            return default

        keys = path.split('.')
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    except:
        return default


def cleanup_sql_processor():
    """Clean up SQL processor connection when session ends"""
    if 'sql_processor' in st.session_state:
        try:
            st.session_state['sql_processor'].close_connection()
        except:
            pass
