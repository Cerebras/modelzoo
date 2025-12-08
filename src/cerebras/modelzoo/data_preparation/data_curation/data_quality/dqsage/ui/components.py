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
Reusable UI components for the data visualizer.
Contains functions for displaying common UI elements and interactions.
"""

import codecs
import json
import re

import pandas as pd
import streamlit as st

from ..utils.helpers import get_nested_value

# -------------------------------------------------------------
# Pretty text decoding utilities (optional display feature)
# -------------------------------------------------------------

# Detect common escaped sequences (\\n, \\t, \\r, \\uXXXX, \\UXXXXXXXX, \\\\)
_ESCAPE_SEQ_PATTERN = re.compile(
    r"\\(u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|[ntr\\\"])"
)


def _decode_escaped_text(value: str) -> str:
    """Best-effort decode of visible escape sequences into real characters.

    Rules:
      - Only attempt if we detect at least one escape pattern.
      - Leaves original string unchanged on failure.
      - Avoid double-decoding: if no patterns, return as-is.
    - Supports: \\n, \\t, \\r, \\\\, \\\" plus Unicode \\uXXXX / \\UXXXXXXXX.
    """
    if not isinstance(value, str):
        return value
    if not _ESCAPE_SEQ_PATTERN.search(value):
        return value
    # Guard: append a space if ending in single backslash (would error in unicode_escape)
    candidate = value
    if candidate.endswith('\\') and not candidate.endswith('\\\\'):
        candidate += ' '
    try:
        # unicode_escape handles the intended sequences; wrap in try for malformed \u
        decoded = codecs.decode(candidate, 'unicode_escape')
        return decoded
    except Exception:
        # Fallback manual replacements (subset) — safer than failing entirely
        basic = (
            candidate.replace('\\n', '\n')
            .replace('\\t', '\t')
            .replace('\\r', '\r')
            .replace('\\\\', '\\')
            .replace('\"', '"')
        )
        return basic


def display_schema_analysis(schema_info):
    """Display the schema analysis results in an expandable format"""
    with st.expander("📋 Data Schema Analysis", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Files Found", schema_info['file_count'])
            st.metric(
                "File Format", schema_info.get('file_format', 'unknown').upper()
            )
            st.write("Structure from first record will be used as reference")

            # Show total field count
            total_fields = len(schema_info.get('all_field_paths', set()))
            st.metric("Total Available Fields", total_fields)

        with col2:
            st.markdown("**Top-level Fields:**")

            scrollable_top_fields = """
                <div style='max-height:200px; overflow-y:auto; padding:0.5em; 
                            border:1px solid var(--secondary-background-color); 
                            border-radius:5px; 
                            background-color: var(--background-color); 
                            margin-bottom:1em; 
                            color: var(--text-color);'>
                """

            for field in sorted(schema_info['top_level_fields']):
                scrollable_top_fields += (
                    f"<div style='margin-bottom: 2px;'>• {field}</div>"
                )
            scrollable_top_fields += "</div>"

            st.markdown(scrollable_top_fields, unsafe_allow_html=True)

        with col3:
            st.markdown("**Nested Dictionary Fields:**")
            nested_fields = schema_info.get('nested_fields', {})
            if nested_fields:
                scrollable_content = """
                    <div style='max-height: 200px; overflow-y: auto; padding: 0.5em; 
                                border: 1px solid var(--secondary-background-color); 
                                border-radius: 5px; 
                                background-color: var(--background-color); 
                                margin-bottom: 1em; 
                                color: var(--text-color);'>
                    """

                for parent_field, nested_keys in sorted(nested_fields.items()):
                    scrollable_content += f"<div style='margin-bottom: 8px;'><strong>{parent_field}:</strong></div>"
                    # Show all nested keys, not just first 5
                    for nested_key in sorted(nested_keys):
                        scrollable_content += f"<div style='margin-left: 15px; margin-bottom: 2px;'>• {nested_key}</div>"
                scrollable_content += "</div>"
                st.markdown(scrollable_content, unsafe_allow_html=True)
            else:
                st.write("No nested dictionary fields found")


def display_batch_navigation_controls(lazy_loader, batch_size):
    """Display batch navigation controls with prev/next buttons"""
    st.markdown("### 🔄 Batch Navigation")
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([2, 2, 2, 2])

    with col_nav1:
        current_batch = st.session_state.get('current_batch', 0)
        if (
            lazy_loader.is_indexing_complete()
            or lazy_loader.file_format == 'parquet'
        ):
            total_batches = lazy_loader.get_total_batches(batch_size)
            st.metric("Current Batch", f"{current_batch} / {total_batches - 1}")
        else:
            st.metric("Current Batch", current_batch)

    with col_nav2:
        current_df = st.session_state.get('current_df', pd.DataFrame())
        st.metric("Records in Batch", len(current_df))
        # Show current batch info if indexed
        if lazy_loader.is_indexing_complete():
            batch_info = lazy_loader.get_batch_info(current_batch)
            if batch_info:
                st.caption(f"From: {batch_info['file_name']}")

    _handle_batch_navigation_buttons(
        lazy_loader, batch_size, col_nav3, col_nav4
    )


def _handle_batch_navigation_buttons(
    lazy_loader, batch_size, col_nav3, col_nav4
):
    """Handle the previous/next batch navigation buttons"""

    with col_nav3:
        if st.button("⏮️ Load Previous Batch"):
            current_batch = st.session_state.get('current_batch', 0)
            if current_batch > 0:
                with st.spinner("Loading previous batch..."):
                    try:
                        target_batch = current_batch - 1
                        batch_df = lazy_loader.get_batch(
                            target_batch, batch_size=batch_size
                        )
                        if not batch_df.empty:
                            st.session_state['current_df'] = batch_df
                            st.session_state['current_batch'] = target_batch

                            # Save column selection before rerun
                            if "raw_data_display_columns" in st.session_state:
                                st.session_state["saved_display_columns"] = (
                                    st.session_state["raw_data_display_columns"]
                                )

                            st.success(f"Loaded previous batch {target_batch}")
                            st.rerun()  # Force immediate rerun
                        else:
                            st.warning("Previous batch is empty")
                    except Exception as e:
                        st.error(f"Error loading previous batch: {e}")
            else:
                st.warning("Already at first batch")

    with col_nav4:
        if st.button("⏭️ Load Next Batch"):
            current_batch_num = st.session_state.get('current_batch', 0)

            # Check if we're at the end (for indexed or parquet files)
            if (
                lazy_loader.is_indexing_complete()
                or lazy_loader.file_format == 'parquet'
            ):
                total_batches = lazy_loader.get_total_batches(batch_size)
                if current_batch_num >= total_batches - 1:
                    st.warning(f"Already at last batch ({total_batches - 1})")
                else:
                    with st.spinner("Loading next batch..."):
                        try:
                            target_batch = current_batch_num + 1
                            batch_df = lazy_loader.get_batch(
                                target_batch, batch_size=batch_size
                            )
                            if not batch_df.empty:
                                st.session_state['current_df'] = batch_df
                                st.session_state['current_batch'] = target_batch

                                # Save column selection before rerun
                                if (
                                    "raw_data_display_columns"
                                    in st.session_state
                                ):
                                    st.session_state[
                                        "saved_display_columns"
                                    ] = st.session_state[
                                        "raw_data_display_columns"
                                    ]

                                st.success(f"Loaded next batch {target_batch}")
                                st.rerun()  # Force immediate rerun
                            else:
                                st.warning("Next batch is empty")
                        except Exception as e:
                            st.error(f"Error loading next batch: {e}")
            else:
                # Sequential mode - try to load next batch
                with st.spinner("Loading next batch..."):
                    try:
                        target_batch = current_batch_num + 1
                        batch_df = lazy_loader.get_batch(
                            target_batch, batch_size=batch_size
                        )
                        if not batch_df.empty:
                            st.session_state['current_df'] = batch_df
                            st.session_state['current_batch'] = target_batch
                            st.success(f"Loaded next batch {target_batch}")
                            st.rerun()  # Force immediate rerun
                        else:
                            st.warning("No more batches to load!")
                    except Exception as e:
                        st.error(f"Error loading next batch: {e}")


def display_individual_record_viewer(display_df_source, selected_fields):
    """Display the individual record viewer component with optional pretty formatting."""
    st.markdown("### 🔍 Individual Record Viewer")

    pretty_mode = st.checkbox(
        "Pretty format text fields",
        value=st.session_state.get('pretty_text_mode', False),
        help="Decode visible escape sequences (\\n, \\t, \\uXXXX) into real characters for display.",
        key='pretty_text_mode',
    )

    if not display_df_source.empty:
        record_options = []
        for idx, row in display_df_source.iterrows():
            # Safe check for ID field
            has_valid_id = False
            try:
                if 'id' in row:
                    id_value = row['id']
                    if id_value is not None and not pd.isna(id_value):
                        # Additional check for empty arrays
                        if hasattr(id_value, 'size') and id_value.size == 0:
                            has_valid_id = False
                        else:
                            has_valid_id = True
            except:
                has_valid_id = False

            if has_valid_id:
                label = f"Record {idx} - {str(row['id'])}"
            else:
                label = (
                    f"Record {idx} - Line {row.get('line_number', 'Unknown')}"
                )
            record_options.append((idx, label))

        selected_idx = st.selectbox(
            "Select a record to view in detail:",
            options=[idx for idx, _ in record_options],
            format_func=lambda x: next(
                label for idx, label in record_options if idx == x
            ),
            key="individual_record_selector",
        )

        selected_record = display_df_source.loc[selected_idx]

        col1, col2 = st.columns([1, 1])

        _display_nested_fields(
            col1, selected_record, selected_fields, pretty_mode
        )
        _display_top_level_fields(
            col2, selected_record, selected_fields, pretty_mode
        )


def _display_nested_fields(
    col, selected_record, selected_fields, pretty_mode: bool = False
):
    """Display nested fields in the individual record viewer; optionally pretty-decode string values."""
    with col:
        st.markdown("**Nested Fields:**")
        # Show only loaded nested fields (any parent.child pattern)
        loaded_nested = {}

        for field in selected_fields:
            if '.' in field:
                value = get_nested_value(selected_record, field)
                if value is not None:
                    if pretty_mode and isinstance(value, str):
                        loaded_nested[field] = _decode_escaped_text(value)
                    else:
                        loaded_nested[field] = value

        if loaded_nested:
            st.json(loaded_nested)
        else:
            st.write("No nested fields were loaded or have values")


def _display_top_level_fields(
    col, selected_record, selected_fields, pretty_mode: bool = False
):
    """Display top-level fields in the individual record viewer; optionally pretty-decode string values."""
    with col:
        st.markdown("**Top-level Fields:**")

        # Show each loaded top-level field in its own box (excluding nested fields)
        top_level_fields_to_show = []
        for field in selected_fields:
            if '.' not in field and field in selected_record:
                top_level_fields_to_show.append(field)

        # Add source info fields
        top_level_fields_to_show.extend(['source_file', 'line_number'])

        # Display each field in its own box
        for field in top_level_fields_to_show:
            if field in ['source_file', 'line_number']:
                field_value = selected_record.get(field, 'Unknown')
            else:
                field_value = selected_record[field]

            st.markdown(f"**{field.replace('_', ' ').title()}:**")

            field_str = _safe_field_to_string(field_value)
            if pretty_mode and isinstance(field_str, str):
                field_str = _decode_escaped_text(field_str)

            # Check if it's a long text field that should be scrollable
            if len(field_str) > 200:
                # Use text_area with scrolling for long content
                st.text_area(
                    f"{field}:",
                    field_str,
                    height=400,
                    key=f"field_display_{field}_{selected_record.name}",
                    label_visibility="collapsed",
                )
                st.caption(f"Length: {len(field_str):,} characters")
            elif len(field_str) > 50:
                # Use text_area with smaller height for medium content
                st.text_area(
                    f"{field}:",
                    field_str,
                    height=80,
                    key=f"field_display_{field}_{selected_record.name}",
                    label_visibility="collapsed",
                )
            else:
                # Use simple text display for short content
                if field_str:
                    st.code(field_str, language=None)
                else:
                    st.write("*Empty*")

            # Add some spacing between fields
            st.write("")


def _safe_field_to_string(value):
    """Safely convert field value to string, handling numpy arrays and other special types"""
    try:
        # Handle None/NaN values
        if value is None:
            return ''

        # Handle numpy arrays
        if hasattr(value, 'size'):
            if hasattr(value, '__len__') and len(value) == 0:
                return ''
            elif hasattr(value, 'size') and value.size == 0:
                return ''

        # Handle pandas NaN values
        if pd.isna(value):
            return ''

        # Convert to string
        return str(value)
    except:
        # Fallback to simple string conversion
        return str(value) if value is not None else ''


# Helper function for formatting display values
def format_display_value(value):
    """Normalize values for Arrow-friendly display.
    - Convert list/dict to JSON string
    - Convert other types to string (empty for None/NaN/empty arrays)
    - Truncate long strings to first 200 chars
    """
    # 1. None / NaN
    if value is None:
        return ''
    try:
        if pd.isna(value):
            return ''
    except Exception:
        pass

    # 2. Empty array-like
    if hasattr(value, '__len__') and not isinstance(
        value, (str, bytes, bytearray)
    ):
        try:
            if len(value) == 0:
                return ''
        except Exception:
            pass
    if hasattr(value, 'size') and getattr(value, 'size', None) == 0:
        return ''

    # 3. Convert to text
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
    elif isinstance(value, (bytes, bytearray)):
        text = value.decode('utf-8', errors='replace')
    else:
        text = str(value)

    # 4. Empty after conversion
    if text == '':
        return ''

    # 5. Truncate safely
    return text if len(text) <= 200 else text[:200] + '…'
