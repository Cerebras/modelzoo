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
Main Streamlit application for the DQSage Data Visualizer.
Contains the refactored main function broken into smaller, manageable pieces.
"""

import atexit

import altair as alt

_ = alt  # prevent unused import removal

import glob
import multiprocessing as mp
import os
import re
import time
import traceback
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
from dqsage.core.data_analyzer import (
    analyze_data_schema,
    analyze_data_structure,
    validate_data_compatibility,
)
from dqsage.core.data_loader import LazyDataLoader

# Import from our modular structure
from dqsage.core.file_handlers import FileFormatRegistry
from dqsage.query.sql_processor import SQLQueryProcessor
from dqsage.ui.components import (
    display_batch_navigation_controls,
    display_individual_record_viewer,
    display_schema_analysis,
    format_display_value,
)
from dqsage.utils.helpers import cleanup_sql_processor, get_nested_value


def setup_sidebar_configuration():
    """Handle all sidebar inputs and configuration"""
    # Sidebar
    st.sidebar.markdown(
        '<p class="sidebar-header">📁 Data Configuration</p>',
        unsafe_allow_html=True,
    )

    # Data path input
    data_path = st.sidebar.text_input(
        "Data Directory Path:",
        value="data",
        help="Enter the path to directory containing data files",
    )

    # File format selection
    available_formats = FileFormatRegistry.get_available_formats()

    if not available_formats:
        st.error("❌ No file formats available! This should not happen.")
        st.info("Both JSONL and Parquet support are built-in using DuckDB.")
        return None, None, None

    # Display format availability info
    format_info = {}
    for fmt in ['jsonl', 'parquet']:
        if fmt in available_formats:
            if fmt == 'parquet':
                format_info[fmt] = "✅ Available (DuckDB-powered)"
            else:
                format_info[fmt] = "✅ Available"
        else:
            format_info[fmt] = "❌ Not available"

    # File format selection
    default_format = available_formats[0] if available_formats else 'jsonl'
    file_format = st.sidebar.selectbox(
        "File Format:",
        options=available_formats,
        index=0,
        help="Select the format of your data files",
    )

    # Dynamic file pattern based on format
    if file_format == 'jsonl':
        default_pattern = "*.jsonl"
        pattern_help = (
            "Pattern to match JSONL files (e.g., chunk_*.jsonl, *.jsonl)"
        )
    elif file_format == 'parquet':
        default_pattern = "*.parquet"
        pattern_help = (
            "Pattern to match Parquet files (e.g., chunk_*.parquet, *.parquet)"
        )
    else:
        default_pattern = f"*.{file_format}"
        pattern_help = f"Pattern to match {file_format.upper()} files"

    file_pattern = st.sidebar.text_input(
        "File Pattern:", value=default_pattern, help=pattern_help
    )

    # Store data path, file format, and file pattern in session state for Full Dataset SQL queries
    st.session_state['data_path'] = data_path
    st.session_state['file_format'] = file_format
    st.session_state['file_pattern'] = file_pattern

    return data_path, file_format, file_pattern


def handle_schema_analysis(data_path, file_format, file_pattern):
    """Handle schema analysis button and display logic"""
    # Step 1: Analyze data schema
    if st.sidebar.button("🔍 Analyze Data Schema", type="secondary"):
        with st.spinner("Analyzing data schema..."):
            # Reset all session state to avoid conflicts when switching datasets
            keys_to_reset = [
                'field_multiselect',
                'lazy_loader',
                'selected_fields',
                'text_field_for_limit',
                'batch_size',
                'current_batch',
                'total_batches_loaded',
                'current_df',
                'estimated_total_records',
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]

            # Analyze the new schema
            schema_info = analyze_data_schema(
                data_path, file_format, file_pattern
            )
            st.session_state['schema_info'] = schema_info

    # Show schema analysis results
    if 'schema_info' in st.session_state and st.session_state['schema_info']:
        schema_info = st.session_state['schema_info']

        # Check for errors in schema analysis
        if 'error' in schema_info:
            st.error(f"❌ Schema Analysis Error: {schema_info['error']}")
            if schema_info['file_count'] == 0:
                st.info(
                    f"💡 Make sure you have {file_format.upper()} files in the directory with the pattern: `{file_pattern}`"
                )
            return None

        # Display schema analysis using the component
        display_schema_analysis(schema_info)

        return schema_info

    return None


def handle_field_selection_and_loader_init(schema_info):
    """Handle field selection and lazy loader initialization"""
    # Step 2: Field Selection
    st.sidebar.markdown(
        '<p class="sidebar-header">⚙️ Load Configuration</p>',
        unsafe_allow_html=True,
    )

    # Build comprehensive field options including nested fields
    field_options = []

    # Add top-level fields
    for field in sorted(schema_info['top_level_fields']):
        field_options.append(field)

    # Add nested fields with dot notation
    nested_fields = schema_info.get('nested_fields', {})
    for parent_field, nested_keys in sorted(nested_fields.items()):
        for nested_key in sorted(nested_keys):
            field_options.append(f"{parent_field}.{nested_key}")

    # Field selection in sidebar
    st.sidebar.markdown("**Select Fields to Load:**")
    st.sidebar.info(
        f"💡 {len(field_options)} fields available (including nested)"
    )

    # Initialize session state if needed
    if 'field_multiselect' not in st.session_state:
        if (
            'id' in schema_info['top_level_fields']
            and 'text' in schema_info['top_level_fields']
        ):
            st.session_state.field_multiselect = ['id', 'text']
        else:
            st.session_state.field_multiselect = field_options[:5]

    # Multiselect for field selection - let Streamlit handle the state automatically
    selected_fields = st.sidebar.multiselect(
        "Fields:",
        options=field_options,
        key="field_multiselect",
        help="Select only the fields you need to improve loading speed",
    )

    # Set default values for commented out performance settings
    max_records = None  # Load all records
    text_field_for_limit = None
    text_length_limit = None

    # Lazy Loading Configuration
    st.sidebar.markdown("**Lazy Loading Settings:**")

    # Predefined batch size options
    batch_size_option = st.sidebar.selectbox(
        "Batch Size:",
        options=["Custom", 10, 50, 100, 200, 500, 1000],
        index=3,  # Default to 100
        help="Number of records to load per batch. Smaller batches use less memory.",
    )

    # If user selects "Custom", show a number input
    if batch_size_option == "Custom":
        batch_size = st.sidebar.number_input(
            "Enter custom batch size:",
            min_value=1,
            step=1,
            value=100,  # Default custom value
        )
    else:
        batch_size = int(batch_size_option)

    if st.sidebar.button("🚀 Initialize Lazy Data Loader", type="primary"):
        if selected_fields:
            try:
                with st.spinner("Initializing lazy data loader..."):
                    # COMPREHENSIVE RESET - Clear all session state and caches
                    # Clear existing lazy loader and its caches if it exists

                    if 'lazy_loader' in st.session_state:
                        old_loader = st.session_state['lazy_loader']
                        if hasattr(old_loader, 'clear_cache'):
                            old_loader.clear_cache()
                        if hasattr(old_loader, 'clear_index'):
                            old_loader.clear_index()

                    # Get configuration from session state BEFORE clearing
                    data_path = st.session_state.get('data_path', 'data')
                    file_format = st.session_state.get('file_format', 'jsonl')
                    file_pattern = st.session_state.get(
                        'file_pattern', '*.jsonl'
                    )

                    # Dynamic reset - Clear all session state except essential app state
                    # Keep only critical keys that should persist across resets
                    keys_to_preserve = {
                        'schema_info',  # Keep schema analysis
                        'field_multiselect',  # Keep field selection (will be updated below anyway)
                        'data_path',  # Keep data configuration
                        'file_format',  # Keep file format
                        'file_pattern',  # Keep file pattern
                    }

                    # Get all current session state keys and reset everything else
                    all_keys = list(st.session_state.keys())
                    for key in all_keys:
                        if key not in keys_to_preserve:
                            del st.session_state[key]

                    # Create fresh lazy loader
                    lazy_loader = LazyDataLoader(
                        data_dir=data_path,
                        file_format=file_format,
                        file_pattern=file_pattern,
                        selected_fields=selected_fields,
                        text_field_for_limit=text_field_for_limit,
                        text_length_limit=text_length_limit,
                        reference_structure=schema_info.get(
                            'reference_structure'
                        ),
                        handler=schema_info.get('handler'),
                    )

                    # Store in session state with fresh values
                    st.session_state['lazy_loader'] = lazy_loader
                    st.session_state['selected_fields'] = selected_fields
                    st.session_state['text_field_for_limit'] = (
                        text_field_for_limit
                    )
                    st.session_state['batch_size'] = batch_size

                    # Initialize pagination state to original values
                    st.session_state['current_batch'] = 0
                    st.session_state['total_batches_loaded'] = 0
                    st.session_state['current_df'] = pd.DataFrame()

                    # Auto-load the first batch
                    try:
                        first_batch_df = lazy_loader.get_batch(
                            0, batch_size=batch_size
                        )

                        if not first_batch_df.empty:
                            st.session_state['current_df'] = first_batch_df
                            st.session_state['current_batch'] = (
                                0  # Set to 0 since we loaded batch 0
                            )
                            st.session_state['total_batches_loaded'] = 1

                            st.info(
                                f"🚀 Auto-loaded first batch with {len(first_batch_df):,} records"
                            )
                        else:
                            st.warning("First batch appears to be empty")
                    except Exception as batch_error:
                        st.warning(
                            f"Could not auto-load first batch: {batch_error}"
                        )

                    st.rerun()  # Refresh to show the loaded data

            except Exception as e:
                st.error(f"❌ Error initializing lazy loader: {e}")
                return None, None, None

        else:
            st.sidebar.error("Please select at least one field to load")
            return None, None, None

    # Return current state
    if 'lazy_loader' in st.session_state:
        lazy_loader = st.session_state['lazy_loader']
        selected_fields = st.session_state.get('selected_fields', [])
        batch_size = st.session_state.get('batch_size', 100)
        return lazy_loader, selected_fields, batch_size

    return None, None, None


def display_lazy_loading_controls(lazy_loader, batch_size):
    """Display all the indexing and batch management controls"""
    # Lazy Loading Controls
    st.markdown("## 🔄 Lazy Loading Controls")

    # Only show indexing section for formats that require it (not Parquet)
    if lazy_loader.file_format != 'parquet':
        display_indexing_section(lazy_loader, batch_size)
    else:
        display_parquet_info_section(lazy_loader, batch_size)

    # Batch management controls
    display_batch_management_controls(lazy_loader)


def display_indexing_section(lazy_loader, batch_size):
    """Display the indexing section for non-Parquet files"""
    # Indexing Section
    st.markdown("### ⚡ Fast Random Access Indexing")

    col_index1, col_index2, col_index3, col_index4 = st.columns([5, 2, 2, 2])

    with col_index1:
        if lazy_loader.is_indexing_complete():
            st.success(
                f"✅ Index Built: {lazy_loader.get_total_batches()} batches"
            )
            total_records = lazy_loader.get_total_estimated_records()
            st.info(f"📊 Total Records: {total_records:,}")
        else:
            st.warning("⏳ No index built - using sequential access")

            # Parallel indexing options
            col_parallel1, col_parallel2 = st.columns([1, 1])

            with col_parallel1:
                use_parallel = st.checkbox(
                    "🚀 Use Parallel Processing",
                    value=True,
                    help="Use multiple CPU cores for faster indexing",
                )

            with col_parallel2:
                max_workers = st.number_input(
                    "Workers:",
                    min_value=1,
                    max_value=mp.cpu_count(),
                    value=mp.cpu_count(),
                    step=1,
                    help="Number of parallel workers (default = all available cores)",
                )

            if st.button("🔧 Build Index for Fast Access", type="primary"):
                # Create a progress placeholder
                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                def progress_callback(message):
                    status_text.text(message)
                    progress = lazy_loader.get_indexing_progress()
                    if progress['total_files'] > 0:
                        file_progress = (
                            progress['current_file'] / progress['total_files']
                        )
                        progress_bar.progress(file_progress)

                try:
                    with st.spinner(
                        "Building index for super-fast random access..."
                    ):
                        lazy_loader.build_batch_index(
                            batch_size=batch_size,
                            progress_callback=progress_callback,
                            use_parallel=use_parallel,
                            max_workers=max_workers,
                        )

                    progress_bar.empty()
                    status_text.empty()

                    total_time = time.time() - start_time
                    total_batches = lazy_loader.get_total_batches()
                    total_records = lazy_loader.get_total_estimated_records()

                    # Show performance metrics
                    if use_parallel:
                        st.success(
                            f"🚀 Parallel indexing completed in {total_time:.1f}s!"
                        )
                        st.balloons()  # Celebration for fast indexing!
                    else:
                        st.success(
                            f"✅ Sequential indexing completed in {total_time:.1f}s!"
                        )

                    st.info(
                        f"📊 Indexed {total_batches} batches with {total_records:,} records"
                    )
                    st.rerun()  # Refresh to show updated status

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Error building index: {e}")

                    # Offer fallback to sequential
                    if use_parallel:
                        st.warning(
                            "Parallel indexing failed. You can try again with sequential processing."
                        )

    with col_index2:
        if lazy_loader.is_indexing_complete():
            # Show index statistics
            st.markdown("**Index Statistics:**")
            st.write(f"• Total batches: {lazy_loader.get_total_batches()}")
            st.write(f"• Files indexed: {len(lazy_loader.data_files)}")

            # Estimate index memory usage
            index_size = (
                len(lazy_loader.batch_index) * 500
            )  # Rough estimate: 500 bytes per batch entry
            st.write(f"• Index size: ~{index_size / (1024*1024):.1f} MB")

        else:
            st.markdown("**Available CPU Cores:**")
            cpu_count = mp.cpu_count()
            st.write(f"🖥️ System: {cpu_count} cores")
            st.write(f"📁 Files: {len(lazy_loader.data_files)}")

            if len(lazy_loader.data_files) >= cpu_count:
                st.write("✅ Parallel processing will be very effective!")
            elif len(lazy_loader.data_files) > 1:
                st.write("⚡ Parallel processing will provide moderate speedup")
            else:
                st.write("ℹ️ Only 1 file - parallel processing not needed")

            st.info("💡 Benefits after indexing:")
            st.write("⚡ O(1) batch lookup")
            st.write("📍 Instant batch jumping")

    with col_index3:
        if lazy_loader.is_indexing_complete():
            if st.button("🗑️ Clear Index"):
                lazy_loader.clear_index()
                st.success("Index cleared!")
                st.rerun()

            st.markdown("**Current Mode:**")
            st.write("🚀 **Indexed Mode**")
            st.write("⚡ O(1) batch lookup")
            st.write("📍 Precise navigation")
            st.write("🎯 Instant random access")
        else:
            st.markdown("**Current Mode:**")
            st.write("🐌 **Sequential Mode**")
            st.write("⏳ O(n) batch lookup")
            st.write("📊 Estimated counts")
            st.write("⌛ Slow random access")

    with col_index4:
        # Cache management
        # Get cache info first
        cache_info = lazy_loader.get_cache_info()

        if st.button("🗑️ Clear Cache", key="lazy_loader_clear_cache"):
            lazy_loader.clear_cache()
            st.success("Cache cleared!")
            # Get updated cache info after clearing
            cache_info = lazy_loader.get_cache_info()
            st.rerun()  # Refresh the page to update cache info

        st.write(f"• Batches: {cache_info['cached_batches']}")
        st.write(f"• Memory: {cache_info['memory_usage_mb']:.1f} MB")


def display_parquet_info_section(lazy_loader, batch_size):
    """Display information section for Parquet files"""
    # For Parquet files, show a note about direct access
    st.markdown("### 🚀 Optimized Direct Access")

    col_info1, col_info2 = st.columns([1, 1])

    with col_info1:
        st.markdown("**Performance Benefits:**")
        st.write("⚡ **Always fast** - no setup time")
        st.write("🧠 **Low memory** - no index overhead")
        st.write("🎯 **Perfect accuracy** - native cross-file access")
        st.write("🚀 **DuckDB powered** - optimal for analytics")

    with col_info2:
        total_records = lazy_loader.get_total_estimated_records()
        total_batches = lazy_loader.get_total_batches(batch_size)
        st.markdown("**Dataset Information:**")
        st.write(f"📊 **Total Records:** {total_records:,}")
        st.write(f"📦 **Total Batches:** {total_batches:,}")
        st.write(f"📁 **Files:** {len(lazy_loader.data_files)}")
        st.write("🔄 **Access Mode:** Direct DuckDB queries")


def display_batch_management_controls(lazy_loader):
    """Display batch management controls (reset, cache, jump)"""
    # Batch management controls

    # Additional batch controls row
    col_load1, col_load2 = st.columns([1, 5])  # Reduced to 2 columns

    with col_load1:
        # Jump to specific batch (enhanced with index support)
        batch_size = st.session_state.get('batch_size', 1000)

        if lazy_loader.is_indexing_complete():
            max_batch = lazy_loader.get_total_batches(batch_size) - 1
            help_text = f"Enter batch number (0-{max_batch}). Index enables instant jumping!"
        elif lazy_loader.file_format == 'parquet':
            max_batch = lazy_loader.get_total_batches(batch_size) - 1
            help_text = f"Enter batch number (0-{max_batch}). Parquet allows instant jumping!"
        else:
            max_batch = 100  # Fallback estimate
            help_text = (
                "Enter batch number to jump to (build index for faster access)"
            )

        # # Ensure the current batch value doesn't exceed max_batch
        # current_batch_value = st.session_state.get('current_batch', 0)
        # if (
        #     lazy_loader.is_indexing_complete()
        #     or lazy_loader.file_format == 'parquet'
        # ) and current_batch_value > max_batch:
        #     current_batch_value = max_batch

        target_batch = st.number_input(
            "Jump to Batch:",
            min_value=0,
            max_value=(
                max_batch
                if (
                    lazy_loader.is_indexing_complete()
                    or lazy_loader.file_format == 'parquet'
                )
                else None
            ),
            # value=current_batch_value,
            step=1,
            help=help_text,
        )

    with col_load2:
        # Create sub-columns for button and status message
        col_jmp_btn, col_rst_btn, col_status = st.columns([2, 2, 10])

        with col_jmp_btn:
            st.markdown(
                "<div style='margin-top: 27.5px;'></div>",
                unsafe_allow_html=True,
            )
            jump_pressed = st.button("🎯 Jump to Batch")

        with col_rst_btn:
            st.markdown(
                "<div style='margin-top: 27.5px;'></div>",
                unsafe_allow_html=True,
            )
            if st.button("🔄 Reset to First Batch"):
                st.session_state['current_df'] = pd.DataFrame()
                st.session_state['current_batch'] = 0
                st.session_state['total_batches_loaded'] = 0
                # Clear cache and inconsistent records
                if hasattr(lazy_loader, 'inconsistent_records'):
                    lazy_loader.inconsistent_records = []
                lazy_loader.clear_cache()

                st.success("Reset to first batch and cleared cache")
                st.rerun()  # Force page refresh to update metrics

        with col_status:
            if jump_pressed:
                with st.spinner(f"Jumping to batch {target_batch}..."):
                    try:
                        batch_df = lazy_loader.get_batch(
                            target_batch, batch_size=batch_size
                        )
                        if not batch_df.empty:
                            st.session_state['current_df'] = batch_df
                            st.session_state['current_batch'] = target_batch

                            access_method = (
                                "⚡ Indexed"
                                if lazy_loader.is_indexing_complete()
                                else "🐌 Sequential"
                            )
                            # Use markdown with custom styling for inline message
                            st.markdown(
                                f"<div style='margin-top: 27.5px; padding: 8px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724; font-size: 14px;'>✅ Jumped to batch {target_batch} ({len(batch_df):,} records) - {access_method}</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            if lazy_loader.is_indexing_complete():
                                st.markdown(
                                    f"<div style='margin-top: 27.5px; padding: 8px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404; font-size: 14px;'>⚠️ Batch {target_batch} doesn't exist (max: {max_batch})</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<div style='margin-top: 27.5px; padding: 8px; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; color: #856404; font-size: 14px;'>⚠️ Batch {target_batch} is empty or doesn't exist</div>",
                                    unsafe_allow_html=True,
                                )
                    except Exception as e:
                        st.markdown(
                            f"<div style='margin-top: 27.5px; padding: 8px; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24; font-size: 14px;'>❌ Error jumping to batch: {e}</div>",
                            unsafe_allow_html=True,
                        )


def handle_sql_query_tab(lazy_loader, selected_fields, batch_size, filtered_df):
    """Handle everything in the SQL Query tab"""
    st.markdown("## 🗄️ SQL Query Interface")

    # Initialize SQL processor in session state
    if 'sql_processor' not in st.session_state:
        # from dqsage.query.sql_processor import SQLQueryProcessor
        st.session_state['sql_processor'] = SQLQueryProcessor()

    sql_processor = st.session_state['sql_processor']

    # Query scope selection and configuration
    query_scope, query_df, data_info = handle_sql_scope_selection(
        lazy_loader, selected_fields, batch_size, filtered_df
    )

    # Show data scope info and SQL interface for Current Batch and Batch Range only
    if query_df is not None and query_scope != "Full Dataset":
        # Register DataFrame with DuckDB
        success, reg_msg = sql_processor.register_dataframe(query_df, 'data')
        if success:
            st.success(f"✅ {reg_msg}")

            # Get table information
            table_info = sql_processor.get_table_info('data')

            # Show table schema
            with st.expander("📋 Table Schema", expanded=False):
                st.markdown("**Available Columns:**")
                if table_info['table_exists']:
                    for col_info in table_info['columns']:
                        col_name = col_info['column_name']
                        col_type = col_info['column_type']
                        st.write(f"• `{col_name}` ({col_type})")
                else:
                    st.write("No columns available")

            # Handle SQL interface for batch queries
            handle_batch_sql_interface()
        else:
            st.error(f"❌ {reg_msg}")
    elif query_scope == "Full Dataset":
        st.info(
            "💡 For Full Dataset queries, use the SQL Query and Export section above to save results directly to files."
        )
    else:
        st.info(
            "👆 Select a data scope and load data to start querying with SQL"
        )


def handle_sql_scope_selection(
    lazy_loader, selected_fields, batch_size, filtered_df
):
    """Handle SQL query scope selection and configuration"""
    # Query scope selection
    st.markdown("### 📊 Query Scope")
    col_scope1, col_scope2, col_scope3 = st.columns([2, 2, 2])

    with col_scope1:
        query_scope = st.selectbox(
            "Data Scope:",
            ["Current Batch", "Batch Range", "Full Dataset"],
            help="Choose the scope of data for your SQL query",
        )

    # Check for query scope changes and clear ALL SQL-related data if scope changed
    previous_query_scope = st.session_state.get('previous_query_scope', None)
    if previous_query_scope is not None and previous_query_scope != query_scope:
        # Query scope changed - clear ALL SQL-related session state completely
        # First, properly close the SQL processor connection before deletion
        if 'sql_processor' in st.session_state:
            try:
                st.session_state['sql_processor'].close_connection()
            except:
                pass  # Ignore cleanup errors

        sql_keys_to_clear = [
            'sql_query_df',
            'sql_result_df',
            'sql_result_msg',
            'sql_result_scope',
            'sql_processor',
        ]
        for key in sql_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

    # Update the stored scope
    st.session_state['previous_query_scope'] = query_scope

    with col_scope2:
        start_batch = 0
        if query_scope == "Batch Range":
            start_batch = st.number_input(
                "Start Batch:",
                min_value=0,
                value=max(0, st.session_state.get('current_batch', 0) - 2),
                help="Starting batch number",
            )

    with col_scope3:
        end_batch = 0
        if query_scope == "Batch Range":
            # Ensure end_batch value is always >= start_batch to avoid min_value error
            default_end_batch = max(
                start_batch, st.session_state.get('current_batch', 0) + 2
            )
            end_batch = st.number_input(
                "End Batch:",
                min_value=start_batch if query_scope == "Batch Range" else 0,
                value=default_end_batch,
                help="Ending batch number (inclusive)",
            )

    # Prepare data based on scope
    query_df = None
    data_info = ""

    if query_scope == "Current Batch":
        query_df = handle_current_batch_sql(filtered_df, selected_fields)
        data_info = f"Current batch ({len(query_df):,} records, {len(query_df.columns)} fields)"

    elif query_scope == "Batch Range":
        query_df = handle_batch_range_sql(
            lazy_loader, selected_fields, batch_size, start_batch, end_batch
        )
        if query_df is not None:
            data_info = f"Batches {start_batch}-{end_batch} ({len(query_df):,} records, {len(query_df.columns)} fields)"

    elif query_scope == "Full Dataset":
        handle_full_dataset_sql()
        data_info = "Full dataset mode - results saved to files"

    return query_scope, query_df, data_info


def handle_current_batch_sql(filtered_df, selected_fields):
    """Handle SQL queries for current batch scope"""
    query_df = filtered_df.copy()
    # Expand nested fields for SQL (data already filtered by LazyDataLoader)
    query_df = expand_nested_fields_for_sql(query_df, selected_fields)
    return query_df


def handle_batch_range_sql(
    lazy_loader, selected_fields, batch_size, start_batch, end_batch
):
    """Handle SQL queries for batch range scope"""
    if st.button("📥 Load Batch Range for SQL", key="sql_load_batch_range"):
        with st.spinner(f"Loading batches {start_batch} to {end_batch}..."):
            try:
                # Optimized approach: Calculate global offset and total records needed
                # instead of loop-based batch loading for better performance
                global_offset = start_batch * batch_size
                total_records_needed = (
                    end_batch - start_batch + 1
                ) * batch_size

                # Use direct batch range loading for optimal performance
                query_df = lazy_loader.get_batch_range_direct(
                    global_offset, total_records_needed
                )

                if not query_df.empty:
                    # Expand nested fields for SQL (data already filtered by LazyDataLoader)
                    query_df = expand_nested_fields_for_sql(
                        query_df, selected_fields
                    )
                    st.session_state['sql_query_df'] = query_df
                    return query_df
                else:
                    st.warning("No data found in the specified batch range")
                    return None
            except Exception as e:
                st.error(f"Error loading batch range: {e}")
                return None

    # Use previously loaded batch range data if available
    if 'sql_query_df' in st.session_state:
        query_df = st.session_state['sql_query_df']
        # Expand nested fields for SQL (data already filtered by LazyDataLoader)
        query_df = expand_nested_fields_for_sql(query_df, selected_fields)
        return query_df

    return None


def handle_full_dataset_sql():
    """Handle SQL queries for full dataset scope with export functionality (optimized with DESCRIBE)."""
    lazy_loader = st.session_state.get('lazy_loader')
    if not lazy_loader:
        st.error("❌ No lazy loader available")
        return

    if 'schema_info' in st.session_state:
        schema_info = st.session_state['schema_info']
        with st.expander("📋 Available Fields for SQL Queries", expanded=False):
            st.markdown("**Available Columns:**")

            def _render_fallback_fields(fields):
                st.session_state['full_dataset_field_types'] = {}
                content = """
                    <div style='max-height: 300px; overflow-y: auto; padding: 0.5em; 
                                border: 1px solid var(--secondary-background-color); 
                                border-radius: 5px; 
                                background-color: var(--background-color); 
                                margin-bottom: 1em; 
                                color: var(--text-color);'>
                    """

                for field in sorted(fields):
                    content += f"<div style='margin-bottom: 2px;'>• <code>{field}</code> (VARCHAR)</div>"
                content += "</div>"
                st.markdown(content, unsafe_allow_html=True)

            all_available_fields = []
            for field in sorted(schema_info['top_level_fields']):
                all_available_fields.append(field)
            nested_fields = schema_info.get('nested_fields', {})
            for parent_field, nested_keys in sorted(nested_fields.items()):
                for nested_key in sorted(nested_keys):
                    all_available_fields.append(f"{parent_field}.{nested_key}")

            data_path = st.session_state.get('data_path', 'data')
            file_format = st.session_state.get('file_format', 'jsonl')
            file_pattern = st.session_state.get('file_pattern', '*.jsonl')
            schema_cache_key = (data_path, file_pattern, file_format)
            cached_key = st.session_state.get('full_ds_schema_key')
            need_refresh = (
                cached_key != schema_cache_key
                or 'full_dataset_field_types' not in st.session_state
            )
            type_map = (
                st.session_state.get('full_dataset_field_types', {})
                if not need_refresh
                else {}
            )

            if need_refresh:
                try:
                    glob_pattern = os.path.join(
                        data_path, file_pattern
                    ).replace('\\', '/')
                    conn = duckdb.connect()
                    if file_format == 'parquet':
                        describe_query = f"DESCRIBE SELECT * FROM read_parquet('{glob_pattern}') LIMIT 0"
                    else:
                        describe_query = f"DESCRIBE SELECT * FROM read_json_auto('{glob_pattern}', format='newline_delimited', sample_size=20) LIMIT 0"
                    schema_df = conn.execute(describe_query).fetchdf()
                    conn.close()
                    excluded_cols = {"source_file", "line_number"}
                    type_map = {
                        str(row['column_name']): str(row['column_type']).upper()
                        for _, row in schema_df.iterrows()
                        if str(row['column_name']) not in excluded_cols
                    }
                    # Parse STRUCT column definitions to derive nested field types so we can display accurate types
                    # and enable proper casting for nested dicts access instead of defaulting to VARCHAR.
                    try:
                        struct_cols = {
                            col: col_type
                            for col, col_type in type_map.items()
                            if 'STRUCT(' in col_type.upper()
                        }
                        for parent_col, parent_type in struct_cols.items():
                            # Extract inner struct definition: STRUCT(field TYPE, field2 TYPE2, ...)
                            m = re.match(
                                r"STRUCT\((.*)\)$",
                                parent_type.strip(),
                                flags=re.IGNORECASE,
                            )
                            if not m:
                                continue
                            inner = m.group(1)
                            # Split on commas not inside quotes (struct definitions here are flat; arrays like VARCHAR[] contain no commas)
                            parts = []
                            buf = []
                            in_quotes = False
                            for ch in inner:
                                if ch == '"':
                                    in_quotes = not in_quotes
                                    buf.append(ch)
                                elif ch == ',' and not in_quotes:
                                    part = ''.join(buf).strip()
                                    if part:
                                        parts.append(part)
                                    buf = []
                                else:
                                    buf.append(ch)
                            last = ''.join(buf).strip()
                            if last:
                                parts.append(last)
                            for part in parts:
                                # part like: FIELD_NAME TYPE  or "FIELD_NAME" TYPE
                                part_match = re.match(
                                    r'"?([A-Za-z0-9_]+)"?\s+(.+)$', part.strip()
                                )
                                if not part_match:
                                    continue
                                child_name_raw, child_type = part_match.groups()
                                child_type_norm = child_type.strip().upper()
                                # Normalize array notation spacing
                                child_type_norm = re.sub(
                                    r'\s+', ' ', child_type_norm
                                )
                                # Compose dot path using lower-case child (schema_info nested fields appear lower-case)
                                dot_path = (
                                    f"{parent_col}.{child_name_raw.lower()}"
                                )
                                # Only set if not already present (allow later overrides if needed)
                                if dot_path not in type_map:
                                    type_map[dot_path] = child_type_norm
                    except Exception:
                        # Non-fatal; fallback to previously built map
                        pass
                    st.session_state['full_dataset_field_types'] = type_map
                    st.session_state['full_ds_schema_key'] = schema_cache_key
                except Exception as e:
                    st.warning(
                        f"⚠️ Fast schema inference failed ({e}); falling back to generic VARCHAR display."
                    )
                    _render_fallback_fields(all_available_fields)
                    type_map = {}

            if type_map:
                content = """
                    <div style='max-height: 300px; overflow-y: auto; padding: 0.5em; 
                                border: 1px solid var(--secondary-background-color); 
                                border-radius: 5px; 
                                background-color: var(--background-color); 
                                margin-bottom: 1em; 
                                color: var(--text-color);'>
                    """

                for field in sorted(all_available_fields):
                    display_type = type_map.get(field, 'VARCHAR')
                    content += f"<div style='margin-bottom: 2px;'>• <code>{field}</code> ({display_type})</div>"
                content += "</div>"
                st.markdown(content, unsafe_allow_html=True)
                if not need_refresh:
                    st.caption(
                        f"Schema cached ({len(type_map)} columns inferred)"
                    )
                else:
                    st.caption(
                        f"Schema inferred via DuckDB DESCRIBE ({len(type_map)} columns)"
                    )

    st.markdown("### 💻 SQL Query for Full Dataset Export")
    sql_query = st.text_area(
        "Enter your SQL query:",
        value='SELECT * FROM data LIMIT 10',
        height=120,
        help="Use 'data' as the table name. Query executes directly on JSONL files using DuckDB. Access nested fields with dot notation in double quotes (e.g., \"metadata.category\").",
        key="full_dataset_sql_query",
    )

    # Output Configuration
    st.markdown("### 📁 Output Configuration")

    col_config1, col_config2 = st.columns([3, 2])

    with col_config1:
        output_path = st.text_input(
            "Output Directory:",
            value=os.getcwd(),
            help="Directory where the filtered results will be saved",
            key="full_dataset_output_path",
        )

    with col_config2:
        output_format = st.selectbox(
            "Output Format:",
            ["jsonl", "csv"],
            help="Format for the exported filtered data",
            key="full_dataset_output_format",
        )

    col_config3, col_config4 = st.columns([2, 2])

    with col_config3:
        add_timestamp = st.checkbox(
            "Add timestamp to filename",
            value=True,
            help="Add current timestamp to output filename",
            key="full_dataset_add_timestamp",
        )

    with col_config4:
        chunk_output = st.checkbox(
            "Chunk large outputs",
            value=False,
            help="Split large result files into smaller chunks",
            key="full_dataset_chunk_output",
        )

    # Query optimization option
    col_config5, col_config6 = st.columns([2, 2])

    with col_config5:
        use_optimized_query = st.checkbox(
            "🚀 Use single query for entire dataset",
            value=True,
            help="Use DuckDB's native glob pattern for 1 query across all files (much faster than N separate queries)",
            key="full_dataset_use_optimized",
        )

    with col_config6:
        if use_optimized_query:
            st.info(
                "🚀 **Optimized mode**: Single query across all files (recommended)"
            )
        else:
            st.info(
                "🐌 **Fallback mode**: Separate query per file (for compatibility)"
            )

    if chunk_output:
        chunk_size = st.number_input(
            "Records per chunk:",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            help="Number of records per output chunk file",
            key="full_dataset_chunk_size",
        )

    # Execute Query and Save Button
    if st.button(
        "🚀 Execute Query and Save Results",
        type="primary",
        key="execute_full_dataset_sql",
    ):
        if not sql_query.strip():
            st.error("Please enter a SQL query")
        elif not output_path.strip():
            st.error("Please specify an output directory")
        else:
            # Prepare output configuration
            output_config = {
                'output_path': output_path,
                'output_format': output_format,
                'add_timestamp': add_timestamp,
                'chunk_output': chunk_output,
                'chunk_size': chunk_size if chunk_output else None,
                'use_optimized_query': use_optimized_query,
            }

            execute_full_dataset_query(sql_query, output_config)


def execute_full_dataset_query(sql_query, output_config):
    """Execute SQL query on full dataset and save results"""
    # Validate output directory
    try:
        if not os.path.exists(output_config['output_path']):
            os.makedirs(output_config['output_path'])
    except Exception as e:
        st.error(f"Error creating output directory: {e}")
        return

    # Progress tracking setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    progress_info = st.empty()

    with st.spinner(
        "Executing SQL query directly on files... This may take a while."
    ):
        try:
            start_time = time.time()

            # Standardize the SQL query syntax using processor helper with optional type map
            type_map = (
                st.session_state.get('full_dataset_field_types', {})
                if 'full_dataset_field_types' in st.session_state
                else {}
            )
            standardized_sql_query = SQLQueryProcessor.standardize_sql_syntax(
                sql_query, type_map
            )

            # Get data files from data directory
            data_path = st.session_state.get('data_path', 'data')
            file_pattern = st.session_state.get('file_pattern', '*.jsonl')
            file_format = st.session_state.get('file_format', 'jsonl')

            # Use the data path from sidebar
            data_files = glob.glob(os.path.join(data_path, file_pattern))

            if not data_files:
                st.error(
                    f"No {file_format.upper()} files found in '{data_path}' matching pattern '{file_pattern}'"
                )
                return

            st.info(
                f"Found {len(data_files)} {file_format.upper()} files to query"
            )

            # Create DuckDB connection
            conn = duckdb.connect()

            try:
                # Calculate total data size for progress estimation
                total_size_mb = sum(
                    os.path.getsize(f) / (1024 * 1024) for f in data_files
                )
                status_text.text(
                    f"📊 Total data size: {total_size_mb:.1f} MB across {len(data_files)} files"
                )

                # OPTIMIZED vs FALLBACK APPROACH based on user preference
                if output_config.get('use_optimized_query', True):
                    # OPTIMIZED SINGLE QUERY APPROACH - Use DuckDB's native glob pattern support
                    # This runs ONE query across ALL files instead of N separate queries

                    # Create glob pattern for all files
                    glob_pattern = os.path.join(
                        data_path, file_pattern
                    ).replace('\\', '/')

                    status_text.text(
                        f"🚀 Executing single optimized query across {len(data_files)} files..."
                    )
                    progress_bar.progress(0.1)

                    # Choose appropriate DuckDB function based on file format
                    if file_format == 'jsonl':
                        # Single query for ALL JSONL files using glob pattern
                        optimized_query = standardized_sql_query.replace(
                            'FROM data',
                            f"FROM read_json_auto('{glob_pattern}', format='newline_delimited', maximum_object_size=1000000000)",
                        )
                    elif file_format == 'parquet':
                        # Single query for ALL Parquet files using glob pattern
                        optimized_query = standardized_sql_query.replace(
                            'FROM data', f"FROM read_parquet('{glob_pattern}')"
                        )
                    else:
                        st.error(
                            f"Unsupported file format for Full Dataset SQL: {file_format}"
                        )
                        conn.close()
                        return

                    status_text.text(
                        f"🔄 DuckDB processing all {len(data_files)} files with single query..."
                    )
                    progress_bar.progress(0.3)

                    try:
                        # Execute the SINGLE optimized query that processes ALL files at once
                        query_start = time.time()
                        result_df = conn.execute(optimized_query).fetchdf()
                        query_time = time.time() - query_start

                        progress_bar.progress(1.0)
                        # throughput = total_size_mb / query_time if query_time > 0 else 0
                        # status_text.text(f"✅ Single-query optimization complete! Throughput: {throughput:.1f} MB/s")
                        # progress_info.text(f"🚀 Processed {len(data_files)} files with 1 query in {query_time:.1f}s - {len(result_df):,} results")

                        # st.success(f"🚀 **Optimized Execution**: Used single DuckDB query instead of {len(data_files)} separate queries!")
                        # st.info(f"💡 **Performance Benefit**: ~{len(data_files)}x reduction in query overhead + optimal memory usage")

                    except Exception as query_error:
                        # Auto-fallback to file-by-file approach if glob pattern fails
                        st.warning(
                            f"⚠️ Optimized single query failed: {query_error}"
                        )
                        st.info(
                            "🔄 Auto-falling back to file-by-file processing..."
                        )

                        # Set flag to use fallback approach
                        output_config['use_optimized_query'] = False

                # FALLBACK APPROACH: Process files individually (either by user choice or auto-fallback)
                if not output_config.get('use_optimized_query', True):
                    status_text.text(
                        f"🔄 Processing {len(data_files)} files individually..."
                    )
                    progress_bar.progress(0.0)

                    # Process files individually with memory optimization
                    result_chunks = []
                    processed_size = 0
                    total_records = 0

                    for i, file_path in enumerate(data_files):
                        file_size_mb = os.path.getsize(file_path) / (
                            1024 * 1024
                        )
                        normalized_path = file_path.replace('\\', '/')

                        # Update progress
                        progress = i / len(data_files)
                        progress_bar.progress(progress)

                        elapsed_time = time.time() - start_time
                        if processed_size > 0:
                            avg_speed = processed_size / elapsed_time
                            remaining_size = total_size_mb - processed_size
                            eta_seconds = (
                                remaining_size / avg_speed
                                if avg_speed > 0
                                else 0
                            )
                            eta_text = (
                                f"ETA: {eta_seconds/60:.1f}m"
                                if eta_seconds > 60
                                else f"ETA: {eta_seconds:.0f}s"
                            )
                        else:
                            eta_text = "Calculating ETA..."

                        status_text.text(
                            f"🔄 File {i+1}/{len(data_files)}: {os.path.basename(file_path)} ({file_size_mb:.1f} MB) - {eta_text}"
                        )
                        progress_info.text(
                            f"📊 Processed: {processed_size:.1f}/{total_size_mb:.1f} MB ({progress*100:.1f}%)"
                        )

                        # Execute query on this file
                        if file_format == 'jsonl':
                            file_query = standardized_sql_query.replace(
                                'FROM data',
                                f"FROM read_json_auto('{normalized_path}', format='newline_delimited', maximum_object_size=1000000000)",
                            )
                        elif file_format == 'parquet':
                            file_query = standardized_sql_query.replace(
                                'FROM data',
                                f"FROM read_parquet('{normalized_path}')",
                            )
                        else:
                            st.warning(
                                f"⚠️ Skipping {os.path.basename(file_path)}: Unsupported format {file_format}"
                            )
                            continue

                        try:
                            file_result = conn.execute(file_query).fetchdf()
                            if not file_result.empty:
                                result_chunks.append(file_result)
                                total_records += len(file_result)

                                # Memory optimization: Combine chunks periodically
                                if len(result_chunks) >= 10:
                                    progress_info.text(
                                        f"📊 Memory optimization: combining {len(result_chunks)} chunks..."
                                    )
                                    combined_chunk = pd.concat(
                                        result_chunks, ignore_index=True
                                    )
                                    result_chunks = [combined_chunk]

                        except Exception as file_error:
                            st.warning(
                                f"⚠️ Error processing {os.path.basename(file_path)}: {file_error}"
                            )
                            continue

                        processed_size += file_size_mb

                    # Combine final results for fallback approach
                    progress_bar.progress(0.95)
                    status_text.text("🔄 Final combination of results...")
                    progress_info.text(
                        f"📊 Combining final {len(result_chunks)} chunks..."
                    )

                    if result_chunks:
                        concat_start = time.time()
                        if len(result_chunks) > 1:
                            result_df = pd.concat(
                                result_chunks, ignore_index=True
                            )
                        else:
                            result_df = result_chunks[0]
                        concat_time = time.time() - concat_start

                        progress_bar.progress(1.0)
                        status_text.text(
                            f"✅ File-by-file processing complete: {len(result_df):,} records in {concat_time:.1f}s"
                        )
                    else:
                        result_df = pd.DataFrame()

                    progress_bar.progress(1.0)
                    elapsed_time = time.time() - start_time
                    overall_throughput = (
                        total_size_mb / elapsed_time if elapsed_time > 0 else 0
                    )
                    status_text.text(
                        f"✅ Query processing completed! Overall throughput: {overall_throughput:.1f} MB/s"
                    )

                processing_time = time.time() - start_time

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                progress_info.empty()

            except Exception as duckdb_error:
                # Clear progress indicators on error
                progress_bar.empty()
                status_text.empty()
                progress_info.empty()

                st.error(f"DuckDB query error: {duckdb_error}")
                st.info(
                    "💡 Make sure your query uses valid SQL syntax and the JSON structure matches your query fields"
                )
                conn.close()
                return

            finally:
                conn.close()

            # Save results to file with progress tracking
            if not result_df.empty:
                # Save file logic
                save_progress_bar = st.progress(0)
                save_status_text = st.empty()
                save_info_text = st.empty()

                save_start_time = time.time()

                # Generate filename
                timestamp = (
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                    if output_config['add_timestamp']
                    else ""
                )
                base_filename = (
                    f"filtered_data{'_' + timestamp if timestamp else ''}"
                )

                # Save based on format and chunking
                if (
                    output_config['chunk_output']
                    and len(result_df) > output_config['chunk_size']
                ):
                    # Save in chunks with progress
                    chunk_size = output_config['chunk_size']
                    num_chunks = (len(result_df) + chunk_size - 1) // chunk_size
                    saved_files = []

                    save_status_text.text(f"💾 Saving {num_chunks} chunks...")

                    for i in range(num_chunks):
                        # Update progress
                        chunk_progress = i / num_chunks
                        save_progress_bar.progress(chunk_progress)

                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(result_df))
                        chunk_df = result_df.iloc[start_idx:end_idx]

                        chunk_filename = f"{base_filename}_chunk_{i+1:03d}.{output_config['output_format']}"
                        chunk_filepath = os.path.join(
                            output_config['output_path'], chunk_filename
                        )

                        save_status_text.text(
                            f"💾 Saving chunk {i+1}/{num_chunks}: {chunk_filename}"
                        )
                        save_info_text.text(
                            f"📁 Records: {start_idx:,} - {end_idx:,}"
                        )

                        # Save chunk
                        chunk_save_start = time.time()
                        if output_config['output_format'] == "jsonl":
                            # Serialize timestamps as ISO-8601 like CSV text representation
                            chunk_df.to_json(
                                chunk_filepath,
                                orient='records',
                                lines=True,
                                date_format='iso',
                                date_unit='ns',
                            )
                        elif output_config['output_format'] == "csv":
                            chunk_df.to_csv(chunk_filepath, index=False)

                        chunk_save_time = time.time() - chunk_save_start
                        save_info_text.text(
                            f"✅ Chunk {i+1} saved in {chunk_save_time:.1f}s"
                        )

                        saved_files.append(chunk_filename)

                    save_progress_bar.progress(1.0)
                    total_save_time = time.time() - save_start_time

                    # Clear save progress indicators
                    save_progress_bar.empty()
                    save_status_text.empty()
                    save_info_text.empty()

                    processing_time = time.time() - start_time
                    st.info(
                        f"📁 Saved {len(result_df):,} records to: {output_config['output_path']}"
                    )
                    st.info(f"📄 Files: {', '.join(saved_files)}")
                    st.info(
                        f"⏱️ Total time: {processing_time:.1f}s (Save: {total_save_time:.1f}s)"
                    )

                else:
                    # Save as single file with progress
                    save_progress_bar.progress(0.1)
                    filename = (
                        f"{base_filename}.{output_config['output_format']}"
                    )
                    filepath = os.path.join(
                        output_config['output_path'], filename
                    )

                    save_status_text.text(f"💾 Saving single file: {filename}")
                    save_info_text.text(
                        f"📊 Format: {output_config['output_format'].upper()}"
                    )

                    save_progress_bar.progress(0.5)
                    file_save_start = time.time()

                    if output_config['output_format'] == "jsonl":
                        save_status_text.text(f"💾 Writing JSONL file...")
                        result_df.to_json(
                            filepath,
                            orient='records',
                            lines=True,
                            date_format='iso',
                            date_unit='ns',
                        )
                    elif output_config['output_format'] == "csv":
                        save_status_text.text(f"💾 Writing CSV file...")
                        result_df.to_csv(filepath, index=False)

                    file_save_time = time.time() - file_save_start
                    save_progress_bar.progress(1.0)

                    # Clear save progress indicators
                    save_progress_bar.empty()
                    save_status_text.empty()
                    save_info_text.empty()

                    processing_time = time.time() - start_time
                    st.success(f"✅ Query executed and results saved!")
                    st.info(
                        f"📁 Saved {len(result_df):,} records to: {filepath}"
                    )
                    st.info(
                        f"⏱️ Total time: {processing_time:.1f}s (Save: {file_save_time:.1f}s)"
                    )
            else:
                st.warning("No results to save - query returned empty dataset")

        except Exception as e:
            st.error(f"Error executing query: {e}")
            st.code(traceback.format_exc())


def handle_batch_sql_interface():
    """Handle SQL interface for current batch and batch range"""
    # SQL Query Input
    st.markdown("### 💻 SQL Query")

    # Function to reset the query text
    def reset_sql_query():
        """Reset the SQL query and clear cache"""
        # Clear SQL-related session state
        sql_keys_to_clear = [
            'sql_query_df',
            'sql_result_df',
            'sql_result_msg',
            'sql_result_scope',
        ]
        for key in sql_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        # Reset SQL processor's cache
        if 'sql_processor' in st.session_state:
            st.session_state['sql_processor'].clear_cache()

        # Delete the existing widget's key value to allow a new default on next run
        if 'sql_query_text' in st.session_state:
            del st.session_state['sql_query_text']

        # Set a flag to use default query on next render
        st.session_state['reset_sql_query'] = True

    # Get default query - check if we need to use the reset default
    if (
        'reset_sql_query' in st.session_state
        and st.session_state['reset_sql_query']
    ):
        default_query = 'SELECT * FROM data LIMIT 10'
        # Clear the reset flag
        del st.session_state['reset_sql_query']
    else:
        default_query = st.session_state.get(
            'sql_query_input', 'SELECT * FROM data LIMIT 10'
        )

    # Create the text area and buttons
    sql_query = st.text_area(
        "Enter your SQL query:",
        value=default_query,
        height=150,
        help="Use \"data\" as the table. Quote nested column names with double quotes, e.g., SELECT \"metadata.category\" FROM data",
        key="sql_query_text",
    )

    # Create two columns for the buttons
    col_btn1, col_btn2 = st.columns([2, 1])

    with col_btn1:
        execute_button = st.button("🚀 Execute Query", type="primary")
    with col_btn2:
        # Use on_click to trigger the reset function
        clear_button = st.button(
            "🗑️ Clear Query", type="secondary", on_click=reset_sql_query
        )

    # Query execution controls
    if execute_button:
        if sql_query.strip():
            with st.spinner("Executing SQL query..."):
                sql_processor = st.session_state['sql_processor']
                result_df, result_msg = sql_processor.execute_query(
                    sql_query, 'data'
                )

                if result_df is not None:
                    # Show results immediately without caching
                    st.success(result_msg)

                    st.markdown("### 📊 Query Results")

                    # Results display options
                    col_res1, col_res2 = st.columns([2, 2])

                    with col_res1:
                        st.metric("Rows Returned", f"{len(result_df):,}")

                    with col_res2:
                        st.metric("Columns", len(result_df.columns))

                    # Display results table
                    st.dataframe(
                        result_df, use_container_width=True, height=400
                    )
                else:
                    st.error(result_msg)
        else:
            st.warning("Please enter a SQL query")


def expand_nested_fields_for_sql(df, selected_fields):
    """Expand nested fields into individual columns for SQL queries"""
    if df.empty:
        return df

    # Early detection of whether we need processing
    has_nested = any('.' in field for field in selected_fields)
    if not has_nested:
        return df  # Return original if no nested fields

    # Identify nested fields and their parents
    nested_fields = {}  # {parent: [(full_path, nested_path), ...]}
    parent_fields = set()

    for field in selected_fields:
        if '.' in field:
            parts = field.split('.')
            parent = parts[0]
            nested_path = '.'.join(parts[1:])

            if parent not in nested_fields:
                nested_fields[parent] = []
                parent_fields.add(parent)

            nested_fields[parent].append((field, nested_path))

    # Create result columns dict - start with non-parent columns
    result_columns = {}
    for col in df.columns:
        if col not in parent_fields:
            result_columns[col] = df[col]

    # Process nested fields efficiently
    for parent, paths in nested_fields.items():
        if parent not in df.columns:
            continue

        parent_data = df[parent]

        for full_path, nested_path in paths:
            try:
                # Create new column directly as Series
                result_columns[full_path] = pd.Series(
                    [
                        (
                            get_nested_value(x, nested_path)
                            if isinstance(x, dict)
                            else None
                        )
                        for x in parent_data
                    ],
                    index=df.index,
                )
            except Exception:
                # Fallback for errors
                result_columns[full_path] = pd.Series(
                    [None] * len(df), index=df.index
                )

    # Create final DataFrame in one operation
    return pd.DataFrame(result_columns)

    # Removed local standardize_sql_syntax in favor of SQLQueryProcessor.standardize_sql_syntax


# ------------------------ Mini Histogram Helpers (Phase 1) ------------------------
def _extract_series_from_column(
    display_df_source: pd.DataFrame, col_name: str
) -> pd.Series:
    """Return a pandas Series for the requested column from the source batch.
    Supports nested dot paths using get_nested_value on each row.
    Always returns a Series of length == len(display_df_source).
    """
    if display_df_source is None or display_df_source.empty:
        return pd.Series([], dtype=object)
    if '.' in col_name and col_name not in display_df_source.columns:
        # Build from nested path
        vals = [
            get_nested_value(row, col_name, None)
            for _, row in display_df_source.iterrows()
        ]
        return pd.Series(vals)
    if col_name in display_df_source.columns:
        return display_df_source[col_name]
    # Unknown column -> empty series of Nones
    return pd.Series([None] * len(display_df_source))


def _build_chart_for_series(
    series_in: pd.Series, title: str, size: str = 'mini'
):
    """Build an Altair chart for a Series and return (chart, caption_text).
    size: 'mini' for small grid thumbnails, 'full' for normal-size charts.
    Only two types are supported: numeric histogram or text length histogram.
    All non-numeric data (including datetime, categorical, bool-as-text) is treated as text.
    """
    try:
        s = series_in.copy()
        total = int(len(s))
        # Normalize missing detection (treat empty strings as missing for texty cols)
        missing_mask = s.isna() | (s.astype(object) == '')
        non_missing = s[~missing_mask]
        missing_count = int(missing_mask.sum())

        # Dimensions
        if size == 'full':
            h_small = 200
            h_medium = 240
            h_cat = 260
            maxbins_cap = 60
            topk_cat = 30
        else:
            h_small = 80
            h_medium = 100
            h_cat = 120
            maxbins_cap = 30
            topk_cat = 20

        # Try numeric
        numeric = pd.to_numeric(non_missing, errors='coerce')
        numeric = numeric.dropna()
        if not numeric.empty:
            df_num = pd.DataFrame({'value': numeric})
            maxbins = int(min(maxbins_cap, max(5, df_num['value'].nunique())))
            chart = (
                alt.Chart(df_num)
                .mark_bar()
                .encode(
                    x=alt.X(
                        'value:Q', bin=alt.Bin(maxbins=maxbins), title=None
                    ),
                    y=alt.Y('count()', title=None),
                )
                .properties(height=h_small if size == 'mini' else h_medium)
            )
            try:
                mn, mx = float(df_num['value'].min()), float(
                    df_num['value'].max()
                )
                mean = float(df_num['value'].mean())
                caption = f"min {mn:.2f} • mean {mean:.2f} • max {mx:.2f} • missing {missing_count/total*100:.1f}%"
            except Exception:
                caption = f"missing {missing_count/total*100:.1f}%"
            return chart, caption

        # Text length histogram for all remaining data
        s_str = non_missing.astype(str)
        lens = s_str.map(lambda v: len(v))
        df_len = pd.DataFrame({'length': lens})
        maxbins = int(min(maxbins_cap, max(5, df_len['length'].nunique())))
        chart = (
            alt.Chart(df_len)
            .mark_bar()
            .encode(
                x=alt.X('length:Q', bin=alt.Bin(maxbins=maxbins), title=None),
                y=alt.Y('count()', title=None),
            )
            .properties(height=h_small if size == 'mini' else h_medium)
        )
        try:
            mean_len = float(df_len['length'].mean())
            p95 = (
                float(np.percentile(df_len['length'], 95)) if len(df_len) else 0
            )
            caption = f"avg len {mean_len:.1f} • p95 {p95:.0f} • missing {missing_count/total*100:.1f}%"
        except Exception:
            caption = f"missing {missing_count/total*100:.1f}%"
        return chart, caption

    except Exception as e:
        # On any error, show nothing but a small caption
        return None, f"error: {e}"


def _mini_chart_for_series(series_in: pd.Series, title: str):
    """Backwards-compatible wrapper for mini-sized charts."""
    return _build_chart_for_series(series_in, title, size='mini')


def render_mini_histograms(
    display_df_source: pd.DataFrame, display_columns: list[str]
):
    """Render a compact grid of mini histograms aligned in the same order as visible columns.
    Designed for batch-sized data (10-1000 rows). Recomputes on batch change.
    """
    if display_df_source is None or display_df_source.empty:
        return
    # Avoid too many plots at once
    columns_to_show = display_columns[:40]
    per_row = (
        4
        if len(columns_to_show) > 8
        else (3 if len(columns_to_show) > 4 else len(columns_to_show))
    )
    # Title row
    st.markdown("#### Column Distributions (current batch)")
    for i in range(0, len(columns_to_show), max(1, per_row)):
        subset = columns_to_show[i : i + per_row]
        row_cols = st.columns(len(subset))
        for sc, col_name in zip(row_cols, subset):
            with sc:
                s = _extract_series_from_column(display_df_source, col_name)
                chart, caption = _mini_chart_for_series(s, col_name)
                # Header with truncation
                label = (
                    col_name if len(col_name) <= 40 else (col_name[:37] + '…')
                )
                st.markdown(f"**{label}**")
                if chart is not None and not s.dropna().empty:
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.caption("No data")
                if caption:
                    st.caption(caption)


def render_full_histograms(
    display_df_source: pd.DataFrame, display_columns: list[str]
):
    """Render normal-size charts, one per selected column, full-width."""
    if display_df_source is None or display_df_source.empty:
        return
    st.markdown("### Column Distributions")
    for col_name in display_columns:
        s = _extract_series_from_column(display_df_source, col_name)
        chart, caption = _build_chart_for_series(s, col_name, size='full')
        label = col_name if len(col_name) <= 80 else (col_name[:77] + '…')
        st.markdown(f"#### {label}")
        if chart is not None and not s.dropna().empty:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("No data")
        if caption:
            st.caption(caption)


def handle_raw_data_tab(lazy_loader, selected_fields, batch_size, filtered_df):
    """Handle everything in the Raw Data tab"""
    st.markdown("## 🔎 Raw Data Explorer")

    if filtered_df.empty:
        st.warning("No data available with current filters.")
        return

    # Batch Navigation Controls using the component
    # from dqsage.ui.components import display_batch_navigation_controls
    display_batch_navigation_controls(lazy_loader, batch_size)

    st.markdown("---")

    # Display entire batch (no sampling needed since batch size is already controlled)
    display_df_source = filtered_df  # Use the entire filtered batch

    # Display the main data table with column selection
    display_data_table(display_df_source, selected_fields)

    # Individual Record Viewer using the component
    # from dqsage.ui.components import display_individual_record_viewer
    display_individual_record_viewer(display_df_source, selected_fields)


def display_data_table(display_df_source, selected_fields):
    """Display the main data table with column selection"""
    # Dynamic column selection based on loaded fields
    available_columns = ['source_file', 'line_number']

    # Add loaded fields to display options
    for field in selected_fields:
        if field not in available_columns:
            available_columns.append(field)

    # Show selected columns
    default_columns = available_columns[: min(4, len(available_columns))]
    saved_columns = st.session_state.get(
        "saved_display_columns", default_columns
    )
    # --- Apply Pattern (Option A implementation: bulk buttons BEFORE widget creation) ---
    # Canonical applied selection
    if "saved_display_columns" not in st.session_state:
        st.session_state["saved_display_columns"] = saved_columns

    # If a bulk operation was requested in previous run, hydrate staging before widget
    if "_pending_staging_columns" in st.session_state:
        # Remove existing widget key so we can safely set new default
        if "raw_data_display_columns_staging" in st.session_state:
            del st.session_state["raw_data_display_columns_staging"]
        st.session_state["raw_data_display_columns_staging"] = (
            st.session_state.pop("_pending_staging_columns")
        )

    # Ensure staging exists (copy of applied selection)
    if "raw_data_display_columns_staging" not in st.session_state:
        st.session_state["raw_data_display_columns_staging"] = st.session_state[
            "saved_display_columns"
        ].copy()

    # Removed top-level bulk buttons (Select All / Clear / Revert) per request; show counts only
    # st.caption(f"Applied columns: {len(st.session_state['saved_display_columns'])} | Staging: {len(st.session_state['raw_data_display_columns_staging'])}")

    # Persistent checkbox-based selector to avoid dropdown closing after each pick
    with st.expander("📊 Column Selector", expanded=False):
        filter_text = st.text_input(
            "Filter columns", key="col_filter", placeholder="type to filter..."
        )
        if filter_text:
            filtered_columns = [
                c for c in available_columns if filter_text.lower() in c.lower()
            ]
        else:
            filtered_columns = available_columns

        # Compute current staging selection early (checkbox values from previous run already in session_state)
        current_staging = (
            set(st.session_state["raw_data_display_columns_staging"])
            if "raw_data_display_columns_staging" in st.session_state
            else set()
        )
        early_staging_selection = [
            c
            for c in available_columns
            if st.session_state.get(f"col_cb_{c}", False)
        ]
        if (
            early_staging_selection
            and set(early_staging_selection) != current_staging
        ):
            st.session_state["raw_data_display_columns_staging"] = (
                early_staging_selection.copy()
            )
            current_staging = set(early_staging_selection)

        # Bulk + Apply row
        # bulk_vis_1, bulk_vis_2, bulk_vis_3, bulk_vis_4 = st.columns([1,1,1,0.5])
        bulk_vis_1, bulk_vis_2, bulk_vis_3 = st.columns([1, 1, 1])
        with bulk_vis_1:
            if st.button("Select All"):
                for c in filtered_columns:
                    st.session_state[f"col_cb_{c}"] = True
                st.session_state["_pending_staging_columns"] = [
                    c
                    for c in available_columns
                    if st.session_state.get(f"col_cb_{c}", False)
                ]
                st.rerun()
        with bulk_vis_2:
            if st.button("Select None"):
                for c in filtered_columns:
                    st.session_state[f"col_cb_{c}"] = False
                st.session_state["_pending_staging_columns"] = [
                    c
                    for c in available_columns
                    if st.session_state.get(f"col_cb_{c}", False)
                ]
                st.rerun()
        with bulk_vis_3:
            if st.button("✅ Apply"):
                # Recompute staging from current checkbox states and apply
                staging_for_apply = [
                    c
                    for c in available_columns
                    if st.session_state.get(f"col_cb_{c}", False)
                ]
                st.session_state["saved_display_columns"] = (
                    staging_for_apply.copy()
                )
                st.success("Columns applied")
                st.rerun()
        # with bulk_vis_4:
        #     st.caption("Toggle select then Apply.")

        # Initialize checkbox states from current (possibly updated) staging selection
        effective_staging = set(
            st.session_state["raw_data_display_columns_staging"]
        )
        for c in available_columns:
            key = f"col_cb_{c}"
            if key not in st.session_state:
                st.session_state[key] = c in effective_staging

        # Display checkboxes grid
        cols_per_row = 3 if len(filtered_columns) > 12 else 2
        for i in range(0, len(filtered_columns), cols_per_row):
            row = filtered_columns[i : i + cols_per_row]
            row_cols = st.columns(len(row))
            for rc, col_name in zip(row_cols, row):
                with rc:
                    st.checkbox(col_name, key=f"col_cb_{col_name}")

        # Reconstruct staging selection AFTER rendering (captures current run choices)
        staging_selection = [
            c
            for c in available_columns
            if st.session_state.get(f"col_cb_{c}", False)
        ]
        if set(staging_selection) != effective_staging:
            st.session_state["raw_data_display_columns_staging"] = (
                staging_selection.copy()
            )

    # Use applied selection for display
    display_columns = st.session_state["saved_display_columns"]

    if display_columns:
        # Prepare display dataframe
        display_df = pd.DataFrame()

        def extract_nested_values(path: str):
            vals = []
            for _, row in display_df_source.iterrows():
                vals.append(get_nested_value(row, path, None))
            return vals

        def is_numeric_value(v):
            if v is None:
                return True  # allow missing within numeric column
            if isinstance(v, bool):
                return False
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return True
            if isinstance(v, str):
                s = v.strip()
                if s == '':
                    return True
                try:
                    float(s)
                    return True
                except Exception:
                    return False
            return False

        def all_numeric(vals):
            any_real = False
            for v in vals:
                if v is None:
                    continue
                if isinstance(v, str) and v.strip() == '':
                    continue
                if not is_numeric_value(v):
                    return False
                any_real = True
            return any_real

        for col in display_columns:
            if '.' in col:
                raw_vals = extract_nested_values(col)
                if all_numeric(raw_vals):
                    # Coerce to numeric (NaN for non-convertible blanks/None)
                    coerced = []
                    for v in raw_vals:
                        if v is None:
                            coerced.append(np.nan)
                        elif isinstance(v, str) and v.strip() == '':
                            coerced.append(np.nan)
                        else:
                            try:
                                coerced.append(float(v))
                            except Exception:
                                coerced.append(np.nan)
                    display_df[col] = pd.Series(coerced)
                else:
                    display_df[col] = [
                        format_display_value(v) for v in raw_vals
                    ]
            else:
                if col in display_df_source.columns:
                    source_series = display_df_source[col]
                    if pd.api.types.is_numeric_dtype(source_series):
                        display_df[col] = source_series
                    else:
                        display_df[col] = source_series.apply(
                            format_display_value
                        )
                else:
                    display_df[col] = ''

    st.dataframe(display_df, use_container_width=True, height=500)


def handle_stats_tab(filtered_df, selected_fields):
    """Stats tab showing per-column histograms for the current batch.
    Recomputes automatically when the batch changes (Streamlit rerun).
    """
    st.markdown("## 📈 Stats (Current Batch)")

    if filtered_df is None or filtered_df.empty:
        st.info("Load a batch to view stats.")
        return

    # Build options: include saved display columns and selected_fields (nested dot paths), plus DataFrame columns
    saved_cols = st.session_state.get("saved_display_columns", [])
    base_cols = list(filtered_df.columns)
    extra_cols = selected_fields or []
    # Merge while preserving order preference: saved -> selected_fields -> df columns
    merged = []
    for lst in (saved_cols, extra_cols, base_cols):
        for c in lst:
            if c not in merged:
                merged.append(c)
    # Exclude technical columns
    options = [c for c in merged if c not in ("source_file", "line_number")]
    # Default selection prefers saved columns (filtered and capped), fallback to options
    default_pool = [c for c in saved_cols if c in options] or options
    default_cols = default_pool[:40]

    cols = st.multiselect(
        "Columns to analyze:",
        options=options,
        default=default_cols,
        help="Choose columns to render histograms for (current batch only).",
        key="stats_columns_selection",
    )
    if not cols:
        st.info("Select at least one column to render stats.")
        return

    try:
        render_full_histograms(filtered_df, cols)
    except Exception as e:
        st.warning(f"Could not render stats: {e}")


def validate_and_setup_data():
    """Validate current data and setup for display"""
    # Get current data for display
    if (
        'current_df' not in st.session_state
        or st.session_state['current_df'].empty
    ):
        return None, None

    df = st.session_state['current_df']

    # Analyze loaded data structure using generic approach
    # from dqsage.core.data_analyzer import analyze_data_structure, validate_data_compatibility
    data_analysis = analyze_data_structure(df)

    # Validate data compatibility
    warnings, errors = validate_data_compatibility(data_analysis)

    # Show errors and warnings
    if errors:
        for error in errors:
            st.error(f"❌ {error}")
        return None, None

    if warnings:
        with st.expander("⚠️ Data Compatibility Warnings"):
            for warning in warnings:
                st.warning(warning)

    # Dynamic filters based on loaded fields
    filtered_df = df.copy()

    return df, filtered_df


def main():
    """Main application function - refactored into smaller components"""
    # Header
    st.markdown(
        '<h1 class="main-header">📊 DQSage Data Visualizer</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    data_path, file_format, file_pattern = setup_sidebar_configuration()

    # Check if data directory exists
    if not os.path.exists(data_path):
        st.error(
            f"❌ Data directory '{data_path}' not found! Please enter a valid directory path."
        )
        st.info(
            f"The app expects {file_format.upper()} files in the specified directory."
        )
        return

    # Schema analysis
    schema_info = handle_schema_analysis(data_path, file_format, file_pattern)
    if not schema_info:
        st.info(
            "👆 Please analyze the data schema first to see available fields and configure loading options."
        )
        return

    # Field selection and loader initialization
    lazy_loader, selected_fields, batch_size = (
        handle_field_selection_and_loader_init(schema_info)
    )
    if not lazy_loader:
        st.warning(
            "⏳ Please initialize the lazy data loader using the sidebar configuration."
        )
        return

    # Lazy loading controls
    display_lazy_loading_controls(lazy_loader, batch_size)

    # Validate and setup current data
    df, filtered_df = validate_and_setup_data()
    if df is None:
        st.info("📊 Load a batch or sample to start exploring the data!")
        return

    # Memory usage info
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    st.info(f"💾 Current batch memory usage: {memory_usage_mb:.1f} MB")

    # Main tabs (add Stats)
    tab_raw, tab_stats, tab_sql = st.tabs(
        ["🔎 Raw Data", "📈 Stats", "🗄️ SQL Query"]
    )

    with tab_stats:
        handle_stats_tab(filtered_df, selected_fields)

    with tab_sql:
        handle_sql_query_tab(
            lazy_loader, selected_fields, batch_size, filtered_df
        )

    with tab_raw:
        handle_raw_data_tab(
            lazy_loader, selected_fields, batch_size, filtered_df
        )


# Register cleanup function
if 'cleanup_registered' not in st.session_state:
    atexit.register(cleanup_sql_processor)
    st.session_state['cleanup_registered'] = True


if __name__ == "__main__":
    main()
