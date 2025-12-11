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
Data schema analysis and structure validation functions.
Provides tools for analyzing data structure and compatibility.
"""

import streamlit as st

from .file_handlers import FileFormatRegistry


@st.cache_data
def analyze_data_schema(data_dir, file_format, file_pattern):
    """Analyze schema of data files by examining only first record structure - optimized for performance"""
    # Get the appropriate handler
    try:
        handler = FileFormatRegistry.get_handler(file_format, file_pattern)
    except ValueError as e:
        return {'error': str(e), 'file_count': 0}
    except ImportError as e:
        return {'error': f"Format not available: {e}", 'file_count': 0}

    # Get matching files
    data_files = handler.get_files(data_dir)

    if not data_files:
        return {
            'error': f"No {file_format.upper()} files found in {data_dir} with pattern {file_pattern}",
            'file_count': 0,
        }

    schema_info = {
        'top_level_fields': set(),
        'nested_fields': {},  # Store nested dictionary fields: {parent_field: set(nested_keys)}
        'all_field_paths': set(),  # All available field paths including nested ones
        'reference_structure': None,  # Store structure of first record as reference
        'sample_records': [],
        'file_count': len(data_files),
        'file_format': file_format,
        'handler': handler,  # Store handler for later use
    }

    def extract_nested_fields(data, parent_path=""):
        """Recursively extract all nested field paths from a dictionary"""
        nested_paths = set()

        if not isinstance(data, dict):
            return nested_paths

        for key, value in data.items():
            current_path = f"{parent_path}.{key}" if parent_path else key
            nested_paths.add(current_path)

            # If value is a dictionary, recurse to get nested paths
            if isinstance(value, dict):
                nested_paths.update(extract_nested_fields(value, current_path))

        return nested_paths

    # Get the first file only
    first_file = data_files[0]

    try:
        first_record = handler.read_first_record(first_file)

        if first_record:
            # Store as reference structure
            schema_info['reference_structure'] = first_record

            # Extract all field paths
            all_paths = extract_nested_fields(first_record)
            schema_info['all_field_paths'] = all_paths

            # Separate top-level and nested fields
            for path in all_paths:
                if '.' not in path:
                    # Top-level field
                    schema_info['top_level_fields'].add(path)
                else:
                    # Nested field - track parent-child relationship
                    parts = path.split('.')
                    parent = parts[0]
                    nested_key = '.'.join(parts[1:])

                    if parent not in schema_info['nested_fields']:
                        schema_info['nested_fields'][parent] = set()
                    schema_info['nested_fields'][parent].add(nested_key)

            # Store sample record for reference
            schema_info['sample_records'] = [
                {
                    'structure': list(first_record.keys()),
                    'nested_structure': {
                        k: (
                            list(v.keys())
                            if isinstance(v, dict)
                            else type(v).__name__
                        )
                        for k, v in first_record.items()
                    },
                }
            ]
        else:
            schema_info['error'] = f"No valid records found in {first_file}"

    except Exception as e:
        schema_info['error'] = f"Error analyzing schema: {e}"

    return schema_info


def analyze_data_structure(df):
    """Return only what's needed for validation and UI setup: total_records  and columns.

    Contract:
    - Input: pandas DataFrame (may be empty)
    - Output: dict with keys 'total_records' (int) and 'columns' (list[str])
    - Guarantees: Keys always present; safe for empty DataFrames.
    """
    if df is None or getattr(df, "empty", True):
        return {"total_records": 0, "columns": []}

    return {
        "total_records": int(len(df)),
        "columns": list(df.columns),
    }


def validate_data_compatibility(analysis):
    """Validate if the data is compatible with the visualizer - generic approach"""
    warnings = []
    errors = []

    if analysis['total_records'] == 0:
        errors.append("No records found in the data")
        return warnings, errors

    # Generic validation - just check if we have any data at all
    if len(analysis['columns']) == 0:
        errors.append("No columns found in the data")
        return warnings, errors

    # Count useful columns (excluding our tracking columns)
    useful_columns = [
        col
        for col in analysis['columns']
        if col not in ['source_file', 'line_number']
    ]

    if len(useful_columns) == 0:
        errors.append("No data columns found (only tracking columns present)")
        return warnings, errors

    return warnings, errors
