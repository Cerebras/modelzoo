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
Lazy data loader for handling large datasets efficiently with caching.
Supports multiple file formats and provides batch-based data access.
"""

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
import streamlit as st

from ..utils.helpers import get_nested_value
from ..utils.workers import process_file_index_worker
from .file_handlers import FileFormatRegistry


class LazyDataLoader:
    """Enhanced lazy data loader for handling large datasets efficiently with caching - supports multiple file formats"""

    def __init__(
        self,
        data_dir,
        file_format="jsonl",
        file_pattern=None,
        selected_fields=None,
        text_field_for_limit=None,
        text_length_limit=None,
        reference_structure=None,
        handler=None,
    ):
        self.data_dir = data_dir
        self.file_format = file_format
        self.file_pattern = file_pattern or f"*.{file_format}"
        self.selected_fields = selected_fields or []
        self.text_field_for_limit = text_field_for_limit
        self.text_length_limit = text_length_limit
        self.reference_structure = reference_structure

        # Get file handler
        if handler:
            self.handler = handler
        else:
            self.handler = FileFormatRegistry.get_handler(
                file_format, self.file_pattern
            )

        # Get all data files using the handler
        self.data_files = self.handler.get_files(data_dir)
        if not self.data_files:
            raise FileNotFoundError(
                f"No {file_format.upper()} files found in {data_dir} with pattern {self.file_pattern}"
            )

        # Keep legacy attribute name for compatibility
        self.jsonl_files = self.data_files

        # Initialize lazy loading state
        self.current_file_idx = 0
        self.current_file_position = 0
        self.total_files = len(self.data_files)
        self.records_per_batch = 1000  # Configurable batch size
        self.inconsistent_records = []

        # Enhanced caching system
        self.batch_cache = {}  # Cache for loaded batches
        self.max_cached_batches = (
            50  # Maximum number of batches to keep in memory
        )
        self.file_iterators = {}  # Cache for file iterators

        # Fast random access indexing system
        self.batch_index = (
            {}
        )  # Maps batch_number -> {'file_idx': int, 'file_path': str, 'byte_offset': int, 'line_start': int, 'record_count': int}
        self.file_line_index = (
            {}
        )  # Maps file_path -> [byte_offsets] for each line
        self.is_indexed = False  # Flag to track if indexing is complete
        self.indexing_progress = {
            'current_file': 0,
            'total_files': 0,
            'current_batch': 0,
        }

        # Metadata for efficient navigation
        self._file_line_counts = {}
        self._batch_metadata = {}  # Store metadata about each batch

    def validate_structure(self, record, reference):
        """Validate record structure against reference"""
        if not reference:
            return True, []

        def validate_nested_dict(record_dict, reference_dict, path=""):
            """Recursively validate nested dictionary structures"""
            nested_issues = []

            if not isinstance(record_dict, dict) or not isinstance(
                reference_dict, dict
            ):
                return nested_issues

            record_keys = set(record_dict.keys())
            reference_keys = set(reference_dict.keys())

            if record_keys != reference_keys:
                missing = reference_keys - record_keys
                extra = record_keys - reference_keys
                prefix = f"{path}." if path else ""

                if missing:
                    nested_issues.append(
                        f"Missing {prefix}keys: {', '.join(missing)}"
                    )
                if extra:
                    nested_issues.append(
                        f"Extra {prefix}keys: {', '.join(extra)}"
                    )

            # Recursively check nested dictionaries
            for key in record_keys.intersection(reference_keys):
                if isinstance(record_dict[key], dict) and isinstance(
                    reference_dict[key], dict
                ):
                    current_path = f"{path}.{key}" if path else key
                    nested_issues.extend(
                        validate_nested_dict(
                            record_dict[key], reference_dict[key], current_path
                        )
                    )

            return nested_issues

        issues = []

        # Validate top-level structure
        record_keys = set(record.keys())
        reference_keys = set(reference.keys())

        if record_keys != reference_keys:
            missing = reference_keys - record_keys
            extra = record_keys - reference_keys
            if missing:
                issues.append(f"Missing top-level keys: {', '.join(missing)}")
            if extra:
                issues.append(f"Extra top-level keys: {', '.join(extra)}")

        # Recursively validate any nested dictionary structures
        for key in record_keys.intersection(reference_keys):
            if isinstance(record[key], dict) and isinstance(
                reference[key], dict
            ):
                issues.extend(
                    validate_nested_dict(record[key], reference[key], key)
                )

        return len(issues) == 0, issues

    def _process_record(self, data, file_path, line_num):
        """Process a single record with field filtering and validation"""

        def set_nested_value_safe(data_dict, field_path, value):
            """Safely set nested value using dot notation - supports unlimited depth"""
            try:
                keys = field_path.split('.')
                current = data_dict

                # Navigate to parent of target key, creating nested dicts as needed
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    elif not isinstance(current[key], dict):
                        # If we encounter a non-dict value, we can't go deeper
                        return False
                    current = current[key]

                # Set the final value
                current[keys[-1]] = value
                return True
            except:
                return False

        # Validate structure if reference provided
        if self.reference_structure:
            is_valid, issues = self.validate_structure(
                data, self.reference_structure
            )
            if not is_valid:
                self.inconsistent_records.append(
                    {
                        'file': os.path.basename(file_path),
                        'line': line_num + 1,
                        'keys': list(data.keys()),
                        'nested_keys': {
                            k: (
                                list(v.keys())
                                if isinstance(v, dict)
                                else type(v).__name__
                            )
                            for k, v in data.items()
                        },
                        'issues': issues,
                    }
                )
                return None  # Skip inconsistent records

        # Filter fields if specified
        if self.selected_fields:
            filtered_data = {}
            for field in self.selected_fields:
                if '.' in field:
                    # Handle nested field using global get_nested_value function
                    value = get_nested_value(data, field)
                    # Always try to set the field, even if value is None (to preserve structure)
                    if (
                        field == self.text_field_for_limit
                        and self.text_length_limit
                        and isinstance(value, str)
                    ):
                        value = value[: self.text_length_limit]
                    set_nested_value_safe(filtered_data, field, value)
                elif field in data:
                    # Handle top-level field
                    value = data[field]
                    # Apply text length limit if specified for this field
                    if (
                        field == self.text_field_for_limit
                        and self.text_length_limit
                        and isinstance(value, str)
                    ):
                        value = value[: self.text_length_limit]
                    filtered_data[field] = value
            data = filtered_data

        # Always add source info
        data['source_file'] = os.path.basename(file_path)
        data['line_number'] = line_num + 1
        return data

    def build_batch_index(
        self,
        batch_size=1000,
        progress_callback=None,
        use_parallel=True,
        max_workers=None,
    ):
        """Build comprehensive index for fast random access to batches using parallel processing"""
        if self.is_indexed:
            return  # Already indexed

        # Skip indexing for formats that don't require it (like Parquet with DuckDB directory queries)
        if (
            hasattr(self.handler, 'requires_indexing')
            and not self.handler.requires_indexing()
        ):
            if progress_callback:
                progress_callback(
                    f"⚡ Skipping indexing for {self.file_format.upper()} - using direct directory access for optimal performance!"
                )

            # Mark as "indexed" but with empty index to indicate direct access mode
            self.is_indexed = True
            self.total_batches = (
                self.get_total_batches()
            )  # Calculate based on estimated records
            return

        start_time = time.time()

        self.batch_index = {}
        self.file_line_index = {}

        self.indexing_progress['total_files'] = len(self.data_files)

        if progress_callback:
            progress_callback(
                f"🚀 Starting parallel indexing of {len(self.data_files)} files..."
            )

        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(len(self.data_files), mp.cpu_count())

        # Prepare file information for workers
        file_info_list = [
            (file_idx, file_path, batch_size, self.file_format)
            for file_idx, file_path in enumerate(self.data_files)
        ]

        # Choose parallel processing method based on file count and user preference
        if use_parallel and len(self.data_files) > 1:
            results = []
            completed_files = 0

            if progress_callback:
                progress_callback(
                    f"⚡ Using {max_workers} parallel workers for super-fast indexing..."
                )

            try:
                # Use ThreadPoolExecutor for better performance with file I/O
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_file = {
                        executor.submit(
                            process_file_index_worker, file_info
                        ): file_info
                        for file_info in file_info_list
                    }

                    # Process completed tasks as they finish
                    for future in as_completed(future_to_file):
                        file_info = future_to_file[future]
                        file_idx, file_path, _, _ = (
                            file_info  # Added extra _ for file_format
                        )

                        try:
                            result = future.result()
                            result['processing_time'] = time.time() - start_time
                            results.append(result)

                            completed_files += 1
                            self.indexing_progress['current_file'] = (
                                completed_files
                            )

                            if progress_callback:
                                progress_callback(
                                    f"✅ Indexed {completed_files}/{len(self.data_files)} files | "
                                    f"File: {os.path.basename(file_path)}"
                                )

                        except Exception as e:
                            if progress_callback:
                                progress_callback(
                                    f"❌ Error indexing {file_path}: {e}"
                                )
                            results.append(
                                {
                                    'file_idx': file_idx,
                                    'file_path': file_path,
                                    'error': str(e),
                                    'line_offsets': [],
                                    'valid_record_count': 0,
                                    'file_batches': [],
                                    'total_batches': 0,
                                }
                            )
                            completed_files += 1

            except Exception as e:
                if progress_callback:
                    progress_callback(
                        f"❌ Parallel processing failed: {e}. Falling back to sequential..."
                    )
                use_parallel = False

        # Fallback to sequential processing if parallel fails or is disabled
        if not use_parallel or len(self.data_files) == 1:
            if progress_callback:
                progress_callback("🔄 Using sequential indexing...")

            results = []
            for file_idx, file_path in enumerate(self.data_files):
                if progress_callback:
                    progress_callback(
                        f"Indexing file {file_idx + 1}/{len(self.data_files)}: {os.path.basename(file_path)}"
                    )

                file_info = (file_idx, file_path, batch_size, self.file_format)
                result = process_file_index_worker(file_info)
                result['processing_time'] = time.time() - start_time
                results.append(result)

                self.indexing_progress['current_file'] = file_idx + 1

        # Consolidate results into global cross-file batches for JSONL; keep Parquet no-op
        batch_number = 0
        total_records = 0
        successful_files = 0

        # Sort results by file_idx to maintain order
        results.sort(key=lambda x: x['file_idx'])

        # Store file order and offsets for cross-file batching
        file_entries = []  # list of dicts per file: {file_path, total_lines}
        for result in results:
            if 'error' in result:
                if progress_callback:
                    progress_callback(
                        f"⚠️ Skipping {result['file_path']} due to error: {result['error']}"
                    )
                continue

            file_path = result['file_path']
            self.file_line_index[file_path] = result['line_offsets']
            total_lines = len(result['line_offsets'])
            file_entries.append(
                {
                    'file_idx': result['file_idx'],
                    'file_path': file_path,
                    'total_lines': total_lines,
                }
            )
            total_records += result.get('valid_record_count', total_lines)
            successful_files += 1

        # Build global batches across files for JSONL; for Parquet we already early-returned
        if self.file_format == 'jsonl':
            current_batch_count = 0
            current_segments = []  # list of segments for current batch
            # Track per-file cursor
            file_cursors = {fe['file_path']: 0 for fe in file_entries}

            # Iterate files in order, filling batches to exact batch_size
            for fe in file_entries:
                fp = fe['file_path']
                total_lines_file = fe['total_lines']
                cursor = file_cursors[fp]
                while cursor < total_lines_file:
                    remaining_in_file = total_lines_file - cursor
                    space_in_batch = batch_size - current_batch_count
                    take = min(remaining_in_file, space_in_batch)

                    if take > 0:
                        # Record a segment for this file
                        seg = {
                            'file_idx': fe['file_idx'],
                            'file_path': fp,
                            'line_start': cursor,
                            'line_end': cursor + take,  # exclusive
                        }
                        # For convenience when reading: compute byte_offset for segment start
                        try:
                            seg['byte_offset'] = self.file_line_index[fp][
                                cursor
                            ]
                        except Exception:
                            seg['byte_offset'] = None
                        current_segments.append(seg)
                        cursor += take
                        file_cursors[fp] = cursor
                        current_batch_count += take

                    # If batch is filled, finalize it
                    if current_batch_count == batch_size:
                        self.batch_index[batch_number] = {
                            'segments': current_segments,
                            'record_count': current_batch_count,
                            # keep a representative file_path for backward UI compatibility
                            'file_path': (
                                current_segments[0]['file_path']
                                if current_segments
                                else fp
                            ),
                        }
                        batch_number += 1
                        current_segments = []
                        current_batch_count = 0

            # Flush trailing partial batch (allowed to be partial per requirement)
            if current_batch_count > 0:
                self.batch_index[batch_number] = {
                    'segments': current_segments,
                    'record_count': current_batch_count,
                    'file_path': (
                        current_segments[0]['file_path']
                        if current_segments
                        else (
                            file_entries[0]['file_path'] if file_entries else ''
                        )
                    ),
                }
                batch_number += 1
        else:
            # For other formats, retain previous behavior (each file’s local batches already suffice)
            # However, Parquet is marked indexed without index content earlier; nothing to add here
            pass

        self.is_indexed = True
        self.total_batches = batch_number

        # Calculate performance metrics
        total_time = time.time() - start_time
        avg_time_per_file = (
            total_time / len(self.data_files) if len(self.data_files) > 0 else 0
        )

        if progress_callback:
            if use_parallel and len(self.data_files) > 1:
                speedup_estimate = len(self.data_files) / max_workers
                progress_callback(
                    f"🎉 Parallel indexing complete! "
                    f"Indexed {successful_files} files in {total_time:.1f}s "
                    f"(~{speedup_estimate:.1f}x speedup) | "
                    f"Total: {batch_number} batches, {total_records:,} records"
                )
            else:
                progress_callback(
                    f"✅ Sequential indexing complete! "
                    f"Indexed {successful_files} files in {total_time:.1f}s | "
                    f"Total: {batch_number} batches, {total_records:,} records"
                )

    def build_batch_index_sequential(
        self, batch_size=1000, progress_callback=None
    ):
        """Original sequential indexing method (kept for compatibility)"""
        if self.is_indexed:
            return  # Already indexed

        self.batch_index = {}
        self.file_line_index = {}
        batch_number = 0

        self.indexing_progress['total_files'] = len(self.data_files)

        for file_idx, file_path in enumerate(self.data_files):
            self.indexing_progress['current_file'] = file_idx + 1

            if progress_callback:
                progress_callback(
                    f"Indexing file {file_idx + 1}/{len(self.data_files)}: {os.path.basename(file_path)}"
                )

            try:
                file_info = (file_idx, file_path, batch_size, self.file_format)
                result = process_file_index_worker(file_info)

                if 'error' in result:
                    if progress_callback:
                        progress_callback(
                            f"Warning: Error indexing {file_path}: {result['error']}"
                        )
                    continue

                # Store line offsets for this file
                self.file_line_index[file_path] = result['line_offsets']

                # Add batches to the global index
                for batch_entry in result['file_batches']:
                    batch_entry_global = batch_entry.copy()
                    del batch_entry_global['local_batch_number']
                    self.batch_index[batch_number] = batch_entry_global
                    batch_number += 1
                    self.indexing_progress['current_batch'] = batch_number

            except Exception as e:
                if progress_callback:
                    progress_callback(
                        f"Warning: Error indexing {file_path}: {e}"
                    )
                continue

        self.is_indexed = True
        self.total_batches = batch_number

        if progress_callback:
            progress_callback(
                f"Indexing complete! Created index for {batch_number} batches across {len(self.data_files)} files"
            )

    def get_batch_from_index(self, batch_number, batch_size=1000):
        """Get batch using the pre-built index for O(1) access"""
        if not self.is_indexed:
            raise RuntimeError(
                "Index not built. Call build_batch_index() first."
            )

        if batch_number not in self.batch_index:
            return pd.DataFrame()

        batch_info = self.batch_index[batch_number]
        batch_records = []

        try:
            if self.file_format == 'jsonl':
                # JSONL batches may span multiple files via segments
                segments = batch_info.get('segments')
                if segments:
                    for seg in segments:
                        fp = seg['file_path']
                        line_start = seg['line_start']
                        line_end = seg['line_end']
                        byte_offset = seg.get('byte_offset')

                        with open(fp, 'r', encoding='utf-8') as f:
                            # Seek to byte offset if available; otherwise skip lines
                            if byte_offset is not None:
                                f.seek(byte_offset)
                            else:
                                for _ in range(line_start):
                                    f.readline()

                            for local_idx in range(line_end - line_start):
                                line = f.readline()
                                if not line:
                                    break
                                try:
                                    data = json.loads(line.strip())
                                    processed_record = self._process_record(
                                        data, fp, line_start + local_idx
                                    )
                                    if processed_record is not None:
                                        batch_records.append(processed_record)
                                except json.JSONDecodeError:
                                    continue
                else:
                    # Backward-compat single-file entry
                    file_path = batch_info['file_path']
                    byte_offset = batch_info['byte_offset']
                    line_start = batch_info['line_start']
                    line_end = batch_info['line_end']
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.seek(byte_offset)
                        for line_idx in range(line_end - line_start):
                            line = f.readline()
                            if not line:
                                break
                            try:
                                data = json.loads(line.strip())
                                processed_record = self._process_record(
                                    data, file_path, line_start + line_idx
                                )
                                if processed_record is not None:
                                    batch_records.append(processed_record)
                            except json.JSONDecodeError:
                                continue

            elif self.file_format == 'parquet':
                # For Parquet files, use directory-level DuckDB querying for true cross-file access
                try:
                    # Use the new directory-level global batch reading method
                    records = self.handler.read_batch_from_directory_global(
                        self.data_dir, batch_number, batch_size
                    )

                    # Process each record with consistent file info (use first file for metadata)
                    first_file = (
                        self.data_files[0] if self.data_files else "unknown"
                    )
                    for idx, record in enumerate(records):
                        global_record_idx = batch_number * batch_size + idx
                        processed_record = self._process_record(
                            record, first_file, global_record_idx
                        )

                        if processed_record is not None:
                            batch_records.append(processed_record)

                except Exception as e:
                    st.warning(
                        f"Error reading Parquet batch {batch_number} from directory: {e}"
                    )
                    return pd.DataFrame()

        except Exception as e:
            st.warning(
                f"Error reading batch {batch_number} from {file_path}: {e}"
            )
            return pd.DataFrame()

        return pd.DataFrame(batch_records)

    def get_batch_direct(self, batch_number, batch_size=1000):
        """Get batch directly using format-optimized approach (no indexing required)"""
        if self.file_format == 'parquet':
            # For Parquet, use directory-level DuckDB querying - no indexing needed!
            try:
                records = self.handler.read_batch_from_directory_global(
                    self.data_dir, batch_number, batch_size
                )

                # Process records with consistent metadata
                batch_records = []
                for idx, record in enumerate(records):
                    global_record_idx = batch_number * batch_size + idx
                    # Use data_dir as source since we're querying across multiple files
                    processed_record = self._process_record(
                        record, f"batch_{batch_number}", global_record_idx
                    )

                    if processed_record is not None:
                        batch_records.append(processed_record)

                return pd.DataFrame(batch_records)

            except Exception as e:
                st.warning(
                    f"Error reading Parquet batch {batch_number} from directory: {e}"
                )
                return pd.DataFrame()
        else:
            # For other formats, fall back to indexed or sequential access
            if self.is_indexed:
                return self.get_batch_from_index(batch_number, batch_size)
            else:
                return self._get_batch_sequential(batch_number, batch_size)

    def get_batch_range_direct(self, global_offset, total_records):
        """
        Optimized batch range loading using single DuckDB query instead of loop.
        Significantly improves performance for large batch ranges.

        Args:
            global_offset: Starting record position across all files
            total_records: Total number of records to retrieve

        Returns:
            DataFrame containing all records in the range
        """
        if self.file_format == 'parquet':
            # For Parquet: Single DuckDB query across all files - optimal performance!
            try:
                # Use DuckDB's glob pattern for directory-level querying with precise OFFSET/LIMIT
                parquet_pattern = os.path.join(
                    self.data_dir, self.file_pattern
                ).replace('\\', '/')

                query = f"""
                    SELECT * 
                    FROM '{parquet_pattern}' 
                    LIMIT {total_records} 
                    OFFSET {global_offset}
                """
                result_df = duckdb.query(query).to_df()

                # Process records with consistent metadata
                batch_records = []
                for idx, record in enumerate(result_df.to_dict('records')):
                    global_record_idx = global_offset + idx
                    # Use descriptive source since we're querying across multiple files
                    processed_record = self._process_record(
                        record, f"range_query", global_record_idx
                    )

                    if processed_record is not None:
                        batch_records.append(processed_record)

                return pd.DataFrame(batch_records)

            except Exception as e:
                st.warning(
                    f"Error reading Parquet batch range (offset={global_offset}, limit={total_records}): {e}"
                )
                return pd.DataFrame()
        else:
            # For other formats, fall back to sequential batch loading
            # Calculate which batches we need
            batch_size = self.records_per_batch
            start_batch = global_offset // batch_size
            end_batch = (global_offset + total_records - 1) // batch_size

            batch_dfs = []
            for batch_num in range(start_batch, end_batch + 1):
                batch_df = self.get_batch_direct(batch_num, batch_size)
                if not batch_df.empty:
                    batch_dfs.append(batch_df)

            if batch_dfs:
                combined_df = pd.concat(batch_dfs, ignore_index=True)

                # Trim to exact range requested (since batches might include extra records)
                start_idx = (
                    global_offset % batch_size
                    if start_batch == (global_offset // batch_size)
                    else 0
                )
                end_idx = start_idx + total_records

                return combined_df.iloc[start_idx:end_idx].reset_index(
                    drop=True
                )

            return pd.DataFrame()

    def get_batch(self, batch_number, batch_size=1000, use_cache=True):
        """Get a specific batch by number with caching and optimized access"""
        if use_cache and batch_number in self.batch_cache:
            return self.batch_cache[batch_number]

        # For Parquet files, use direct directory-level access for best performance
        if self.file_format == 'parquet':
            batch_df = self.get_batch_direct(batch_number, batch_size)
        elif self.is_indexed:
            # Use indexed access for other formats if available
            batch_df = self.get_batch_from_index(batch_number, batch_size)
        else:
            # Fallback to sequential access
            batch_df = self._get_batch_sequential(batch_number, batch_size)

        # Cache the batch if caching is enabled
        if use_cache and not batch_df.empty:
            self.batch_cache[batch_number] = batch_df

            # Memory management - remove oldest if cache is full
            if len(self.batch_cache) > self.max_cached_batches:
                oldest_batch = min(self.batch_cache.keys())
                del self.batch_cache[oldest_batch]
                if oldest_batch in self._batch_metadata:
                    del self._batch_metadata[oldest_batch]

        return batch_df

    def _get_batch_sequential(self, batch_number, batch_size=1000):
        """Original sequential batch access method (fallback)"""
        # Generate cross-file full batches until we reach the requested one
        batch_iterator = self.get_batch_iterator(batch_size=batch_size)
        current_batch = 0

        for batch_df, file_idx, line_num, batch_num in batch_iterator:
            if current_batch == batch_number:
                return batch_df
            current_batch += 1

        # If we reach here, batch doesn't exist
        return pd.DataFrame()

    def get_batch_iterator(self, batch_size=1000):
        """Get iterator that yields batches of records"""
        self.records_per_batch = batch_size
        batch_number = 0

        if self.file_format == 'jsonl':
            # Cross-file global batching: fill each batch to batch_size except possibly the last
            batch = []
            last_file_idx = -1
            last_line_num = -1
            for file_idx, file_path in enumerate(self.data_files):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            try:
                                data = json.loads(line.strip())
                            except json.JSONDecodeError:
                                continue
                            processed_record = self._process_record(
                                data, file_path, line_num
                            )
                            if processed_record is not None:
                                batch.append(processed_record)
                                last_file_idx = file_idx
                                last_line_num = line_num + 1
                            if len(batch) >= batch_size:
                                yield pd.DataFrame(
                                    batch
                                ), last_file_idx, last_line_num, batch_number
                                batch = []
                                batch_number += 1
                except Exception as e:
                    st.warning(f"Error reading {file_path}: {e}")
                    continue
            if batch:
                # Yield final partial batch
                yield pd.DataFrame(
                    batch
                ), last_file_idx, last_line_num, batch_number
                batch_number += 1
        else:
            # For other formats (like Parquet), read per-file chunks (global batching handled elsewhere)
            for file_idx, file_path in enumerate(self.data_files):
                try:
                    file_handler = FileFormatRegistry.get_handler(
                        self.file_format, self.file_pattern
                    )
                    total_records = file_handler.count_records_in_file(
                        file_path
                    )
                    start_idx = 0
                    line_num = 0
                    while start_idx < total_records:
                        records = file_handler.read_batch_from_file(
                            file_path, start_idx, batch_size
                        )
                        if not records:
                            break
                        batch_local = []
                        for record in records:
                            processed_record = self._process_record(
                                record, file_path, line_num
                            )
                            if processed_record is not None:
                                batch_local.append(processed_record)
                            line_num += 1
                        if batch_local:
                            yield pd.DataFrame(
                                batch_local
                            ), file_idx, line_num, batch_number
                            batch_number += 1
                        start_idx += batch_size
                except Exception as e:
                    st.warning(f"Error reading {file_path}: {e}")
                    continue

    def get_total_estimated_records(self):
        """Get estimated total number of records across all files"""
        if self.is_indexed:
            # If indexed, we can provide exact count
            return sum(
                batch_info['record_count']
                for batch_info in self.batch_index.values()
            )

        if not hasattr(self, '_estimated_total'):
            # Quick estimation using file handler
            try:
                if self.file_format == 'jsonl':
                    # For JSONL, count lines in first file and estimate
                    first_file = self.data_files[0]
                    with open(first_file, 'r', encoding='utf-8') as f:
                        total_lines_first = sum(1 for _ in f)
                    # Estimate total across all files
                    self._estimated_total = total_lines_first * len(
                        self.data_files
                    )
                elif self.file_format == 'parquet':
                    # For Parquet, use directory-level counting for exact total
                    try:
                        self._estimated_total = (
                            self.handler.count_records_in_directory(
                                self.data_dir
                            )
                        )
                    except Exception:
                        # Fallback to file-by-file estimation if directory query fails
                        first_file = self.data_files[0]
                        first_file_count = self.handler.count_records_in_file(
                            first_file
                        )
                        self._estimated_total = first_file_count * len(
                            self.data_files
                        )
                else:
                    # For other formats, use handler to get exact count from first file and estimate
                    first_file = self.data_files[0]
                    file_handler = FileFormatRegistry.get_handler(
                        self.file_format, self.file_pattern
                    )
                    first_file_count = file_handler.count_records_in_file(
                        first_file
                    )
                    self._estimated_total = first_file_count * len(
                        self.data_files
                    )
            except Exception as e:
                st.warning(f"Error estimating total records: {e}")
                self._estimated_total = 1000  # Fallback estimate

        return self._estimated_total

    def get_total_batches(self, batch_size=None):
        """Get total number of batches available"""
        if self.is_indexed:
            return len(self.batch_index)
        else:
            # Use current batch size from parameter or fallback to original
            current_batch_size = (
                batch_size if batch_size is not None else self.records_per_batch
            )
            # Estimate based on total records and current batch size
            estimated_records = self.get_total_estimated_records()
            return (
                estimated_records + current_batch_size - 1
            ) // current_batch_size

    def get_batch_info(self, batch_number):
        """Get information about a specific batch without loading it"""
        if self.is_indexed and batch_number in self.batch_index:
            batch_info = self.batch_index[batch_number].copy()
            batch_info['file_name'] = os.path.basename(batch_info['file_path'])
            return batch_info
        elif self.file_format == 'parquet':
            # For Parquet files using direct access, provide estimated info
            total_batches = self.get_total_batches()
            if 0 <= batch_number < total_batches:
                return {
                    'batch_number': batch_number,
                    'file_name': 'directory_query',
                    'access_method': 'DuckDB_global_directory_query',
                    'record_count': min(
                        1000,
                        self.get_total_estimated_records()
                        - (batch_number * 1000),
                    ),
                }
        return None

    def get_indexing_progress(self):
        """Get current indexing progress"""
        return self.indexing_progress.copy()

    def is_indexing_complete(self):
        """Check if indexing is complete"""
        return self.is_indexed

    def get_cache_info(self):
        """Get information about current cache state"""
        total_cached_records = sum(len(df) for df in self.batch_cache.values())

        # Calculate actual memory usage of cached DataFrames
        total_memory_usage = 0
        for df in self.batch_cache.values():
            total_memory_usage += df.memory_usage(deep=True).sum()

        return {
            'cached_batches': len(self.batch_cache),
            'cached_records': total_cached_records,
            'memory_usage_bytes': total_memory_usage,
            'memory_usage_mb': total_memory_usage / (1024 * 1024),
        }

    def clear_cache(self):
        """Clear all cached batches"""
        self.batch_cache.clear()
        self._batch_metadata.clear()

        # Clear file line counts if it exists
        if hasattr(self, '_file_line_counts'):
            self._file_line_counts.clear()

        # Handle legacy attributes that might exist from old versions
        if hasattr(self, 'batch_positions'):
            self.batch_positions.clear()
        if hasattr(self, 'file_handles'):
            for handle in self.file_handles.values():
                try:
                    handle.close()
                except:
                    pass
            self.file_handles.clear()

    def clear_index(self):
        """Clear the batch index (useful for memory management or rebuilding)"""
        self.batch_index.clear()
        self.file_line_index.clear()
        self.is_indexed = False
        self.indexing_progress = {
            'current_file': 0,
            'total_files': 0,
            'current_batch': 0,
        }
