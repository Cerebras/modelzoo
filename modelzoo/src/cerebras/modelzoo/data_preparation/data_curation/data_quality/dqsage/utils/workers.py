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
Worker functions for parallel processing in the data visualizer.
Contains functions that run in separate processes for indexing and data processing.
"""

import json

import duckdb


def process_file_index_worker(file_info):
    """Worker function for parallel file indexing - runs in separate process"""
    file_idx, file_path, batch_size, file_format = file_info

    try:
        if file_format == 'jsonl':
            # Original JSONL processing logic
            line_offsets = []
            valid_record_count = 0

            with open(file_path, 'rb') as f:
                byte_offset = 0
                while True:
                    line_offsets.append(byte_offset)
                    line = f.readline()
                    if not line:
                        break

                    # Quick validation - check if line can be parsed as JSON
                    try:
                        json.loads(line.decode('utf-8').strip())
                        valid_record_count += 1
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

                    byte_offset = f.tell()

            # Remove the last empty offset (EOF)
            line_offsets = line_offsets[:-1]

            # Create batch entries for this file
            file_batches = []
            line_start = 0
            local_batch_number = 0

            while line_start < len(line_offsets):
                records_in_batch = min(
                    batch_size, len(line_offsets) - line_start
                )

                if records_in_batch > 0:
                    batch_entry = {
                        'file_idx': file_idx,
                        'file_path': file_path,
                        'byte_offset': line_offsets[line_start],
                        'line_start': line_start,
                        'record_count': records_in_batch,
                        'line_end': line_start + records_in_batch,
                        'local_batch_number': local_batch_number,
                    }
                    file_batches.append(batch_entry)
                    local_batch_number += 1

                line_start += batch_size

        elif file_format == 'parquet':
            # Parquet processing logic
            line_offsets = (
                []
            )  # Not applicable for Parquet, but keep for compatibility

            try:
                # Use DuckDB to get record count efficiently
                query = f"SELECT COUNT(*) as count FROM '{file_path}'"
                result_df = duckdb.query(query).to_df()
                valid_record_count = int(result_df.iloc[0]['count'])
            except Exception as e:
                # Fallback if DuckDB query fails
                valid_record_count = 10000  # Estimate

            # Create batch entries for this file
            file_batches = []
            record_start = 0
            local_batch_number = 0

            while record_start < valid_record_count:
                records_in_batch = min(
                    batch_size, valid_record_count - record_start
                )

                if records_in_batch > 0:
                    batch_entry = {
                        'file_idx': file_idx,
                        'file_path': file_path,
                        'byte_offset': record_start,  # Use record offset instead of byte offset
                        'line_start': record_start,
                        'record_count': records_in_batch,
                        'line_end': record_start + records_in_batch,
                        'local_batch_number': local_batch_number,
                    }
                    file_batches.append(batch_entry)
                    local_batch_number += 1

                record_start += batch_size

        else:
            # Unknown format - return error
            return {
                'file_idx': file_idx,
                'file_path': file_path,
                'error': f"Unsupported file format: {file_format}",
                'line_offsets': [],
                'valid_record_count': 0,
                'file_batches': [],
                'total_batches': 0,
                'processing_time': 0,
            }

        return {
            'file_idx': file_idx,
            'file_path': file_path,
            'line_offsets': line_offsets,
            'valid_record_count': valid_record_count,
            'file_batches': file_batches,
            'total_batches': len(file_batches),
            'processing_time': 0,  # Will be set by caller
        }

    except Exception as e:
        return {
            'file_idx': file_idx,
            'file_path': file_path,
            'error': str(e),
            'line_offsets': [],
            'valid_record_count': 0,
            'file_batches': [],
            'total_batches': 0,
            'processing_time': 0,
        }
