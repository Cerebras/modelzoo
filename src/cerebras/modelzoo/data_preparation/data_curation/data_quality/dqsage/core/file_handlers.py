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
File format handlers and registry for the DQSage Data Visualizer.
Supports extensible file format handling with built-in support for JSONL and Parquet files.
"""

import glob
import json
import os

import duckdb


class FileFormatHandler:
    """Base class for file format handlers - designed for extensibility"""

    def __init__(self, file_pattern):
        self.file_pattern = file_pattern

    def get_files(self, data_dir):
        """Get list of files matching the pattern"""
        return glob.glob(os.path.join(data_dir, self.file_pattern))

    def read_first_record(self, file_path):
        """Read first record from file for schema analysis"""
        raise NotImplementedError("Subclasses must implement read_first_record")

    def read_batch_from_file(self, file_path, start_idx=0, batch_size=1000):
        """Read a batch of records from file"""
        raise NotImplementedError(
            "Subclasses must implement read_batch_from_file"
        )

    def count_records_in_file(self, file_path):
        """Count total records in file"""
        raise NotImplementedError(
            "Subclasses must implement count_records_in_file"
        )

    def get_file_extension(self):
        """Get file extension for this format"""
        raise NotImplementedError(
            "Subclasses must implement get_file_extension"
        )

    def supports_parallel_indexing(self):
        """Whether this format supports parallel indexing"""
        return True

    def requires_indexing(self):
        """Whether this format requires indexing for optimal performance"""
        return True


class JSONLHandler(FileFormatHandler):
    """Handler for JSONL files"""

    def read_first_record(self, file_path):
        """Read first record from JSONL file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line:
                    return json.loads(first_line)
        except Exception as e:
            raise ValueError(
                f"Error reading first record from {file_path}: {e}"
            )
        return None

    def read_batch_from_file(self, file_path, start_idx=0, batch_size=1000):
        """Read a batch of records from JSONL file"""
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Skip to start_idx
                for _ in range(start_idx):
                    f.readline()

                # Read batch_size records
                for i in range(batch_size):
                    line = f.readline()
                    if not line:
                        break
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise ValueError(f"Error reading batch from {file_path}: {e}")
        return records

    def count_records_in_file(self, file_path):
        """Count total records in JSONL file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:
            raise ValueError(f"Error counting records in {file_path}: {e}")

    def get_file_extension(self):
        return "jsonl"


class ParquetHandler(FileFormatHandler):
    """Handler for Parquet files using DuckDB for optimized performance"""

    def __init__(self, file_pattern):
        super().__init__(file_pattern)
        # DuckDB is already imported and used elsewhere in the app, so we don't need additional dependencies

    def read_first_record(self, file_path):
        """Read first record from Parquet file using DuckDB - extremely fast"""
        try:
            # Use DuckDB to read only the first record
            query = f"SELECT * FROM '{file_path}' LIMIT 1"
            result_df = duckdb.query(query).to_df()

            if len(result_df) > 0:
                # Convert first row to dict
                return result_df.iloc[0].to_dict()
        except Exception as e:
            raise ValueError(
                f"Error reading first record from {file_path}: {e}"
            )
        return None

    def read_batch_from_file(self, file_path, start_idx=0, batch_size=1000):
        """Read a batch of records from Parquet file using DuckDB with LIMIT/OFFSET"""
        try:
            # Use DuckDB for efficient batch reading with true lazy loading
            query = f"""
                SELECT * 
                FROM '{file_path}' 
                LIMIT {batch_size} 
                OFFSET {start_idx}
            """
            result_df = duckdb.query(query).to_df()
            return result_df.to_dict('records')
        except Exception as e:
            raise ValueError(f"Error reading batch from {file_path}: {e}")

    def read_batch_from_directory(self, data_dir, start_idx=0, batch_size=1000):
        """Read a batch across all Parquet files in directory using DuckDB pattern matching"""
        try:
            # Use DuckDB's glob pattern to read from all parquet files
            parquet_pattern = os.path.join(data_dir, self.file_pattern).replace(
                '\\', '/'
            )
            query = f"""
                SELECT * 
                FROM '{parquet_pattern}' 
                LIMIT {batch_size} 
                OFFSET {start_idx}
            """
            result_df = duckdb.query(query).to_df()
            return result_df.to_dict('records')
        except Exception as e:
            raise ValueError(
                f"Error reading batch from directory {data_dir}: {e}"
            )

    def count_records_in_directory(self, data_dir):
        """Count total records across all Parquet files in directory using DuckDB"""
        try:
            # Use DuckDB's glob pattern to count across all parquet files
            parquet_pattern = os.path.join(data_dir, self.file_pattern).replace(
                '\\', '/'
            )
            query = f"SELECT COUNT(*) as count FROM '{parquet_pattern}'"
            result_df = duckdb.query(query).to_df()
            return int(result_df.iloc[0]['count'])
        except Exception as e:
            raise ValueError(
                f"Error counting records in directory {data_dir}: {e}"
            )

    def count_records_in_file(self, file_path):
        """Count total records in Parquet file using DuckDB"""
        try:
            # Use DuckDB to count records efficiently
            query = f"SELECT COUNT(*) as count FROM '{file_path}'"
            result_df = duckdb.query(query).to_df()
            return int(result_df.iloc[0]['count'])
        except Exception as e:
            raise ValueError(f"Error counting records in {file_path}: {e}")

    def get_file_extension(self):
        return "parquet"

    def requires_indexing(self):
        """Parquet files don't need indexing - DuckDB directory queries are faster and more accurate"""
        return False

    def read_batch_from_directory_global(
        self, data_dir, batch_number, batch_size=1000
    ):
        """Read a global batch across all Parquet files using DuckDB with true cross-file indexing"""
        try:
            # Use DuckDB's glob pattern to query across all parquet files with global OFFSET
            parquet_pattern = os.path.join(data_dir, self.file_pattern).replace(
                '\\', '/'
            )
            global_offset = batch_number * batch_size

            query = f"""
                SELECT * 
                FROM '{parquet_pattern}' 
                LIMIT {batch_size} 
                OFFSET {global_offset}
            """
            result_df = duckdb.query(query).to_df()
            return result_df.to_dict('records')
        except Exception as e:
            raise ValueError(
                f"Error reading global batch {batch_number} from directory {data_dir}: {e}"
            )


class FileFormatRegistry:
    """Registry for file format handlers - easily extensible for new formats"""

    _handlers = {}

    @classmethod
    def register_handler(cls, format_name, handler_class, file_patterns):
        """Register a new file format handler"""
        cls._handlers[format_name] = {
            'handler_class': handler_class,
            'file_patterns': file_patterns,
            'display_name': format_name.upper(),
        }

    @classmethod
    def get_available_formats(cls):
        """Get list of available formats"""
        available = []
        for format_name, info in cls._handlers.items():
            # Check if dependencies are available
            try:
                # Try to instantiate with a dummy pattern to check dependencies
                info['handler_class']("*." + info['file_patterns'][0])
                available.append(format_name)
            except ImportError:
                # Format not available due to missing dependencies
                continue
        return available

    @classmethod
    def get_handler(cls, format_name, file_pattern):
        """Get handler instance for a format"""
        if format_name not in cls._handlers:
            raise ValueError(f"Unknown format: {format_name}")

        handler_info = cls._handlers[format_name]
        return handler_info['handler_class'](file_pattern)

    @classmethod
    def get_format_info(cls, format_name):
        """Get format information"""
        return cls._handlers.get(format_name, {})


# Register built-in file format handlers
FileFormatRegistry.register_handler('jsonl', JSONLHandler, ['jsonl'])
# DuckDB-based Parquet support is always available (no additional dependencies needed)
FileFormatRegistry.register_handler('parquet', ParquetHandler, ['parquet'])
