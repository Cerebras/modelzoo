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
"""
KenLM Reader Module

This module provides a reader for KenLM pipeline that can read input data from various sources.
"""
from typing import Callable

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers import (
    CSVReader,
    HuggingFaceDatasetReader,
    JsonlReader,
    ParquetReader,
)
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.utils.logging import logger


class TextReader(BaseDiskReader):
    """Read data from text files.
        Will read each line as a separate document.
    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
    """

    name = "Text"
    _requires_dependencies = []

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )

    def read_file(self, filepath: str):
        with self.data_folder.open(filepath, "r", encoding="utf-8") as f:
            li = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = {self.text_key: line, self.id_key: f"{filepath}:{li}"}
                document = self.get_document_from_dict(data, filepath, li)
                if not document:
                    continue

                yield document
                li += 1


class KenLMReader(BaseDiskReader):
    """Custom reader that handles the case where there are fewer files than ranks.
    Will read each line as a separate document.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
        input_format: the format of the input files (jsonl, parquet, csv)
    """

    name = "KenLM Reader"
    _requires_dependencies = []
    input_format: str = "jsonl"

    def __init__(
        self,
        data_folder: DataFolderLike,
        paths_file: DataFileLike | None = None,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
        input_format: str = "jsonl",
    ):
        # Initialize the appropriate reader based on input format
        reader_map = {
            'text': TextReader,
            'jsonl': JsonlReader,
            'parquet': ParquetReader,
            'csv': CSVReader,
            'huggingface': HuggingFaceDatasetReader,
        }
        self.input_format = input_format
        if input_format not in reader_map:
            raise ValueError(f"Unsupported input format: {input_format}")

        # Set glob pattern based on input format if not provided
        if glob_pattern is None:
            glob_pattern = (
                f"**/*.{input_format}" if recursive else f"*.{input_format}"
            )

        self.format_reader = reader_map[input_format](
            data_folder=data_folder,
            paths_file=paths_file,
            limit=limit,
            skip=skip,
            file_progress=file_progress,
            doc_progress=doc_progress,
            adapter=adapter,
            text_key=text_key,
            id_key=id_key,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
        )
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        """Override run method to handle the case where there are fewer files than ranks."""
        if data:
            yield from data

        # Get all files matching the pattern
        files = list(self.data_folder.glob(self.glob_pattern))
        if not files:
            logger.warning(
                f"No files found matching pattern {self.glob_pattern} in {self.data_folder.path}"
            )
            return

        # If we have fewer files than ranks, distribute them evenly
        if len(files) < world_size:
            # Calculate which files this rank should process
            files_per_rank = len(files) / world_size
            start_idx = int(rank * files_per_rank)
            end_idx = int((rank + 1) * files_per_rank)
            files_shard = files[start_idx:end_idx]
        else:
            # Use the original sharding logic
            files_shard = files[rank::world_size]
        files = [f for f in files_shard if f.endswith(self.input_format)]
        if not files:
            logger.warning(f"No files assigned to rank {rank}")
            return

        # Use the format-specific reader to process the files
        for doc in self.format_reader.read_files_shard(files_shard):
            self.update_doc_stats(doc)
            yield doc

    def read_file(self, filepath: str):
        """Delegate to the format-specific reader's read_file method."""
        return self.format_reader.read_file(filepath)
