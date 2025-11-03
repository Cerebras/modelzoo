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
This module contains helper functions and classes to read data from different
formats, process them, and save in HDF5 format. It supports JSONL, GZipped JSON,
Parquet, ZST compressed JSONL, and TAR archives of ZST compressed JSONL files.

Classes:
    DataFrame:
        An object to hold and process data with the ability to serialize itself
        into an HDF5 format.

    Reader:
        Provides a mechanism to read data from multiple file formats, process it,
        and yield in manageable chunks.
"""

import copy
import inspect
import logging
import numbers
import os
import sys
import traceback
from bisect import bisect_left
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from datasets import IterableDataset, IterableDatasetDict, load_dataset

from cerebras.modelzoo.data_preparation.data_preprocessing.custom_dataset_loader.custom_dataset_loader import (
    DatasetLoader,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.vsl_finetuning_token_generator import (
    VSLFinetuningTokenGenerator,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.vsl_pretraining_token_generator import (
    VSLPretrainingTokenGenerator,
)

logger = logging.getLogger("data_reader")
logger.setLevel(logging.INFO)


def get_data_size(data: Any) -> int:
    """
    Compute the size of the given data.

    Args:
        data (Any): Data whose size needs to be determined.

    Returns:
        int: Size of the given data in bytes.
    """

    if data is None:
        return 0
    if isinstance(data, str):
        return len(data.encode("utf-8"))
    elif isinstance(data, bytes):
        return len(data)
    elif isinstance(data, (tuple, list)):
        # Calculate the size by summing up the sizes of individual elements.
        return sum(get_data_size(item) for item in data)
    elif isinstance(data, dict):
        # Calculate the size of the dictionary by summing the sizes of keys and values.
        return sum(
            get_data_size(key) + get_data_size(value)
            for key, value in data.items()
        )
    elif isinstance(data, (int, float, bool, complex)):
        # For basic data types, rely on sys.getsizeof
        return sys.getsizeof(data)

    # If the data type is not supported, you could raise a TypeError.
    raise TypeError("Unsupported data type for size calculation")


@contextmanager
def optional_lock(lock):
    if lock:
        with lock:
            yield
    else:
        yield


def find_last_paragraph_or_sentence_end(buffer: str) -> int:
    """
    Find the last end of a paragraph (denoted by '\n\n') or a sentence in the buffer.

    Args:
        buffer (str): The text buffer.

    Returns:
        int: The position of the last end of the paragraph or sentence.
    """
    # Check for the end of a paragraph first
    para_end_pos = buffer.rfind('\n\n')
    if para_end_pos != -1:
        return para_end_pos + 2

    # If no end of paragraph is found, check for the end of a sentence
    sentence_endings = ['.', '!', '?']
    for i in range(len(buffer) - 1, -1, -1):
        if buffer[i] in sentence_endings:
            return i + 1

    return None


def split_entry_by_paragraph_or_sentence(
    entry: str, entry_size: int, chunk_size: int
) -> Iterator[str]:
    """
    Split a large entry into chunks by sentence or paragraph end.

    Args:
        entry (str): The text entry.
        entry_size (int): Size of the input entry.
        chunk_size (int): The desired chunk size.

    Returns:
        Iterator[str]: Yields chunks of the text.
    """
    while entry_size > chunk_size:
        end_pos = find_last_paragraph_or_sentence_end(entry[:chunk_size])

        if end_pos:
            yield entry[:end_pos]
            entry = entry[end_pos:]
            entry_size -= end_pos
        else:
            # Fallback: chunk at chunk_size if no sentence-ending is found
            yield entry[:chunk_size]
            entry = entry[chunk_size:]
            entry_size -= chunk_size

    if entry:
        yield entry


class DataFrame:
    def __init__(
        self,
        keys: Optional[Dict] = None,
        read_hook_fn=None,
        file_index=0,
        df_index_in_file=0,
        global_df_index=0,
    ):
        """
        Initialize the DataFrame object.

        Args:
            keys (Dict): Keys for the data entries.
        """

        self.data_keys = keys
        self.read_hook_fn = read_hook_fn
        self.raw_data = []  ## Stores a list of dictionaries
        self.stats_checkpoint_list = []
        self.tokenized_data = defaultdict(list)
        self.size = 0
        self.data_stats = defaultdict(int)
        self.file_index = file_index
        self.df_index_in_file = df_index_in_file
        self.global_df_index = global_df_index

    def set_df_indices(self, file_index, df_index_in_file, global_df_index):
        """
        Set dataframe indices from checkpoint arguments

        Args:
            checkpoint_args (dict): Dict containing (file_index, df_index_in_file, global_df_index)
        """
        self.file_index = file_index
        self.df_index_in_file = df_index_in_file
        self.global_df_index = global_df_index

    def add(self, value: Dict[str, Any]) -> None:
        """
        Add an entry to the DataFrame.

        Args:
            value (Union[Dict[str, Any], Any]): Entry to be added.
        """
        if not isinstance(value, dict):
            value = {self.data_keys.values()[0]: value}

        self.raw_data.append(self.read_hook_fn(value))
        self.size += get_data_size(value)

    def clear(self) -> None:
        """
        Clear the raw data after tokenizing.
        """
        self.raw_data.clear()

    def enforce_token_limit(self, max_tokens: int, cum_num_tokens: int):
        """
        Enforce max_tokens by finding the closest index in stats_list_checkpoint
        where num_tokens + cum_num_tokens >= max_tokens using binary search.
        Trim tokenized_data accordingly.

        Args:
            max_tokens (int): Maximum allowed number of tokens.
            cum_num_tokens (int): Cumulative token count before this chunk.
        """
        # Create cumulative num_tokens list
        num_tokens_list = [
            stats["num_tokens"] + cum_num_tokens
            for stats in self.stats_checkpoint_list
        ]

        # Perform binary search for the first index where num_tokens >= max_tokens
        closest_index = bisect_left(num_tokens_list, max_tokens)

        # Update data_stats with the selected checkpoint
        for key in self.stats_checkpoint_list[closest_index]:
            if key in self.data_stats:
                self.data_stats[key] = self.stats_checkpoint_list[
                    closest_index
                ][key]

        # Trim tokenized_data for all keys
        for key in self.tokenized_data:
            self.tokenized_data[key] = self.tokenized_data[key][
                : closest_index + 1
            ]

    def tokenize(self, token_generator: Any) -> None:
        """
        Tokenize the data values.

        Args:
            token_generator: Token generator to be used for processing the data.
        """
        for doc in self.raw_data:
            tokenized_doc, data_stats = token_generator.encode(doc)

            for key in data_stats:
                self.data_stats[key] += data_stats[key]
            if not tokenized_doc:
                continue
            else:
                for key, value in tokenized_doc.items():
                    self.tokenized_data[key].append(value)

            self.stats_checkpoint_list.append(copy.deepcopy(self.data_stats))
        self.clear()

        if self.tokenized_data and isinstance(
            token_generator,
            (VSLPretrainingTokenGenerator, VSLFinetuningTokenGenerator),
        ):
            self.tokenized_data = token_generator.append_within_max_length(
                self.tokenized_data
            )
            self.tokenized_data, stats_checkpoint_list = (
                token_generator.process_chunks(self.tokenized_data)
            )

            data_stats = stats_checkpoint_list[-1]
            for key in data_stats:
                self.data_stats[key] = data_stats[key]

            self.stats_checkpoint_list = self.stats_checkpoint_list[
                : len(stats_checkpoint_list)
            ]
            for i, stats in enumerate(stats_checkpoint_list):
                for key, value in stats.items():
                    self.stats_checkpoint_list[i][key] = stats[key]

    def __repr__(self) -> str:
        """
        String representation of the DataFrame object.

        Returns:
            str: Description of the DataFrame.
        """
        output = [f"DataFrame(size={self.size}):"]
        for values in self.tokenized_data:
            if len(values) >= 6:
                # Get the first and last three values
                start_values = values[:3]
                end_values = values[-3:]
                output.append(
                    f"[{', '.join(map(str, start_values))}, ... , {', '.join(map(str, end_values))}] ({len(values)} entries)"
                )
            elif values:
                # If there are less than 6 values, print them all
                output.append(
                    f"[{', '.join(map(str, values))}] ({len(values)} entries)"
                )
            else:
                output.append(f"[] (0 entries)")

        # Adding statistics to the representation
        output.append("\nStatistics:")
        for stat, value in self.data_stats.items():
            output.append(f"{stat}: {value}")

        return "\n".join(output)


class Reader:
    def __init__(
        self,
        file_list: List[str],
        read_chunk_size: int,
        keys: Dict,
        read_hook_fn: Callable,
        checkpoint_args: Dict,
        num_nodes: int = 1,
        rank: int = 0,
        **kwargs: Optional[Dict],
    ) -> None:
        """
        Initialize the Reader instance.

        Args:
            file_list (List[str]): List of file paths to be read.
            read_chunk_size (int): Maximum read chunk size for accumulated data.
            keys (Dict): Dictionary containing the type of key and it's name.
        """
        self.file_list = file_list
        self.read_chunk_size = read_chunk_size
        self.keys = keys
        self.read_hook_fn = read_hook_fn
        self.prefix_df = DataFrame(self.keys, self.read_hook_fn)
        self.skip_jsonl_decoding_error = (
            kwargs.get("skip_jsonl_decoding_error", False) if kwargs else False
        )
        self.checkpoint_args = checkpoint_args
        self.num_nodes = num_nodes
        self.rank = rank

    def accumulate_and_yield(
        self,
        data_gen: IterableDataset,
    ) -> Iterator[Any]:
        """
        Accumulate data and yield in chunks.

        Args:
            data_gen (Iterator[Dict[str, Any]]): Generator yielding data entries.
        Returns:
            Iterator[Any]: Yields accumulated data chunks.
        """
        df = copy.deepcopy(self.prefix_df)

        file_index = self.checkpoint_args.get("file_index", 0)
        curr_df_global_index = self.checkpoint_args.get("global_df_index", 0)
        df_index_in_file = 0
        df.set_df_indices(file_index, df_index_in_file, curr_df_global_index)

        for entry in data_gen:
            entry_size = sum(get_data_size(val) for val in entry.values())
            # If there's only one key and its size exceeds the chunk size
            if len(entry) == 1 and entry_size > self.read_chunk_size:
                if df.size > 0:
                    yield df
                    self.prefix_df.clear()
                    df_index_in_file += 1
                    curr_df_global_index += 1
                    df = DataFrame(
                        self.keys,
                        self.read_hook_fn,
                        file_index,
                        df_index_in_file,
                        curr_df_global_index,
                    )
                key = next(iter(entry))
                for chunk in split_entry_by_paragraph_or_sentence(
                    entry[key], entry_size, self.read_chunk_size
                ):
                    new_entry = {key: chunk}
                    df.add(new_entry)
                    yield df
                    self.prefix_df.clear()
                    df_index_in_file += 1
                    curr_df_global_index += 1
                    df = DataFrame(
                        self.keys,
                        self.read_hook_fn,
                        file_index,
                        df_index_in_file,
                        curr_df_global_index,
                    )

                continue
            elif df.size + entry_size > self.read_chunk_size and df.size != 0:
                yield df
                self.prefix_df.clear()
                df_index_in_file += 1
                curr_df_global_index += 1
                df = DataFrame(
                    self.keys,
                    self.read_hook_fn,
                    file_index,
                    df_index_in_file,
                    curr_df_global_index,
                )

            df.add(entry)

        if df.size > 0:

            self.prefix_df = copy.deepcopy(df)

    def number_to_string_ds(self, filtered_dataset):
        return filtered_dataset.map(
            lambda x: {
                k: str(x[k]) if isinstance(x[k], numbers.Number) else x[k]
                for k in self.keys.values()
            }
        )

    def parse_jsonl_zst_tar(self, filepath, fmt, num_nodes=1, rank=0):
        import io
        import tarfile

        import jsonlines
        import zstandard
        from datasets import IterableDataset

        # Step 1: Open the .tar and find the .jsonl.zst file automatically
        def stream_jsonl_from_tar_zst(
            tar_path, selected_features, num_nodes, rank
        ):
            with tarfile.open(tar_path, "r") as archive:
                for member in archive:
                    if member.name.endswith(".jsonl.zst"):
                        with archive.extractfile(member) as f:
                            cctx = zstandard.ZstdDecompressor()
                            reader = io.BufferedReader(cctx.stream_reader(f))
                            rdr = jsonlines.Reader(reader)
                            for idx, obj in enumerate(rdr.iter(type=dict)):
                                # Filter only selected keys
                                if idx % num_nodes == rank:
                                    record = {
                                        k: obj.get(k, None)
                                        for k in selected_features
                                    }
                                    yield record

        generator = lambda: stream_jsonl_from_tar_zst(
            filepath, self.keys.values(), num_nodes, rank
        )
        ds = IterableDataset.from_generator(generator)
        return IterableDatasetDict({"train": self.number_to_string_ds(ds)})

    def load_supported_format(self, filepath, fmt):
        ds = load_dataset(fmt, data_files={"train": [filepath]}, streaming=True)

        filtered_dataset = ds.map(
            lambda x: {k: x.get(k, None) for k in self.keys.values()}
        )
        return self.number_to_string_ds(filtered_dataset)

    def load_fasta_dataset(self, path, data_files, selected_features):
        ds = load_dataset(
            path=path,
            data_files=data_files,
            streaming=True,
            trust_remote_code=True,
            selected_features=selected_features,
            num_nodes=self.num_nodes,
            rank=self.rank,
        )
        return self.number_to_string_ds(ds)

    def shard_dataset(self, dataset):
        for i, sample in enumerate(dataset):
            if i % self.num_nodes == self.rank:
                yield sample

    def stream_data(
        self, get_meta: bool = False, output_dir: str = None
    ) -> Iterator[Any]:
        """
        Stream and process data from multiple file formats.

        Args:
            get_meta (bool): Flag to determine if meta data should be extracted.
            output_dir (str, optional): Directory to save error logs. Logs to console if None.
        Returns:
            Iterator[Any]: Yields processed data chunks.
        """
        file_list = self.file_list[self.checkpoint_args["file_index"] :]
        custom_loader_path = os.path.abspath(inspect.getfile(DatasetLoader))

        def build_load_supported_format_args(filepath, fmt):
            return dict(
                filepath=filepath,
                fmt=fmt,
            )

        def build_load_fasta_dataset_args(filepath, fmt):
            return dict(
                path=custom_loader_path,
                data_files={"train": [filepath]},
                selected_features=list(self.keys.values()),
            )

        format_map = {
            ".jsonl": (
                self.load_supported_format,
                "json",
                build_load_supported_format_args,
            ),
            ".jsonl.zst": (
                self.load_supported_format,
                "json",
                build_load_supported_format_args,
            ),
            ".json.gz": (
                self.load_supported_format,
                "json",
                build_load_supported_format_args,
            ),
            ".parquet": (
                self.load_supported_format,
                "parquet",
                build_load_supported_format_args,
            ),
            ".txt": (
                self.load_supported_format,
                "text",
                build_load_supported_format_args,
            ),
            ".jsonl.zst.tar": (
                self.parse_jsonl_zst_tar,
                "dummy_arg",
                build_load_supported_format_args,
            ),
            ".fasta": (
                self.load_fasta_dataset,
                "dummy_arg",
                build_load_fasta_dataset_args,
            ),
        }

        for f in file_list:
            try:
                match_found = False
                for suffix, (func, fmt, kwargs_fn) in format_map.items():
                    if f.endswith(suffix):
                        match_found = True
                        # enforce one-feature rule on text files
                        if suffix == ".txt" or suffix == ".fasta":
                            assert (
                                len(self.keys.values()) == 1
                            ), f"{suffix} inputs require exactly one selected_feature"
                        kwargs = kwargs_fn(f, fmt)
                        ds = func(**kwargs)
                        yield from self.accumulate_and_yield(
                            self.shard_dataset(ds['train'])
                        )
                        break
                if not match_found:
                    logger.warning(
                        f"Unsupported file format for {f}. Skipping this file"
                    )
            except Exception as e:
                logger.error(f"Error reading file {f}: {e}. Skipping this file")
                # Log error details with line number and traceback
                error_details = (
                    f"Error reading file {f} at line {sys.exc_info()[2].tb_lineno}: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                if output_dir:
                    error_log_path = os.path.join(output_dir, "error_log.txt")
                    with open(error_log_path, "a") as error_log:
                        error_log.write(error_details + "\n")
                else:
                    logger.error(error_details)

        # If there is anything pending in the prefix_df, we need to yield that as well
        if self.prefix_df.size > 0:
            yield self.prefix_df
