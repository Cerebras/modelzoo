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
import gzip
import io
import json
import logging
import numbers
import sys
import tarfile
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

import jsonlines
import pyarrow.parquet as pq
import zstandard

from cerebras.modelzoo.data_preparation.data_preprocessing.vsl_finetuning_token_generator import (
    VSLFinetuningTokenGenerator,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.vsl_pretraining_token_generator import (
    VSLPretrainingTokenGenerator,
)

logger = logging.getLogger("data_reader")
logger.setLevel(logging.INFO)


def set_doc_idx(df, file_idx, start_doc_idx, end_doc_idx) -> None:
    """
    This is used to set metadata for a given dataframe

    Args:
        file_idx: The file index of the current dataframe
        start_doc_idx: The starting doc index of the current dataframe
        end_doc_idx: The ending doc index of the current dataframe

    """
    df.file_idx = file_idx
    df.start_doc_idx = start_doc_idx
    df.end_doc_idx = end_doc_idx
    assert (
        df.end_doc_idx >= df.start_doc_idx
    ), "Dataframe's  ending document idx must not be less than it's starting document idx"


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
    def __init__(self, keys: Optional[Dict] = None, read_hook_fn=None):
        """
        Initialize the DataFrame object.

        Args:
            keys (Dict): Keys for the data entries.
        """

        self.data_keys = keys
        self.read_hook_fn = read_hook_fn
        self.raw_data = []  ## Stores a list of dictionaries
        self.tokenized_data = defaultdict(list)
        self.file_idx = None  ##  stores the file idx from the list of files to which current dataframe belongs
        self.start_doc_idx = None  ## stores the starting doc index of the current df in the current file
        self.end_doc_idx = None  ## stores the ending doc index of the current df in the current file
        self.size = 0
        self.data_stats = defaultdict(int)

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
        self.clear()

        if self.tokenized_data and isinstance(
            token_generator,
            (VSLPretrainingTokenGenerator, VSLFinetuningTokenGenerator),
        ):
            self.tokenized_data = token_generator.append_within_max_length(
                self.tokenized_data
            )
            self.tokenized_data, data_stats = token_generator.process_chunks(
                self.tokenized_data
            )
            for key in data_stats:
                self.data_stats[key] += data_stats[key]

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
        max_chunk_size: int,
        keys: Optional[Dict],
        read_hook_fn: Callable,
        **kwargs: Optional[Dict],
    ) -> None:
        """
        Initialize the Reader instance.

        Args:
            file_list (List[str]): List of file paths to be read.
            max_chunk_size (int): Maximum chunk size for accumulated data.
            keys (Optional[Dict]): Dictionary containing the type of key and it's name.
        """
        self.file_list = file_list
        self.max_chunk_size = max_chunk_size
        self.keys = keys
        self.read_hook_fn = read_hook_fn
        self.prefix_df = DataFrame(self.keys, self.read_hook_fn)
        self.skip_jsonl_decoding_error = (
            kwargs.get("skip_jsonl_decoding_error", False) if kwargs else False
        )

    def handle_jsonl(
        self,
        jsonl_reader: Any,
        start_doc_idx: int,
        get_meta: bool,
        autojoin_paragraphs: bool,
        para_joiner: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Handle JSONL data and yield processed entries.

        Args:
            jsonl_reader (Any): The JSONL reader object.
            start_doc_idx (int): Contains the current document starting index
            get_meta (bool): Flag to determine if meta data should be extracted.
            autojoin_paragraphs (bool): Flag to auto join paragraphs.
            para_joiner (str): Paragraph joiner string.

        Returns:
            Iterator[Dict[str, Any]]: Yields processed data entries.
        """
        for idx, ob in enumerate(
            jsonl_reader.iter(
                type=dict, skip_invalid=self.skip_jsonl_decoding_error
            )
        ):
            if (
                idx < start_doc_idx
            ):  ## resume streaming data from the current document starting index
                continue
            if isinstance(ob, str):
                assert not get_meta
                yield {"text": ob, "doc_idx": idx}
                continue

            entry = {}
            # Check if all required keys are missing from ob
            if all(value not in ob for value in self.keys.values()):
                raise ValueError(
                    f"Fields {list(self.keys.values())} do not exist in the input entry"
                )

            for key, value in self.keys.items():
                entry[value] = (
                    str(ob[value])
                    if isinstance(ob.get(value), numbers.Number)
                    else ob.get(value)
                )

            if get_meta and "meta" in ob:
                entry["meta"] = ob["meta"]
            entry["doc_idx"] = idx

            yield entry

    def accumulate_and_yield(
        self, data_gen: Iterator[Dict[str, Any]], file_idx
    ) -> Iterator[Any]:
        """
        Accumulate data and yield in chunks.

        Args:
            data_gen (Iterator[Dict[str, Any]]): Generator yielding data entries.
            file_idx (int): Current file index
        Returns:
            Iterator[Any]: Yields accumulated data chunks.
        """
        df = copy.deepcopy(self.prefix_df)
        start_doc_idx = None
        previous_doc_idx = None

        for entry in data_gen:
            doc_idx = entry.pop("doc_idx", 0)
            entry.pop("meta", None)
            if start_doc_idx is None:
                start_doc_idx = doc_idx
                previous_doc_idx = doc_idx

            entry_size = sum(get_data_size(val) for val in entry.values())
            # If there's only one key and its size exceeds the chunk size
            if len(entry) == 1 and entry_size > self.max_chunk_size:
                if df.size > 0:
                    set_doc_idx(df, file_idx, start_doc_idx, previous_doc_idx)
                    yield df
                    self.prefix_df.clear()
                    df = DataFrame(self.keys, self.read_hook_fn)
                    start_doc_idx = doc_idx

                key = next(iter(entry))
                for chunk in split_entry_by_paragraph_or_sentence(
                    entry[key], entry_size, self.max_chunk_size
                ):
                    new_entry = {key: chunk}
                    df.add(new_entry)
                    set_doc_idx(df, file_idx, start_doc_idx, start_doc_idx)
                    yield df
                    self.prefix_df.clear()
                    df = DataFrame(self.keys, self.read_hook_fn)

                start_doc_idx = None
                continue
            elif df.size + entry_size > self.max_chunk_size and df.size != 0:
                set_doc_idx(
                    df,
                    file_idx,
                    start_doc_idx,
                    (
                        previous_doc_idx
                        if previous_doc_idx != None
                        else start_doc_idx
                    ),
                )
                yield df
                self.prefix_df.clear()
                start_doc_idx = doc_idx
                df = DataFrame(self.keys, self.read_hook_fn)
            df.add(entry)
            previous_doc_idx = doc_idx

        if df.size > 0:

            set_doc_idx(df, file_idx, start_doc_idx, previous_doc_idx)
            self.prefix_df = copy.deepcopy(df)

    def read_txt(self, file: str, checkpoint_args: tuple) -> Iterator[Any]:
        """
        Read and process text file.

        Args:
            file (str): Path to the .txt file.
            checkpoint_args (tuple): Contains the current file starting index , current document starting index
        Returns:
            Iterator[Any]: Yields processed data lines.

        """

        current_file_idx, start_doc_idx = checkpoint_args

        def entry_gen(start_doc_idx):
            with open(file, "r") as fh:
                text = fh.read()
                entry = {self.keys["text_key"]: text}
                entry["doc_idx"] = start_doc_idx
                yield entry

        yield from self.accumulate_and_yield(
            entry_gen(start_doc_idx), current_file_idx
        )

    def read_jsongz(
        self,
        file: str,
        checkpoint_args: tuple,
    ) -> Iterator[Any]:
        """
        Read and process gzipped JSON file.

        Args:
            file (str): Path to the .json.gz file.
            checkpoint_args (tuple): Contains the current file starting index , current document starting index
        Returns:
            Iterator[Any]: Yields processed data entries.
        """
        current_file_idx, start_doc_idx = checkpoint_args
        with gzip.open(file, "rb") as f:
            text_key = self.keys["text_key"]
            data_gen = (
                {
                    text_key: json.loads(line.decode("utf-8").strip())[
                        text_key
                    ],
                    "doc_idx": idx,
                }
                for idx, line in enumerate(f)
                if idx >= start_doc_idx
            )
            yield from self.accumulate_and_yield(data_gen, current_file_idx)

    def read_jsonl(
        self,
        file: str,
        checkpoint_args: tuple,
        get_meta: bool = False,
        autojoin_paragraphs: bool = True,
        para_joiner: str = "\n\n",
    ) -> Iterator[Any]:
        """
        Read and process JSONL file.

        Args:
            file (str): Path to the .jsonl file.
            checkpoint_args (tuple): Contains the current file starting index , current document starting index
            get_meta (bool): Flag to determine if meta data should be extracted.
            autojoin_paragraphs (bool): Flag to auto join paragraphs.
            para_joiner (str): Paragraph joiner string.

        Returns:
            Iterator[Any]: Yields processed data entries.
        """
        current_file_idx, start_doc_idx = checkpoint_args
        with open(file, "r", errors='ignore') as fh:
            rdr = jsonlines.Reader(fh)
            data_gen = self.handle_jsonl(
                rdr, start_doc_idx, get_meta, autojoin_paragraphs, para_joiner
            )
            yield from self.accumulate_and_yield(data_gen, current_file_idx)

    def read_jsonl_zst(
        self,
        file: str,
        checkpoint_args: tuple,
        get_meta: bool = False,
        autojoin_paragraphs: bool = True,
        para_joiner: str = "\n\n",
    ) -> Iterator[Any]:
        """
        Read and process ZST compressed JSONL file.

        Args:
            file (str): Path to the .jsonl.zst file.
            checkpoint_args (tuple): Contains the current file starting index , current document starting index
            get_meta (bool): Flag to determine if meta data should be extracted.
            autojoin_paragraphs (bool): Flag to auto join paragraphs.
            para_joiner (str): Paragraph joiner string.

        Returns:
            Iterator[Any]: Yields processed data entries.
        """
        current_file_idx, start_doc_idx = checkpoint_args

        with open(file, "rb") as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            rdr = jsonlines.Reader(reader)
            data_gen = self.handle_jsonl(
                rdr, start_doc_idx, get_meta, autojoin_paragraphs, para_joiner
            )
            yield from self.accumulate_and_yield(data_gen, current_file_idx)

    def read_jsonl_tar(
        self,
        file: str,
        checkpoint_args: tuple,
        get_meta: bool = False,
        autojoin_paragraphs: bool = True,
        para_joiner: str = "\n\n",
    ) -> Iterator[Any]:
        """
        Read and process TAR archive containing ZST compressed JSONL files.

        Args:
            file (str): Path to the .jsonl.zst.tar file.
            checkpoint_args (tuple): Contains the current file starting index , current document starting index
            get_meta (bool): Flag to determine if meta data should be extracted.
            autojoin_paragraphs (bool): Flag to auto join paragraphs.
            para_joiner (str): Paragraph joiner string.

        Returns:
            Iterator[Any]: Yields processed data entries.
        """
        current_file_idx, start_doc_idx = checkpoint_args
        with tarfile.open(file, "r") as archive:
            for member in archive:
                with archive.extractfile(member) as f:
                    cctx = zstandard.ZstdDecompressor()
                    reader = io.BufferedReader(cctx.stream_reader(f))
                    rdr = jsonlines.Reader(reader)
                    data_gen = self.handle_jsonl(
                        rdr,
                        start_doc_idx,
                        get_meta,
                        autojoin_paragraphs,
                        para_joiner,
                    )
                    yield from self.accumulate_and_yield(
                        data_gen, current_file_idx
                    )

    def read_parquet(self, file: str, checkpoint_args: tuple) -> Iterator[Any]:
        """
        Read and process Parquet file.

        Args:
            file (str): Path to the .parquet file.
            checkpoint_args (tuple): Contains the current file starting index , current document starting index
        Returns:
            Iterator[Any]: Yields processed data rows.
        """
        current_file_idx, start_doc_idx = checkpoint_args
        parquet_file = pq.ParquetFile(file)

        def entry_gen(start_doc_idx) -> Iterator[Dict[str, Any]]:
            global_doc_idx = 0
            for row_group_index in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_index)
                columns = {
                    value: table.column(value)
                    for key, value in self.keys.items()
                    if value != None
                }

                for i in range(table.num_rows):
                    if global_doc_idx < start_doc_idx:
                        global_doc_idx += 1
                        continue
                    else:
                        entry = {
                            key: (
                                str(col[i].as_py())
                                if isinstance(col[i].as_py(), numbers.Number)
                                else col[i].as_py()
                            )
                            for key, col in columns.items()
                        }
                        entry["doc_idx"] = global_doc_idx
                        yield entry
                        global_doc_idx += 1

        yield from self.accumulate_and_yield(
            entry_gen(start_doc_idx), current_file_idx
        )

    def read_fasta(
        self, file: str, checkpoint_args: tuple
    ) -> Iterator[Dict[str, Any]]:
        """
        Read and process Fasta file without using BioPython.
        Args:
            file (str): Path to the .fasta file.
            checkpoint_args (tuple): Contains the current file starting index, current document starting index
        Returns:
            Iterator[Dict[str, Any]]: Yields processed data rows.
        """
        current_file_idx, start_doc_idx = checkpoint_args

        def entry_gen(start_doc_idx):
            with open(file, 'r') as fasta_file:
                record_id = None
                sequence_lines = []
                idx = -1  # Initialize sequence index
                for line in fasta_file:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    if line.startswith(">"):
                        if record_id is not None and idx >= start_doc_idx:
                            # Yield the previous record
                            yield {
                                "text": ''.join(sequence_lines),
                                "doc_idx": idx,
                            }
                        record_id = line[
                            1:
                        ]  # Remove the ">" symbol and store the record ID
                        sequence_lines = (
                            []
                        )  # Reset the sequence for a new record
                        idx += 1  # Increment sequence index when a new record is found
                    else:
                        sequence_lines.append(line)
                # Don't forget to yield the last record in the file
                if record_id is not None and idx >= start_doc_idx:
                    yield {"text": ''.join(sequence_lines), "doc_idx": idx}

        yield from self.accumulate_and_yield(
            entry_gen(start_doc_idx), current_file_idx
        )

    def stream_data(
        self, checkpoint_args, get_meta: bool = False
    ) -> Iterator[Any]:
        """
        Stream and process data from multiple file formats.

        Args:
            get_meta (bool): Flag to determine if meta data should be extracted.
            checkpoint_args (tuple): Contains the current file starting index, current document starting index.
        Returns:
            Iterator[Any]: Yields processed data chunks.
        """
        file_idx, start_doc_idx = checkpoint_args
        zipped_file_list = list(zip(range(len(self.file_list)), self.file_list))
        file_list = zipped_file_list[file_idx:]

        for idx, f in file_list:
            checkpoint_args = (idx, start_doc_idx)

            try:
                if f.endswith(".jsonl"):
                    yield from self.read_jsonl(f, checkpoint_args, get_meta)
                elif f.endswith(".jsonl.zst"):
                    yield from self.read_jsonl_zst(f, checkpoint_args, get_meta)
                elif f.endswith(".jsonl.zst.tar"):
                    yield from self.read_jsonl_tar(f, checkpoint_args, get_meta)
                elif f.endswith(".txt"):
                    assert not get_meta
                    yield from self.read_txt(f, checkpoint_args)
                elif f.endswith(".json.gz"):
                    assert not get_meta
                    yield from self.read_jsongz(f, checkpoint_args)
                elif f.endswith(".parquet"):
                    assert not get_meta
                    yield from self.read_parquet(f, checkpoint_args)
                elif f.endswith(".fasta"):
                    assert not get_meta
                    yield from self.read_fasta(f, checkpoint_args)
                else:
                    logger.warning(
                        f"Skipping {f} as streaming for that filetype is not implemented"
                    )
            except Exception as e:
                logger.error(f"Error reading file {f}: {e}. Skipping this file")

        # If there is anything pending in the prefix_df, we need to yield that as well
        if self.prefix_df.size > 0:
            yield self.prefix_df
