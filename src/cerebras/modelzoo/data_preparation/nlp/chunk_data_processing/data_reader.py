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

import gzip
import io
import json
import logging
import math
import numbers
import os
import tarfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import h5py
import jsonlines
import numpy as np
import pyarrow.parquet as pq
import zstandard

from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.lm_vsl_data_token_generator import (
    VSLLMDataTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.summarization_data_token_generator import (
    SummarizationTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.summarization_vsl_data_token_generator import (
    VSLSummarizationTokenGenerator,
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
    elif isinstance(data, int):
        # If you wish to support integers as well, represent them in bytes.
        return 1 if data == 0 else math.ceil(data.bit_length() / 8)

    # Handle other data types like floats or custom objects here if needed.

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
    def __init__(self, keys: Optional[Dict] = None):
        """
        Initialize the DataFrame object.

        Args:
            keys (Dict): Keys for the data entries.
        """

        self.data_keys = keys
        self.multi_turn_content_key = self.data_keys.get(
            'multi_turn_content_key', None
        )
        self.raw_data = {key: [] for key in self.data_keys.values()}
        self.tokenized_data = (
            []
        )  ## assuming that we get a single tokenized list for each entry in the df
        self.file_idx = None  ##  stores the file idx from the list of files to which current dataframe belongs
        self.start_doc_idx = None  ## stores the starting doc index of the current df in the current file
        self.end_doc_idx = None  ## stores the ending doc index of the current df in the current file
        self.entry_sizes = {}
        self.size = 0
        self.data_stats = {
            "discarded": 0,
            "processed": 0,
            "successful": 0,
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "num_pad_tokens": 0,
            "non_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
        }

    def save_to_hdf5(
        self,
        h5file: Any,
        write_in_batch: bool,
        dtype: str = "i4",
        compression: str = "gzip",
    ) -> None:
        """
        Save the DataFrame object to an HDF5 file.

        Args:
            h5file: An HDF5 file handle.
            data_frame_num (int): Unique identifier for the data frame.
        """
        data_label = "data"
        _data = np.concatenate(self.tokenized_data, axis=0)
        n_examples, features, max_seq_length = _data.shape

        if write_in_batch:
            h5file.attrs["n_examples"] = n_examples
            h5file.create_dataset(
                data_label,
                data=_data,
                dtype=dtype,
                chunks=(1, features, max_seq_length),
                compression=compression,
            )
        else:
            h5file.attrs["n_examples"] = n_examples
            dset = h5file.create_dataset(
                data_label,
                shape=_data.shape,
                dtype=dtype,
                chunks=(1, features, max_seq_length),
                compression=compression,
            )
            for idx, f in enumerate(_data):
                dset[idx] = f

    def append_to_hdf5(
        self,
        output_dir,
        total_chunks,
        pid,
        chunk_locks,
        dtype="i4",
        compression="gzip",
    ):
        """
        Appends the different examples in a dataFrame object to different HDF5 files.
        This API is called when online shuffling is used

        Args:
            output_dir: Output dir where HDF5 data is supposed to be dumped.
            total_chunks: Total number of estimated output chunks.
            pid: Process id of the writer process.
            chunk_locks: The list of file specific chunk locks used while appending to a output file.

        """
        if len(self.tokenized_data) == 0:
            return 0

        _data = np.concatenate(self.tokenized_data, axis=0)
        n_examples, features, max_seq_length = _data.shape
        shuffled_indices = np.random.choice(
            np.arange(total_chunks), _data.shape[0]
        )
        for idx, sequence in enumerate(_data):
            idx_seq = shuffled_indices[idx]
            ## shuffled files don't have an associated file index and a doc starting index. They only have the chunk index
            output_file_name = os.path.join(
                output_dir,
                f"output_chunk_{idx_seq}.h5",
            )
            if chunk_locks:
                lock = chunk_locks[idx_seq % len(chunk_locks)]
            else:
                ## There is no lock when there is only 1 writer process
                lock = None
            with optional_lock(lock):
                with h5py.File(output_file_name, "a") as h5f:
                    data = sequence.reshape(1, features, max_seq_length)
                    data_label = "data"

                    if data_label not in h5f:
                        h5f.attrs["n_examples"] = 1
                        h5f.create_dataset(
                            data_label,
                            data=data,
                            dtype='i4',
                            chunks=(1, features, max_seq_length),
                            maxshape=(None, features, max_seq_length),
                            compression='gzip',
                        )
                    else:
                        h5f.attrs["n_examples"] += 1
                        h5f[data_label].resize(
                            (h5f[data_label].shape[0] + data.shape[0]), axis=0
                        )
                        h5f[data_label][-data.shape[0] :] = data

        return n_examples

    def add(self, value: Dict[str, Any]) -> None:
        """
        Add an entry to the DataFrame.

        Args:
            value (Union[Dict[str, Any], Any]): Entry to be added.
        """
        if not isinstance(value, dict):
            value = {self.data_keys.values()[0]: value}

        for k, v in value.items():
            if k in self.raw_data:
                size_before = get_data_size(v)
                self.raw_data[k].append(v)
                self.size += size_before

    def clear(self) -> None:
        """
        Clear the raw data after tokenizing.
        """
        for k in self.raw_data:
            self.raw_data[k].clear()
        self.entry_sizes.clear()

    def check_valid_multi_turn_dialogue(self, doc):
        """
        Checks if the document is corrupted in the case of summarization tasks
        """

        if (
            self.data_keys.get("multi_turn_key", None)
            and self.multi_turn_content_key
        ):
            if self.multi_turn_content_key not in doc[0]:
                logger.warning(
                    "multi_turn_content_key not in file, file may be corrupted"
                )
                return False
        return True

    def tokenize(self, dataset_processor: Any) -> None:
        """
        Tokenize the data values.

        Args:
            dataset_processor: Dataset Processor to be used for processing the data.
        """
        if "jsonl_key" in self.data_keys:
            doc_list = self.raw_data[self.data_keys["jsonl_key"]]
        elif (
            "chosen_key" in self.data_keys and "rejected_key" in self.data_keys
        ):
            if len(self.data_keys.values()) == 2:
                none_values = [None] * len(
                    self.raw_data[self.data_keys["rejected_key"]]
                )
                doc_list = list(
                    zip(
                        none_values,
                        self.raw_data[self.data_keys["chosen_key"]],
                        self.raw_data[self.data_keys["rejected_key"]],
                    )
                )
            else:
                doc_list = list(
                    zip(
                        self.raw_data[self.data_keys["prompt_key"]],
                        self.raw_data[self.data_keys["chosen_key"]],
                        self.raw_data[self.data_keys["rejected_key"]],
                    )
                )
        elif "multi_turn_key" in self.data_keys:
            doc_list = self.raw_data[self.data_keys["multi_turn_key"]]
        else:
            doc_list = list(
                zip(
                    self.raw_data[self.data_keys["prompt_key"]],
                    self.raw_data[self.data_keys["completion_key"]],
                )
            )

        if isinstance(dataset_processor, SummarizationTokenGenerator):
            for idx, doc in enumerate(doc_list):
                if not self.check_valid_multi_turn_dialogue(doc):
                    continue

                if "multi_turn_key" in self.data_keys:
                    assert (
                        len(doc) % 2 == 0
                    ), "We assume that every prompt has a response"
                    doc = [x[self.multi_turn_content_key] for x in doc]
                    doc_list[idx] = [
                        (doc[i], doc[i + 1]) for i in range(0, len(doc), 2)
                    ]
                else:
                    doc_list[idx] = [tuple(doc)]

        for doc in doc_list:
            tokenized_doc, data_stats = dataset_processor.encode(doc)
            for key in data_stats:
                self.data_stats[key] += data_stats[key]
            if tokenized_doc == []:
                continue
            else:
                self.tokenized_data.append(tokenized_doc)

        if self.tokenized_data and isinstance(
            dataset_processor,
            (VSLLMDataTokenGenerator, VSLSummarizationTokenGenerator),
        ):
            self.tokenized_data = dataset_processor.append_within_max_length(
                self.tokenized_data
            )
            self.tokenized_data, data_stats = dataset_processor.process_chunks(
                self.tokenized_data
            )
            for key in data_stats:
                self.data_stats[key] += data_stats[key]

        self.clear()

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
        keys: Optional[Dict] = None,
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
        for idx, ob in enumerate(jsonl_reader):
            if (
                idx < start_doc_idx
            ):  ## resume streaming data from the current document starting index
                continue
            if isinstance(ob, str):
                assert not get_meta
                yield {"text": ob, "doc_idx": idx}
                continue

            entry = {}
            for key in self.keys.values():
                if key in ob:
                    text = ob[key]
                    if not text:
                        continue
                    ## Only autojoin paragraphs if the text is a list of strings. If it is a list of integers(token ids)
                    ## as in the case of NLG datasets then don't autojoin
                    if (
                        autojoin_paragraphs
                        and isinstance(text, list)
                        and isinstance(text[0], str)
                    ):
                        text = para_joiner.join(text)
                    ## Special Case: If the data is an integer typecast it to a string
                    if isinstance(text, numbers.Number):
                        text = str(text)
                    entry[key] = text
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
        df = DataFrame(self.keys)
        start_doc_idx = None
        previous_doc_idx = -1

        for entry in data_gen:
            if start_doc_idx is None:
                start_doc_idx = entry["doc_idx"]

            entry_size = sum(get_data_size(val) for val in entry.values())
            # If there's only one key and its size exceeds the chunk size
            if len(entry) == 1 and entry_size > self.max_chunk_size:
                if df.size > 0:
                    set_doc_idx(df, file_idx, start_doc_idx, previous_doc_idx)
                    yield df
                    df = DataFrame(self.keys)
                    start_doc_idx = entry["doc_idx"]

                key = next(iter(entry))
                set_doc_idx(df, file_idx, start_doc_idx, start_doc_idx)
                for chunk in split_entry_by_paragraph_or_sentence(
                    entry[key], entry_size, self.max_chunk_size
                ):
                    new_entry = {key: chunk}
                    df.add(new_entry)
                    yield df
                    df = DataFrame(self.keys)

                start_doc_idx = None
                continue
            elif df.size + entry_size > self.max_chunk_size and df.size != 0:
                set_doc_idx(df, file_idx, start_doc_idx, previous_doc_idx)
                yield df
                start_doc_idx = entry["doc_idx"]
                df = DataFrame(self.keys)
            df.add(entry)
            previous_doc_idx = entry["doc_idx"]

        if df.size > 0:
            df.file_idx = file_idx
            df.start_doc_idx = start_doc_idx
            df.end_doc_idx = previous_doc_idx
            assert (
                df.end_doc_idx >= df.start_doc_idx
            ), "Dataframe's  ending document idx must not be less than it's starting document idx"
            yield df

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
        with open(file, "r") as fh:
            text = fh.read()
            entry = {"text": text, "doc_idx": start_doc_idx}

            df = DataFrame(self.keys)
            set_doc_idx(df, current_file_idx, start_doc_idx, start_doc_idx)
            df.add(entry)

            yield df

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
            data_gen = (
                {
                    "text": json.loads(line.decode("utf-8")).strip(),
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
        with open(file, "r") as fh:
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
                columns = {key: table.column(key) for key in self.keys.values()}

                for i in range(table.num_rows):
                    if global_doc_idx < start_doc_idx:
                        global_doc_idx += 1
                        continue
                    else:
                        entry = {
                            key: str(col[i].as_py())
                            if isinstance(col[i].as_py(), numbers.Number)
                            else col[i].as_py()
                            for key, col in columns.items()
                        }
                        entry["doc_idx"] = global_doc_idx
                        yield entry
                        global_doc_idx += 1

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
            checkpoint_args (tuple): Contains the current file starting index , current document starting index
        Returns:
            Iterator[Any]: Yields processed data chunks.
        """
        file_idx, start_doc_idx = checkpoint_args
        zipped_file_list = list(zip(range(len(self.file_list)), self.file_list))
        file_list = zipped_file_list[file_idx:]
        for idx, f in file_list:
            checkpoint_args = (idx, start_doc_idx)
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
            else:
                logger.warning(
                    f"Skipping {f} as streaming for that filetype is not implemented"
                )
