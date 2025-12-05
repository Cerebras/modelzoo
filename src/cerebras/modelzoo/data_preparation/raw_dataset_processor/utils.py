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

import gzip
import io
import json
import logging
import numbers
import tarfile
import types
from typing import Any, Callable, Dict, Iterator, List, Optional

import jsonlines
import pyarrow.parquet as pq
import zstandard

logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)


class Reader:
    def __init__(
        self,
        file_list: List[str],
        keys: Optional[Dict],
        read_hook_fn: Callable,
    ) -> None:
        """
        Initialize the Reader instance.

        Args:
            file_list (List[str]): List of file paths to be read.
            keys (Optional[Dict]): Dictionary containing the type of key and it's name.
        """
        self.file_list = file_list
        self.keys = keys
        self.read_hook_fn = read_hook_fn

    def handle_jsonl(
        self,
        jsonl_reader: Any,
        get_meta: bool,
        autojoin_paragraphs: bool,
        para_joiner: str,
    ) -> Iterator[Dict[str, Any]]:
        """
        Handle JSONL data and yield processed entries.

        Args:
            jsonl_reader (Any): The JSONL reader object.
            get_meta (bool): Flag to determine if meta data should be extracted.
            autojoin_paragraphs (bool): Flag to auto join paragraphs.
            para_joiner (str): Paragraph joiner string.

        Returns:
            Iterator[Dict[str, Any]]: Yields processed data entries.
        """
        for idx, ob in enumerate(
            jsonl_reader.iter(type=dict, skip_invalid=True)
        ):
            if isinstance(ob, str):
                assert not get_meta
                yield {"text": ob, "doc_idx": idx}
                continue

            entry = {}
            for key, value in self.keys.items():
                if value in ob:
                    text = ob[value]
                    if not text:
                        entry[value] = None
                        continue
                    # ## Special Case: If the data is an integer typecast it to a string
                    if isinstance(text, numbers.Number):
                        text = str(text)
                    entry[value] = text
                else:
                    entry[value] = None
            if get_meta and "meta" in ob:
                entry["meta"] = ob["meta"]
            entry["doc_idx"] = idx

            yield entry

    def read_txt(self, file: str) -> Iterator[Any]:
        """
        Read and process text file.

        Args:
            file (str): Path to the .txt file.
        Returns:
            Iterator[Any]: Yields processed data lines.

        """
        with open(file, "r") as fh:
            text = fh.read()
            entry = {self.keys["text_key"]: text}

            yield entry

    def read_jsongz(
        self,
        file: str,
    ) -> Iterator[Any]:
        """
        Read and process gzipped JSON file.

        Args:
            file (str): Path to the .json.gz file.
        Returns:
            Iterator[Any]: Yields processed data entries.
        """
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
            )
            yield from data_gen

    def read_jsonl(
        self,
        file: str,
        get_meta: bool = False,
        autojoin_paragraphs: bool = True,
        para_joiner: str = "\n\n",
    ) -> Iterator[Any]:
        """
        Read and process JSONL file.

        Args:
            file (str): Path to the .jsonl file.
            get_meta (bool): Flag to determine if meta data should be extracted.
            autojoin_paragraphs (bool): Flag to auto join paragraphs.
            para_joiner (str): Paragraph joiner string.

        Returns:
            Iterator[Any]: Yields processed data entries.
        """
        with open(file, "r") as fh:
            rdr = jsonlines.Reader(fh)
            data_gen = self.handle_jsonl(
                rdr, get_meta, autojoin_paragraphs, para_joiner
            )
            assert isinstance(data_gen, types.GeneratorType) == True
            for data in data_gen:
                yield data

    def read_jsonl_zst(
        self,
        file: str,
        get_meta: bool = False,
        autojoin_paragraphs: bool = True,
        para_joiner: str = "\n\n",
    ) -> Iterator[Any]:
        """
        Read and process ZST compressed JSONL file.

        Args:
            file (str): Path to the .jsonl.zst file.
            get_meta (bool): Flag to determine if meta data should be extracted.
            autojoin_paragraphs (bool): Flag to auto join paragraphs.
            para_joiner (str): Paragraph joiner string.

        Returns:
            Iterator[Any]: Yields processed data entries.
        """
        with open(file, "rb") as fh:
            cctx = zstandard.ZstdDecompressor()
            reader = io.BufferedReader(cctx.stream_reader(fh))
            rdr = jsonlines.Reader(reader)
            data_gen = self.handle_jsonl(
                rdr, get_meta, autojoin_paragraphs, para_joiner
            )
            assert isinstance(data_gen, types.GeneratorType) == True
            for data in data_gen:
                yield data

    def read_jsonl_tar(
        self,
        file: str,
        get_meta: bool = False,
        autojoin_paragraphs: bool = True,
        para_joiner: str = "\n\n",
    ) -> Iterator[Any]:
        """
        Read and process TAR archive containing ZST compressed JSONL files.

        Args:
            file (str): Path to the .jsonl.zst.tar file.
            get_meta (bool): Flag to determine if meta data should be extracted.
            autojoin_paragraphs (bool): Flag to auto join paragraphs.
            para_joiner (str): Paragraph joiner string.

        Returns:
            Iterator[Any]: Yields processed data entries.
        """
        with tarfile.open(file, "r") as archive:
            for member in archive:
                with archive.extractfile(member) as f:
                    cctx = zstandard.ZstdDecompressor()
                    reader = io.BufferedReader(cctx.stream_reader(f))
                    rdr = jsonlines.Reader(reader)
                    data_gen = self.handle_jsonl(
                        rdr,
                        get_meta,
                        autojoin_paragraphs,
                        para_joiner,
                    )
                    assert isinstance(data_gen, types.GeneratorType) == True
                    for data in data_gen:
                        yield data

    def read_parquet(self, file: str) -> Iterator[Any]:
        """
        Read and process Parquet file.

        Args:
            file (str): Path to the .parquet file.
        Returns:
            Iterator[Any]: Yields processed data rows.
        """
        parquet_file = pq.ParquetFile(file)

        def entry_gen() -> Iterator[Dict[str, Any]]:
            for row_group_index in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_index)
                columns = {
                    value: table.column(value)
                    for key, value in self.keys.items()
                    if value != None
                }
                for i in range(table.num_rows):
                    entry = {
                        key: (
                            str(col[i].as_py())
                            if isinstance(col[i].as_py(), numbers.Number)
                            else col[i].as_py()
                        )
                        for key, col in columns.items()
                    }
                    yield entry

        yield from entry_gen()

    def read_fasta(self, file: str) -> Iterator[Dict[str, Any]]:
        """
        Read and process Fasta file without using BioPython.
        Args:
            file (str): Path to the .fasta file.
        Returns:
            Iterator[Dict[str, Any]]: Yields processed data rows.
        """

        def entry_gen():
            with open(file, 'r') as fasta_file:
                record_id = None
                sequence_lines = []
                for line in fasta_file:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    if line.startswith(">"):
                        if record_id is not None:
                            # Yield the previous record
                            yield {
                                "text": ''.join(sequence_lines),
                            }
                        record_id = line[
                            1:
                        ]  # Remove the ">" symbol and store the record ID
                        sequence_lines = (
                            []
                        )  # Reset the sequence for a new record
                    else:
                        sequence_lines.append(line)
                # Don't forget to yield the last record in the file
                if record_id is not None:
                    yield {"text": ''.join(sequence_lines)}

        yield from entry_gen()

    def stream_data(self, get_meta: bool = False) -> Iterator[Any]:
        """
        Stream and process data from multiple file formats.

        Args:
            get_meta (bool): Flag to determine if meta data should be extracted.
        Returns:
            Iterator[Any]: Yields processed data chunks.
        """
        zipped_file_list = list(zip(range(len(self.file_list)), self.file_list))
        for idx, f in zipped_file_list:
            try:
                if f.endswith(".jsonl"):
                    yield from self.read_jsonl(f, get_meta)
                elif f.endswith(".jsonl.zst"):
                    yield from self.read_jsonl_zst(f, get_meta)
                elif f.endswith(".jsonl.zst.tar"):
                    yield from self.read_jsonl_tar(f, get_meta)
                elif f.endswith(".txt"):
                    assert not get_meta
                    yield from self.read_txt(f)
                elif f.endswith(".json.gz"):
                    assert not get_meta
                    yield from self.read_jsongz(f)
                elif f.endswith(".parquet"):
                    assert not get_meta
                    yield from self.read_parquet(f)
                elif f.endswith(".fasta"):
                    assert not get_meta
                    yield from self.read_fasta(f)
                else:
                    logger.warning(
                        f"Skipping {f} as streaming for that filetype is not implemented"
                    )
            except Exception as e:
                logger.error(f"Error reading file {f}: {e}. Skipping this file")
