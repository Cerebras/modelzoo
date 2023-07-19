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

import logging
import random
from abc import ABC, abstractmethod

import h5py
import numpy as np
from tqdm import tqdm

from modelzoo.transformers.data_processing.scripts.hdf5_preprocessing.utils import (
    read_checkpoint,
)
from modelzoo.transformers.data_processing.tokenizers.BPETokenizer import (
    BPETokenizer,
)
from modelzoo.transformers.data_processing.tokenizers.HFTokenizer import (
    HFTokenizer,
)
from modelzoo.transformers.data_processing.utils import split_list

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class HDF5BasePreprocessor(ABC):
    """
    This module defines how to process a dataset, tokenize it and write into HDF5 format.

    Args:
        params (Dict): Dictionary contains the parameters that configures
            the processing of the dataset.
    """

    def __init__(self, params):
        self.output_dir = params["setup"].get("output_dir", "./data_dir/")
        params = params["processing"]

        self.tokenizer_type = params.pop("tokenizer_type", "none").lower()
        assert (
            self.tokenizer_type != "none"
        ), "`tokenizer_type` is missing, please provide it using `args.tokenizer_type`."
        if self.tokenizer_type == "gpt2tokenizer":
            vocab_file = params.pop("vocab_file", None)
            encoder_file = params.pop("encoder_file", None)
            assert (
                vocab_file
            ), "`vocab_file` is missing, please provide it using `args.vocab_file`."
            assert (
                encoder_file
            ), "`encoder_file` is missing, please provide it using `args.encoder_file`."
            self.tokenizer = BPETokenizer(vocab_file, encoder_file)
            self.eos_id = [self.tokenizer.get_token_id("<|endoftext|>")]
            self.pad_id = self.tokenizer.get_token_id("<|endoftext|>")
        elif self.tokenizer_type == "neoxtokenizer":
            encoder_file = params.pop("encoder_file", None)
            assert (
                encoder_file
            ), "`encoder_file` is missing, please provide it using `args.encoder_file`."
            self.tokenizer = HFTokenizer(encoder_file)
            self.eos_id = [self.tokenizer.eos]
            self.pad_id = self.tokenizer.pad
        else:
            raise NotImplementedError(
                f"{self.tokenizer_type} is not implemented."
            )
        if not self.eos_id:
            self.eos_id = [params.pop("eos_id", 0)]
        if not self.pad_id:
            self.pad_id = params.pop("pad_id", 0)

        self.max_seq_length = params.pop("max_seq_length", 2048)
        self.short_seq_prob = params.pop("short_seq_prob", 0.0)

        self.ftfy = params.pop("ftfy", False)
        self.ftfy_normalizer = params.pop("ftfy_normalizer", "NFC")
        if self.ftfy_normalizer == "None":
            self.ftfy_normalizer = None
        self.wikitext_detokenize = params.pop("wikitext_detokenize", False)

        self.output_name = params.pop("output_name", "examples")
        self.files_per_record = params.pop("files_per_record", 50000)

        self.write_remainder = params.pop("write_remainder", True)
        self.resume_from_checkpoint = params.pop(
            "resume_from_checkpoint", False
        )
        self.display_pbar = params.pop("display_pbar", True)

        self.seed = params.pop("seed", 0)
        self.write_in_batch = params.pop("write_in_batch", False)

        if params:
            logger.warning(
                "The following processing params are unused: "
                + ", ".join(params.keys())
            )

        self.files_processed = 0
        self.discarded_files = 0

    @abstractmethod
    def file_read_generator(self, file):
        """ Read file and generates content
        Args:
            file (str): path to data file
        Returns:
            docs_read (tuple): a tuple of intermediate results read from files
        """
        raise NotImplementedError

    @abstractmethod
    def preprocessing_generator(self, *doc_read_results):
        """ Takes in content read from files and generates samples
        Args:
            dos_read (tuple): return results of function file_read_generator
        Returns:
            sample (np.array): one or multiple training samples
        """
        raise NotImplementedError

    def generate_sample(self, file):
        for index, doc in enumerate(self.file_read_generator(file)):

            self.files_processed += 1
            if self.files_processed < self.resume_files_processed:
                continue  # enable resuming from checkpoint

            for preprocessed in self.preprocessing_generator(doc):
                yield preprocessed

    def add_token(self, token):
        """ Add token to the tokenizer
        Args:
            token (str): token to be added to the tokenizer
        """
        if self.tokenizer_type == "gpt2tokenizer":
            self.tokenizer.add_token(token)
        elif self.tokenizer_type == "neoxtokenizer":
            self.tokenizer.add_token([token])

    def get_vocab_size(self):
        """ Get tokenizer vocabulary size
        Returns:
            vocab_size (int): text to tokenize
        """
        if self.tokenizer_type == "gpt2tokenizer":
            vocab_size = len(self.tokenizer.encoder)
        elif self.tokenizer_type == "neoxtokenizer":
            vocab_size = self.tokenizer.tokenizer.get_vocab_size()

        return vocab_size

    def seed_runs(self, rank=0):
        """Set seed for run based on user provided seed and rank.

        Args:
            rank (int): Rank to set, based on process number for execution.
                Defaults to 0.

        Returns:
            Object of type random.Random, with seed set.
        """
        rng = random.Random()
        rng.seed(self.seed + rank)
        np.random.seed(self.seed + rank)

        return rng

    def write_hdf5_file(
        self,
        file_path,
        files,
        rng,
        n_examples,
        chunks,
        dtype="i4",
        compression="gzip",
    ):
        """Write data to HDF5 file.

        Args:
            file_path (string): HDF5 file path.
            files (sequence): List of lists containing tokenized data to write.
            rng (random.Random obj): Instance of random object, with states set.
            n_examples (int): Number of examples that will be written in the file.
            chunks (tuple or bool): Chunk shape, or True to enable auto-chunking.
            dtype (string): Data type for the HDF5 dataset.
            compression (string): Compression strategy.
        """
        data_label = "data"
        data_shape = (n_examples, 3, self.max_seq_length)
        data_buffer = files
        if self.write_in_batch:
            # Below will convert list of strings into numpy 'U' type and h5py
            # doesn't allow storing such format
            # https://docs.h5py.org/en/stable/strings.html#what-about-numpy-s-u-type
            _data = np.stack(data_buffer)
            with h5py.File(file_path, mode="w") as h5_file:
                h5_file.attrs["n_examples"] = n_examples
                h5_file.create_dataset(
                    data_label,
                    data=_data,
                    dtype=dtype,
                    chunks=chunks,
                    compression=compression,
                )
        else:
            with h5py.File(file_path, mode="w") as h5_file:
                h5_file.attrs["n_examples"] = n_examples
                dset = h5_file.create_dataset(
                    data_label,
                    shape=data_shape,
                    dtype=dtype,
                    chunks=chunks,
                    compression=compression,
                )
                for idx, f in enumerate(data_buffer):
                    dset[idx] = f

    def write_hdf5_files(
        self,
        files,
        start_number,
        write_remainder=False,
        process_number=None,
        rng=random.Random(),
    ):
        """Writes a list of files to HDF5.

        Args:
            files (sequence): List of lists containing tokenized data to write.
            start_number (int): Continual count of HDF5 files written out.
            write_remainder (bool): Write out remaining data from files, if
                files per record is not met. Defaults to `False`.
            process_number (int): Process number for execution. Defaults to `None`.
            rng (random.Random obj): Instance of random object, with states set.
                Defaults to new instance created for write.

        Returns:
            start_number (int): Continual count of HDF5 files written out.
            remainder (list): Remaining sequences not written out, if length of
                files to write is greater than the file per record.
        """
        if not files:
            return start_number, []

        files_per_record = self.files_per_record
        file_chunks = split_list(files, files_per_record)
        if not file_chunks:
            return start_number, []

        if len(file_chunks[-1]) != files_per_record and not write_remainder:
            remainder = file_chunks.pop(-1)
        else:
            remainder = []
            files_per_record = len(file_chunks[-1])

        hdf5_chunk_size = (1, 3, self.max_seq_length)
        hdf5_dtype = "i4"

        for files in file_chunks:
            fp = f"{self.output_dir}/{self.output_name}_{start_number}"
            if process_number is not None:
                fp += f"_{process_number}"

            self.write_hdf5_file(
                file_path=fp + f".h5",
                files=files,
                rng=rng,
                n_examples=files_per_record,
                chunks=hdf5_chunk_size,
                dtype=hdf5_dtype,
            )

            start_number += 1

        return start_number, remainder

    def create_dataset(self, params):
        """Creates HDF5 dataset from given parameters.

        Args:
            files (list): List of files to process.
            process_no (int): process id

        Returns:
            Dictionary containing results of execution, specifically as number of
                processed, discarded, and successful files as well as number of examples.
        """

        files, process_no = params
        self.rng = self.seed_runs(process_no)

        self.discarded_files = 0
        self.files_processed = 0
        pbar = tqdm(
            desc=f"Parsed 0 input files. Files written ",
            disable=not self.display_pbar,
        )
        checkpoint_path = f"{self.output_dir}/checkpoint_{process_no}.txt"
        self.resume_files_processed, df_count = read_checkpoint(
            checkpoint_path, self.resume_from_checkpoint
        )

        doc_object_array = []

        for _file in files:
            for doc_object in self.generate_sample(_file):
                if doc_object == []:
                    continue

                # add tokenized files > chunk size to main array
                doc_object_array.append(doc_object)

                if len(doc_object_array) >= self.files_per_record:
                    _df_count, remainder = self.write_hdf5_files(
                        doc_object_array,
                        start_number=df_count,
                        process_number=process_no,
                        rng=self.rng,
                    )
                    pbar.update(_df_count - df_count)
                    pbar.set_description(
                        f"Parsed {self.files_processed} input files. Files written "
                    )

                    df_count = _df_count
                    doc_object_array = (
                        remainder  # add remaining files to next chunk
                    )
                    with open(checkpoint_path, "w") as checkpoint_file:
                        checkpoint_file.write(
                            f"{self.files_processed}, {df_count}"
                        )

        remainder = doc_object_array

        n_examples = df_count * self.files_per_record
        if self.write_remainder and len(remainder) > 0:
            n_examples += len(remainder)
            _df_count, _ = self.write_hdf5_files(
                remainder,
                start_number=df_count,
                write_remainder=True,
                process_number=process_no,
                rng=self.rng,
            )
            pbar.update(_df_count - df_count)
            pbar.set_description(
                f"Parsed {self.files_processed} input files. Files written "
            )
            with open(checkpoint_path, "w") as checkpoint_file:
                checkpoint_file.write(f"{self.files_processed}, {_df_count}")

            successful_files = self.files_processed - self.discarded_files
            return {
                "discarded": self.discarded_files,
                "processed": self.files_processed,
                "successful": successful_files,
                "examples": n_examples,
            }
