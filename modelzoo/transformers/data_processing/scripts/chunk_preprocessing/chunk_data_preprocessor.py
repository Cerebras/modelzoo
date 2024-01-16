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
This module implements a generic data preprocessor called `ChunkDataPreprocessor`.
It internally uses `DataFrame` and `DataReader` to read and process data.
"""

import math
import multiprocessing
import os
import sys
import time
from multiprocessing import Event, Process, Queue, Value, cpu_count
from multiprocessing.synchronize import Event
from threading import Event, Thread
from typing import Any, Dict, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from modelzoo.common.input.utils import check_and_create_output_dirs
from modelzoo.transformers.data_processing.scripts.chunk_preprocessing.data_reader import (
    DataFrame,
    Reader,
)
from modelzoo.transformers.data_processing.scripts.chunk_preprocessing.lm_data_token_generator import (
    LMDataTokenGenerator,
)
from modelzoo.transformers.data_processing.scripts.hdf5_preprocessing.utils import (
    dump_args,
    get_files,
)
from modelzoo.transformers.data_processing.tokenizers.BPETokenizer import (
    BPETokenizer,
)
from modelzoo.transformers.data_processing.tokenizers.HFTokenizer import (
    HFTokenizer,
)


def get_compression_factor(filename: str) -> int:
    """
    Calculate and return the compression factor based on a file's extension.

    Args:
        filename (str): The name of the file.

    Returns:
        int: Compression factor. Returns 3 for all compressed and parquet formats,
             otherwise returns 1 for uncompressed formats.
    """
    compressed_formats = [
        ".jsonl.zst",
        ".jsonl.zst.tar",
        ".json.gz",
        ".parquet",
    ]

    for format in compressed_formats:
        if filename.endswith(format):
            return 3  # compression factor for compressed/parquet formats

    return 1  # default factor for uncompressed formats


def update_progress(
    pbar: tqdm,
    progress_counter: Value,
    total_chunks: int,
    start_time: float,
    stop_event: Event,
) -> None:
    """
    Update the progress bar based on the current progress.

    Args:
        pbar (tqdm): The progress bar instance.
        progress_counter (Value): A shared counter to track progress across processes.
        total_chunks (int): Total chunks to process.
        start_time (float): The start time of the process.
        stop_event (Event): Event to signal when to stop updating progress.

    Returns:
        None
    """
    while not stop_event.is_set():
        if progress_counter.value > pbar.n:
            num_processed = progress_counter.value - pbar.n
            pbar.update(num_processed)

            elapsed_time = time.time() - start_time
            avg_time_per_chunk = elapsed_time / pbar.n
            estimated_remaining = avg_time_per_chunk * (total_chunks - pbar.n)

            # Update progress bar description with processed/total chunks
            pbar.set_description(f"Processing {pbar.n}/{total_chunks} chunks")

            # Update the progress bar postfix with avg processing time and estimated time
            pbar.set_postfix(
                avg_time=f"{avg_time_per_chunk:.4f}s/chunk",
                est_remaining=f"{estimated_remaining:.2f}s",
                refresh=True,
            )

        time.sleep(0.5)


class ChunkDataPreprocessor:
    def __init__(self, params, logger):
        """
        Initialize the class with given parameters and logger.

        Args:
            params (dict): Configuration parameters.
            logger (Logger): Logging interface.
        """
        self.params = params
        self.logger = logger
        self.json_params_file = None
        self.running_avg_processing_time = 0
        self.chunks_processed = 0
        self.process_params()

    def process_params(self) -> None:
        """
        Process parameters by calling various initialization methods.
        """
        self.setup_output_directory()
        self.handle_metadata_files()
        self.process_setup_params()
        self.process_dataset_params()
        self.process_processing_params()
        self.initialize_miscellaneous_attributes()
        self.check_unused_params()

    def setup_output_directory(self) -> None:
        """
        Set up the output directory based on provided configuration.
        """
        self.output_dir = self.params["setup"].get("output_dir", "./output/")
        if not self.params["processing"].get("resume_from_checkpoint", False):
            check_and_create_output_dirs(self.output_dir, filetype="h5")
        self.logger.info(f"\nWriting data to {self.output_dir}.\n")
        self.max_chunk_size = self.params["processing"].get(
            "max_chunk_size", 64
        )
        self.logger.info(f"\nChunk size in kB: {self.max_chunk_size}.\n")
        self.json_params_file = os.path.join(
            self.output_dir, "data_params.json"
        )
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint.txt")
        dump_args(self.params, self.json_params_file)

    def handle_metadata_files(self) -> None:
        """
        Handle metadata files based on provided configuration.
        """
        metadata_files = self.params["setup"].pop("metadata_files", None)
        if metadata_files:
            metadata_files = metadata_files.split(",")
        input_dir = self.params["setup"].pop("input_dir", None)
        self.input_files = sorted(
            get_files(input_dir=input_dir, metadata_files=metadata_files)
        )

    def process_setup_params(self) -> None:
        """
        Setup the number of processes based on provided configuration.
        """
        self.processes = self.params["setup"].pop("processes", 0)
        if self.processes == 0:
            self.processes = cpu_count()

        self.reader_process_num = 1
        self.tokenize_process_num = self.processes - 2
        self.writer_process_num = 1

        # Create tokenizer queues for each tokenizer process
        self.tokenizer_queues = None
        if self.tokenize_process_num > 0:
            # Set up communication queues
            self.tokenizer_queues = [
                Queue() for _ in range(self.tokenize_process_num)
            ]
            self.writer_queues = [
                Queue() for _ in range(self.tokenize_process_num)
            ]
        self.stats_queue = Queue()

        ds_processor = self.params["setup"].pop(
            "dataset_processor", "LMDataPreprocessor"
        )
        token_generator_map = {
            "LMDataPreprocessor": "LMDataTokenGenerator",
            "SummarizationPreprocessor": "SummarizationTokenGenerator",
            "FIMDataPreprocessor": "FIMTokenGenerator",
        }
        self.token_generator_name = token_generator_map[ds_processor]
        if (
            self.token_generator_name == "LMDataTokenGenerator"
            or self.token_generator_name == "FIMTokenGenerator"
        ):
            self.prefix_queue = Queue()

    def check_unused_params(self) -> None:
        """
        Check for any unused parameters and log them as warnings.
        """
        unused_setup_params = [
            key for key in self.params["setup"].keys() if key != "output_dir"
        ]
        if unused_setup_params:
            self.logger.warning(
                f"\nThe following setup params are unused: {', '.join(unused_setup_params)}"
            )

        if self.params["dataset"]:
            self.logger.warning(
                "The following dataset params are unused: "
                + ", ".join(self.params["dataset"].keys())
            )

    def process_dataset_params(self) -> None:
        """
        Process dataset specific parameters.
        """
        dataset_params = self.params["dataset"]

        self.jsonl_key = dataset_params.pop("jsonl_key", None)

        # Summarization specific fields
        self.prompt_key = dataset_params.pop("prompt_key", None)
        self.completion_key = dataset_params.pop("completion_key", None)

        if (
            self.token_generator_name == "LMDataTokenGenerator"
            or self.token_generator_name == "FIMTokenGenerator"
        ):
            assert (
                self.prompt_key is None and self.completion_key is None
            ), f"Prompt/Completion key can't be provided when performing LM or FIM tasks. Provided prompt_key: {self.prompt_key}, completion_key: {self.completion_key}"
        elif self.token_generator_name == "SummarizationTokenGenerator":
            assert (
                self.jsonl_key is None
            ), f"jsonl key can't be provided when performing summarization tasks. Provided jsonl_key: {self.jsonl_key}"
        self.sep_token = dataset_params.get("sep_token", None)
        self.data_keys = dataset_params.get("keys", None)
        if self.data_keys:
            self.data_keys = [key.strip() for key in self.data_keys.split(",")]
        elif self.prompt_key and self.completion_key:
            self.data_keys = [self.prompt_key, self.completion_key]
        elif self.jsonl_key:
            self.data_keys = [self.jsonl_key]
        else:
            self.data_keys = ["text"]  ## default jsonl key is text

    def process_processing_params(self) -> None:
        """
        Process the processing parameters and initialize relevant class attributes.
        """
        processing_params = self.params["processing"]
        self.output_name = processing_params.pop("output_name", "examples")
        self.resume_from_checkpoint = processing_params.pop(
            "resume_from_checkpoint", False
        )
        self.max_seq_length = processing_params.get("max_seq_length", 2048)
        self.display_pbar = processing_params.pop("display_pbar", True)
        self.write_in_batch = processing_params.pop("write_in_batch", False)
        self.shuffle = processing_params.get("shuffle", False)
        if self.shuffle:
            self.shuffle_seed = processing_params.get("shuffle_seed", 0)
            self.set_shuffle_seed()
        self.initialize_tokenizer(processing_params)
        if self.sep_token:
            self.add_token(self.sep_token)
            self.logger.warning(
                f"A sep token {self.sep_token} was added to tokenizer. This "
                "will change the vocab size. If you are using a pretrained "
                "model, you will need to avoid adding this."
            )

    def set_shuffle_seed(self):
        "Sets shuffle seed for numpy"
        np.random.seed(self.shuffle_seed)

    def add_token(self, token):
        """ Add token to the tokenizer
        Args:
            token (str): token to be added to the tokenizer
        """
        if self.tokenizer_type == "gpt2tokenizer":
            self.tokenizer.add_token(token)
        elif self.tokenizer_type == "neoxtokenizer":
            self.tokenizer.add_token([token])
        elif self.tokenizer_type == "huggingfacetokenizer":
            self.tokenizer.add_token([token])

    def initialize_tokenizer(self, processing_params: Dict[str, Any]) -> None:
        """
        Initialize tokenizer based on the provided `tokenizer_type` parameter.

        Args:
            processing_params (Dict[str, Any]): Dictionary of processing parameters.
        """
        self.tokenizer_type = processing_params.pop(
            "tokenizer_type", "none"
        ).lower()
        assert (
            self.tokenizer_type != "none"
        ), "`tokenizer_type` is missing, please provide it using `args.tokenizer_type`."

        if self.tokenizer_type == "gpt2tokenizer":
            self.initialize_gpt2tokenizer(processing_params)
        elif self.tokenizer_type == "neoxtokenizer":
            self.initialize_neoxtokenizer(processing_params)
        elif self.tokenizer_type == "huggingfacetokenizer":
            self.initialize_huggingfacetokenizer(processing_params)
        else:
            raise NotImplementedError(
                f"{self.tokenizer_type} is not implemented. Acceptable values are: `gpt2tokenizer`, `neoxtokenizer`, `huggingfacetokenizer`."
            )

        # Override eos id and pad id from user args
        if (
            processing_params.get("eos_id") is not None
        ):  # important as id could be set to 0
            self.logger.info(
                f"Overriding the eos id {self.eos_id} from the tokenizer with supplied eos id: {processing_params['eos_id']}."
            )
            self.eos_id = processing_params["eos_id"]
            self.pad_id = processing_params[
                "eos_id"
            ]  # set pad id same as eos id
        if processing_params.get("pad_id") is not None:
            self.logger.info(
                f"Overriding the pad id {self.pad_id} from the tokenizer with supplied pad id: {processing_params['pad_id']}."
            )
            self.pad_id = processing_params["pad_id"]
            if (
                self.pad_id != self.eos_id
                and self.tokenizer_type == "gpt2tokenizer"
            ):
                self.logger.info(
                    f"Pad id {self.pad_id} supplied from command line is different from eos id {self.eos_id}. For GPT2 tokenizer, pad id and eos id must be the same. Setting pad id to eos id."
                )
                self.pad_id = self.eos_id

        self.token_generator = getattr(
            sys.modules[__name__], self.token_generator_name
        )(self.params, self.tokenizer, self.eos_id, self.pad_id)

    def initialize_gpt2tokenizer(
        self, processing_params: Dict[str, Any]
    ) -> None:
        """
        Initialize GPT-2 tokenizer.

        Args:
            processing_params (Dict[str, Any]): Dictionary of processing parameters.
        """
        vocab_file = processing_params.pop("vocab_file", None)
        encoder_file = processing_params.pop("encoder_file", None)
        self.tokenizer = BPETokenizer(vocab_file, encoder_file)
        assert (
            vocab_file
        ), "`vocab_file` is missing, please provide it using `args.vocab_file`."
        assert (
            encoder_file
        ), "`encoder_file` is missing, please provide it using `args.encoder_file`."

        self.eos_id = self.tokenizer.get_token_id("<|endoftext|>")
        self.pad_id = self.tokenizer.get_token_id("<|endoftext|>")

    def initialize_neoxtokenizer(
        self, processing_params: Dict[str, Any]
    ) -> None:
        """
        Initialize Neox tokenizer.

        Args:
            processing_params (Dict[str, Any]): Dictionary of processing parameters.
        """
        encoder_file = processing_params.pop("encoder_file", None)
        assert (
            encoder_file
        ), "`encoder_file` is missing, please provide it using `args.encoder_file`."
        self.tokenizer = HFTokenizer(encoder_file)
        self.eos_id = self.tokenizer.eos_id
        self.pad_id = (
            self.eos_id
            if self.tokenizer.pad_id is None
            else self.tokenizer.pad_id
        )

    def initialize_huggingfacetokenizer(
        self, processing_params: Dict[str, Any]
    ) -> None:
        """
        Initialize Hugging Face tokenizer.

        Args:
            processing_params (Dict[str, Any]): Dictionary of processing parameters.
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            processing_params.pop("huggingface_tokenizer")
        )
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = (
            self.eos_id
            if self.tokenizer.pad_token_id is None
            else self.tokenizer.pad_token_id
        )

    def initialize_miscellaneous_attributes(self) -> None:
        """
        Initialize miscellaneous attributes.
        """
        self.n_examples = (
            0  ## stores the total number of sequences in the current dataset
        )

    def get_params_file(self) -> str:
        """
        Retrieve the path to the JSON parameters file.

        Returns:
            str: Path to the JSON parameters file.
        """
        return self.json_params_file

    def get_output_dir(self) -> str:
        """
        Retrieve the output directory path.

        Returns:
            str: Path to the output directory.
        """
        return self.output_dir

    def calculate_total_size(self) -> int:
        """
        Calculate the total size of all input files, taking compression
        factors into consideration.

        Returns:
            int: The total size of all input files in bytes.
        """
        self.logger.info(f"Input files = {self.input_files}")
        total_size = sum(
            os.path.getsize(file) * get_compression_factor(file)
            for file in self.input_files
        )
        return total_size

    def calculate_total_chunks(self, total_size: int) -> int:
        """
        Calculate the total number of chunks based on the given total size
        and the predefined max chunk size.

        Parameters:
            total_size (int): The total size of the data in bytes.

        Returns:
            int: Total number of chunks.
        """
        max_chunk_size_bytes = self.max_chunk_size * 1024
        return math.ceil(total_size / max_chunk_size_bytes)

    def task_split_read_checkpoint(self):

        if self.resume_from_checkpoint and os.path.isfile(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r") as file:
                    file_idx, doc_idx, hdf5_idx = [
                        int(i) for i in file.read().split(", ")
                    ]
                self.logger.info(
                    f"Resuming from file number: {file_idx}, "
                    f"with number of documents processed: {doc_idx} and number of hdf5 files written = {hdf5_idx}"
                )
                return file_idx, doc_idx, hdf5_idx
            except Exception as e:
                # if checkpoint path is at initialization,
                # file may exist, but no data might be written in the file
                # in that event, do not do anything, go to the final return
                self.logger.error(e)
        return 0, 0, 0

    def file_split_read_checkpoint(self):

        process_checkpoints = [(0, 0, 0) for process in range(self.processes)]
        root, extension = os.path.splitext(self.checkpoint_path)

        for pid in range(self.processes):
            process_checkpoint_path = root + f'_process_{pid}.txt'
            if self.resume_from_checkpoint and os.path.isfile(
                process_checkpoint_path
            ):
                try:
                    with open(process_checkpoint_path, "r") as file:
                        file_idx, doc_idx, hdf5_idx = [
                            int(i) for i in file.read().split(", ")
                        ]
                    process_checkpoints[pid] = (file_idx, doc_idx, hdf5_idx)

                    self.logger.info(
                        f"Process {pid} resuming from file number: {file_idx}, "
                        f"with number of documents processed: {doc_idx} and number of hdf5 files written = {hdf5_idx}"
                    )
                except Exception as e:
                    # if checkpoint path is at initialization,
                    # file may exist, but no data might be written in the file
                    # in that event, do not do anything, go to the final return
                    self.logger.error(e)
        return process_checkpoints

    def verify_hdf5_files(self, chunk_data):

        ## check that the expected shape of data matches and
        ## the indices in the data is less than voacb size
        chunk_number, df_chunk = chunk_data
        vocab_size = self.get_vocab_size()
        expected_shape = (3, self.max_seq_length)
        data_arr = np.concatenate(df_chunk.tokenized_data, axis=0)
        data_shape = data_arr.shape
        assert data_shape[1:] == expected_shape or self.max_seq_length == -1, (
            f"Error in dataframe with number {chunk_number}, conversion is corrupted as the "
            f"shape of example is unexpected. Expected:"
            f" {expected_shape}, received {data_shape[1:]}."
        )
        assert (data_arr < vocab_size).all(), (
            f"Error in dataframe with number {chunk_number}, conversion is corrupted as the "
            f"input ids are greater than vocab size."
            f"Please ensure that a correct tokenizer is used "
            f"and the eos_id and pad_id are correct within the "
            f"tokenizer vocabulary size."
        )

    def write_remaining_prefix(self, output_file_name):

        prefix_sequences = 0
        prefix_stats = {
            "num_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
        }
        pid = os.getpid()
        if isinstance(self.token_generator, LMDataTokenGenerator):
            prefix = []
            sentinels_received = 0
            while True:
                curr_prefix = self.prefix_queue.get()
                if curr_prefix is None:
                    sentinels_received += 1
                    if sentinels_received == self.tokenize_process_num:
                        break
                else:
                    prefix = prefix + curr_prefix

            if len(prefix) > 0:
                (
                    encoded_prefix,
                    prefix_stats,
                ) = self.token_generator.encode_leftover_prefix(prefix)
                if len(encoded_prefix) > 0:
                    base_name, extension = os.path.splitext(output_file_name)
                    chunk_number = int(base_name.split('_')[-1])
                    df_chunk = DataFrame(self.data_keys)
                    df_chunk.tokenized_data.append(encoded_prefix)
                    chunk_data = chunk_number, df_chunk
                    self.verify_hdf5_files(chunk_data)
                    if not self.shuffle:
                        with h5py.File(output_file_name, "w") as h5f:
                            df_chunk.save_to_hdf5(h5f, self.write_in_batch)
                            prefix_sequences += int(h5f.attrs["n_examples"])
                    else:
                        n_examples = df_chunk.append_to_hdf5(
                            self.output_dir, self.total_chunks, pid
                        )
                        prefix_sequences += n_examples

        return prefix_sequences, prefix_stats

    def process_files(
        self, file_paths, process_idx, checkpoint_args, progress_counter,
    ) -> int:
        """
        Process the given files, tokenize the data chunks, and save to HDF5 format.

        Parameters:
        - file_paths: list of file_paths.
        - process_idx: Index of current process among all process spawned for file split

        Returns:
        - int: The count of processed chunks.
        """

        cum_data_stats = {
            "discarded": 0,
            "processed": 0,
            "successful": 0,
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "num_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
        }
        total_examples = 0

        # Initial setup
        reader = Reader(
            file_paths,
            max_chunk_size=self.max_chunk_size * 1024,
            logger=self.logger,
            keys=self.data_keys,
        )

        file_idx, doc_start_idx, hdf5_written = checkpoint_args
        process_chunk_number = hdf5_written
        checkpoint_args = (file_idx, doc_start_idx)
        for df_chunk in reader.stream_data(checkpoint_args):
            # Tokenize chunk
            df_chunk.tokenize(
                self.token_generator, self.pad_id,
            )

            cum_data_stats = {
                key: cum_data_stats[key] + df_chunk.data_stats[key]
                for key in df_chunk.data_stats
            }

            if df_chunk.tokenized_data == []:
                process_chunk_number += 1
                # Update progress counter
                progress_counter.value += 1
                continue

            # Save chunk to HDF5
            if not self.shuffle:
                output_file_name = os.path.join(
                    self.output_dir,
                    f"output_chunk_{process_idx}_{df_chunk.file_idx}_{df_chunk.start_doc_idx}_{process_chunk_number}.h5",
                )
                with h5py.File(output_file_name, "w") as h5f:
                    df_chunk.save_to_hdf5(h5f, self.write_in_batch)
                    total_examples += int(h5f.attrs["n_examples"])
            else:
                n_examples = df_chunk.append_to_hdf5(
                    self.output_dir, self.total_chunks, process_idx
                )
                total_examples += n_examples
            root, extension = os.path.splitext(self.checkpoint_path)
            process_checkpoint_path = root + f'_process_{process_idx}.txt'
            with open(process_checkpoint_path, "w") as file:
                file.write(
                    f"{df_chunk.file_idx}, {df_chunk.end_doc_idx + 1}, {process_chunk_number+1}"
                )
            process_chunk_number += 1
            # Update progress counter
            progress_counter.value += 1

        if isinstance(self.token_generator, LMDataTokenGenerator):
            if self.token_generator.prefix != []:
                self.prefix_queue.put(self.token_generator.prefix)
            self.prefix_queue.put(None)

        cum_data_stats["examples"] = total_examples
        self.stats_queue.put(cum_data_stats)
        self.stats_queue.put(None)

    def file_split_process_dataset(self) -> None:
        """
        Process the dataset by splitting files across multiple processes.
        """
        start_time = time.time()
        self.tokenize_process_num = self.processes
        total_size = self.calculate_total_size()
        self.total_chunks = self.calculate_total_chunks(total_size)
        final_data_stats = {
            "discarded": 0,
            "processed": 0,
            "successful": 0,
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "num_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
            "examples": 0,
        }
        process_file_lists = [[] for _ in range(self.processes)]
        process_checkpoints = self.file_split_read_checkpoint()
        hdf5_files_written = sum(
            [checkpoint_args[-1] for checkpoint_args in process_checkpoints]
        )
        for idx, file in enumerate(self.input_files):
            target_process = idx % self.processes
            process_file_lists[target_process].append(file)

        # Setup the shared progress counter
        progress_counter = multiprocessing.Value("i", hdf5_files_written)

        # Using tqdm for progress bar
        with tqdm(
            total=self.total_chunks, desc="Processing", dynamic_ncols=True
        ) as pbar:
            stop_event = (
                Event()
            )  # To signal the progress update thread when to stop
            progress_thread = Thread(
                target=update_progress,
                args=(
                    pbar,
                    progress_counter,
                    self.total_chunks,
                    start_time,
                    stop_event,
                ),
            )
            progress_thread.start()
            processes = [
                Process(
                    target=self.process_files,
                    args=(
                        files,
                        pid,
                        process_checkpoints[pid],
                        progress_counter,
                    ),
                )
                for pid, files in enumerate(process_file_lists)
            ]
            for p in processes:
                p.start()

            # Wait for all processes to finish
            for p in processes:
                # TODO: We had to add a timeout here
                # as a workaround to avoid hanging at the
                # join. We need to figure out a better
                # solution.
                p.join(timeout=1e-6)

            ## write remaining prefix from all processes
            if isinstance(self.token_generator, LMDataTokenGenerator):
                ## Set prefix df chunk file index to be same as last chunks file index
                ## and it's starting doc be one greater then previous chunk's ending doc index.
                (
                    prefix_df_chunk_file_idx,
                    prefix_df_chunk_start_doc_idx,
                    prefix_df_chunk_idx,
                ) = self.file_split_read_checkpoint()[-1]
                output_file_name = os.path.join(
                    self.output_dir,
                    f"output_chunk_{self.processes-1}_{prefix_df_chunk_file_idx}_{prefix_df_chunk_start_doc_idx}_{prefix_df_chunk_idx}.h5",
                )
                prefix_examples, stats = self.write_remaining_prefix(
                    output_file_name
                )
                final_data_stats["loss_valid_tokens"] += stats[
                    "loss_valid_tokens"
                ]
                final_data_stats["num_tokens"] += stats["num_tokens"]
                final_data_stats["num_pad_tokens"] += stats["num_pad_tokens"]
                final_data_stats["num_masked_tokens"] += stats[
                    "num_masked_tokens"
                ]
                final_data_stats["examples"] += prefix_examples
            end_time = time.time()
            elapsed_time = end_time - start_time
            stop_event.set()  # Signal the progress update thread to stop
            progress_thread.join()  # Wait for the progress update thread to finish

        self.logger.info(
            f"The process_dataset function took {elapsed_time:.2f} seconds to complete."
        )

        sentinels_received = 0
        while True:
            data_stats = self.stats_queue.get()
            if data_stats == None:
                sentinels_received += 1
                if sentinels_received == self.processes:
                    break  # Exit loop after receiving all sentinels
            else:
                final_data_stats = {
                    key: final_data_stats[key] + data_stats[key]
                    for key in data_stats
                }

        final_data_stats["average_chars_per_sequence"] = 0
        final_data_stats["average_bytes_per_sequence"] = 0

        try:
            final_data_stats["average_chars_per_sequence"] = math.ceil(
                final_data_stats["raw_chars_count"]
                / final_data_stats["examples"]
            )
            final_data_stats["average_bytes_per_sequence"] = math.ceil(
                final_data_stats["raw_bytes_count"]
                / final_data_stats["examples"]
            )
        except ZeroDivisionError:
            # Handle the division by zero error
            self.logger.info("No output hdf5 files were created .")

        return final_data_stats

    def reader_process(self, checkpoint_args: Tuple) -> None:
        """
        Reads data from input files and distributes them to the tokenizer queues.

        Args:
            checkpoint_args (Tuple[int, int, int]): File index, doc start index, and hdf5 index.
        """
        reader = Reader(
            self.input_files,
            max_chunk_size=self.max_chunk_size * 1024,
            logger=self.logger,
            keys=self.data_keys,
        )

        file_idx, doc_start_idx, hdf5_idx = checkpoint_args
        checkpoint_args = (file_idx, doc_start_idx)

        chunk_number = hdf5_idx  # Initialize chunk number counter
        for df_chunk in reader.stream_data(checkpoint_args):
            # Distribute chunks in a round-robin fashion across tokenizer queues
            # while making sure the first chunk is given to first tokenizer process
            tokenizer_queue = self.tokenizer_queues[
                (chunk_number - hdf5_idx) % (self.tokenize_process_num)
            ]
            tokenizer_queue.put(
                (chunk_number, df_chunk)
            )  # Send chunk number with df_chunk
            chunk_number += 1

        # Place sentinel values in each tokenizer queue to indicate end of reading
        for tq in self.tokenizer_queues:
            tq.put(None)

    def tokenizer_process(self, idx: int) -> None:
        """
        Tokenizes data and forwards the tokenized data to the writer queue.

        Args:
            tokenizer_queue (Queue): Queue containing chunks of data for tokenization.
            idx (int): Queue ID to forward tokenized chunks of data.
        """

        while True:
            chunk_data = self.tokenizer_queues[idx].get()
            if chunk_data is None:  # Sentinel value indicates termination
                if isinstance(self.token_generator, LMDataTokenGenerator):
                    if self.token_generator.prefix != []:
                        self.prefix_queue.put(self.token_generator.prefix)
                    self.prefix_queue.put(None)

                self.writer_queues[idx].put(None)
                break
            (
                chunk_number,
                df_chunk,
            ) = chunk_data  # Unpack chunk number and data frame chunk
            df_chunk.tokenize(self.token_generator, self.pad_id)
            # Forward the chunk number along with the df_chunk to the writer
            self.writer_queues[idx].put((chunk_number, df_chunk))

    def writer_process(self, progress_counter: "Value[int]",) -> None:
        """
        Process that writes tokenized data to HDF5 format.

        Args:
            writer_queue (Queue): Queue from which tokenized chunks of data are taken for writing.
            progress_counter (Value[int]): Shared counter tracking number of processed chunks.
        """
        cum_data_stats = {
            "discarded": 0,
            "processed": 0,
            "successful": 0,
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "num_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
            "examples": 0,
        }
        total_examples = 0
        sentinels_received = 0
        tokenizer_idx = 0
        while True:
            chunk_data = self.writer_queues[tokenizer_idx].get()
            tokenizer_idx = (tokenizer_idx + 1) % (self.tokenize_process_num)
            if chunk_data is None:
                sentinels_received += 1
                if sentinels_received == self.tokenize_process_num:
                    ## write remaining prefix from all processes
                    if isinstance(self.token_generator, LMDataTokenGenerator,):
                        ## Get prefix df chunk's file index and start doc index
                        ## from checkpoint file
                        (
                            prefix_df_chunk_file_idx,
                            prefix_df_chunk_start_doc_idx,
                            prefix_df_chunk_idx,
                        ) = self.task_split_read_checkpoint()
                        output_file_name = os.path.join(
                            self.output_dir,
                            f"output_chunk_{0}_{prefix_df_chunk_file_idx}_{prefix_df_chunk_start_doc_idx}_{prefix_df_chunk_idx}.h5",
                        )

                        prefix_examples, stats = self.write_remaining_prefix(
                            output_file_name
                        )
                        cum_data_stats["loss_valid_tokens"] += stats[
                            "loss_valid_tokens"
                        ]
                        cum_data_stats["num_tokens"] += stats["num_tokens"]
                        cum_data_stats["num_pad_tokens"] += stats[
                            "num_pad_tokens"
                        ]
                        cum_data_stats["num_masked_tokens"] += stats[
                            "num_masked_tokens"
                        ]
                        total_examples += prefix_examples

                    break
                continue

            chunk_number, df_chunk = chunk_data
            output_file_name = os.path.join(
                self.output_dir,
                f"output_chunk_{0}_{df_chunk.file_idx}_{df_chunk.start_doc_idx}_{chunk_number}.h5",
            )

            cum_data_stats = {
                key: cum_data_stats[key] + df_chunk.data_stats[key]
                for key in df_chunk.data_stats
            }

            if df_chunk.tokenized_data == []:
                progress_counter.value += 1
                continue
            else:
                self.verify_hdf5_files(chunk_data)
                if not self.shuffle:
                    with h5py.File(output_file_name, "w") as h5f:
                        df_chunk.save_to_hdf5(h5f, self.write_in_batch)
                        total_examples += int(h5f.attrs["n_examples"])
                else:
                    n_examples = df_chunk.append_to_hdf5(
                        self.output_dir, self.total_chunks, 0
                    )
                    total_examples += n_examples

            with open(self.checkpoint_path, "w") as file:
                file.write(
                    f"{df_chunk.file_idx}, {df_chunk.end_doc_idx + 1}, {chunk_number+1}"
                )

            progress_counter.value += 1

        cum_data_stats["examples"] = total_examples
        self.stats_queue.put(cum_data_stats)
        self.stats_queue.put(None)

    def task_split_process_dataset(self) -> None:
        """
        Split the dataset processing tasks across multiple processes.
        """
        start_time = time.time()
        total_size = self.calculate_total_size()
        self.logger.info(f"Total size of dataset: {total_size} bytes")
        self.total_chunks = self.calculate_total_chunks(total_size)
        self.logger.info(f"Total chunks to process: {self.total_chunks}")
        final_data_stats = {
            "discarded": 0,
            "processed": 0,
            "successful": 0,
            "raw_chars_count": 0,
            "raw_bytes_count": 0,
            "num_pad_tokens": 0,
            "num_masked_tokens": 0,
            "loss_valid_tokens": 0,
            "num_tokens": 0,
            "normalized_chars_count": 0,
            "normalized_bytes_count": 0,
            "examples": 0,
        }

        (
            file_index,
            doc_start_index,
            hdf5_written,
        ) = self.task_split_read_checkpoint()
        checkpoint_args = (file_index, doc_start_index, hdf5_written)
        progress_counter = multiprocessing.Value("i", hdf5_written)

        # Log process information
        self.logger.info(f"Total processes: {self.processes}")
        self.logger.info(f"Reader processes: 1")
        self.logger.info(f"Tokenizer processes: {self.processes - 2}")
        self.logger.info(f"Writer processes: 1")

        # Initialize and start processes
        tokenizers = [
            Process(target=self.tokenizer_process, args=(idx,),)
            for idx in range(self.tokenize_process_num)
        ]
        writer = Process(target=self.writer_process, args=(progress_counter,),)
        for t in tokenizers:
            t.start()
        writer.start()

        # Use tqdm for the progress bar
        with tqdm(
            total=self.total_chunks, desc="Processing", dynamic_ncols=True
        ) as pbar:
            stop_event = Event()
            progress_thread = Thread(
                target=update_progress,
                args=(
                    pbar,
                    progress_counter,
                    self.total_chunks,
                    start_time,
                    stop_event,
                ),
            )
            progress_thread.start()
            self.reader_process(checkpoint_args)
            for t in tokenizers:
                t.join()
            writer.join()
            stop_event.set()
            progress_thread.join()

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.logger.info(
                f"The process_dataset function took {elapsed_time:.2f} seconds to complete."
            )

        sentinels_received = 0
        while True:
            data_stats = self.stats_queue.get()
            if data_stats == None:
                break  # Exit loop after receiving all sentinels
            else:
                final_data_stats = {
                    key: final_data_stats[key] + data_stats[key]
                    for key in data_stats
                }
        final_data_stats["average_chars_per_sequence"] = 0
        final_data_stats["average_bytes_per_sequence"] = 0

        try:
            final_data_stats["average_chars_per_sequence"] = math.ceil(
                final_data_stats["raw_chars_count"]
                / final_data_stats["examples"]
            )
            final_data_stats["average_bytes_per_sequence"] = math.ceil(
                final_data_stats["raw_bytes_count"]
                / final_data_stats["examples"]
            )
        except ZeroDivisionError:
            # Handle the division by zero error
            self.logger.info("No output hdf5 files were created .")

        return final_data_stats

    def process_dataset(self) -> dict:
        """
        Process the dataset either through file split or task split methods.
        """
        data_stats = None
        if self.processes < 3:
            data_stats = self.file_split_process_dataset()
        else:
            data_stats = self.task_split_process_dataset()
        return data_stats

    def get_vocab_size(self):
        """ Get tokenizer vocabulary size
        Returns:
            vocab_size (int): text to tokenize
        """
        if self.tokenizer_type == "gpt2tokenizer":
            vocab_size = len(self.tokenizer.encoder)
        elif self.tokenizer_type == "neoxtokenizer":
            vocab_size = self.tokenizer.tokenizer.get_vocab_size()
        elif self.tokenizer_type == "huggingfacetokenizer":
            return self.tokenizer.vocab_size

        return vocab_size
