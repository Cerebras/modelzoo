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

import glob
import json
import math
import multiprocessing
import os
import sys
import time
from multiprocessing import Event, Lock, Process, Queue, Value, cpu_count
from multiprocessing.synchronize import Event
from threading import Event, Thread
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
from tqdm import tqdm

from cerebras.modelzoo.common.utils.utils import check_and_create_output_dirs
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.data_reader import (
    DataFrame,
    Reader,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.lm_data_token_generator import (
    LMDataTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    dump_args,
    get_files,
)
from cerebras.modelzoo.data_preparation.nlp.tokenizers.BPETokenizer import (
    BPETokenizer,
)
from cerebras.modelzoo.data_preparation.nlp.tokenizers.HFTokenizer import (
    HFTokenizer,
)

from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.dpo_data_token_generator import (  # noqa
    DPOTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.fim_data_token_generator import (  # noqa
    FIMTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.lm_vsl_data_token_generator import (  # noqa
    VSLLMDataTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.nlg_token_generator import (  # noqa
    NLGTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.summarization_data_token_generator import (  # noqa
    SummarizationTokenGenerator,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.summarization_vsl_data_token_generator import (  # noqa
    VSLSummarizationTokenGenerator,
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
        progress = progress_counter.value
        if progress > pbar.n:
            num_processed = progress - pbar.n
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

        ds_processor = self.params["setup"].pop(
            "dataset_processor", "LMDataPreprocessor"
        )
        token_generator_map = {
            "LMDataPreprocessor": "LMDataTokenGenerator",
            "SummarizationPreprocessor": "SummarizationTokenGenerator",
            "CurationCorpusPreprocessor": "SummarizationTokenGenerator",
            "FIMDataPreprocessor": "FIMTokenGenerator",
            "NLGPreprocessor": "NLGTokenGenerator",
            "VSLLMDataPreprocessor": "VSLLMDataTokenGenerator",
            "VSLSummarizationPreprocessor": "VSLSummarizationTokenGenerator",
            "DPOPreprocessor": "DPOTokenGenerator",
        }
        self.token_generator_name = token_generator_map[ds_processor]

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
        self.chosen_key = dataset_params.pop("chosen_key", None)
        self.rejected_key = dataset_params.pop("rejected_key", None)
        self.multi_turn_key = dataset_params.pop("multi_turn_key", None)
        self.multi_turn_content_key = dataset_params.pop(
            "multi_turn_content_key", None
        )
        if self.token_generator_name in [
            "LMDataTokenGenerator",
            "FIMTokenGenerator",
            "VSLLMDataTokenGenerator",
        ]:
            assert (
                self.prompt_key is None and self.completion_key is None
            ), f"Prompt/Completion key can't be provided when performing LM or FIM tasks. Provided prompt_key: {self.prompt_key}, completion_key: {self.completion_key}"
        elif self.token_generator_name in [
            "SummarizationTokenGenerator",
            "VSLSummarizationTokenGenerator",
        ]:
            assert (
                self.jsonl_key is None
            ), f"jsonl key can't be provided when performing summarization tasks. Provided jsonl_key: {self.jsonl_key}"
        self.sep_token = dataset_params.get("sep_token", None)

        self.data_keys = {}
        if self.prompt_key and self.completion_key:
            self.data_keys = {
                "prompt_key": self.prompt_key,
                "completion_key": self.completion_key,
            }
        elif self.jsonl_key:
            self.data_keys = {"jsonl_key": self.jsonl_key}
        elif self.multi_turn_key:
            self.data_keys = {"multi_turn_key": self.multi_turn_key}
            if self.multi_turn_content_key:
                self.data_keys.update(
                    {"multi_turn_content_key": self.multi_turn_content_key}
                )
        elif self.chosen_key and self.rejected_key:
            self.data_keys = {
                "chosen_key": self.chosen_key,
                "rejected_key": self.rejected_key,
            }
            if self.prompt_key:
                self.data_keys.update({"prompt_key": self.prompt_key})

        ## Set default keys if keys not available
        if not self.data_keys:
            if self.token_generator_name == "DPOTokenGenerator":
                self.data_keys = {
                    "prompt_key": "prompt",
                    "chosen_key": "chosen",
                    "rejected_key": "rejected",
                }  ## these are default dpo keys
            elif self.token_generator_name == "NLGTokenGenerator":
                self.data_keys = {
                    "prompt_key": "context",
                    "completion_key": "completion",
                }
            elif self.token_generator_name == "LMDataTokenGenerator":
                self.data_keys = {
                    "jsonl_key": "text"
                }  ## default jsonl key is text

        ## initialize the final data statistics
        self.final_data_stats = {
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
            "examples": 0,
            "average_chars_per_sequence": 0,
            "average_bytes_per_sequence": 0,
        }

        # Initialize checkpoint data stats
        self.checkpoint_data_stats = {
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
            "average_chars_per_sequence": 0,
            "average_bytes_per_sequence": 0,
        }

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
        self.max_chunk_size = processing_params.pop("max_chunk_size", 1024)
        self.logger.info(f"\nChunk size in kB: {self.max_chunk_size}.\n")
        self.display_pbar = processing_params.pop("display_pbar", True)
        self.write_in_batch = processing_params.pop("write_in_batch", False)
        self.shuffle = processing_params.get("shuffle", False)
        if self.shuffle:
            self.shuffle_seed = processing_params.get("shuffle_seed", 0)
            self.writer_process_num = (self.processes - 1) // 2
        else:
            self.writer_process_num = math.ceil((self.processes - 1) / 10)

        self.tokenize_process_num = self.processes - 1 - self.writer_process_num
        self.initialize_tokenizer(processing_params)
        if self.sep_token:
            self.add_token(self.sep_token)
            self.logger.warning(
                f"A sep token {self.sep_token} was added to tokenizer. This "
                "will change the vocab size. If you are using a pretrained "
                "model, you will need to avoid adding this."
            )
            # Create tokenizer queues for each tokenizer process
        self.tokenizer_queues = None
        if self.tokenize_process_num > 0:
            # Set up communication queues
            self.tokenizer_queues = [
                Queue(maxsize=50) for _ in range(self.tokenize_process_num)
            ]
            self.writer_queues = [
                Queue(maxsize=50) for _ in range(self.tokenize_process_num)
            ]
        self.stats_queue = Queue()
        if isinstance(
            self.token_generator, LMDataTokenGenerator
        ) and not isinstance(self.token_generator, VSLLMDataTokenGenerator):
            self.prefix_queue = Queue()

    def add_token(self, token):
        """Add token to the tokenizer
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

        if self.token_generator_name == "NLGTokenGenerator":
            self.token_generator = getattr(
                sys.modules[__name__], self.token_generator_name
            )(self.max_seq_length)
        else:
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
            processing_params.pop("huggingface_tokenizer"),
            token=processing_params.pop("auth_token"),
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

    def read_checkpoint(self, num_writers) -> List[Tuple[int, int, int]]:
        """
        This function reads the checkpoint args from the created checkpoint file.
        Parameters:
            num_writers: The number of writer processes
        """

        # Collect all the stats from previous processing
        # Pattern to find checkpoint stats JSON files
        checkpoint_pattern = os.path.join(
            self.output_dir, 'checkpoint_process_stats*.json'
        )
        # Read and aggregate stats from each file
        for stats_file in glob.glob(checkpoint_pattern):
            with open(stats_file, 'r') as file:
                stats_data = json.load(file)
                for key in self.checkpoint_data_stats:
                    if key in stats_data:
                        self.checkpoint_data_stats[key] += stats_data[key]

        process_checkpoints = [(0, 0, 0) for process in range(num_writers)]
        root, extension = os.path.splitext(self.checkpoint_path)

        for pid in range(num_writers):
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
                        f"and number of hdf5 files written = {hdf5_idx}"
                    )
                except Exception as e:
                    # if checkpoint path is at initialization,
                    # file may exist, but no data might be written in the file
                    # in that event, do not do anything, go to the final return
                    self.logger.error(e)
        return process_checkpoints

    def verify_hdf5_files(self, chunk_data) -> None:
        """
        This function verifies the whether the hdf5 files have been written correctly.
        Parameters:
            chunk_data: The df chunk which needs to be written to a hdf5 file
        """

        ## check that the expected shape of data matches and
        ## the indices in the data is less than voacb size
        chunk_number, df_chunk = chunk_data
        vocab_size = self.get_vocab_size()
        if isinstance(
            self.token_generator,
            (VSLLMDataTokenGenerator, VSLSummarizationTokenGenerator),
        ):
            expected_shape = (5, self.max_seq_length)
        elif isinstance(
            self.token_generator,
            DPOTokenGenerator,
        ):
            expected_shape = (6, self.max_seq_length)
        else:
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

    def write_remaining_prefix(self, chunk_locks, pid) -> Tuple[int, Dict]:
        """
        This function writes the prefix remaining after processing LMData when pack_sequences is set to true.
        Parameters:
            chunk_locks : List of locks for appending to hdf5 files during shuffling
            pid: Process id of the current process

        """

        ## write remaining prefix from all processes for LMData tasks when pack sequences is set to true
        if not (
            self.token_generator_name == "LMDataTokenGenerator"
            or self.token_generator_name == "FIMTokenGenerator"
        ):
            return
        else:
            output_file_name = os.path.join(
                self.output_dir,
                f"output_chunk_{pid}_{0}_{0}_{0}.h5",
            )
            prefix_sequences = 0
            prefix_stats = {
                "num_pad_tokens": 0,
                "non_pad_tokens": 0,
                "num_masked_tokens": 0,
                "loss_valid_tokens": 0,
                "num_tokens": 0,
            }
            if self.shuffle:
                np.random.seed(self.shuffle_seed)
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
                            self.output_dir,
                            self.total_chunks,
                            pid,
                            chunk_locks,
                        )
                        prefix_sequences += n_examples

                    self.final_data_stats["loss_valid_tokens"] += prefix_stats[
                        "loss_valid_tokens"
                    ]
                    self.final_data_stats["num_tokens"] += prefix_stats[
                        "num_tokens"
                    ]
                    self.final_data_stats["num_pad_tokens"] += prefix_stats[
                        "num_pad_tokens"
                    ]
                    self.final_data_stats["non_pad_tokens"] += prefix_stats[
                        "non_pad_tokens"
                    ]
                    self.final_data_stats["num_masked_tokens"] += prefix_stats[
                        "num_masked_tokens"
                    ]
                    self.final_data_stats["examples"] += prefix_sequences

    def shuffle_second_pass(self, file_list, progress_counter, pid) -> None:
        """
        This function performs the second pass of shuffling.
        Parameters:
            file_list:  List of hdf5 file paths to shuffle
            progress_counter: A shared counter to track progress across processes.
        """

        np.random.seed(self.shuffle_seed + pid)
        for file_path in file_list:
            with h5py.File(file_path, 'r+') as hf:
                data = hf["data"]
                data_array = data[:]
                np.random.shuffle(data_array)
                data[...] = data_array
            progress_counter.value += 1

    def split_shuffle_second_pass(self):
        """
        This function divides the output hdf5 files into different processes and prepares them for the
        second pass of shuffling.
        """

        ## Perform the second pass of shuffling:
        self.logger.info("The second pass of shuffling has started")
        hdf5_file_list = sorted(
            glob.glob(os.path.join(self.output_dir, "*.h5"))
        )
        hdf5_file_list_length = len(hdf5_file_list)
        chunk_size = hdf5_file_list_length // self.processes
        file_list_per_process = [
            hdf5_file_list[i * chunk_size : (i + 1) * chunk_size]
            for i in range(self.processes)
        ]

        remainder = hdf5_file_list_length % self.processes
        for i in range(remainder):
            file_list_per_process[i].append(
                hdf5_file_list[self.processes * chunk_size + i]
            )

        progress_counter = multiprocessing.Value("i", 0)
        start_time = time.time()
        shuffle_processes = [
            Process(
                target=self.shuffle_second_pass,
                args=(
                    file_list_per_process[i],
                    progress_counter,
                    i,
                ),
            )
            for i in range(self.processes)
        ]
        for proc in shuffle_processes:
            proc.start()
        with tqdm(
            total=hdf5_file_list_length, desc="Processing", dynamic_ncols=True
        ) as pbar:
            stop_event = Event()
            progress_thread = Thread(
                target=update_progress,
                args=(
                    pbar,
                    progress_counter,
                    hdf5_file_list_length,
                    start_time,
                    stop_event,
                ),
            )
            progress_thread.start()
            for i, proc in enumerate(shuffle_processes):
                proc.join()
            stop_event.set()
            progress_thread.join()

    def stats_collation(self, num_writer_processes) -> None:
        """
        This function collates the stats obtained from the different writer processes into a combined final stats
        Parameters:
            num_writer_processes:  Number of writer processes
        """

        sentinels_received = 0
        while True:
            data_stats = self.stats_queue.get()
            if data_stats == None:
                sentinels_received += 1
                if sentinels_received == num_writer_processes:
                    break  # Exit loop after receiving all sentinels
            else:
                for key in data_stats:
                    self.final_data_stats[key] = (
                        self.final_data_stats[key] + data_stats[key]
                    )

        if self.resume_from_checkpoint:
            # Update final_data_stats with aggregated values
            for key in self.checkpoint_data_stats:
                self.final_data_stats[key] += self.checkpoint_data_stats[key]

        # Calculate average bytes and chars per sequence if examples > 0 to avoid division by zero
        if self.final_data_stats["examples"] > 0:
            self.final_data_stats["average_chars_per_sequence"] = math.ceil(
                self.final_data_stats["raw_chars_count"]
                / self.final_data_stats["examples"]
            )
            self.final_data_stats["average_bytes_per_sequence"] = math.ceil(
                self.final_data_stats["raw_bytes_count"]
                / self.final_data_stats["examples"]
            )

    def process_files(
        self,
        file_paths,
        process_idx,
        checkpoint_args,
        progress_counter,
        chunk_locks,
    ) -> None:
        """
        Process the given files, tokenize the data chunks, and save to HDF5 format.

        Parameters:
            file_paths: list of file_paths.
            process_idx: Index of current process among all process spawned for file split
            checkpoint_args (Tuple[int, int, int]): File index, doc start index, and hdf5 index.
            progress_counter (Value[int]): Shared counter tracking number of processed chunks.
            chunk_locks : List of locks for appending to hdf5 files during shuffling

        """

        cum_data_stats = {
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
            "examples": 0,
        }
        total_examples = 0
        if self.shuffle:
            np.random.seed(self.shuffle_seed + process_idx)
        # Initial setup
        reader = Reader(
            file_paths,
            max_chunk_size=self.max_chunk_size * 1024,
            keys=self.data_keys,
        )

        file_idx, doc_start_idx, hdf5_written = checkpoint_args
        process_chunk_number = hdf5_written
        checkpoint_args = (file_idx, doc_start_idx)
        for df_chunk in reader.stream_data(checkpoint_args):
            # Tokenize chunk
            df_chunk.tokenize(self.token_generator)
            for key in df_chunk.data_stats:
                cum_data_stats[key] += df_chunk.data_stats[key]

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
                    cum_data_stats["examples"] += int(h5f.attrs["n_examples"])
            else:
                n_examples = df_chunk.append_to_hdf5(
                    self.output_dir,
                    self.total_chunks,
                    process_idx,
                    chunk_locks,
                )
                cum_data_stats["examples"] += n_examples
            root, extension = os.path.splitext(self.checkpoint_path)
            process_checkpoint_path = root + f'_process_{process_idx}.txt'
            process_stats_path = root + f'_process_stats_{process_idx}.json'
            checkpoint_doc_idx = df_chunk.end_doc_idx + 1
            if isinstance(
                self.token_generator,
                (VSLLMDataTokenGenerator, VSLSummarizationTokenGenerator),
            ):
                checkpoint_doc_idx = df_chunk.start_doc_idx

            with open(process_checkpoint_path, "w") as file:
                file.write(
                    f"{df_chunk.file_idx}, {checkpoint_doc_idx}, {process_chunk_number+1}"
                )
            dump_args(cum_data_stats, process_stats_path)
            process_chunk_number += 1
            # Update progress counter
            progress_counter.value += 1
        if isinstance(
            self.token_generator, LMDataTokenGenerator
        ) and not isinstance(self.token_generator, VSLLMDataTokenGenerator):
            if self.token_generator.prefix != []:
                self.prefix_queue.put(self.token_generator.prefix)
            self.prefix_queue.put(None)

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
        process_file_lists = [[] for _ in range(self.processes)]
        process_checkpoints = self.read_checkpoint(self.processes)
        hdf5_files_written = sum(
            [checkpoint_args[-1] for checkpoint_args in process_checkpoints]
        )
        for idx, file in enumerate(self.input_files):
            target_process = idx % self.processes
            process_file_lists[target_process].append(file)

        # Setup the shared progress counter
        progress_counter = multiprocessing.Value("i", hdf5_files_written)
        if self.shuffle and self.processes > 1:
            lock_pool_size = cpu_count()
            chunk_locks = [Lock() for _ in range(lock_pool_size)]
        else:
            chunk_locks = None

        processes = [
            Process(
                target=self.process_files,
                args=(
                    files,
                    pid,
                    process_checkpoints[pid],
                    progress_counter,
                    chunk_locks,
                ),
            )
            for pid, files in enumerate(process_file_lists)
        ]
        for p in processes:
            p.start()
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
            self.stats_collation(self.processes)
            self.write_remaining_prefix(chunk_locks, self.processes)
            # Wait for all processes to finish
            for p in processes:
                # TODO: We had to add a timeout here
                # as a workaround to avoid hanging at the
                # join. We need to figure out a better
                # solution.
                p.join(timeout=1e-6)

            end_time = time.time()
            elapsed_time = end_time - start_time
            stop_event.set()  # Signal the progress update thread to stop
            progress_thread.join()  # Wait for the progress update thread to finish

        if self.shuffle:
            self.split_shuffle_second_pass()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(
            f"The process_dataset function took {elapsed_time:.2f} seconds to complete."
        )

    def reader_process(self, checkpoint_args: Tuple) -> None:
        """
        Reads data from input files and distributes them to the tokenizer queues.

        Args:
            checkpoint_args (Tuple[int, int, int]): File index, doc start index, and hdf5 index.
        """

        reader = Reader(
            self.input_files,
            max_chunk_size=self.max_chunk_size * 1024,
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
            idx (int): Queue ID to forward tokenized chunks of data.
        """
        while True:
            try:
                chunk_data = self.tokenizer_queues[idx].get()
                if chunk_data is None:  # Sentinel value indicates termination
                    if isinstance(
                        self.token_generator, LMDataTokenGenerator
                    ) and not isinstance(
                        self.token_generator, VSLLMDataTokenGenerator
                    ):
                        if self.token_generator.prefix != []:
                            self.prefix_queue.put(self.token_generator.prefix)
                        self.prefix_queue.put(None)
                    self.writer_queues[idx].put(None)
                    break
                (
                    chunk_number,
                    df_chunk,
                ) = chunk_data  # Unpack chunk number and data frame chunk
                df_chunk.tokenize(self.token_generator)
                self.writer_queues[idx].put((chunk_number, df_chunk))

            except Exception as e:
                self.logger.error(
                    f'Exception in tokenizer process {os.getpid()}: {e}'
                )

    def writer_process(
        self,
        progress_counter: "Value[int]",
        num_sentinels: int,
        writer_idx: int,
        chunk_locks,
    ) -> None:
        """
        Process that writes tokenized data to HDF5 format.

        Args:
            progress_counter (Value[int]): Shared counter tracking number of processed chunks.
            num_sentinels : Number of sentinels to be received for the current writer process
            writer_idx : The index of the current writer process
            chunk_locks : List of locks for appending to hdf5 files during shuffling
        """
        cum_data_stats = {
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
            "examples": 0,
        }
        total_examples = 0
        sentinels_received = 0
        tokenizer_idx = writer_idx
        if self.shuffle:
            np.random.seed(self.shuffle_seed + writer_idx)
        while True:
            try:
                chunk_data = self.writer_queues[tokenizer_idx].get()
                ## We need to allocate the writer queues to writer processes in a round robin fashion.
                ## When the writer queue index (aka tokenizer_idx) goes beyond the total writer queues(aka self.tokenize_process_num) reset it to the current writer process index(aka writer_idx).
                tokenizer_idx = tokenizer_idx + self.writer_process_num
                if tokenizer_idx >= self.tokenize_process_num:
                    tokenizer_idx = writer_idx
                if chunk_data is None:
                    sentinels_received += 1
                    if sentinels_received == num_sentinels:
                        break
                    continue
                chunk_number, df_chunk = chunk_data
                if self.resume_from_checkpoint:
                    h5_files = sorted(
                        glob.glob(self.output_dir + f"/output_chunk_*.h5")
                    )
                    for h5file_name in h5_files:
                        (
                            hdf5_file_idx,
                            hdf5_start_doc_idx,
                            hdf5_idx,
                        ) = os.path.splitext(h5file_name)[0].split("_")[-3:]
                        if hdf5_idx == chunk_number:
                            ## remove the previously file and over ride it
                            os.remove(h5file_name)
                            break
                for key in df_chunk.data_stats:
                    cum_data_stats[key] += df_chunk.data_stats[key]
                if df_chunk.tokenized_data == []:
                    progress_counter.value += 1
                    continue
                else:
                    self.verify_hdf5_files(chunk_data)
                    if not self.shuffle:
                        output_file_name = os.path.join(
                            self.output_dir,
                            f"output_chunk_{writer_idx}_{df_chunk.file_idx}_{df_chunk.start_doc_idx}_{chunk_number}.h5",
                        )
                        with h5py.File(output_file_name, "w") as h5f:
                            df_chunk.save_to_hdf5(h5f, self.write_in_batch)
                            cum_data_stats["examples"] += int(
                                h5f.attrs["n_examples"]
                            )
                    else:
                        n_examples = df_chunk.append_to_hdf5(
                            self.output_dir,
                            self.total_chunks,
                            writer_idx,
                            chunk_locks,
                        )
                        cum_data_stats["examples"] += n_examples

                root, extension = os.path.splitext(self.checkpoint_path)
                process_checkpoint_path = root + f'_process_{writer_idx}.txt'
                process_stats_path = root + f'_process_stats_{writer_idx}.json'
                checkpoint_doc_idx = df_chunk.end_doc_idx + 1
                if isinstance(
                    self.token_generator,
                    (VSLLMDataTokenGenerator, VSLSummarizationTokenGenerator),
                ):
                    checkpoint_doc_idx = df_chunk.start_doc_idx

                with open(process_checkpoint_path, "w") as file:
                    file.write(
                        f"{df_chunk.file_idx}, {checkpoint_doc_idx}, {chunk_number+1}"
                    )
                dump_args(cum_data_stats, process_stats_path)
                progress_counter.value += 1
            except Exception as e:
                self.logger.error(
                    f'Exception in writer process {os.getpid()}: {e}'
                )
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
        self.logger.info(
            f"Approximate number of chunks to process: {self.total_chunks}"
        )

        process_checkpoints = self.read_checkpoint(self.writer_process_num)
        sorted_process_checkpoints = sorted(
            process_checkpoints, key=lambda x: x[-1]
        )
        checkpoint_args = sorted_process_checkpoints[0]
        hdf5_files_written = checkpoint_args[2]
        progress_counter = multiprocessing.Value("i", hdf5_files_written)
        # Log process information
        self.logger.info(f"Total processes: {self.processes}")
        self.logger.info(f"Reader processes: 1")
        self.logger.info(f"Tokenizer processes: {self.tokenize_process_num}")
        self.logger.info(f"Writer processes: {self.writer_process_num}")

        if self.shuffle and self.writer_process_num > 1:
            lock_pool_size = 128
            chunk_locks = [Lock() for _ in range(lock_pool_size)]
        else:
            chunk_locks = None
        chunks_per_writer = [
            (self.tokenize_process_num // self.writer_process_num + 1)
            if i < self.tokenize_process_num % self.writer_process_num
            else (self.tokenize_process_num // self.writer_process_num)
            for i in range(self.writer_process_num)
        ]
        # Initialize and start processes
        tokenizers = [
            Process(
                target=self.tokenizer_process,
                args=(idx,),
            )
            for idx in range(self.tokenize_process_num)
        ]
        writers = []
        for idx in range(self.writer_process_num):
            writers.append(
                Process(
                    target=self.writer_process,
                    args=(
                        progress_counter,
                        chunks_per_writer[idx],
                        idx,
                        chunk_locks,
                    ),
                )
            )
        for t in tokenizers:
            t.start()
        for w in writers:
            w.start()

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
                # TODO: We had to add a timeout here
                # as a workaround to avoid hanging at the
                # join. We need to figure out a better
                # solution.
                t.join(timeout=1e-6)
            self.stats_collation(self.writer_process_num)
            self.write_remaining_prefix(chunk_locks, self.writer_process_num)
            for w in writers:
                w.join()

            stop_event.set()
            progress_thread.join()

        if self.shuffle:
            self.split_shuffle_second_pass()

        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(
            f"The process_dataset function took {elapsed_time:.2f} seconds to complete."
        )

    def process_dataset(self) -> dict:
        """
        Process the dataset either through file split or task split methods.
        """
        data_stats = None
        if self.processes < 3:
            self.file_split_process_dataset()
        else:
            self.task_split_process_dataset()
        return self.final_data_stats

    def get_vocab_size(self):
        """Get tokenizer vocabulary size
        Returns:
            vocab_size (int): text to tokenize
        """
        if self.tokenizer_type == "gpt2tokenizer":
            vocab_size = len(self.tokenizer.encoder)
        elif self.tokenizer_type == "neoxtokenizer":
            vocab_size = self.tokenizer.tokenizer.get_vocab_size()
        elif self.tokenizer_type == "huggingfacetokenizer":
            return len(self.tokenizer)

        return vocab_size
