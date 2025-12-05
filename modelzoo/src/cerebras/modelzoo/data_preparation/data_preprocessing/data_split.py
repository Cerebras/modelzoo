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

import copy
import json
import logging
import math
import os
import time
from functools import partial
from multiprocessing import Process, Value
from threading import Event, Thread

import numpy as np
import zstandard as zstd
from tqdm import tqdm

from cerebras.modelzoo.data_preparation.data_preprocessing.data_reader import (
    Reader,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    calculate_total_size,
    check_and_create_output_dirs,
    convert_fractions_or_floats,
    format_time,
    get_files,
    get_size,
    load_dataset_wrapper,
    normalize_msl,
    update_progress,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataSplit:
    """
    A class to handle dataset splitting into training/validation/testing sets
    and splitting the dataset to be preprocessed with different context lengths.

    The `DataSplit` class encapsulates the logic for validating configuration parameters,
    splitting datasets into predefined splits (e.g., train/val/test), and generating
    parameter configurations for downstream preprocessing.

    Attributes:
        params (dict): Configuration dictionary containing setup and processing parameters.
    """

    def __init__(self, params):
        """
        Initialize the DataSplit instance with the provided parameters.

        Args:
            params (dict): Configuration dictionary containing setup and processing parameters.
        """
        self.params = params
        self.data_splits = params["setup"].pop("data_splits", None)
        self.shared_context_split = self.params["setup"].pop(
            "context_splits", None
        )
        if not (self.data_splits or self.shared_context_split):
            ## No dataset splitting needs to be done.
            self.do_split = False
        else:
            self.do_split = True
            self.context_len_splits_list = (
                []
            )  ## This stores list of context splits for each data split. In case there is no data split it will store a singleton context split element for the entire data.
            self.input_dir_list = []
            self.input_files = []
            self.processes = params["setup"].get("processes", 1)
            self.split_seed = params["processing"].pop("split_seed", 0)
            self.data_splits_dir = params["setup"].pop(
                "data_splits_dir",
                os.path.join(
                    os.path.dirname(params["setup"]["output_dir"]),
                    "data_splits_dir",
                ),
            )
            check_and_create_output_dirs(
                self.data_splits_dir,
                filetype="jsonl.zst",
                dir_name="Data Splits",
            )
            self.read_chunk_size = (
                params["processing"].get("read_chunk_size", 1024) * 1024
            )  # By default, set to 1 MB.
            self.write_chunk_size = (
                params["processing"].get("write_chunk_size", 1024) * 1024
            )  # Use the write chunk size for determining the number of docs to split by doing batch append.
            self.input_files = self.get_input_files(self.params["setup"])
            total_size = calculate_total_size(self.input_files)
            self.total_chunks = math.ceil(total_size / self.read_chunk_size)

            self.progress_counter = Value(
                "i", 0
            )  ## Variable to store progress of dataset splitting
            self.split_stats = {"split_docs": Value("Q", 0)}

    def validate_context_split(self, context_split, split_name=None):
        """
        Validate the context split for the presence of required keys (MSL_List and split_fractions).

        Args:
            context_split (dict): Context split configuration dictionary.
            split_name (str, optional): Name of the data split (e.g., 'train', 'val').

        Returns:
            list: The MSL_List (list of sequence lengths).

        Raises:
            ValueError: If MSL_List or split_fractions is missing in the context split.
        """
        msl_list = context_split.get("MSL_List")
        if msl_list is None:
            raise ValueError(
                f"MSL_List argument is not provided for the context split{f' in the {split_name} data split' if split_name else ''}."
            )
        normalized_msl_list = [normalize_msl(item) for item in msl_list]

        if len(normalized_msl_list) != len(set(normalized_msl_list)):
            raise ValueError(
                f"There are duplicate entries in the MSL_List argument. Please specify unique values"
            )

        split_fractions = context_split.get("split_fractions")
        if split_fractions is None:
            raise ValueError(
                f"split_fractions argument is not provided for the context split{f' in the {split_name} data split' if split_name else ''}."
            )
        if len(split_fractions) != len(msl_list):
            raise ValueError(
                f"The number of elements in argument 'split_fractions' should match that in 'msl_list'."
            )
        return msl_list

    def prepare_splits(self):
        """
        Prepare the splits for the dataset based on the configuration parameters.

        This method validates and organizes the splits (train/val/test) and context splits.
        Raises exceptions for invalid configurations.
        """

        if self.data_splits:
            split_sum = sum(
                convert_fractions_or_floats(
                    split_value.get("split_fraction", 0)
                )
                for split_value in self.data_splits.values()
            )
            if not math.isclose(split_sum, 1.0):
                raise ValueError(
                    "Please make sure that 'split_fraction' field is present and the 'split_fraction' values for data split add upto 1."
                )

            for split_name, split_value in self.data_splits.items():
                split_fraction = convert_fractions_or_floats(
                    split_value.get("split_fraction")
                )
                if split_fraction is None:
                    raise ValueError(
                        f"split_fraction is not provided for the data split: {split_name}."
                    )
                context_split = split_value.get("context_splits")
                if context_split:
                    msl_list = self.validate_context_split(
                        context_split, split_name
                    )
                    self.input_dir_list.extend(
                        [(split_name, msl) for msl in msl_list]
                    )
                    self.context_len_splits_list.append(context_split)
                elif not self.shared_context_split:
                    self.input_dir_list.append((split_name, None))

        if self.context_len_splits_list and self.shared_context_split:
            raise ValueError(
                "Context splits have been provided indivually under each of the data splits and as a shared context split. Only one of the above is allowed. "
            )

        if self.shared_context_split:
            msl_list = self.validate_context_split(self.shared_context_split)
            if self.data_splits:
                for split_name in self.data_splits.keys():
                    self.input_dir_list.extend(
                        [(split_name, msl) for msl in msl_list]
                    )
                    self.context_len_splits_list.append(
                        self.shared_context_split
                    )
            else:
                self.input_dir_list.extend([(None, msl) for msl in msl_list])
                self.context_len_splits_list.append(self.shared_context_split)

    def split(self):
        """
        Perform dataset splitting based on the configuration.

        This method distributes the input files across processes in a round-robin manner
        and invokes the splitting logic using multiprocessing.Process.
        """

        # Assign files in a round-robin manner to each process
        file_list = [[] for _ in range(self.processes)]
        for idx, file in enumerate(self.input_files):
            file_list[idx % self.processes].append(file)

        # Prepare the splitting function
        splitting_func = partial(
            self.dataset_splitting,
            data_splits=self.data_splits,
            final_splits=self.context_len_splits_list,
            data_splits_dir=self.data_splits_dir,
        )

        logger.info("Dataset Splitting Started")
        start_time = time.time()

        # Create and start processes manually
        processes = []
        for i in range(self.processes):
            process = Process(
                target=self.split_per_process,
                args=(
                    file_list[i],  # List of files for this process
                    self.get_data_keys(self.params),  # Data keys
                    self.read_chunk_size,  # Chunk size
                    self.split_seed + i,  # Seed for splitting
                    splitting_func,  # Splitting function
                ),
            )
            processes.append(process)
            process.start()

        with tqdm(
            total=self.total_chunks,
            desc="Dataset Splitting Progress",
            dynamic_ncols=True,
        ) as pbar:
            stop_event = Event()  # Signal to stop progress update
            progress_thread = Thread(
                target=update_progress,
                args=(
                    pbar,
                    self.progress_counter,
                    self.total_chunks,
                    start_time,
                    stop_event,
                    None,
                    "splitting",
                    self.split_stats,
                ),
            )
            progress_thread.start()
            # Wait for all processes to finish
            for process in processes:
                process.join()

            # Final update of the progress bar to make sure it reaches `progress_counter.value`
            pbar.n = self.progress_counter.value
            pbar.total = self.progress_counter.value
            pbar.refresh()
            stop_event.set()  # Signal the progress update thread to stop
            progress_thread.join()  # Wait for the progress update thread to finish

        end_time = time.time()
        logger.info(
            f"Dataset Splitting Ended. Time taken = {format_time(end_time - start_time)}"
        )

    def get_params_list(self):
        """
        Generate a list of updated parameter configurations based on the splits.

        Returns:
            list: A list of dictionaries containing updated parameter configurations for each split.
        """

        params_list = []
        output_dir = self.params["setup"]["output_dir"]
        for data_split_name, msl in self.input_dir_list:
            updated_params = copy.deepcopy(self.params)
            ## The huggingface dataset has already been downloaded after splitting,so preprocess it as a regular local dataset
            if updated_params["setup"]["data"]["type"] == "huggingface":
                source = self.params["setup"]["data"]["source"]
                updated_params["setup"]["data"] = {
                    "type": "local",
                    "source": source,
                }

            base_input_path = (
                os.path.join(self.data_splits_dir, data_split_name)
                if data_split_name
                else self.data_splits_dir
            )
            updated_params["setup"]["data"]["source"] = (
                os.path.join(base_input_path, f"msl_{msl}")
                if msl
                else base_input_path
            )
            base_output_path = (
                os.path.join(output_dir, data_split_name)
                if data_split_name
                else output_dir
            )
            updated_params["setup"]["output_dir"] = (
                os.path.join(base_output_path, f"msl_{msl}")
                if msl
                else base_output_path
            )
            if msl:
                updated_params["processing"]["max_seq_length"] = normalize_msl(
                    msl
                )
            params_list.append(updated_params)

        return params_list

    @staticmethod
    def get_data_keys(params):
        """
        Extract and validate data keys from the processing parameters.

        Args:
            params (dict): Configuration dictionary containing processing parameters.

        Returns:
            dict: A dictionary of data keys.

        Raises:
            ValueError: If no valid data keys are found.
        """
        read_hook_kwargs = params["processing"].get("read_hook_kwargs")
        if not read_hook_kwargs:
            raise ValueError("Read hook kwargs is missing.")
        data_keys = {
            key: value
            for key, value in read_hook_kwargs.items()
            if key.endswith('_key')
        }
        if not data_keys:
            raise ValueError(
                "No data keys found. Please provide data keys in 'read_hook_kwargs'"
            )
        return data_keys

    def get_input_files(self, setup_params):
        """
        Retrieve input files based on the setup parameters.

        Args:
            setup_params (dict): Configuration dictionary containing setup parameters.

        Returns:
            list: A sorted list of input file paths.

        Raises:
            ValueError: If the input data source type is invalid.
        """
        metadata_files = setup_params.get("metadata_files")
        if metadata_files:
            metadata_files = metadata_files.split(",")
        input_data_params = setup_params.get("data")
        data_source_type = input_data_params.get("type")
        if data_source_type == 'huggingface':
            load_dataset_params = copy.deepcopy(input_data_params)
            load_dataset_params.pop("type", None)
            self.input_dir = load_dataset_wrapper(load_dataset_params)
        elif data_source_type == 'local':
            self.input_dir = input_data_params.get("source")
            if not self.input_dir:
                raise ValueError(
                    "Input data directory must be specified in source for local data"
                )
        else:
            raise ValueError(
                f"Invalid data source type: {data_source_type}. Source type can be ['huggingface', 'local']"
            )

        return sorted(
            get_files(input_dir=self.input_dir, metadata_files=metadata_files)
        )

    @staticmethod
    def save(doc_buffer, file_name):
        """
        Save a list of docs (list of dictionaries) to separate JSON Lines (.jsonl.zst) file in append mode.

        Args:
            doc_buffer (list[dict]): A list of docs, each representing a row.
            file_name (str): The file path with `.jsonl.zst` extension to save the data.
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        mode = 'ab' if os.path.exists(file_name) else 'wb'
        with open(file_name, mode) as f:
            with zstd.ZstdCompressor(level=3).stream_writer(f) as compressor:
                for doc in doc_buffer:
                    json_line = json.dumps(doc).encode('utf-8') + b'\n'
                    compressor.write(json_line)

    def dataset_splitting(
        self,
        doc_buffer,
        file_idx_list,
        rng,
        data_splits,
        final_splits,
        data_splits_dir,
    ):
        """
        Perform dataset splitting for document buffer containing a list of documents to be split.

        Based on the configuration, this method determines the appropriate split
        (e.g., train/val/test) and context length for the document. The document is
        then saved into the corresponding directory.

        Args:
            doc_buffer (List[dict]): A list of documents which need to be split.
            file_idx_list (List[str]): List of file names that the current process is splitting.
            rng (numpy.random.Generator): Random number generator for deterministic splits.
            data_splits (dict): Dictionary containing data split configurations (e.g., fractions).
            final_splits (list): List of context splits for different sequence lengths.
            data_splits_dir (str): Directory to store the split data.

        Raises:
            ValueError: If no valid splits are defined in the configuration.
        """
        from collections import defaultdict

        ## This stores the list of docs which will be mapped to the same file name.
        ## Each value in this mapping will be appended to the file name in 1 shot.
        file_name_to_doc_mapping = defaultdict(list)
        for doc in doc_buffer:
            index = rng.integers(0, len(file_idx_list))
            filename = file_idx_list[index]
            basename = os.path.basename(filename)  # Extract filename
            doc_name = basename.replace(
                ".", "_"
            )  ## .../test.jsonl.zst converted to .../test_jsonl_zst

            # Get relative directory (excluding filename)
            relative_dir = os.path.relpath(
                os.path.dirname(filename), self.input_dir
            )

            # Join back to get relative doc path without extensions
            relative_doc_path = os.path.normpath(
                os.path.join(relative_dir, doc_name)
            )

            if not data_splits and final_splits:
                context_split = final_splits[0]
                # Convert and normalize split fractions
                split_fractions = convert_fractions_or_floats(
                    context_split["split_fractions"]
                )

                context_split_index = rng.choice(
                    np.arange(len(split_fractions)), p=split_fractions
                )
                file_name = os.path.join(
                    data_splits_dir,
                    f"msl_{context_split['MSL_List'][context_split_index]}",
                    f"{relative_doc_path}.jsonl.zst",
                )
            else:
                # Convert and normalize data split probabilities
                data_splits_probs = convert_fractions_or_floats(
                    [
                        split_value.get("split_fraction")
                        for split_value in data_splits.values()
                    ]
                )
                data_split_index = rng.choice(
                    np.arange(len(data_splits.keys())), p=data_splits_probs
                )
                if final_splits:
                    context_split = final_splits[data_split_index]
                    # Convert and normalize context split fractions
                    split_fractions = convert_fractions_or_floats(
                        context_split["split_fractions"]
                    )

                    context_split_index = rng.choice(
                        np.arange(len(split_fractions)), p=split_fractions
                    )
                    file_name = os.path.join(
                        data_splits_dir,
                        list(data_splits.keys())[data_split_index],
                        f"msl_{context_split['MSL_List'][context_split_index]}",
                        f"{relative_doc_path}.jsonl.zst",
                    )
                else:
                    file_name = os.path.join(
                        data_splits_dir,
                        list(data_splits.keys())[data_split_index],
                        f"{relative_doc_path}.jsonl.zst",
                    )
            file_name_to_doc_mapping[file_name].append(doc)
        for file_name, doc_list in file_name_to_doc_mapping.items():
            self.save(doc_list, file_name)

    def split_per_process(
        self,
        input_files,
        data_keys,
        read_chunk_size,
        process_seed,
        splitting_func,
    ):
        """
        Perform dataset splitting for a subset of input files in a specific process.

        This method reads chunks of data from the input files, applies the provided
        splitting function to each document, and ensures deterministic splits using
        a process-specific random seed.

        Args:
            input_files (list): List of input files to process.
            data_keys (dict): Dictionary containing the keys for reading data.
            read_chunk_size (int): Maximum size (in bytes) for each read chunk.
            process_seed (int): Random seed for ensuring deterministic splits in this process.
            splitting_func (function): Function to handle the actual splitting logic for each document.

        Raises:
            ValueError: If the input files are invalid or cannot be read.
        """

        rng = np.random.default_rng(process_seed)
        checkpoint_args = {
            "file_index": 0,
            "global_df_index": 0,
        }
        reader = Reader(
            input_files,
            keys=data_keys,
            read_chunk_size=read_chunk_size,
            read_hook_fn=self.identity_function,
            checkpoint_args=checkpoint_args,
        )
        doc_buffer = []  ## Stores the list of docs to split
        current_buffer_size = 0
        df_index_ptr = 0
        for current_df_index, dataframe in enumerate(
            reader.stream_data(False, output_dir=self.data_splits_dir)
        ):
            for doc_idx, doc in enumerate(dataframe.raw_data):
                doc_buffer.append(doc)
                doc_size = get_size(doc)
                if current_buffer_size + doc_size >= self.write_chunk_size:
                    splitting_func(doc_buffer, input_files, rng)
                    self.split_stats["split_docs"].value += len(doc_buffer)
                    ## The number of data frames split equals the difference of the index of last processed dataframe and current dataframe
                    self.progress_counter.value += (
                        current_df_index - df_index_ptr
                    )
                    if doc_idx != len(dataframe.raw_data) - 1:
                        df_index_ptr = current_df_index
                    else:
                        ## The current dataframe has been fully split and put in buffer.
                        self.progress_counter.value += 1
                        df_index_ptr = current_df_index + 1
                    doc_buffer = []
                    current_buffer_size = 0
                else:
                    current_buffer_size += doc_size

        ## Split the remaining items in doc buffer.
        if doc_buffer:
            splitting_func(doc_buffer, input_files, rng)
            self.split_stats["split_docs"].value += len(doc_buffer)
            self.progress_counter.value += current_df_index - df_index_ptr + 1
            doc_buffer = []

    @staticmethod
    def identity_function(x):
        return x
