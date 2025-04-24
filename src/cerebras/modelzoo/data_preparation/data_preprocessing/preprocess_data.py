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
Script to generate an HDF5 dataset for GPT Models.
"""

# isort: off
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../"))
# isort: on
import logging
import signal
from multiprocessing import Event

from cerebras.modelzoo.data_preparation.data_preprocessing.data_preprocessor import (
    DataPreprocessor,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.data_split import (
    DataSplit,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.subset_split import (
    SubsetSplit,
)
from cerebras.modelzoo.data_preparation.data_preprocessing.utils import (
    get_params,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

exit_event = Event()


def handle_signal(signum, frame):
    logger.info(f"Received signal {signum}. Preparing to exit gracefully...")
    exit_event.set()
    sys.exit(0)


## Register signal handlers.
signal.signal(
    signal.SIGINT,
    handle_signal,
)
signal.signal(
    signal.SIGTERM,
    handle_signal,
)
signal.signal(
    signal.SIGHUP,
    handle_signal,
)
signal.signal(
    signal.SIGQUIT,
    handle_signal,
)


def generate_params_list(params):
    subset_class = SubsetSplit(params)
    params_list = subset_class.generate_subsets()
    final_params_list = []
    ## Perform a dataset split on each of the data subsets individually
    for subset_param in params_list:
        split_class = DataSplit(subset_param)
        if not split_class.do_split:
            ## If no split needs to be done just return with subset params list. If a split needs to be done it will be done on all subsets.
            return params_list
        split_class.prepare_splits()
        split_class.split()
        final_params_list.extend(split_class.get_params_list())

    return final_params_list


def main():
    """Main function for execution."""

    params = get_params(desc="Create HDF5 dataset for language models")
    preprocess_data(params)


def preprocess_data(params):
    params_list = generate_params_list(params)

    for updated_params in params_list:
        dataset_processor = DataPreprocessor(updated_params, exit_event)
        dataset_processor.process_dataset()
        output_dir = dataset_processor.get_output_dir()
        json_params_file = dataset_processor.get_params_file()
        # Retrieve vocab size and log completion
        vocab_size = dataset_processor.get_vocab_size()
        logger.info(
            f"\nFinished writing data to {output_dir}."
            f" Args & outputs can be found at {json_params_file}."
        )


if __name__ == "__main__":
    main()
