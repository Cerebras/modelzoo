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

from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.chunk_data_preprocessor import (
    ChunkDataPreprocessor,
)
from cerebras.modelzoo.data_preparation.nlp.chunk_data_processing.utils import (
    dump_result,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
    get_params,
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main():
    """Main function for execution."""
    params = get_params(desc="Create HDF5 dataset for language models")
    dataset_processor = ChunkDataPreprocessor(params, logger)
    results = dataset_processor.process_dataset()
    output_dir = dataset_processor.get_output_dir()
    json_params_file = dataset_processor.get_params_file()

    vocab_size = dataset_processor.get_vocab_size()
    logger.info(
        f"\nFinished writing data to {output_dir}."
        f" Args & outputs can be found at {json_params_file}."
    )
    dump_result(
        results,
        json_params_file,
        dataset_processor.eos_id,
        dataset_processor.pad_id,
        vocab_size,
    )


if __name__ == "__main__":
    main()
