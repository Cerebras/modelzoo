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
Script that generates a dataset in HDF5 format for GPT Models.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../../.."))
from modelzoo.common.input.utils import check_and_create_output_dirs
from modelzoo.transformers.data_processing.scripts.hdf5_preprocessing.utils import (
    create_dataset,
    create_dataset_mp,
    dump_args,
    dump_result,
    get_parser,
    set_defaults,
    verify_saved_hdf5_files,
    verify_saved_hdf5_files_mp,
    write_hdf5_files,
)
from modelzoo.transformers.data_processing.scripts.utils import get_files

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main():
    """Main function for execution."""
    parser = get_parser("Create HDF5 dataset for raw text dataset.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not args.resume_from_checkpoint:
        check_and_create_output_dirs(output_dir, filetype="h5")

    if args.metadata_files:
        metadata_files = args.metadata_files.split(",")
    else:
        metadata_files = None
    input_files = get_files(
        input_dir=args.input_dir, metadata_files=metadata_files
    )

    logger.info(f"\nWriting data to {output_dir}.")

    set_defaults(args)

    json_params_file = os.path.join(output_dir, "data_params.json")
    dump_args(args, json_params_file)

    if args.processes > 1:
        results = create_dataset_mp(input_files, args, write_hdf5_files)
    else:
        # Run only single process run, with process number set as 0.
        results = create_dataset((input_files, args, write_hdf5_files, 0))

    if args.mode == "raw_text":
        dump_result(results, json_params_file)
    else:
        dump_result(results, json_params_file, args.eos_id, args.pad_id)

    logger.info(
        f"\nFinished writing data to {args.output_dir}."
        f" Runtime arguments and outputs can be found at {json_params_file}."
    )

    logger.info(f"Verifying the converted dataset at: {output_dir}")
    output_files = list(Path(output_dir).glob("*.h5"))
    if args.processes > 1:
        verify_saved_hdf5_files_mp(output_files, args)
    else:
        # Run only single process run, with process number set as 0.
        verify_saved_hdf5_files((output_files, args))
    logger.info("Done verifying the converted dataset.")


if __name__ == "__main__":
    main()
