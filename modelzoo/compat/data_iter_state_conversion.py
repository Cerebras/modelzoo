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

"""Script to convert R1.9 DataLoader state files to R2.0 DataLoader checkpoint format
for the new map and iterable DataLoaders in MZ. This is useful to provide
backwards comptability for deterministic restart on 2.0 runs from old dataloader state
files.
"""
import argparse
import os
from pathlib import Path

import cerebras_pytorch as cstorch


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Convert R1.9 DataLoader state to R2.0 DataLoader state.'
    )
    parser.add_argument(
        "--old_checkpoint",
        "-c",
        help="Path to the r1.9 checkpoint file",
        required=True,
    )
    parser.add_argument(
        "--worker_data_iter_files_dir",
        "-w",
        help=(
            "Path to directory containing data step file `data_iter_checkpoint_state_file_global` "
            "and worker checkpoint files of the format `data_iter_state_file_worker_*_step_*.txt`"
        ),
        required=True,
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help=(
            "Path where the output R2.0 checkpoint file with the converted "
            "DataLoader state should be saved."
        ),
        required=True,
    )
    parser.add_argument(
        "--dataloader_type",
        "-d",
        help=(
            "The MZ DataLoader for which state is being converted. Use `map` for "
            "the map-style dataloader and `iterable` for the iterable-style "
            "dataloader. Defaults to map-style dataloader."
        ),
        choices=["map", "iterable"],
        default="map",
    )
    parser.add_argument(
        "--shuffle_seed",
        "-s",
        help=(
            "The seed value to be captured in the DataLoader state for "
            "the map-style dataloader. Note that the seed is only relevant "
            "for deterministically restarting the map-style dataloader if "
            "dataset shuffling/mixing is enabled."
        ),
        type=int,
        default=0,
    )
    return parser


def convert_dataloader_checkpoint(
    old_checkpoint_path,
    data_checkpoints_dir,
    output_file_path,
    dataloader_type="map",
    shuffle_seed=0,
):
    # Check if data_checkpoints_dir contains file specifying the
    # data step
    global_data_iter_state_file = os.path.join(
        data_checkpoints_dir, "data_iter_checkpoint_state_file_global"
    )
    assert os.path.isfile(global_data_iter_state_file), (
        f"File `{global_data_iter_state_file}` does not exist. "
        f"Please ensure that the specified dir `{data_checkpoints_dir}` "
        "has file `data_iter_checkpoint_state_file_global` that records "
        "the data step for the R1.9 Dataloader state being converted."
    )

    # Read the data step
    with open(global_data_iter_state_file, "r") as f:
        data_step = int(f.readline())

    state_dict = cstorch.load(old_checkpoint_path)

    total_samples_streamed = 0
    wrk_states = []

    dir = Path(data_checkpoints_dir)

    for f in os.listdir(dir):
        if f.endswith('.txt'):
            # WRK data iter files names follow the format:
            # `data_iter_state_file_worker_{wrk_id}_step_{step}.txt`
            # where `wrk_id` and `step` are ints.
            file_name_split = f.split('_')
            wrk_id = int(file_name_split[-3])
            step = int(file_name_split[-1].split('.')[0])

            # Each worker will have a single checkpoint at the data step
            if step == data_step:
                wrk_ckpt_file = os.path.join(data_checkpoints_dir, f)

                with open(wrk_ckpt_file, "r") as ckpt:
                    samples_streamed = int(ckpt.readline())
                    if dataloader_type == "iterable":
                        wrk_state_dict = {
                            "samples_streamed": samples_streamed,
                            "shard_index": wrk_id,
                        }
                        wrk_states.append(wrk_state_dict)
                    total_samples_streamed += samples_streamed

    if (
        dataloader_type == "map"
    ):  # State dict aggregation of map-style dataloader
        aggregated_state_dict = {
            "samples_streamed": total_samples_streamed,
            "seed": shuffle_seed,
        }
    else:  # State dict aggregation of iterable-style dataloader
        wrk_states.sort(key=lambda x: x["shard_index"])
        aggregated_state_dict = {"all_worker_states": wrk_states}

    # Add DL state to previously loaded checkpoint state dict
    state_dict["dataloader"] = aggregated_state_dict

    # Save the save dict to a new checkpoint
    cstorch.save(state_dict, Path(output_file_path))


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    old_checkpoint_path = args.old_checkpoint
    assert os.path.isfile(old_checkpoint_path), (
        f"Path `{old_checkpoint_path}` is invalid. "
        "Please provide the correct checkpoint file path for the R1.9 checkpoint."
    )

    data_ckpts_dir = args.worker_data_iter_files_dir
    assert os.path.isdir(data_ckpts_dir), (
        f"Path `{data_ckpts_dir}` is not a directory. "
        "Please provide the correct directory path "
        "where the worker checkpoint files of the "
        "format `data_iter_state_file_worker_*_step_*.txt` "
        "are saved."
    )

    output_file_path = args.output_file
    dataloader_type = args.dataloader_type
    shuffle_seed = args.shuffle_seed

    convert_dataloader_checkpoint(
        old_checkpoint_path,
        data_ckpts_dir,
        output_file_path,
        dataloader_type,
        shuffle_seed,
    )


if __name__ == "__main__":
    main()
