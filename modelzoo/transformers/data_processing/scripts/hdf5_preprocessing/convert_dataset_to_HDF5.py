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

import json
import os
from typing import Union

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from modelzoo.common.input.utils import check_and_create_output_dirs


def write_hdf5_file(
    file_path, data, n_examples, chunks, dtype="i4", compression="gzip"
):
    """Write data to HDF5 file.

    Args:
        file_path (string): HDF5 file path.
        data (numpy array): Input features and labels that will be written to HDF5.
        n_examples (int): Number of examples that will be written in the file.
        chunks (tuple or bool): Chunk shape, or True to enable auto-chunking.
        dtype (string): Data type for the HDF5 dataset.
        compression (string): Compression strategy.
    """
    with h5py.File(file_path, mode='w') as h5_file:
        h5_file.attrs["n_examples"] = n_examples
        h5_file.create_dataset(
            "data",
            data=data,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )


def convert_dataset_to_HDF5(
    dataset: Union[IterableDataset, Dataset],
    output_dir="./hdf5_dataset/",
    name="dataset-partition",
    samples_per_file=2000,
    num_workers=8,
    batch_size=64,
    data_collator=None,
    dtype="i4",
    compression="gzip",
):
    """Iterates PyTorch dataset and writes the data to HDF5 files.

    Args:
        dataset (IterableDataset, Dataset): PyTorch dataset to fetch the data from.
        output_dir (string): directory where HDF5 will be stored. Defaults to './hdf5_dataset/'
        name (string): name of the dataset; i.e. prefix to use for HDF5 file names. Defaults to 'dataset-partition'
        samples_per_file (int): number of samples written to each HDF5 file
        (last file can have less samples if the dataset isn't divisible). Defaults to 2000
        num_workers (int): number of Python processes to use for generating data. Defaults to 8
        batch_size (int): The batch size to use fetching the data. Defaults to 64
        data_collator (Callable): merges a list of samples to form a mini-batch of Tensor(s).
        Used when using batched loading from a map-style dataset.
        dtype (string): Data type for the HDF5 dataset.
        compression (string): Compression strategy.
    """
    check_and_create_output_dirs(output_dir, filetype="h5")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=data_collator,
        drop_last=False,
    )

    writer_index = 1
    total_written = 0
    example_index = 0
    data_buffer = []

    for batch in tqdm(dataloader):
        # Examine shapes and features
        if total_written == 0:
            features_len = len(batch)
            for _, value in batch.items():
                max_seq_length = value.shape[1]

        batch_np_array = np.concatenate(
            [np.expand_dims(batch[feature], axis=1) for feature in batch],
            axis=1,
        )
        batch_np_array = np.split(
            batch_np_array, batch_np_array.shape[0], axis=0
        )

        for example in batch_np_array:
            data_buffer.append(np.squeeze(example, axis=0))
            example_index += 1
            if example_index == samples_per_file:
                file_samples = np.stack(data_buffer, axis=0)
                fp = os.path.join(output_dir, f"{name}-{writer_index}.h5")
                write_hdf5_file(
                    file_path=fp,
                    data=file_samples,
                    n_examples=example_index,
                    chunks=(1, features_len, max_seq_length),
                    dtype=dtype,
                    compression=compression,
                )
                example_index = 0
                data_buffer = []
                writer_index += 1
            total_written += 1

    # write the last file if there are examples in data_buffer,
    # then break the infinite loop and exit
    if example_index > 0:
        file_samples = np.stack(data_buffer, axis=0)
        fp = os.path.join(output_dir, f"{name}-{writer_index}.h5")
        write_hdf5_file(
            file_path=fp,
            data=file_samples,
            n_examples=example_index,
            chunks=(1, features_len, max_seq_length),
            dtype=dtype,
            compression=compression,
        )

    params = {}
    params["n_examples"] = total_written
    params["features"] = [key for key in batch.keys()]
    params["output_dir"] = output_dir
    params["features_len"] = features_len
    params["max_seq_length"] = max_seq_length
    params["dtype"] = dtype
    params["compression"] = compression
    params["samples_per_file"] = samples_per_file
    json_params_file = os.path.join(output_dir, "data_params.json")
    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout, indent=4)

    print(f"Done! Wrote total of {total_written} examples.")
