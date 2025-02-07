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
import logging
import os
from collections import OrderedDict, defaultdict
from typing import Union

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from cerebras.modelzoo.common.utils.utils import check_and_create_output_dirs

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def write_hdf5_file(
    file_path,
    dataset_name,
    data,
    n_examples,
    chunks,
    dtype="i4",
    compression="gzip",
):
    """Write data to HDF5 file.

    Args:
        dataset_name (string): HDF5 dataset name
        file_path (string): HDF5 file path.
        data (numpy array): Input features and labels that will be written to HDF5.
        n_examples (int): Number of examples that will be written in the file.
        chunks (tuple or bool): Chunk shape, or True to enable auto-chunking.
        dtype (string): Data type for the HDF5 dataset.
        compression (string): Compression strategy.
    """
    with h5py.File(file_path, mode='a') as h5_file:
        h5_file.attrs["n_examples"] = n_examples
        h5_file.create_dataset(
            dataset_name,
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
    feature_groups = defaultdict(list)

    for batch in tqdm(dataloader):
        # Examine shapes and features to determine how many datasets/groups are required in the HDF5 file
        if total_written == 0:
            # store all feature names together that have same shape
            for feature, value in batch.items():
                # Drop the batch dimension from shape
                shape = value.shape[1:]
                feature_groups[shape].append(feature)

            feature_groups = OrderedDict(feature_groups)
            num_groups = len(feature_groups)
            data_buffer = [[] for _ in range(num_groups)]

        feature_groups_np_arrays = []
        for group, features in feature_groups.items():
            # batch starts as dictionary of key (str): value (torch.Tensor)
            # where value may have arbitrary dimensions
            # Then we stack all features that have the same shape
            batch_np_array = np.concatenate(
                [
                    np.expand_dims(batch[feature], axis=1)
                    for feature in features
                ],
                axis=1,
            )
            # split the batch into a list of single elements
            batch_np_array = np.split(
                batch_np_array, batch_np_array.shape[0], axis=0
            )
            feature_groups_np_arrays.append(batch_np_array)

        # collect elements from batch into a buffer until we reach
        # samples_per_file, then clear buffer and continue adding
        for group_examples in zip(*feature_groups_np_arrays):
            for i, example in enumerate(group_examples):
                data_buffer[i].append(np.squeeze(example, axis=0))

            example_index += 1
            if example_index == samples_per_file:
                fp = os.path.join(output_dir, f"{name}-{writer_index}.h5")

                for i in range(num_groups):
                    file_samples = np.stack(data_buffer[i], axis=0)
                    chunk_size = (1, *file_samples.shape[1:])

                    write_hdf5_file(
                        dataset_name="data" if num_groups == 1 else f"data_{i}",
                        file_path=fp,
                        data=file_samples,
                        n_examples=example_index,
                        chunks=chunk_size,
                        dtype=dtype,
                        compression=compression,
                    )

                example_index = 0
                data_buffer = [[] for _ in range(num_groups)]
                writer_index += 1
            total_written += 1

    # write the last file if there are examples in data_buffer
    if example_index > 0:
        fp = os.path.join(output_dir, f"{name}-{writer_index}.h5")
        for i in range(num_groups):
            file_samples = np.stack(data_buffer[i], axis=0)
            chunk_size = (1, *file_samples.shape[1:])
            write_hdf5_file(
                dataset_name="data" if num_groups == 1 else f"data_{i}",
                file_path=fp,
                data=file_samples,
                n_examples=example_index,
                chunks=chunk_size,
                dtype=dtype,
                compression=compression,
            )

    params = {}
    params["n_examples"] = total_written
    # For backward compatibility
    if num_groups == 1:
        params[f"features"] = list(feature_groups.values())[0]
        params["features_len"] = len(params[f"features"])
        params["max_sequence_length"] = list(feature_groups.keys())[0][0]
    else:
        for i, (group, features) in enumerate(feature_groups.items()):
            params[f"data_{i}_features"] = features
            params[f"data_{i}_features_len"] = len(features)
            params[f"data_{i}_max_sequence_length"] = list(group)[-1]
            params[f"data_{i}_shape"] = list(group)

    params["output_dir"] = output_dir
    params["dtype"] = dtype
    params["compression"] = compression
    params["samples_per_file"] = samples_per_file
    json_params_file = os.path.join(output_dir, "data_params.json")
    with open(json_params_file, 'w') as _fout:
        json.dump(params, _fout, indent=4)

    print(f"Done! Wrote total of {total_written} examples.")
