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

"""Specifies `SamplesSaver` class for writing numpy array datasets."""

import logging
import os
import shutil
import sys
from typing import List, Optional, Tuple

import numpy as np


class SamplesSaver:
    """Manages data samples chunking and saving for numpy arrays."""

    def __init__(
        self,
        data_dir: str,
        max_file_size: int,
        filename_prefix: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
    ):
        """Constructs a `SamplesSaver` instance.

        Args:
            data_dir: Path to mounted dir where the samples are dumped
            max_file_size: Maximum file size (in bytes) for the .npy samples file(s)
            filename_prefix: (Optional) filename prefix for the .npy file(s)
            dtype: (Optional) numpy dtype for the array. If unspecified, the dtype is np.int32
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.max_file_size = max_file_size
        self.filename_prefix = (
            filename_prefix if filename_prefix is not None else "samples"
        )
        self.dtype = dtype if dtype is not None else np.int32

        self._data_samples = []
        self._samples_file_list = []
        self._samples_len = 0
        self._chunk_idx = 0
        self._samples_sizeb = 0

    @property
    def dataset_size(self) -> int:
        """Returns the total number of data samples."""
        return self._samples_len

    @property
    def samples_files(self) -> List[Tuple[str, int]]:
        """Returns the list of `.npy` file(s)."""
        return self._samples_file_list

    def add_sample(self, data_sample: np.array) -> None:
        """Adds the np array to internally maintained list of data
        samples and dumps these to file if the total size exceeds
        `max_file_size` threshold.

        Args:
            data_sample: np array data sample
        """
        self._samples_sizeb += sys.getsizeof(data_sample)
        # If total size of samples (in bytes) exceeds threshold,
        # write the chunk to file to prevent possible OOM issues
        # with holding a very large dataset
        if self._samples_sizeb >= self.max_file_size:
            self._dump_samples_file()

        self._data_samples.append(data_sample)
        self._samples_len += 1

    def _dump_samples_file(self) -> None:
        """Saves the data samples in the current chunk as a `.npy` file
        and resets attributes for writing the next chunk.
        """
        if not self._data_samples:  # Skip dumping if previously chunked
            return

        samples_file_path = os.path.join(
            self.data_dir, f"{self.filename_prefix}_{self._chunk_idx}.npy"
        )
        try:
            with open(samples_file_path, 'wb') as f:
                np.save(f, np.array(self._data_samples, dtype=self.dtype))
        except Exception as e:
            raise RuntimeError(
                f"Failed to save samples in file {samples_file_path} "
                f"due to error: {e}"
            ) from e

        # Save file path and number of samples in the file
        self._samples_file_list.append(
            (samples_file_path, len(self._data_samples))
        )

        # Increment chunk no. and reset samples' list and size attrs
        self._chunk_idx += 1
        self._data_samples = []
        self._samples_sizeb = 0

    def flush(self):
        """Dumps any remaining data samples not yet written to file."""
        if self._data_samples:
            self._dump_samples_file()

    def delete_data_dumps(self) -> None:
        """Cleans up by deleting all dumped data."""
        try:
            if os.path.exists(self.data_dir):
                shutil.rmtree(self.data_dir)
        except Exception as e:  # pylint: disable=broad-except
            logging.error(
                f"Failed to delete samples data directory due to: {e}"
            )
