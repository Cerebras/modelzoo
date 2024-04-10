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

"""Specifies `SamplesSaver` and `SamplesViewer` classes for writing and reading numpy array datasets."""
import logging
import os
import shutil
import sys
from typing import List, Optional

import numpy as np


class SamplesViewer:
    """Handles iteraing over data samples of numpy arrays from .npy files."""

    def __init__(self, samples_file_list: List[str]):
        if not samples_file_list:
            raise RuntimeError(
                "No samples files to load. Please provide a list of "
                "valid paths to .npy files to load data samples from "
                "when initializing this class."
            )
        self._samples_file_list = samples_file_list

    def __iter__(self):
        for samples_file in self._samples_file_list:
            try:
                with open(samples_file, 'rb') as f:
                    samples = np.load(f)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to read requests file: {samples_file}"
                ) from e

            for sample in samples:
                yield sample


class SamplesSaver:
    """Manages data samples chunking and saving for numpy arrays."""

    def __init__(
        self,
        data_dir: str,
        max_file_size: int,
        filename_prefix: Optional[str] = None,
    ):
        """
        Args:
            data_dir: Path to mounted dir where the samples are dumped
            max_file_size: Maximum file size (in bytes) for the .npy samples file(s)
            filename_prefix: (Optional) filename prefix for the .npy file(s)
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        self.max_file_size = max_file_size
        self.filename_prefix = (
            filename_prefix if filename_prefix is not None else "samples"
        )

        self._data_samples = []
        self._samples_file_list = []
        self._samples_len = 0
        self._chunk_idx = 0
        self._samples_sizeb = 0

    @property
    def dataset_size(self) -> int:
        """Returns the total numer of data samples."""
        return self._samples_len

    @property
    def samples_files(self) -> List[str]:
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
                np.save(f, np.array(self._data_samples))
        except Exception as e:
            raise RuntimeError(
                f"Failed to save samples in file {samples_file_path} "
                f"due to error: {e}"
            ) from e

        # Add filename to list of files
        self._samples_file_list.append(samples_file_path)

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
            shutil.rmtree(self.data_dir)
        except Exception as e:
            logging.error(
                f"Failed to delete samples data directory due to: {e}"
            )
