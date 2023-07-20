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

import contextlib
from pathlib import Path

import h5py
import numpy as np


@contextlib.contextmanager
def trivial_context_manager(f):
    yield f


class H5Reader:
    """
    An abstraction for reading individual sequences from h5 files on disk.

    Supports 2 formats of data on disk. The first is a rank-1 tensor of
    concatenated tokenized documents. The second is a rank > 1 tensor of
    preprocessed samples where the 0th index of the data on disk indexes the
    data by sample.
    """

    _MAX_OPEN_FILES = 35

    def __init__(
        self,
        data_dirs,
        sequence_length=None,
        read_extra_token=False,
        data_subset=None,
    ):
        """
        Creates a reader for an h5 corpus

        Args:
            data_dirs (list[str]): Directories containing h5 files to read from
            sequence_length (int): The number of tokens per sample if reading
                from a corpus. Must be `None` if the data has already been
                preprocessed into samples.
            read_extra_token (bool): Whether to read and return one extra token
                after the end of the sequence. This can be useful for language
                modeling tasks where you want to construct the labels as an
                shifted version of the inputs. Setting this to `True` differs
                from increasing `sequence_length` by one in that the extra token
                returned due to this flag will be included in some other
                sequence as the first token. Will be ignored if
                `sequence_length` is `None`.
            data_subset (str): A string specifying the subset of the corpus to
                consider. E.g. if `data_subset="0.0-0.75"` is specified, only
                samples in the first 3/4 of the dataset will be considered and
                the last 1/4 of the dataset will be completely untouched. The
                self reported length will be the length of the valid portion
                of the dataset (e.g. the first 3/4), and any attempt to access
                an element beyond this length will result in an exception.
        """
        self.msl = sequence_length
        self._num_extra_tokens = 1 if read_extra_token else 0

        self._files = []
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]
        for data_dir in data_dirs:
            p = Path(data_dir)
            if not p.is_dir():
                raise ValueError(
                    f"The path {p} does not exist or is not a directory"
                )
            self._files.extend(p.glob("*.h5"))

        self._by_sample = False
        with h5py.File(self._files[0], "r") as f:
            data_shape = f["data"].shape
        if self.msl is None:
            if len(data_shape) < 2:
                raise ValueError(
                    "If you don't specify `sequence_length`, then the data "
                    "being read must be preprocessed by sample, but the data "
                    f"written to {self._files[0]} has rank 1"
                )
            self._by_sample = True
        elif len(data_shape) > 1:
            if self.msl is not None:
                raise ValueError(
                    "If loading data that has been preprocessed into sequences "
                    "the sequence length provided must either be None or match "
                    "dimension 1 of the data on disk. Got sequence length "
                    f"{self.msl}, but the shape of the data in {self._files[0]}"
                    f" is {data_shape}"
                )
            self._by_sample = True

        sequence_counts = []
        for f in self._files:
            with h5py.File(f, "r") as f:
                if self._by_sample:
                    sequence_counts.append(len(f["data"]))
                else:
                    sequence_counts.append(
                        (len(f["data"]) - self._num_extra_tokens) // self.msl
                    )
        self._num_sequences = sum(sequence_counts)
        self._file_end_counts = np.cumsum(sequence_counts)
        self._file_start_indices = np.insert(self._file_end_counts, 0, 0)[:-1]

        if self._by_sample or len(self._files) > self._MAX_OPEN_FILES:
            self.keep_files_open = False
        else:
            self.keep_files_open = True
            self._files = [h5py.File(f, "r") for f in self._files]
        if self._by_sample:
            self.prev_file_descriptor = None
            self.prev_file_name = None

        # handle data subsetting
        self.offsets_full_dataset = []
        self.offsets_skipped_dataset = []
        if data_subset is not None:
            try:
                segments = [
                    (float(seg.split("-")[0]), float(seg.split("-")[1]))
                    for seg in data_subset.strip().split(",")
                ]
            except Exception as e:
                raise RuntimeError(
                    f"There was a problem parsing data subset {data_subset}. "
                    "data_subset must be a string of comma separated ranges of "
                    "floats, for example '0.0-0.2,0.5-0.7'"
                ) from e
            prev_end = 0
            segments = [(0, 0)] + segments + [(1, 1)]
            n = self._num_sequences
            for start, end in segments:
                if start < 0:
                    raise ValueError(
                        f"data_subset must contain only non-negative bounds. "
                        f"Got {data_subset} which contains {start}."
                    )
                if end < start:
                    raise ValueError(
                        f"the end of each range in data_subset must be at "
                        f"least as large as the start of the range, but "
                        f"start={start} and end={end} are present in provided "
                        f"data subset {data_subset}"
                    )
                if end > 1:
                    raise ValueError(
                        f"data_subset can only contain ranges which are subsets"
                        f" of the range [0, 1], but found end={end} in "
                        f"data_subset {data_subset}"
                    )
                if start < prev_end:
                    raise ValueError(
                        f"ranges in data_subset must be monotonically "
                        f"increasing. Got {data_subset}"
                    )
                self.offsets_skipped_dataset.append(
                    int(n * end) - int(n * start)
                )
                self.offsets_full_dataset.append(
                    int(n * start) - int(n * prev_end)
                )
                prev_end = end
            self.offsets_skipped_dataset = np.cumsum(
                self.offsets_skipped_dataset
            )
            self.offsets_full_dataset = np.cumsum(self.offsets_full_dataset)
            self._num_sequences -= self.offsets_full_dataset[-1]

    @property
    def by_sample(self):
        return self._by_sample

    def _maybe_open_file(self, f):
        if self._by_sample:
            # if we're loading data by sample, we expect the common access
            # pattern to be sequential as data should be shuffled during
            # pre-processing, so we keep the current file open until a new
            # file is requested so that we can save time on file opens
            if f == self.prev_file_name:
                return trivial_context_manager(self.prev_file_descriptor)
            else:
                if self.prev_file_descriptor is not None:
                    self.prev_file_descriptor.close()
                self.prev_file_name = f
                self.prev_file_descriptor = h5py.File(f, "r")
                return trivial_context_manager(self.prev_file_descriptor)
        elif self.keep_files_open:
            return trivial_context_manager(f)
        return h5py.File(f, "r")

    def __getitem__(self, i):
        """
        Reads a single item of the dataset from disk

        Args:
            i (int): The index of the item to return. Samples are indexed in
                order of file name (sorted alphabetically) then location within
                that file.
        Returns:
            The `i`th sample element of the corpus, i.e. a numpy array of shape
            `(sequence_length + 1, )` if `read_extra_token` is `True` or of
            shape `(sequence_length, )` otherwise. The dtype of the returned
            array is `np.int32` regardless of how the data was written to disk.
        """
        if len(self.offsets_full_dataset):
            chunk_idx = self.offsets_full_dataset.searchsorted(i, side="right")
            i += self.offsets_skipped_dataset[chunk_idx]
        file_index = np.searchsorted(self._file_end_counts, i, side="right")
        f = self._files[file_index]
        with self._maybe_open_file(f) as f:
            sequence_index = i - self._file_start_indices[file_index]
            if self._by_sample:
                x = f["data"][sequence_index]
            else:
                tok_index = self.msl * sequence_index
                x = f["data"][
                    tok_index : tok_index + self.msl + self._num_extra_tokens
                ]
        return x.astype(np.int32)

    def __len__(self):
        return self._num_sequences

    def __del__(self):
        # not necessary for most python runtimes, but good for completeness
        if self._by_sample and self.prev_file_descriptor is not None:
            self.prev_file_descriptor.close()
        elif self.keep_files_open:
            for f in self._files:
                f.close()


class Mixture:
    """
    Mix several map-style datasets according to provided weights.

    Args:
        datasets: a list of objects implementing `__len__` and `__getitem__`
        weights: a list of weights associated with each dataset. `weights`
            must have the same length as `datasets` and contain only nonnegative
            values. All weights will be normalized to sum to 1.
        interleave: whether or not samples of different datasets should be
            interleaved together. If all the datasets are preprocessed into
            sequences and shuffled before being written to disk, then setting
            this flag will allow you to avoid doing any shuffling at run time
            while still having samples from the different datasets intermingled,
            which may be desirable for enabling sequential disk reads. This is
            implemented in a way that samples within a dataset are not shuffled
            in relation to each other, i.e. sample 0 of dataset 0 will always
            have a smaller index than sample 1 of dataset 0.
        seed: the random seed used for interleaving. Ignored if `interleave`
            is `False`.
    """

    def __init__(self, datasets, weights, interleave=False, seed=0):
        self.interleave = interleave

        self._by_sample = all(d.by_sample for d in datasets)
        if not self._by_sample and any(d.by_sample for d in datasets):
            raise ValueError(
                "Datasets given to a Mixture must either all read data by "
                "sample or all read data by slicing a corpus, but got datasets "
                "that use a mixture"
            )
        num_datasets = len(datasets)
        if len(weights) != len(datasets):
            raise ValueError(
                f"weights must have same length as datasets, got {weights}"
            )
        if any(w < 0 for w in weights):
            raise ValueError(f"weights must be nonnegative, got {weights}")
        if all(w == 0 for w in weights):
            raise ValueError(
                f"at least one weight must be greater than 0, got {weights}"
            )

        self.datasets = []
        new_weights = []
        for d, w in zip(datasets, weights):
            if w > 0:
                self.datasets.append(d)
                new_weights.append(w)
        weights = new_weights

        s = sum(weights)
        weights = [w / s for w in weights]

        # 1 epoch of a mixture is defined to be the number of samples required
        # to see every sample in each sub-dataset of weight at least 5% at least
        # once. Note that this means that some samples will be seen multiple
        # times in each epoch
        total_samples = max(
            len(d) / w for (d, w) in zip(datasets, weights) if w > 0.05
        )
        if self.interleave:
            self.dataset_indices = [
                np.full(int(total_samples * w), i, dtype=np.uint16)
                for i, w in enumerate(weights)
            ]
            self.dataset_samples = [
                np.arange(int(total_samples * w)) % len(d)
                for d, w in zip(self.datasets, weights)
            ]
            self.dataset_indices = np.concatenate(self.dataset_indices)
            self.dataset_samples = np.concatenate(self.dataset_samples)
            self.total_samples = len(self.dataset_indices)
            indices = np.arange(self.total_samples)
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
            # we want samples within a dataset to appear in order to take
            # advantage of sequential read patterns, so we sort the
            # sub-components after the shuffle
            boundaries = [int(total_samples * w) for w in weights]
            boundaries = np.insert(np.cumsum(boundaries), 0, 0)
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                indices[
                    np.where((start <= indices) & (indices < end))
                ] = np.arange(start, end)
            self.dataset_indices = self.dataset_indices[indices]
            self.dataset_samples = self.dataset_samples[indices]
        else:
            self.boundaries = [int(total_samples * w) for w in weights]
            self.boundaries = np.cumsum(self.boundaries)
            self.total_samples = self.boundaries[-1]
            self.boundaries = self.boundaries[:-1]

    @property
    def by_sample(self):
        return self._by_sample

    def __getitem__(self, i):
        if self.interleave:
            dataset = self.datasets[self.dataset_indices[i]]
            return dataset[self.dataset_samples[i]]
        else:
            dataset_index = np.searchsorted(self.boundaries, i, side="right")
            dataset = self.datasets[dataset_index]
            offset = self.boundaries[dataset_index - 1] if dataset_index else 0
            sample_index = (i - offset) % len(dataset)
            return dataset[sample_index]

    def __len__(self):
        return self.total_samples
