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

import atexit
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import h5py
import numpy as np


class H5Reader:
    """Class for reading individual sequences from HDF5 files stored on disk.

    Supports 2 formats of data on disk:
        1. rank-1 tensor of concatenated tokenized documents.
        2. rank > 1 tensor of preprocessed samples where the 0th index of the
           data on disk indexes the data by sample.
    """

    def __init__(
        self,
        data_dirs: Union[str, List[str]],
        extra_data_keys: Optional[List[str]] = None,
        sequence_length: Optional[int] = None,
        read_extra_token: bool = False,
        data_subset: Optional[str] = None,
        sort: bool = True,
        use_vsl: bool = False,
    ):
        """Creates a reader for an HDF5 corpus.

        Args:
            data_dirs: Directories containing h5 files to read from.
            extra_data_keys: Additional HDF5 keys containing data to read from.
            sequence_length: The number of tokens per sample if reading
                from a corpus. Must be `None` if the data has already been
                preprocessed into samples.
            read_extra_token: Whether to read and return one extra token
                after the end of the sequence. This can be useful for language
                modeling tasks where you want to construct the labels as an
                shifted version of the inputs. Setting this to `True` differs
                from increasing `sequence_length` by one in that the extra token
                returned due to this flag will be included in some other
                sequence as the first token. Will be ignored if
                `sequence_length` is `None`.
            data_subset: A string specifying the subset of the corpus to
                consider. E.g. if `data_subset="0.0-0.75"` is specified, only
                samples in the first 3/4 of the dataset will be considered and
                the last 1/4 of the dataset will be completely untouched. The
                self reported length will be the length of the valid portion
                of the dataset (e.g. the first 3/4), and any attempt to access
                an element beyond this length will result in an exception.
            sort: Whether to sort the file paths after reading them. This flag
                is included for backwards compatibility and should almost always
                be set to `True`. It will be removed in the future.
            use_vsl: Flag to enable variable sequence length training.
                It requires the dataset to have two extra features: the
                `attention_span` of keys and the `position_ids` of tokens.
        """
        data_keys = ["data"]
        if extra_data_keys:
            data_keys += extra_data_keys

        files = []
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]
        for data_dir in data_dirs:
            p = Path(data_dir)
            if not p.is_dir():
                raise ValueError(
                    f"The path {p} does not exist or is not a directory. "
                    f"Please specify a valid directory containing h5 files "
                    f"and ensure that the directory is mounted."
                )
            files.extend(p.glob("*.h5"))
        if not files:
            raise ValueError(
                f"No *.h5 files found in specified data directories: "
                f"{data_dirs}."
            )

        if sort:
            files.sort()

        by_sample = False
        with h5py.File(files[0], "r") as f:
            data_shape = f["data"].shape
        if sequence_length is None:
            if len(data_shape) < 2:
                raise ValueError(
                    "If you don't specify `sequence_length`, then the data "
                    "being read must be preprocessed by sample, but the data "
                    f"written to {files[0]} has rank 1"
                )
            by_sample = True
        elif len(data_shape) > 1:
            if sequence_length is not None and sequence_length != data_shape[1]:
                raise ValueError(
                    "If loading data that has been preprocessed into sequences "
                    "the sequence length provided must either be None or match "
                    "dimension 1 of the data on disk. Got sequence length "
                    f"{sequence_length}, but the shape of the data in "
                    f"{files[0]} is {data_shape}"
                )
            by_sample = True

        if by_sample and use_vsl and data_shape[1] not in [4, 5]:
            raise ValueError(
                f"Expected all dataset H5 files to have [4-5] features for "
                f"variable sequence length training, but got "
                f"{data_shape[1]} features in {files[0]}."
            )

        if extra_data_keys and not by_sample:
            raise ValueError(
                f"Reading extra data keys is not currently supported for "
                "'corpus' data format. Please switch to 'sample' format."
            )

        if by_sample:
            self._impl = _SequencedH5Reader(
                files, data_keys=data_keys, data_subset=data_subset
            )
        else:
            self._impl = _CorpusH5Reader(
                files,
                sequence_length=sequence_length,
                read_extra_token=read_extra_token,
                data_subset=data_subset,
            )

    @property
    def by_sample(self) -> bool:
        return isinstance(self._impl, _SequencedH5Reader)

    def __getitem__(self, i: int) -> np.ndarray:
        """Reads a single sequence of the dataset from disk.

        Args:
            i: The index of the item to return. Samples are indexed in
               order of file name (sorted alphabetically) then location within
               that file.
        Returns:
            The `i`th sample element of the corpus, i.e. a numpy array of shape
            `(sequence_length + 1, )` if `read_extra_token` is `True` or of
            shape `(sequence_length, )` otherwise. The dtype of the returned
            array is `np.int32` regardless of how the data was written to disk.
        """
        return self._impl[i]

    def __len__(self) -> int:
        """Returns total number of sequences in the dataset."""
        return len(self._impl)

    @property
    def vdataset(self):
        v = getattr(self._impl, "vdataset", None)
        if v is None:
            raise AttributeError(
                "Trying to access virtual dataset attribute, but none was found"
            )
        return v

    @property
    def vdataset_full(self):
        v = getattr(self._impl, "vdataset_full", None)
        if v is None:
            raise AttributeError(
                "Trying to access virtual dataset attribute, but none was found"
            )
        return v


class _SequencedH5Reader:
    """Class for reading preprocessed samples from HDF5 files stored on disk."""

    def __init__(
        self,
        files: List[str],
        data_keys: List[str],
        data_subset: Optional[str] = None,
    ):
        """Creates an HDF5 reader for preprocessed sequences.

        Args:
            files: HDF5 files to read from.
            data_keys: HDF5 keys to read data from.
            data_subset: A string specifying the subset of the corpus to
                consider.
        """
        vsources: Dict[str, List[h5py.VirtualSource]] = {
            data_key: [] for data_key in data_keys
        }
        for filepath in files:
            with h5py.File(filepath, "r") as f:
                for data_key in data_keys:
                    dataset = f[data_key]
                    vsources[data_key].append(h5py.VirtualSource(dataset))

                    data_dtype = vsources[data_key][0].dtype
                    data_shape = vsources[data_key][0].shape
                    if data_dtype.kind != "S" and dataset.dtype != data_dtype:
                        raise ValueError(
                            f"Expected all dataset H5 files to have the same "
                            f"dtype, but got {data_dtype} in {files[0]} and "
                            f"{dataset.dtype} in {filepath}."
                        )

                    if dataset.shape[1:] != data_shape[1:]:
                        raise ValueError(
                            f"Expected all dataset H5 files to have the same "
                            f"shape beyond the first axis, but got "
                            f"{data_shape} in {files[0]} and {dataset.shape} "
                            f"in {filepath}."
                        )

        self._vdataset = {
            data_key: _VirtualDataset(vsource)
            for data_key, vsource in vsources.items()
        }
        self._num_sequences = len(self._vdataset["data"])

        if data_subset is not None:
            self._segmenter = _DatasetSegmenter(
                self._num_sequences, data_subset
            )
            self._num_sequences -= self._segmenter.num_skipped_sequences
        else:
            self._segmenter = None

    @property
    def vdataset(self):
        return self._vdataset["data"]

    @property
    def vdataset_full(self):
        return self._vdataset

    def __getitem__(self, i: int) -> np.ndarray:
        """Reads a single item of the dataset from disk."""
        if self._segmenter:
            i = self._segmenter.map_index(i)

        if len(self._vdataset) == 1:
            return self._vdataset["data"][i].astype(np.int32)
        else:
            data = {
                data_key: data[i] for data_key, data in self._vdataset.items()
            }
            data["data"] = data["data"].astype(np.int32)
            return data

    def __len__(self) -> int:
        """Returns total number of sequences in the dataset."""
        return self._num_sequences


class _CorpusH5Reader:
    """Class for reading samples from HDF5 corpus stored on disk."""

    def __init__(
        self,
        files: List[str],
        sequence_length: Optional[int] = None,
        read_extra_token: bool = False,
        data_subset: Optional[str] = None,
    ):
        """Creates an HDF5 reader for an HDF5 corpus.

        Args:
            files: HDF5 files to read from.
            sequence_length: The number of tokens per sample.
            read_extra_token: Whether to read and return one extra token
                after the end of the sequence.
            data_subset: A string specifying the subset of the corpus to
                consider.
        """
        vsources: List[h5py.VirtualSource] = []
        for idx, filepath in enumerate(files):
            with h5py.File(filepath, "r") as f:
                dataset = f["data"]

                if len(dataset.shape) != 1:
                    raise ValueError(
                        f"Expected all dataset H5 files in corpus format to "
                        f"have rank 1, but got rank {len(dataset.shape)} in "
                        f"{filepath}."
                    )

                if idx == 0:
                    data_dtype = dataset.dtype
                else:
                    if dataset.dtype != data_dtype:
                        raise ValueError(
                            f"Expected all dataset H5 files to have the same "
                            f"dtype, but got {data_dtype} in {files[0]} and "
                            f"{dataset.dtype} in {filepath}."
                        )

                vsources.append(h5py.VirtualSource(dataset))

        self._vdataset = _VirtualDataset(vsources)

        self._msl = sequence_length
        self._num_extra_tokens = 1 if read_extra_token else 0
        self._num_sequences = (
            len(self._vdataset) - self._num_extra_tokens
        ) // self._msl

        if data_subset is not None:
            self._segmenter = _DatasetSegmenter(
                self._num_sequences, data_subset
            )
            self._num_sequences -= self._segmenter.num_skipped_sequences
        else:
            self._segmenter = None

    @property
    def vdataset(self):
        return self._vdataset

    def __getitem__(self, i: int) -> np.ndarray:
        """Reads a single item of the dataset from disk."""
        if self._segmenter:
            i = self._segmenter.map_index(i)

        tok_idx = self._msl * i
        return self._vdataset[
            tok_idx : tok_idx + self._msl + self._num_extra_tokens
        ].astype(np.int32)

    def __len__(self) -> int:
        """Returns total number of sequences in the dataset."""
        return self._num_sequences


class _VirtualDataset:
    """Class that represents a virtual dataset over multiple HDF5 files."""

    def __init__(self, sources: List[h5py.VirtualSource]):
        """Constructs a virtual dataset from a list of virtual sources.

        Args:
            sources: A list of virtual sources to construct the dataset from.
                It is expected that all virtual sources have the same shape
                (except for the first axis) and dtype.
        """

        length = sum(s.shape[0] for s in sources)

        self._shape = (length, *sources[0].shape[1:])
        # Max for possibly different shaped virtual sources, eg file path strings
        self._dtype = max(source.dtype for source in sources)

        layout = h5py.VirtualLayout(shape=self._shape, dtype=self._dtype)

        start = 0
        for vsource in sources:
            end = start + vsource.shape[0]
            layout[start:end:1, ...] = vsource
            start = end

        self._dataset_tmpfile = tempfile.NamedTemporaryFile(
            "w", prefix="virtual_dataset", suffix=".h5"
        )
        with h5py.File(self._dataset_tmpfile.name, "w", libver="latest") as f:
            f.create_virtual_dataset("data", layout)

        self.__dataset_file = None
        self.__dataset = None

    @property
    def _dataset(self) -> h5py.Dataset:
        """Returns the underlying dataset.

        The underlying dataset is lazily loaded from disk the first time it is
        accessed. This is to avoid loading the dataset then forking when using
        multiprocessing. The loaded dataset is then cached to avoid reloading
        the dataset on every access, which has a high overhead. This is safe
        because the dataset is opened in read-only mode and is not expected
        to be modified while this object is alive.
        """
        if self.__dataset is None:
            self.__dataset_file = h5py.File(self._dataset_tmpfile.name, "r")
            self.__dataset = self.__dataset_file["data"]

            # h5py >= 3.4 hits a segfault on exit deep within hdf5 libraries if the dataset isn't
            # freed up before the file is closed and hdf5 atexit handlers run. Just clearing
            # `self.__dataset` fixes the segfault, but while we're at it, let's also manually close
            # the file.
            @atexit.register
            def _close_at_exit():
                self.__dataset = None
                if self.__dataset_file is not None:
                    self.__dataset_file.close()

        return self.__dataset

    def __getitem__(self, i) -> np.ndarray:
        """Returns the `i`th element of the dataset."""
        return self._dataset[i]

    def __len__(self):
        """Returns the length of the dataset."""
        return self._dataset.shape[0]

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype


class _DatasetSegmenter:
    def __init__(self, num_sequences: int, data_subset: str):
        offsets_full_dataset = []
        offsets_skipped_dataset = []

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
        n = num_sequences
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
            offsets_full_dataset.append(int(n * end) - int(n * start))
            offsets_skipped_dataset.append(int(n * start) - int(n * prev_end))
            prev_end = end

        self._offsets_skipped_dataset = np.cumsum(offsets_skipped_dataset)
        self._offsets_full_dataset = np.cumsum(offsets_full_dataset)

    @property
    def num_skipped_sequences(self) -> int:
        return self._offsets_skipped_dataset[-1]

    def map_index(self, i):
        if len(self._offsets_full_dataset):
            chunk_idx = self._offsets_full_dataset.searchsorted(i, side="right")
            i += self._offsets_skipped_dataset[chunk_idx]
        return i


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

    def __init__(
        self,
        datasets: List[H5Reader],
        weights: List[int],
        interleave: bool = False,
        seed: int = 0,
    ):
        self.interleave = interleave

        self._by_sample = all(d.by_sample for d in datasets)
        if not self._by_sample and any(d.by_sample for d in datasets):
            raise ValueError(
                "Datasets given to a Mixture must either all read data by "
                "sample or all read data by slicing a corpus, but got datasets "
                "that use a mixture"
            )
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

        # Normalize the weights to sum up to 1
        s = sum(weights)
        weights = [w / s for w in weights]

        # 1 epoch of a mixture is defined to be the number of samples required
        # to see every sample at least once in each sub-dataset with mixture
        # weight > 5%. Note that this means that some samples will be seen
        # multiple times in each epoch.
        total_samples = max(
            len(d) / w for (d, w) in zip(datasets, weights) if w > 0.05
        )

        # Find sample index boundaries of each dataset
        self.boundaries = np.cumsum([int(total_samples * w) for w in weights])
        self.boundaries = np.insert(self.boundaries, 0, 0)

        if self.interleave:
            size = len(self)
            dtype = np.min_scalar_type(size)
            self.indices = np.arange(size, dtype=dtype)
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)

            # we want samples within a dataset to appear in order to take
            # advantage of sequential read patterns, so we sort the
            # sub-components after the shuffle
            for start, end in zip(self.boundaries[:-1], self.boundaries[1:]):
                self.indices[
                    np.where((start <= self.indices) & (self.indices < end))
                ] = np.arange(start, end, dtype=dtype)

    @property
    def by_sample(self):
        return self._by_sample

    def __getitem__(self, i):
        if i < 0 or i >= len(self):
            raise RuntimeError(
                f"Index ({i} is out of bounds for mixture with length {len(self)}"
            )

        idx = self.indices[i] if self.interleave else i
        dataset_idx = np.searchsorted(self.boundaries, idx, side="right") - 1
        dataset = self.datasets[dataset_idx]
        offset = self.boundaries[dataset_idx]
        sample_index = (idx - offset) % len(dataset)
        return dataset[sample_index]

    def __len__(self):
        return self.boundaries[-1]
