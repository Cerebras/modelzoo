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

import glob
import logging
import os

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.vision.pytorch.input.utils import create_worker_cache

from .readers import H5Reader, Mixture
from .samplers import CBSampler


class HDF5Dataset(torch.utils.data.Dataset):
    """
    Dynamically read samples from disk for using mapping paradigms.

    It supports two different data formats on disk. The first is data stored
    in an H5 file in the shape `(num_tokens,)`, i.e. a series of documents
    tokenized and concatenated together. We call this format the 'corpus' format
    The second format is H5 data of shape `(num_sequences, ...)`, i.e. data has
    already been tokenized and split into sequences. We call this format the
    'sample' format.

    The corpus format supports flexible choice of MSL backed by a single copy of
    the data on disk. Both formats support deterministic restart, and a data
    order that is independent of the configuration of the cluster you are
    running on. I.e. you can pause a run, increase or decrease the number of
    systems you are running on, and restart the run with no change in data
    order.

    When used in combination with shuffling, this implementation relies on
    random access reads to disk to dynamically split samples into sequences
    and shuffle. Users with unusually slow storage should look out for data
    loading bottlenecks and might consider using `use_worker_cache=True` if
    disk access is indeed a bottleneck.

    Args:
        params (dict): a dictionary containing the following fields:
            - "data_dir" (str or list[str]): the path to the HDF5 files.
                Exactly one of "data_dir" or "mixture" must be specified.
            - "batch_size" (int): batch size
            - "shuffle" (bool): whether or not to shuffle the dataset. Defaults
                to `True`
            - "shuffle_seed" (int): seed used for deterministic shuffling.
                Defaults to 0.
            - "use_worker_cache" (bool): whether or not to copy data to storage
                that is directly attached to each individual worker node.
                Useful when your network storage is unusually slow, but
                otherwise discouraged.
            - "max_sequence_length" (int): the sequence length of samples
                produced by the dataloader. When using the 'corpus' data format,
                the same preprocessed data will work with any max sequence
                length, so this may be set at runtime. When using the 'sample'
                format this must be set to `None`.
            - "data_subset" (str): an optional specification to only consider a
                subset of the full dataset, useful for sequence length
                scheduling and multi-epoch testing. Expected to be a comma
                separated list of ranges, e.g. '0.0-0.5' or '0.1-0.3,0.7-1.0'.
                Specifying '0.0-0.5' creates a dataset from the first half of
                the data on disk and disregards the second half.
            - "mixture" list[dict]: an optional specification of multiple
                datasets to mix over to create one single weighted combination.
                Each element must be a dictionary containing keys `data_dir`
                and `weight`. `data_dir` serves the same purpose as mentioned
                above. `weight` defines the probability with which this dataset
                should be sampled from. Weights are normalized to sum to 1.
                Optionally, the dictionary may also contain a `data_subset`
                field which functions the same as the `data_subset` argument
                above.
            - "drop_last" (bool): similar to the PyTorch drop_last setting
                except that samples that when set to `True`, samples that would
                have been dropped at the end of one epoch are yielded at the
                start of the next epoch so that there is no data loss. This is
                necessary for a data ordering that is independent of the
                distributed setup being used.
    """

    def __init__(self, params):
        self.use_worker_cache = params.get("use_worker_cache", False)
        self.msl = params.get("max_sequence_length", None)
        shuffle = params.get("shuffle", True)
        seed = params.get("shuffle_seed", 0)
        data_dir = params.get("data_dir", None)
        mixture_params = params.get("mixture", None)
        batch_size = params["batch_size"]
        drop_last = params.get("drop_last", True)

        if data_dir and mixture_params:
            raise ValueError(
                "you can't specify `data_dir` and `mixture` at the same time"
            )
        if data_dir is not None:
            self.reader = self._set_up_reader(
                data_dir, params.get("data_subset", None)
            )
        else:
            self.reader = Mixture(
                [
                    self._set_up_reader(
                        x["data_dir"], x.get("data_subset", None)
                    )
                    for x in mixture_params
                ],
                [x["weight"] for x in mixture_params],
                interleave=not shuffle,
                seed=seed,
            )

        start_index = self._get_global_samples_from_ckpt(
            params.get("cerebras", {})
        )
        logging.info(f"Starting dataloader from sample {start_index}")
        self.sampler = CBSampler(
            self,
            shuffle=shuffle,
            seed=seed,
            start_index=start_index,
            shard=True,
            batch_size=batch_size,
            drop_last=drop_last,
        )

        self.map_fn = None

    @property
    def by_sample(self):
        return self.reader.by_sample

    def map(self, fn):
        if self.map_fn is not None:
            raise ValueError(
                f"You may only apply one map function to a H5MapDataset"
            )
        self.map_fn = fn

    def _set_up_reader(self, data_dir, subset):
        if not isinstance(data_dir, list):
            data_dir = [data_dir]
        if self.use_worker_cache and cm.use_cs() and cm.is_streamer():
            if not cm.is_appliance():
                raise RuntimeError(
                    "use_worker_cache not supported for non-appliance runs"
                )
            else:
                data_dir = [create_worker_cache(d) for d in data_dir]

        reader = H5Reader(data_dir, self.msl, True, subset)
        return reader

    def _get_global_samples_from_ckpt(self, dataloader_state):
        base_path = dataloader_state.get("save_iter_state_path", None)
        if base_path is None:
            return 0
        global_state_path = os.path.join(
            base_path, "data_iter_checkpoint_state_file_global"
        )
        if not os.path.exists(global_state_path):
            return 0
        try:
            with open(global_state_path, "r") as f:
                contents = f.readline()
        except Exception as e:
            raise RuntimeError(
                f"Could not access dataloader checkpoint file "
                f"{global_state_path}"
            ) from e
        try:
            global_step = int(contents)
        except Exception as e:
            raise RuntimeError(
                "Invalid contents of global dataloader checkpoint file. "
                f"File {global_state_path} has first line {contents}, but the "
                "first line of the global checkpoint file should be an integer "
                "representing the global step of the previous checkpoint."
            ) from e

        worker_ckpts = glob.glob(
            os.path.join(
                base_path,
                f"data_iter_state_file_worker_*_step_{global_step}.txt",
            )
        )
        total_samples = 0
        # TODO(william): improve error messages for the following code
        for ckpt_path in worker_ckpts:
            with open(ckpt_path, "r") as f:
                total_samples += int(f.readline())

        return total_samples

    def __getitem__(self, i):
        x = self.reader[i]
        if self.map_fn is not None:
            return self.map_fn(x)
        return x

    def __len__(self):
        return len(self.reader)
