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

import numpy as np

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.data.common.h5_map_dataset import (
    HDF5Dataset,
    RestartableDataLoader,
)


@registry.register_datasetprocessor("GptHDF5MapDataProcessor")
class GptHDF5MapDataProcessor:
    """
    A map style dataset for GPT style models.

    Supports data saved on disk in either of the following formats:
        - `(num_tokens,)`, i.e. a set of documents tokenized and concatenated.
            We refer to this as the 'corpus' format in what follows.
        - `(num_sequences, 3, sequence_length)`, i.e. data that has already
            been preprocessed into sequences. We refer to this as the
            'sample' format in what follows.

    Args:
        params (dict): a dictionary containing the following fields:
            - "data_dir" (str or list[str]): the path to the HDF5 files.
                Exactly one of "data_dir" or "mixture" must be specified.
            - "batch_size" (int): batch size
            - "shuffle" (bool): whether or not to shuffle the dataset. Defaults
                to `False`
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
            - "num_samples" (int): the number of samples to shuffle over (if
                shuffling is enabled). In multi-epoch training, it is common to
                set this to the total number of samples that you plan to train
                on so that epochs are not sequential but instead shuffled
                together for potentially improved convergence.
            - "num_workers" (int): the number of PyTorch processes used in the
                dataloader. Defaults to 0.
            - "prefetch_factor" (int): the number of batches to prefetch in the
                dataloader. Defaults to 10.
            - "persistent_workers" (bool): whether or not to keep workers
                persistent between epochs. Defaults to True.
            - "sort_files" (bool): whether or not the reader should sort the input
                files. This is included for backwards compatibility and should
                almost always be set to `True`.
            - "use_vsl" (bool): Flag to enable variable sequence length training.
                It requires the dataset to have two extra features: the
                `attention_span` of keys and the `position_ids` of tokens.
                Defaults to `False`.
    """

    def __init__(self, params):
        # Note: attention_mask is a misnomer and serves as a loss mask in the
        # model itself. This naming will change in 2.0.
        self.dataset = HDF5Dataset(params)

        use_vsl = params.get("use_vsl", False)
        features_list = ["input_ids", "attention_mask", "labels"]
        if use_vsl:
            if self.dataset.by_sample:
                features_list.extend(["attention_span", "position_ids"])
            else:
                raise NotImplementedError(
                    "Variable sequence length (VSL) training is not "
                    "currently supported with 'corpus' format data. Please "
                    "switch to 'sample' format data to use VSL."
                )

        if "dataset_map_fn" in params:
            self.dataset.map(params["dataset_map_fn"])
        elif self.dataset.by_sample:
            self.dataset.map(
                lambda x: {
                    feature: x[idx] for idx, feature in enumerate(features_list)
                }
            )
        else:
            self.dataset.map(
                lambda x: {
                    "input_ids": x[:-1],
                    "labels": x[1:],
                    "attention_mask": np.ones_like(x[:-1]),
                }
            )
        self.num_workers = params.get("num_workers", 0)
        self.prefetch_factor = params.get("prefetch_factor", 10)
        self.persistent_workers = params.get("persistent_workers", True)
        if not self.num_workers:
            self.prefetch_factor = None  # the default value in DataLoader
            self.persistent_workers = False

    def create_dataloader(self):
        return RestartableDataLoader(
            self.dataset,
            batch_sampler=self.dataset.sampler,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
