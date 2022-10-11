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
Processors for handling TF records data format for BERT
"""
import os
from abc import ABC, abstractmethod

import tensorflow as tf

from modelzoo.common.tf.input.utils import transform_dataset
from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset


class TfRecordsProcessor(ABC):
    """
    Creates dataset from pre-compiled TF records using the map_fn provided in
    the child class.

    All files in the data_dir(s) matching '*.tfrecord*' will be read.

    :param data_dir: a path or list of paths for data to be gathered from
    :param int batch_size: batch size for the dataset
    :param bool shuffle: whether the data should be shuffled
    :param int shuffle_seed: seed to use for shuffling
    :param int shuffle_buffer: buffer size for call to tf.data.Dataset.shuffle
    :param bool repeat: whether the dataset should be repeated
    :param bool use_multiple_workers: if True, dataset will be sharded
    :param int n_parallel_reads: for call to tf.data.Dataset.interleave
    :param bool map_before_batch: if True, mapping will happen before batching.
    :param int skip_steps: Number of steps to skip the dataset after batching.
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        shuffle_seed=None,
        shuffle_buffer=None,
        repeat=True,
        use_multiple_workers=False,
        n_parallel_reads=4,
        map_before_batch=False,
        skip_steps=0,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.shuffle_buffer = shuffle_buffer
        self.repeat = repeat
        self.use_multiple_workers = use_multiple_workers
        self.n_parallel_reads = n_parallel_reads
        self.map_before_batch = map_before_batch
        self.skip_steps = skip_steps

        assert self.batch_size > 0, "Batch size should be positive."

    @abstractmethod
    def map_fn(self, raw_record):
        """
        Parses a batch of serialized examples into data formatted correctly for
        the model being used.
        """
        raise NotImplementedError("map_fn must be implemented in child class")

    def batch_fn(self, dataset):
        return dataset.batch(self.batch_size, drop_remainder=True)

    def create_tf_dataset(self, is_training=True, input_context=None):
        """
        Create tf dataset.

        :param bool is_training: Specifies whether the data is for training
        :param dict input_context: Given by distributed strategy for training
        :returns: tf dataset
        """

        if not isinstance(self.data_dir, list):
            self.data_dir = [self.data_dir]

        file_pattern = []

        for file in self.data_dir:
            file_pattern.append(os.path.join(file, '*.tfrecord*'))

        filelist = tf.data.Dataset.list_files(
            file_pattern, shuffle=self.shuffle, seed=self.shuffle_seed
        )

        filelist = shard_dataset(
            filelist, self.use_multiple_workers, input_context
        )

        dataset = filelist.interleave(
            map_func=tf.data.TFRecordDataset,
            cycle_length=self.n_parallel_reads,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            # only allow nondeterminism when shuffling unseeded
            deterministic=not (self.shuffle and self.shuffle_seed is None),
        )

        return transform_dataset(
            dataset,
            self.map_fn,
            self.batch_size,
            is_training,
            shuffle=self.shuffle,
            shuffle_buffer=self.shuffle_buffer,
            repeat=self.repeat,
            seed=self.shuffle_seed,
            map_before_batch=self.map_before_batch,
            batch_fn=self.batch_fn,
            post_batch_map_fn=getattr(self, "post_batch_map_fn", None),
            skip_steps=self.skip_steps,
        )
