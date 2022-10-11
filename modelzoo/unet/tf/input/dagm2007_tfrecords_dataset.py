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
DAGM2007 TFRecords Dataset class
"""

import os

import tensorflow as tf

from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset


class DAGM2007TFRecordsDataset:
    """
    DAGM2007 TFRecords Dataset class
    """

    def __init__(self, params=None):

        self.data_dir = params["train_input"]["dataset_path"]
        self.mixed_precision = params["model"]["mixed_precision"]
        self.image_shape = params["train_input"]["image_shape"]
        self.num_classes = params["train_input"]["num_classes"]
        self.data_format = params["model"]["data_format"]
        self.seed = params["train_input"].get("seed", None)

        self.shuffle_buffer_size = params["train_input"]["shuffle_buffer_size"]
        self.num_parallel_reads = params["train_input"].get(
            "num_parallel_reads", 16
        )
        # For sharding on the Cerebras System, we need to explicitly retrieve `TF_CONFIG`.
        self.use_multiple_workers = params["train_input"].get(
            "use_multiple_workers", False
        )

    def dataset_fn(
        self, batch_size, augment_data=True, shuffle=True, is_training=True,
    ):
        if is_training:
            file_pattern = os.path.join(
                self.data_dir, "train/", "train-*.tfrecords"
            )
        else:
            file_pattern = os.path.join(
                self.data_dir, "test/", "test-*.tfrecords"
            )

        file_list = tf.data.Dataset.list_files(
            file_pattern, seed=self.seed, shuffle=is_training
        )

        file_list = shard_dataset(file_list, self.use_multiple_workers)

        dataset = tf.compat.v1.data.TFRecordDataset(
            file_list, num_parallel_reads=self.num_parallel_reads,
        )

        def _parse_records(record):
            feature_description = {
                "image": tf.io.FixedLenSequenceFeature(
                    self.image_shape, tf.float32, allow_missing=True
                ),
                "mask": tf.io.FixedLenSequenceFeature(
                    self.image_shape, tf.float32, allow_missing=True
                ),
            }

            example = tf.io.parse_single_example(record, feature_description)
            input_image = example["image"]
            mask_image = example["mask"]

            input_image = tf.squeeze(input_image, axis=0)
            mask_image = tf.squeeze(mask_image, axis=0)

            return input_image, mask_image

        dataset = dataset.map(
            map_func=_parse_records,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        dataset = dataset.cache()

        if is_training and shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size, self.seed)

        def _resize_augment_images(input_image, mask_image):
            if augment_data:
                horizontal_flip = (
                    tf.random.uniform(shape=(), seed=self.seed) > 0.5
                )
                input_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(input_image),
                    false_fn=lambda: input_image,
                )

                mask_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(mask_image),
                    false_fn=lambda: mask_image,
                )

                n_rots = tf.random.uniform(
                    shape=(), dtype=tf.int32, minval=0, maxval=3, seed=self.seed
                )

                if self.image_shape[0] != self.image_shape[1]:
                    n_rots = n_rots * 2

                input_image = tf.image.rot90(input_image, k=n_rots)
                mask_image = tf.image.rot90(mask_image, k=n_rots)

                input_image = tf.image.resize_with_crop_or_pad(
                    input_image,
                    target_height=self.image_shape[0],
                    target_width=self.image_shape[1],
                )

                mask_image = tf.image.resize_with_crop_or_pad(
                    mask_image,
                    target_height=self.image_shape[0],
                    target_width=self.image_shape[1],
                )

            if self.data_format == "channels_first":
                input_image = tf.transpose(a=input_image, perm=[2, 0, 1])

            reshaped_mask_image = tf.reshape(mask_image, [-1])

            # handle mixed precision for float variables
            # int variables remain untouched
            if self.mixed_precision:
                input_image = tf.cast(input_image, dtype=tf.float16)
                reshaped_mask_image = tf.cast(
                    reshaped_mask_image, dtype=tf.float16
                )

            return input_image, reshaped_mask_image

        dataset = dataset.map(
            map_func=_resize_augment_images,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if is_training:
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
