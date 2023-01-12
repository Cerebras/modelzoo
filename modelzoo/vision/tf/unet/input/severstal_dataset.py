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
Severstal Dataset class

Owners: {vithu, kamran}@cerebras.net
"""

import os
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf


class SeverstalDataset:
    """
    Severstal Dataset class
    """

    def __init__(self, params=None):

        self.data_dir = params["train_input"]["dataset_path"]
        self.class_id = params["train_input"]["class_id"]
        self.train_test_split = params["train_input"]["train_test_split"]
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                "The dataset directory `%s` does not exist." % self.data_dir
            )

        self.mixed_precision = params["model"]["mixed_precision"]
        self.image_shape = params["train_input"]["image_shape"]
        self.num_classes = params["train_input"]["num_classes"]

        self.normalize_data_method = params["train_input"][
            "normalize_data_method"
        ]

        self.data_format = params["model"]["data_format"]
        self.seed = params["train_input"].get("seed", None)

        self.shuffle_buffer_size = params["train_input"]["shuffle_buffer_size"]

        assert self.class_id <= 4, "Maximum 4 available classes."
        self.image_dir, csv_file = self._get_data_dirs()
        dataset = pd.read_csv(csv_file, index_col=0)

        dataset = dataset[dataset["ClassId"] == self.class_id]
        self.total_rows = len(dataset.index)

        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            self.class_id_dataset_path = temp_file.name
        finally:
            temp_file.close()

        dataset.to_csv(self.class_id_dataset_path, index=True)

    def _get_data_dirs(self):
        csv_file = os.path.join(self.data_dir, "train.csv")
        image_dir = os.path.join(self.data_dir, "train_images")
        return image_dir, csv_file

    def dataset_fn(
        self, batch_size, augment_data=True, shuffle=True, is_training=True,
    ):
        dataset = tf.data.TextLineDataset(self.class_id_dataset_path)

        # Get train-test splits.
        train_rows = int(np.floor(self.train_test_split * self.total_rows))
        dataset = dataset.skip(1)  # Skip CSV Header
        if is_training:
            dataset = dataset.take(train_rows)
        else:
            dataset = dataset.skip(train_rows)

        def _load_severstal_data(line):
            input_image_name, _, encoded_pixels = tf.io.decode_csv(
                records=line,
                record_defaults=[[""], [0], [""]],
                field_delim=",",
            )

            def decode_image(filepath, resize_shape, normalize_data_method):
                # Load the raw data from the file as a string
                image_content = tf.io.read_file(filepath)

                # Convert the compressed string to a 3D uint8 tensor
                image = tf.image.decode_jpeg(image_content, channels=1)

                image = tf.image.resize(
                    image,
                    size=resize_shape[:2],
                    # [BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA]
                    method=tf.image.ResizeMethod.BICUBIC,
                    preserve_aspect_ratio=False,
                )

                image.set_shape(resize_shape)
                image = tf.cast(image, tf.float32)

                if normalize_data_method == "zero_centered":
                    image = tf.divide(image, 127.5) - 1

                elif normalize_data_method == "zero_one":
                    image = tf.divide(image, 255.0)

                return image

            def decode_mask_image(mask_rle, resize_shape):
                # Decoding of the encdoed RLE
                shape = tf.convert_to_tensor(value=(1600, 256), dtype=tf.int64)
                size = tf.math.reduce_prod(input_tensor=shape)

                # Split string
                rle_list = tf.strings.split([mask_rle])
                rle_numbers = tf.strings.to_number(rle_list.values, tf.int64)

                # Get starts and lengths
                starts = rle_numbers[::2] - 1
                lens = rle_numbers[1::2]

                # Make ones to be scattered
                total_ones = tf.reduce_sum(input_tensor=lens)
                ones = tf.ones([total_ones], tf.uint8)

                # Make scattering indices
                r = tf.range(total_ones)
                lens_cum = tf.math.cumsum(lens)
                s = tf.searchsorted(lens_cum, r, "right")

                idx = r + tf.gather(
                    starts - tf.pad(tensor=lens_cum[:-1], paddings=[(1, 0)]), s
                )

                # Scatter ones into flattened mask
                mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])

                # Reshape into mask
                mask = tf.reshape(mask_flat, shape)
                mask = tf.transpose(a=mask)
                mask = tf.expand_dims(mask, 2)

                # Resize mask
                mask = tf.image.resize(
                    mask,
                    size=resize_shape[:2],
                    # [BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA]
                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                    preserve_aspect_ratio=False,
                )

                mask.set_shape(resize_shape)
                mask = tf.cast(mask, tf.float32)

                return mask

            input_image = decode_image(
                filepath=tf.strings.join(
                    [self.image_dir, input_image_name], separator="/"
                ),
                resize_shape=self.image_shape,
                normalize_data_method=self.normalize_data_method,
            )

            mask_image = decode_mask_image(
                mask_rle=encoded_pixels, resize_shape=self.image_shape
            )

            return tf.data.Dataset.from_tensor_slices(
                ([input_image], [mask_image])
            )

        dataset = dataset.interleave(
            _load_severstal_data,
            cycle_length=1,
            block_length=16,
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
                adjust_brightness = (
                    tf.random.uniform(shape=(), seed=self.seed) > 0.5
                )

                input_image = tf.cond(
                    pred=horizontal_flip,
                    true_fn=lambda: tf.image.flip_left_right(input_image),
                    false_fn=lambda: input_image,
                )

                input_image = tf.cond(
                    pred=adjust_brightness,
                    true_fn=lambda: tf.image.adjust_brightness(
                        input_image, delta=0.2
                    ),
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

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        if is_training:
            dataset = dataset.repeat()

        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
