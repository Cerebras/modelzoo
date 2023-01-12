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
DAGM 2007 Dataset class
"""

import os
import tempfile

import pandas as pd
import tensorflow as tf


class DAGM2007Dataset:
    """
    DAGM 2007 Dataset class
    """

    def __init__(self, params=None):

        self.data_dir = params["train_input"]["dataset_path"]
        self.class_id = params["train_input"]["class_id"]
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"The dataset directory `{self.data_dir}` does not exist."
            )
        self.data_dir = os.path.join(self.data_dir, f"Class{self.class_id}")

        self.mixed_precision = params["model"]["mixed_precision"]
        self.image_shape = params["train_input"]["image_shape"]
        self.num_classes = params["train_input"]["num_classes"]
        self.normalize_data_method = params["train_input"][
            "normalize_data_method"
        ]
        self.only_defective_images = params["train_input"][
            "only_defective_images"
        ]

        self.data_format = params["model"]["data_format"]
        self.seed = params["train_input"].get("seed", None)

        self.shuffle_buffer_size = params["train_input"]["shuffle_buffer_size"]

        if self.class_id is None:
            raise ValueError("The parameter `class_id` cannot be set to None")

    def dataset_fn(
        self, batch_size, augment_data=True, shuffle=True, is_training=True,
    ):
        if is_training:
            image_dir = os.path.join(self.data_dir, "Train")
        else:
            image_dir = os.path.join(self.data_dir, "Test")

        mask_image_dir = os.path.join(image_dir, "Label")
        csv_file = os.path.join(mask_image_dir, "Labels.txt")
        dataset = pd.read_csv(csv_file, sep="\t")

        try:
            temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            csv_file = temp_file.name
        finally:
            temp_file.close()

        dataset.to_csv(csv_file, sep=",", index=True)

        dataset = tf.data.TextLineDataset(csv_file)
        dataset = dataset.skip(1)  # Skip CSV Header

        if self.only_defective_images:
            dataset = dataset.filter(
                lambda line: tf.not_equal(
                    tf.strings.split(input=line, sep=",")[1], "0"
                )
            )

        def _load_dagm_data(line):
            _, _, input_image_name, _, image_mask_name, _ = tf.io.decode_csv(
                records=line,
                record_defaults=[[""], [""], [""], [""], [""], [""]],
                field_delim=",",
            )

            def decode_image(filepath, resize_shape, normalize_data_method):
                image_content = tf.io.read_file(filepath)

                image = tf.image.decode_png(
                    contents=image_content,
                    channels=resize_shape[-1],
                    dtype=tf.uint8,
                )

                image = tf.image.resize(
                    image,
                    size=resize_shape[:2],
                    # [BILINEAR, NEAREST_NEIGHBOR, BICUBIC, AREA]
                    method=tf.image.ResizeMethod.BILINEAR,
                    preserve_aspect_ratio=True,
                )

                image.set_shape(resize_shape)
                image = tf.cast(image, tf.float32)

                if normalize_data_method == "zero_centered":
                    image = tf.divide(image, 127.5) - 1

                elif normalize_data_method == "zero_one":
                    image = tf.divide(image, 255.0)

                return image

            input_image = decode_image(
                filepath=tf.strings.join(
                    [image_dir, input_image_name], separator="/"
                ),
                resize_shape=self.image_shape,
                normalize_data_method=self.normalize_data_method,
            )

            mask_image = tf.cond(
                pred=tf.equal(image_mask_name, "0"),
                true_fn=lambda: tf.zeros(self.image_shape, dtype=tf.float32),
                false_fn=lambda: decode_image(
                    filepath=tf.strings.join(
                        [mask_image_dir, image_mask_name], separator="/"
                    ),
                    resize_shape=self.image_shape,
                    normalize_data_method="zero_one",
                ),
            )

            return tf.data.Dataset.from_tensor_slices(
                ([input_image], [mask_image])
            )

        dataset = dataset.interleave(
            _load_dagm_data,
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
