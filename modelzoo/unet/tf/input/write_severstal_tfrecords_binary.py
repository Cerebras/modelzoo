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
Script to generate TFRecords files for Severstal dataset
"""

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from modelzoo.unet.tf.utils import get_params


def parse_args():
    # Parse command line ars
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default="configs/params_severstal.yaml",
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "--images_per_file",
        type=int,
        default=1000,
        help="Number of images in each TFRecords file",
    )
    parser.add_argument(
        "--output_directory",
        default="./severstal_tfrecords/",
        help="Path to store tfrecords files",
    )
    args = parser.parse_args()
    return args


class SeverstalTFRecordsWriter:
    """
    TFRecoders writer for Severstal Dataset
    """

    def __init__(
        self,
        params=None,
        output_directory="./severstal_tfrecords/",
        images_per_file=1000,
    ):

        self.output_directory = output_directory

        self.images_per_file = images_per_file

        self.data_dir = params["train_input"]["dataset_path"]
        self.train_test_split = params["train_input"]["train_test_split"]
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                "The dataset directory `%s` does not exist." % self.data_dir
            )

        self.image_shape = params["train_input"]["image_shape"]
        self.normalize_data_method = params["train_input"][
            "normalize_data_method"
        ]

        self.class_id = params["train_input"]["class_id"]

        self.image_dir, csv_file = self._get_data_dirs()
        dataset = pd.read_csv(csv_file, index_col=0)

        dataset = dataset[dataset["ClassId"] == self.class_id]
        self.total_rows = len(dataset.index)
        # Get train-test splits.
        self.train_rows = int(np.floor(self.train_test_split * self.total_rows))

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
        self, is_training=True, shuffle=True,
    ):
        dataset = tf.data.TextLineDataset(self.class_id_dataset_path)

        dataset = dataset.skip(1)  # Skip CSV Header
        if is_training:
            dataset = dataset.take(self.train_rows)
        else:
            dataset = dataset.skip(self.train_rows)

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

        dataset = dataset.map(
            map_func=_load_severstal_data,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        return dataset.flat_map(lambda x: x.batch(8))

    def generate_tfrecords(self, is_training=True):
        dataset = self.dataset_fn(is_training=is_training)
        dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_element = dataset_iterator.get_next()

        prefix_suffix = "train" if is_training else "test"
        recordPath = self.output_directory + "/" + prefix_suffix + "/"

        if os.path.exists(recordPath):
            shutil.rmtree(recordPath)

        os.makedirs(recordPath)

        num = 1
        recordFileNum = 0
        recordFileName = prefix_suffix + "-%.3d.tfrecords" % recordFileNum
        writer = tf.io.TFRecordWriter(recordPath + recordFileName)

        def _float_feature(value):
            """Returns a float_list from a float / double."""
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))

        with tf.compat.v1.Session() as sess:
            print(f"Creating the tfrecord file {recordFileName}")
            while True:
                try:
                    image, mask = sess.run(next_element)
                    num += 1

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                "image": _float_feature(image.flatten()),
                                "mask": _float_feature(mask.flatten()),
                            }
                        )
                    )

                    writer.write(example.SerializeToString())

                    if num > self.images_per_file:
                        writer.close()
                        num = 1
                        recordFileNum += 1
                        recordFileName = (
                            prefix_suffix + "-%.3d.tfrecords" % recordFileNum
                        )
                        print(f"Creating the tfrecord file {recordFileName}")
                        writer = tf.io.TFRecordWriter(
                            recordPath + recordFileName
                        )

                except tf.errors.OutOfRangeError:
                    break

            writer.close()


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tf.compat.v1.disable_eager_execution()

    args = parse_args()
    params = get_params(args.params)

    tfrecoders_write = SeverstalTFRecordsWriter(
        params=params,
        output_directory=args.output_directory,
        images_per_file=args.images_per_file,
    )

    # Generate training dataset
    tfrecoders_write.generate_tfrecords(is_training=True)

    # Generate test dataset
    tfrecoders_write.generate_tfrecords(is_training=False)
