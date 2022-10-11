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
Script to generate TFRecords files for DAGM2007 dataset
"""

import argparse
import os
import shutil
import sys
import tempfile

import pandas as pd
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from modelzoo.unet.tf.utils import get_params


def parse_args():
    # Parse command line ars
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params",
        default="configs/params_dagm.yaml",
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "--images_per_file",
        type=int,
        default=100,
        help="Number of images in each TFRecords file",
    )
    parser.add_argument(
        "--output_directory",
        default="./dagm_tfrecords/",
        help="Path to store tfrecords files",
    )
    args = parser.parse_args()
    return args


class DAGM2007TFRecordsWriter:
    """
    TFRecoders writer for DAGM2007 Dataset
    """

    def __init__(
        self,
        params=None,
        output_directory="./dagm2007_tfrecords/",
        images_per_file=1000,
    ):

        self.output_directory = output_directory

        self.images_per_file = images_per_file

        self.data_dir = params["train_input"]["dataset_path"]
        self.class_id = params["train_input"]["class_id"]
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(
                f"The dataset directory `{self.data_dir}` does not exist."
            )
        self.data_dir = os.path.join(self.data_dir, f"Class{self.class_id}")

        self.image_shape = params["train_input"]["image_shape"]
        self.normalize_data_method = params["train_input"][
            "normalize_data_method"
        ]
        self.only_defective_images = params["train_input"][
            "only_defective_images"
        ]

        if self.class_id is None:
            raise ValueError("The parameter `class_id` cannot be set to None")

    def dataset_fn(
        self, is_training=True, shuffle=True,
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

        dataset = dataset.map(
            map_func=_load_dagm_data,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        return dataset.flat_map(lambda x: x.batch(8))

    def generate_tfrecords(self, is_training=True):
        dataset = self.dataset_fn(is_training=is_training)
        dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        next_element = dataset_iterator.get_next()

        prefix_suffix = "train" if is_training else "test"
        recordPath = os.path.join(self.output_directory, prefix_suffix)

        if os.path.exists(recordPath):
            shutil.rmtree(recordPath)

        os.makedirs(recordPath)

        num = 1
        recordFileNum = 0
        recordFileName = prefix_suffix + "-%.3d.tfrecords" % recordFileNum
        writer = tf.io.TFRecordWriter(os.path.join(recordPath, recordFileName))

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

    tfrecoders_write = DAGM2007TFRecordsWriter(
        params=params,
        output_directory=args.output_directory,
        images_per_file=args.images_per_file,
    )

    # Generate training dataset
    tfrecoders_write.generate_tfrecords(is_training=True)

    # Generate test dataset
    tfrecoders_write.generate_tfrecords(is_training=False)
