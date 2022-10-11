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
MNIST input function
"""
import tensorflow as tf
import tensorflow_datasets as tfds


def _parse_batch_samples(samples):
    images = tf.image.convert_image_dtype(samples["image"], tf.float16,)
    batch_size = images.shape[0]
    images = tf.reshape(images, [batch_size, -1])
    labels = tf.cast(samples["label"], tf.int32)
    return images, labels


def input_fn(params, mode=tf.estimator.ModeKeys.TRAIN):
    """
    :param <dict> params: dict containing input parameters for creating dataset.
    Expects the following fields:
    
    - "data_dir" (string): path to the data files to use.
    - "batch_size" (int): batch size
    - "to_float16" (bool): whether to convert to float16 or not
    - "drop_last_batch" (bool): whether to drop the last batch or not
    """
    training = mode == tf.estimator.ModeKeys.TRAIN
    ds = None
    input_params = params["train_input"]
    data_dir = input_params["data_dir"]

    # setting num_parallel_calls to 0 implies AUTOTUNE
    num_parallel_calls = input_params.get("num_parallel_calls", 0)

    batch_size = (
        input_params.get("train_batch_size")
        if training
        else input_params.get("eval_batch_size")
    )
    if batch_size is None:
        batch_size = input_params["batch_size"]

    data, info = tfds.load("mnist", data_dir=data_dir, with_info=True)
    ds = data["train"] if training else data["test"]

    if training and input_params["shuffle"]:
        ds = ds.shuffle(info.splits["train"].num_examples)
    if training:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(
        _parse_batch_samples,
        num_parallel_calls=num_parallel_calls
        if num_parallel_calls > 0
        else tf.data.experimental.AUTOTUNE,
    )
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def train_input_fn(params):
    return input_fn(params, mode=tf.estimator.ModeKeys.TRAIN)


def eval_input_fn(params):
    return input_fn(params, mode=tf.estimator.ModeKeys.EVAL)
