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
Input functions for LLNL CRETIN proxy model.
Owner: ryan@cerebras.net
"""

import tensorflow as tf
from llnl_shared.cretin.tf.utils import read_tfrecord

from modelzoo.common.tf.model_utils.shard_dataset import shard_dataset


def input_fn(params):
    batch_size = params["train_input"]["batch_size"]
    if params["runconfig"]["mode"] in [
        tf.estimator.ModeKeys.TRAIN,
        "compile_only",
        "validate_only",
    ]:
        tfrecord = params["train_input"]["tfrecord"]
    elif params["runconfig"]["mode"] == tf.estimator.ModeKeys.EVAL:
        tfrecord = params["evaluation"]["eval_input"]
    elif params["runconfig"]["mode"] == tf.estimator.ModeKeys.PREDICT:
        tfrecord = params["inference"]["infer_input"]
    else:
        raise ValueError("Mode not supported")

    ds = tf.data.TFRecordDataset(tfrecord)
    if (params["runconfig"]["cs_ip"] != None) and (
        params["runconfig"]["mode"] == "train"
    ):
        ds = shard_dataset(ds, True)
    ds = ds.map(
        lambda x: read_tfrecord(
            x, params["model"]["mixed_precision"], params["runconfig"]["mode"],
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.shuffle(buffer_size=100 * batch_size, seed=100,)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    if (
        params["runconfig"].get("mode", "train") == tf.estimator.ModeKeys.TRAIN
        or params["train_input"].get('ds_repeat', False) == True
    ):
        ds = ds.repeat()
    return ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
