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
UNet Data Input Pipeline
"""

import sys

from modelzoo.unet.tf.input import (  # noqa
    DAGM2007Dataset,
    DAGM2007TFRecordsDataset,
    SeverstalDataset,
    SeverstalTFRecordsDataset,
)


def train_input_fn(params=None):
    """
    Dataset Train input function
    """
    return getattr(sys.modules[__name__], params["train_input"]["dataset"])(
        params
    ).dataset_fn(
        batch_size=params["train_input"]["train_batch_size"],
        augment_data=params["train_input"]["augment_data"],
        shuffle=params["train_input"]["shuffle"],
        is_training=True,
    )


def eval_input_fn(params=None):
    """
    Dataset Eval input function
    """
    return getattr(sys.modules[__name__], params["train_input"]["dataset"])(
        params
    ).dataset_fn(
        batch_size=params["train_input"]["eval_batch_size"],
        augment_data=False,
        shuffle=False,
        is_training=False,
    )
