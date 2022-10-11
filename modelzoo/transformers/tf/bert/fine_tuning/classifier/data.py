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

import sys

from modelzoo.transformers.tf.bert.fine_tuning.classifier.input.TfdsTextDataProcessor import (  # noqa
    MnliDataProcessor,
    Sst2DataProcessor,
)


def train_input_fn(params):
    return getattr(
        sys.modules[__name__], params["train_input"]["data_processor"]
    )(params["train_input"]).create_tf_dataset(is_training=True)


def eval_input_fn(params):
    return getattr(
        sys.modules[__name__], params["eval_input"]["data_processor"]
    )(params["eval_input"]).create_tf_dataset(is_training=False)


def predict_input_fn(params):
    return getattr(
        sys.modules[__name__], params["predict_input"]["data_processor"]
    )(params["predict_input"]).create_tf_dataset(is_training=False)
