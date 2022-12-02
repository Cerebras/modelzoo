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

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from modelzoo.transformers.tf.bert.input.BertMlmOnlyTfRecordsDynamicMaskProcessor import (  # noqa
    BertMlmOnlyTfRecordsDynamicMaskProcessor,
)
from modelzoo.transformers.tf.bert.input.BertMlmOnlyTfRecordsStaticMaskProcessor import (  # noqa
    BertMlmOnlyTfRecordsStaticMaskProcessor,
)

from modelzoo.transformers.tf.bert.input.BertTfRecordsProcessor import (  # noqa
    BertTfRecordsProcessor,
)


def train_input_fn(params, input_context=None):
    if input_context:
        return getattr(
            sys.modules[__name__], params["train_input"]["data_processor"]
        )(params["train_input"]).create_tf_dataset(
            is_training=True, input_context=input_context
        )
    return getattr(
        sys.modules[__name__], params["train_input"]["data_processor"]
    )(params["train_input"]).create_tf_dataset(is_training=True)


def eval_input_fn(params, input_context=None):
    if input_context:
        return getattr(
            sys.modules[__name__], params["eval_input"]["data_processor"]
        )(params["eval_input"]).create_tf_dataset(
            is_training=False, input_context=input_context
        )
    return getattr(
        sys.modules[__name__], params["eval_input"]["data_processor"]
    )(params["eval_input"]).create_tf_dataset(is_training=False)
