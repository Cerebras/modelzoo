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

from modelzoo.transformers.pytorch.transformer.input.TransformerDynamicDataProcessor import (  # noqa
    TransformerDynamicDataProcessor,
)


def train_input_dataloader_fn(params):
    return getattr(
        sys.modules[__name__], params["train_input"]["data_processor"]
    )(params["train_input"]).create_dataloader(is_training=True)


def eval_input_dataloader_fn(params):
    return getattr(
        sys.modules[__name__], params["eval_input"]["data_processor"]
    )(params["eval_input"]).create_dataloader(is_training=False)


def train_input_dataloader(params):
    enable_distributed = params["runconfig"].get("enable_distributed", False)
    if enable_distributed:
        return train_input_dataloader_fn
    return train_input_dataloader_fn(params)


def eval_input_dataloader(params):
    enable_distributed = params["runconfig"].get("enable_distributed", False)
    if enable_distributed:
        return eval_input_dataloader_fn
    return eval_input_dataloader_fn(params)
