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


def _summarize_tensor(x, is_gradient_tensor=False, op_name_if_grad=None):
    """Outside of Cerebras Package. NoOp for compatibility"""


def _summary_layer(x):
    """Outside of Cerebras Package. NoOp for compatibility"""
    return x


def _boundary_cast(x, name=None, back_cast=True):
    """Outside of Cerebras Package. NoOp for compatibility"""
    return x


try:
    try:
        from cerebras.tf.tf_helper import (
            boundary_cast,
            summarize_tensor,
            summary_layer,
        )
    except:
        from cerebras_tensorflow.summary import (
            boundary_cast,
            summarize_tensor,
            summary_layer,
        )
except:
    summarize_tensor = _summarize_tensor
    summary_layer = _summary_layer
    boundary_cast = _boundary_cast

__all__ = [
    'summarize_tensor',
    'summary_layer',
    'boundary_cast',
]
