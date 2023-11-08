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

"""Module which provides utilities for selecting half dtype between float16 and bfloat16"""

import torch


class HalfDType:
    def __init__(self, use_bfloat16=False):
        self.use_bfloat16 = use_bfloat16

    @property
    def half_dtype(self):
        if self.use_bfloat16:
            return torch.bfloat16
        return torch.float16


half_dtype_instance = HalfDType()
