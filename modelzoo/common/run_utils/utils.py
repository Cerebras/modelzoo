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

"""Contains common run related utilies"""


class DeviceType:
    """Supported Devices for Running Modelzoo Scripts"""

    CSX = "CSX"
    CPU = "CPU"
    GPU = "GPU"
    # to be used to reference when device type does not matter
    ANY = "ANY"

    @classmethod
    def devices(cls):
        """Valid strategies"""
        return [cls.CSX, cls.CPU, cls.GPU]


class ExecutionStrategy:
    """Supported Cerebras Execution Strategies"""

    pipeline = "pipeline"
    weight_streaming = "weight_streaming"

    @classmethod
    def strategies(cls):
        """Valid strategies"""
        return [cls.pipeline, cls.weight_streaming]
