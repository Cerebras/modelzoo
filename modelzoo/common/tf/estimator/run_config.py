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

try:
    from cerebras.tf.run_config import CSRunConfig as RunConfig
except ImportError:
    from tensorflow_estimator.python.estimator.run_config import RunConfig


class CSRunConfig(RunConfig):
    def __init__(
        self,
        cs_ip=None,
        system_name=None,
        cs_config=None,
        stack_params=None,
        **kwargs,
    ):
        if RunConfig.__name__ == "CSRunConfig":
            kwargs["cs_ip"] = cs_ip
            kwargs["system_name"] = system_name
            kwargs["cs_config"] = cs_config
            kwargs["stack_params"] = stack_params

        super(CSRunConfig, self).__init__(**kwargs)
