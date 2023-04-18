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

""" CSRunConfig

Wrapper for allowing users with different environments to
create an estimator run config when running in Modelzoo.
"""

from modelzoo import CSOFT_PACKAGE, CSoftPackage

if CSOFT_PACKAGE == CSoftPackage.WHEEL:
    from cerebras_appliance.cs_run_config import CSRunConfig as RunConfig
elif CSOFT_PACKAGE == CSoftPackage.NONE:
    from tensorflow_estimator.python.estimator.run_config import RunConfig
else:
    assert False, f"Invalid value for `CSOFT_PACKAGE {CSOFT_PACKAGE}"


class CSRunConfig(RunConfig):
    """ Wrapper class for Run Config objects. """

    def __init__(
        self, cs_config=None, **kwargs,
    ):
        if CSOFT_PACKAGE == CSoftPackage.WHEEL:
            kwargs["cs_config"] = cs_config

        super(CSRunConfig, self).__init__(**kwargs)
