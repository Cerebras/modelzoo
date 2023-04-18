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

from modelzoo import CSOFT_PACKAGE, CSoftPackage

if CSOFT_PACKAGE == CSoftPackage.SRC:
    appliance_environ = os.environ
elif CSOFT_PACKAGE == CSoftPackage.WHEEL:
    from cerebras_appliance.environment import appliance_environ
elif CSOFT_PACKAGE == CSoftPackage.NONE:
    appliance_environ = os.environ
else:
    # We should never get here
    assert False, f"Invalid value for `CSOFT_PACKAGE {CSOFT_PACKAGE}"
