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

"""Determine where Cerebras Software package resides and set CSOFT_PACKAGE"""
import importlib
from enum import Enum


class CSoftPackage(Enum):
    """Location where Cerebras Software package can be found."""

    SRC = 1
    WHEEL = 2
    NONE = 3


def _find_spec(name):
    # Note: Importing a package directly runs the `__init__.py` file, which
    # we don't want, since the imports in that file may have issues themselves
    # and we want to catch them as opposed to falling back to `NONE`.
    try:
        return importlib.util.find_spec(name)
    except ImportError:
        return None


# Find out where Cerebras software package resides and set `CSOFT_PACKAGE`
# accordingly.
if _find_spec("cerebras.framework") is not None:
    CSOFT_PACKAGE = CSoftPackage.SRC
elif _find_spec("cerebras_appliance") is not None:
    CSOFT_PACKAGE = CSoftPackage.WHEEL
else:
    CSOFT_PACKAGE = CSoftPackage.NONE
