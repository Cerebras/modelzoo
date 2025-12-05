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
import re

from pydantic import BeforeValidator
from typing_extensions import Annotated


def resolve_path(path, pattern=r"\$MODELZOO_ROOT"):
    import cerebras.modelzoo as mz

    repl = os.path.dirname(mz.__file__)
    return re.sub(pattern, repl, path)


def validate_path(path):

    paths = path if isinstance(path, list) else [path]
    for xpath in paths:
        if not os.path.exists(xpath):
            raise ValueError(f"Path {xpath} does not exist.")
    return path


AliasedPath = Annotated[
    str,
    BeforeValidator(resolve_path),
]

ValidatedPath = Annotated[
    str,
    BeforeValidator(validate_path),
]
