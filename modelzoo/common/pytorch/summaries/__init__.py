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

"""Package for providing summaries that work on CPU/GPU/CS-X."""

from .cb_summary import (
    CBSummary,
    discard_cached_summaries,
    get_all_summaries,
    save_all_summaries,
)
from .scalar_summary import scalar_summary
from .tensor_summary import TensorSummaryReader, tensor_summary
