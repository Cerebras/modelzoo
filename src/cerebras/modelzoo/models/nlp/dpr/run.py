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

# isort: off
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../.."))
# isort: on

if __name__ == '__main__':
    import warnings

    from cerebras.modelzoo.common.run_utils import run

    warnings.warn(
        "Running models using run.py is deprecated. Please switch to using the ModelZoo CLI. "
        "See https://training-docs.cerebras.ai/model-zoo/cli-overview for more details."
    )

    run()
