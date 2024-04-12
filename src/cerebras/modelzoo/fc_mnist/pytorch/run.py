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
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
# isort: on

from cerebras.modelzoo.common.utils.run.cli_pytorch import get_params_from_args


def main():
    params = get_params_from_args()
    from cerebras.modelzoo.fc_mnist.pytorch.utils import set_defaults

    set_defaults(params)

    from cerebras.modelzoo.common.run_utils import main
    from cerebras.modelzoo.fc_mnist.pytorch.data import (
        get_eval_dataloader,
        get_train_dataloader,
    )
    from cerebras.modelzoo.fc_mnist.pytorch.model import MNISTModel

    main(params, MNISTModel, get_train_dataloader, get_eval_dataloader)


if __name__ == '__main__':
    main()
