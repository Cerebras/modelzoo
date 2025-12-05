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

"""Contains the SelectiveGrad callback class."""

from typing import List, Union

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback
from cerebras.pytorch.sparse.configure import default_sparse_param_filter


class SelectiveGrad(Callback):
    """Callback class that selectively applies gradient computation."""

    def __init__(self, selective_grads: Union[dict, List[dict]]):
        """Constructs a `SelectiveGrad` instance.

        Args:
            selective_grads: Configuration for selective gradient computation.
                It may be initialized with a configuration dict or list of
                dicts.
        """
        if selective_grads:

            def get_selective_grad_from_dict(single_config):
                # use the sparsity filter as well as a default filter
                param_filter = single_config.get("param_filter", None)
                if param_filter is None:
                    param_filter = default_sparse_param_filter

                # make init_method an optional field
                if "init_method" in single_config:
                    return cstorch.nn.SelectiveGrad(
                        param_filter, single_config["init_method"]
                    )

                return cstorch.nn.SelectiveGrad(param_filter)

            if isinstance(selective_grads, dict):
                self.selective_grads = [
                    get_selective_grad_from_dict(selective_grads)
                ]
            elif isinstance(selective_grads, list):
                self.selective_grads = list(
                    map(get_selective_grad_from_dict, selective_grads)
                )
            else:
                raise ValueError(
                    "Expected `selective_grads` to be a dict or a list of dicts."
                )
        else:
            self.selective_grads = []

    def setup(self, trainer):
        for selective_grad in self.selective_grads:
            trainer.model.apply(selective_grad)

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass
