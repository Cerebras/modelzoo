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

from cerebras.modelzoo.trainer.callbacks import Callback


class SelectiveGrad(Callback):
    """Callback class that selectively applies gradient computation."""

    def __init__(self, selective_grads: dict):
        """SelectiveGrad may be initialized with a configuration dictionary or
        keyword arguments.

        Args:
            selective_grads: Configuration for selective gradient computation.
        """
        if selective_grads:
            # TODO: Move this configure_selective_gradient function to this file
            from cerebras.modelzoo.common.utils.utils import (
                configure_selective_gradient,
            )

            self.selective_grads = configure_selective_gradient(selective_grads)
        else:
            self.selective_grads = []

    def setup(self, trainer):
        for selective_grad in self.selective_grads:
            trainer.model.apply(selective_grad)
