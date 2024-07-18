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

"""
This module contains the implementation of the Listener callback class.
"""

from cerebras.modelzoo.trainer.callbacks import Callback


class Listener(Callback):
    """
    Callback class that handles registering listeners to the training step.
    """

    def __init__(self, listeners: dict):
        """
        Args:
            listeners: A dictionary containing the listener configuration.
        """

        from cerebras.pytorch.experimental.listener import (
            ListenerMode,
            create_listener,
        )

        if isinstance(listeners, dict):
            listeners = [listeners]

        self.listeners = [
            create_listener(**listener_params) for listener_params in listeners
        ]
        self.listener_mode = ListenerMode(self.listeners)

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self.listener_mode.__enter__()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.listener_mode.__exit__()

    def on_validate_batch_start(self, trainer, model, batch, batch_idx):
        self.listener_mode.__enter__()

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.listener_mode.__exit__()
