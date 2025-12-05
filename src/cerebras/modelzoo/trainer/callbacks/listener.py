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
This module contains various listeners for automatically summarizing tensors.
"""

import fnmatch
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import List, Union

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback
from cerebras.pytorch.experimental.listener import register_traced_tensor_hook


class _ListenerCallback(Callback, ABC):
    """Base class that handles registering listeners for traced tensors."""

    def __init__(self):
        self.trainer = None

    @abstractmethod
    def traced_tensor_hook(self, tensor: torch.Tensor, name: str):
        """Hook that's called when a tensor is created during tracing."""

    def on_enter_train(self, trainer, stack, train_dataloader, loop, loop_idx):
        stack.enter_context(self._register_listener(trainer))

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        stack.enter_context(self._register_listener(trainer))

    @contextmanager
    def _register_listener(self, trainer):
        handles = [register_traced_tensor_hook(self.traced_tensor_hook)]
        self.trainer = trainer

        try:
            yield
        finally:
            self.trainer = None
            for handle in handles:
                handle.remove()

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class DumpAvailableTensorNames(_ListenerCallback):
    def __init__(self):
        super().__init__()
        self.tensors = []

    def traced_tensor_hook(self, tensor: torch.Tensor, name: str):
        self.tensors.append((name, tensor.shape, tensor.dtype))

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self.tensors = []

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if batch_idx == 0:
            self._dump_names_to_file()

    def on_validate_batch_start(self, trainer, model, batch, batch_idx):
        self.tensors = []

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if batch_idx == 0:
            self._dump_names_to_file()

    def _dump_names_to_file(self):
        executor = cstorch.current_executor()
        outfile = Path(executor.artifact_dir) / f"available_tensor_names.txt"
        outfile.write_text(
            '\n'.join(
                [
                    ", ".join(str(x) for x in items)
                    for items in sorted(self.tensors, key=lambda item: item[0])
                ]
            )
        )


class SummaryTensorListener(_ListenerCallback):
    """Tensor listener that summarizes every tensor."""

    def __init__(
        self,
        listener_name: str,
        tensor_names: Union[str, List[str]],
    ):
        """
        Constructs named tensor listener.

        Args:
            listener_name: a listener name to be used in summarized tensor name.
            tensor_names: a list of tensor names to be captured. It also supports
                glob patterns to match group of tensors using pattern.
                See https://docs.python.org/3/library/fnmatch.html for more details.
        """
        super().__init__()

        self.listener_name = listener_name
        self.tensor_names = tensor_names
        if isinstance(self.tensor_names, str):
            self.tensor_names = [self.tensor_names]

    def traced_tensor_hook(self, tensor: torch.Tensor, name: str):
        if any(fnmatch.fnmatch(name, pattern) for pattern in self.tensor_names):
            self.trainer.log_metrics(**{f"{self.listener_name}/{name}": tensor})


class NormTensorListener(_ListenerCallback):
    """Tensor listener that computes tensor norms."""

    def __init__(
        self,
        listener_name: str,
        tensor_names: Union[str, List[str]],
    ):
        """
        Constructs named tensor listener.

        Args:
            listener_name: a listener name to be used in summarized tensor name.
            tensor_names: a list of tensor names to be captured. It also supports
                glob patterns to match group of tensors using pattern.
                See https://docs.python.org/3/library/fnmatch.html for more details.
        """
        super().__init__()

        self.listener_name = listener_name
        self.tensor_names = tensor_names
        self.total_norm = None
        if isinstance(self.tensor_names, str):
            self.tensor_names = [self.tensor_names]

    def traced_tensor_hook(self, tensor: torch.Tensor, name: str):
        if any(fnmatch.fnmatch(name, pattern) for pattern in self.tensor_names):
            norm = torch.norm(tensor)

            self.trainer.log_metrics(**{f"{self.listener_name}/{name}": norm})

            if self.total_norm is None:
                self.total_norm = torch.pow(norm, 2.0)
            else:
                self.total_norm += torch.pow(norm, 2.0)

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self.total_norm = None

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if self.total_norm is not None:
            self.trainer.log_metrics(
                **{f"{self.listener_name}/__all__": torch.sqrt(self.total_norm)}
            )

    def on_validate_batch_start(self, trainer, model, batch, batch_idx):
        self.total_norm = None

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        if self.total_norm is not None:
            self.trainer.log_metrics(
                **{f"{self.listener_name}/__all__": torch.sqrt(self.total_norm)}
            )
