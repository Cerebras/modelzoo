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
This module contains the BackendCallback class which is used to set the backend
for the trainer.
"""

from contextlib import contextmanager

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import CoreCallback


class BackendCallback(CoreCallback):
    """Callback to set the backend for the trainer."""

    def __init__(self, backend, device):
        """
        Args:
            backend (cstorch.Backend or None): The backend object to be used for the trainer.
                If None, the device argument must be provided.
                If both are provided, an error is raised.
            device (str or None): The device type to be used for the trainer. If None, the
                backend argument must be provided.
        """
        self.backend = backend
        self.device = device

        self._workflow_started = False

    def pre_setup(self, trainer):
        if self.backend is None:
            if isinstance(self.device, str):
                accepted_device_types = {"CSX", "CPU", "GPU"}
                if self.device.upper() not in accepted_device_types:
                    raise ValueError(
                        f"Invalid device type: {self.device}. "
                        f"Expected one of {accepted_device_types}."
                    )

                backend = cstorch.current_backend(
                    raise_exception=False, raise_warning=False
                )
                if backend is None:
                    trainer.backend = cstorch.backend(
                        self.device, trainer.artifact_dir
                    )
                else:
                    if backend.backend_type.name.lower() != self.device.lower():
                        raise ValueError(
                            "Cannot instantiate multiple trainers with different device types"
                        )

                    trainer.backend = backend
                    trainer.backend.artifact_dir = trainer.artifact_dir
            else:
                raise RuntimeError(
                    f"Trainer expected a backend object or a device string"
                )
        else:
            from cerebras.pytorch.backend import Backend

            if not isinstance(self.backend, Backend):
                raise TypeError(
                    f"Expected backend to be a cstorch.Backend object. "
                    f"Got: {type(self.backend)}"
                )
            elif self.device is not None:
                raise ValueError(
                    f"backend and device are mutually exclusive arguments of Trainer. "
                    "Please only provide one or the other"
                )

            if self.backend.is_csx:
                self._workflow_started = (
                    self.backend.cluster.workflow_id is not None
                )

            trainer.backend = self.backend
            # Set the backend's artifact directory to be the same as the
            # trainer's artifact directory
            trainer.backend.artifact_dir = trainer.artifact_dir

    @contextmanager
    def workflow_context(self, trainer, lock_resources=True):
        """Context manager to start and stop the workflow for the trainer.
        Args:
            trainer (Trainer): The trainer object.
            lock_resources (bool): Whether to reserve CSX cluster resources for the entire run.
        """
        if (
            not self._workflow_started
            and trainer.backend.is_csx
            and trainer.backend.is_e2e_execution
        ):
            self._workflow_started = trainer.backend.cluster.start_workflow(
                lock_resources=lock_resources,
            )

            yield
        else:
            yield

    def on_enter_fit(
        self, trainer, stack, train_dataloader, val_dataloader, loop
    ):
        # Disable resource locking for train-only runs (no validation)
        stack.enter_context(
            self.workflow_context(trainer, lock_resources=loop.train_only)
        )

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        # Disable resource locking for standalone `validate`
        # No-op if `trainer.fit` was called or if running in
        # auto-restart mode (workflow already started)
        stack.enter_context(
            self.workflow_context(trainer, lock_resources=False)
        )

    def on_enter_validate_all(
        self, trainer, stack, val_dataloaders, loop, ckpt_paths
    ):
        # Disable resource locking for standalone `validate_all`
        # if no custom downstream validation logic is specified
        # No-op if called from within `trainer.fit`or if running
        # in auto-restart mode (workflow already started)
        stack.enter_context(
            self.workflow_context(
                trainer, lock_resources=bool(trainer.validation_callbacks)
            )
        )
