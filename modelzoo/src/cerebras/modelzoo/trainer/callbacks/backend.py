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

            trainer.backend = self.backend
            # Set the backend's artifact directory to be the same as the
            # trainer's artifact directory
            trainer.backend.artifact_dir = trainer.artifact_dir
