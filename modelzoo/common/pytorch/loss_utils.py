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

"""Module which provides utilities for aggregating and saving losses"""

from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from modelzoo import CSOFT_PACKAGE, CSoftPackage
from modelzoo.common.pytorch import cb_model as cm


class LossSaver:
    """Helper class for storing losses during training/eval."""

    def __init__(self, writer: Optional[SummaryWriter] = None):
        """Constructs a `LossSaver` instance.

        Args:
            writer: Tensorboard summary writer for writing losses.
        """
        self.writer = writer
        self._name = "loss_reduce"
        self._last_saved_loss = -1
        self._total_loss = 0
        self._total_size = 0

    def add(self, loss: torch.Tensor, step: int, epoch: int = None):
        """Store loss value. This method will reduce losses across workers.

        Args:
            loss: The loss tensor whose value will be stored.
            step: Global step at which loss was computed.
            epoch: The current epoch.
        """
        if cm.use_cs():
            # Get mean loss across workers using a mesh reduce
            reduced = cm.mesh_reduce(self._name, loss.item(), self.mean_reduce)
            self._last_saved_loss = reduced
            cm.write_to_summary(
                self.writer, global_step=step, dict_to_write={"loss": reduced}
            )
        elif self.writer:
            self._last_saved_loss = loss.item()
            scalar_name = "loss" if epoch is None else f"loss (epoch {epoch})"
            self.writer.add_scalar(scalar_name, self._last_saved_loss, step)

    def accumulate(self, loss: torch.Tensor):
        """Accumulates loss values. This method will reduce losses across workers
        and update a total_loss

        Args:
            loss: The loss tensor whose value will be stored.
        """
        if self._last_saved_loss != -1:
            self._total_loss += self._last_saved_loss
            self._last_saved_loss = -1
        elif cm.use_cs():
            # Get mean loss across workers using a mesh reduce
            reduced = cm.mesh_reduce(self._name, loss.item(), self.mean_reduce)
            self._total_loss += reduced
        else:
            self._total_loss += loss.item()

        self._total_size += 1

    def clear(self):
        """Clears the total_loss value"""
        self._total_loss = 0
        self._total_size = 0

    @property
    def total_loss(self) -> float:
        """Return the total accumulated loss"""
        return self._total_loss

    @property
    def average_loss(self) -> float:
        """Return the total accumulated loss"""
        if self._total_size == 0:
            return float("nan")
        return self._total_loss / float(self._total_size)

    @staticmethod
    def mean_reduce(vals: list):
        """Apply mean reduction over values.

        Args:
            vals: List of values to apply mean reduction over.
        Returns:
            The mean reduction of values.
        """
        return sum(vals) / len(vals)


if CSOFT_PACKAGE == CSoftPackage.SRC:
    # pylint: disable=unused-import
    from cerebras.framework.torch.utils import extract_loss
elif CSOFT_PACKAGE == CSoftPackage.WHEEL:
    # pylint: disable=unused-import
    from cerebras_pytorch.utils import extract_loss
elif CSOFT_PACKAGE == CSoftPackage.NONE:

    def extract_loss(model_outputs):
        """Extracts the loss from the model outputs"""
        if isinstance(model_outputs, torch.Tensor):
            loss = model_outputs
        elif isinstance(model_outputs, (list, tuple)) and model_outputs:
            loss = model_outputs[0]
        elif hasattr(model_outputs, "loss"):
            loss = model_outputs.loss
        else:
            raise TypeError(f"Invalid outputs type: {type(model_outputs)}.")
        return loss


else:
    assert False, f"Invalid value for `CSOFT_PACKAGE`: {CSOFT_PACKAGE}"
