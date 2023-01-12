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
Utilities for saving scalar summaries.
"""
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from modelzoo.common.pytorch.summaries.cb_summary import (
    CBSummary,
    get_all_summaries,
)


class ScalarSummary(CBSummary):
    """A class for providing scalar summaries on CS/CPU/GPU devices."""

    _REDUCTION_FN_MAP = {
        "mean": torch.mean,
        "max": torch.max,
        "min": torch.min,
        "sum": torch.sum,
    }

    def __init__(self, name: str, reduction: Optional[str] = None):
        """Constructs a `ScalarSummary` instances.

        Args:
            name: Name of the summary. This is the tag that appears in
                TensorBoard.
            reduction: The reduction function to apply to the input tensors.
                Defaults to no reduction.
        """
        super().__init__(name)

        self._output_idx = None

        if reduction is None:
            self._reduction_fn = lambda x: x  # no-op
        else:
            self._reduction_fn = self._REDUCTION_FN_MAP.get(reduction)
            if self._reduction_fn is None:
                raise ValueError(
                    f"Unknown reduction `{reduction}`. Available reduction "
                    f"types are: {list(self._REDUCTION_FN_MAP)}."
                )

    def run_on_host(self, tensor: torch.Tensor) -> float:
        """Runs the host portion of the summary computation.

        Args:
            tensor: The tensor to be summarized.
        Returns:
            The summarized float value.
        """
        return float(self._reduction_fn(tensor))

    def save_on_host(
        self, host_outputs: float, writer: SummaryWriter, step: int,
    ) -> None:
        """Saves the scalar summary to events file.

        host_outputs: The summarized float value to write to events file.
        writer: A writer for writing summaries to events files.
        step: The current global step.
        """
        writer.add_scalar(self.name, host_outputs, global_step=step)


def scalar_summary(
    name: str, tensor: torch.Tensor, reduction: Optional[str] = None
):
    """Convenience method for creating and running scalar summaries.

    This method searches registered summaries for the given name. If one is
    found, it uses it. Otherwise, it creates a new summary and runs the tensor
    through that summary.

    Args:
        name: Name of the summary. This is the tag that appears in TensorBoard.
        tensor: The tensor to be summarized.
        reduction: The reduction function to apply to the tensor. Defaults to
            no reduction.
    """
    # See if a summary with the given name already exists
    summary = get_all_summaries().get(name)
    if summary is None:
        # Create one if it doesn't exist
        summary = ScalarSummary(name, reduction=reduction)
    # Run the summary op
    summary(tensor)
