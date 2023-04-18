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
Base class for creating summaries compatible with CPU/GPU/CS-X.
"""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch

# Keeps track of all registered summaries
_SUMMARIES: Dict[str, "CBSummary"] = dict()


@dataclass
class DeviceOutputs:
    """Class for encapsulating the outputs of `CBSummary.run_on_device`.

    Args:
        args: postional arguments which are passed to `CBSummary.run_on_host`
            once they are converted to CPU tensors.
        kwargs: keyword arguments which are passed to `CBSummary.run_on_host`
            once they are converted to CPU tensors.
    """

    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


class CBSummaryMeta(ABCMeta):
    """Metaclass for CBSummary to handle instance creation."""

    def __call__(cls, name, *args, **kwargs):
        # See if a summary with the given name already exists
        instance = _SUMMARIES.get(name)
        if name not in _SUMMARIES:
            # Create one if it doesn't exist
            instance = super().__call__(name, *args, **kwargs)
            # Register the summary in the global list
            _SUMMARIES[instance.name] = instance
        return instance


class CBSummary(metaclass=CBSummaryMeta):
    """Base class for creating summaries on CS devices.

    Subclasses must override methods to provide the full functionality of the
    metric. These methods are meant to split the summary operations into 3
    portions:
        1. run_on_device: Compiles and runs on the device (i.e., Cerebras).
        2. run_on_host: Runs on the host (i.e., CPU).
        3. save: Saves the results as summaries.

    These summaries also support running on CPU and GPU.
    """

    def __init__(self, name: str):
        """Constructs a `CBSummary` instance.

        This also registers the summary in the global pool of summaries.
        Therefore, it is import for subclasses to call `super().__init__()`.
        Otherwise, the summaries will not run.

        Args:
            name: Name of the summary. Generally, this is the name used to tag
                the summaries with in TensorBoard.
        """
        super().__init__()

        # SYNC mode not supported yet
        if cm.num_receivers() > 1:
            raise RuntimeError(
                "Summaries with multiple receiver ordinals are currently "
                f"not supported. `num_receivers` was {cm.num_receivers()}"
            )

        self._name = name

        # Variable for storing the received summaries
        self._cached_cpu_activations = []

        self._is_appliance = cm.is_appliance()

    @property
    def name(self):
        """Returns the name of the metric."""
        return self._name

    # pylint: disable=no-self-use
    def run_on_device(self, *args, **kwargs) -> DeviceOutputs:
        """Define the portion of the summary computation that runs on device.

        This method must return a `DeviceOutputs` object whose args/kwargs
        can only contain a item/list/tuple/dict of torch tensors or Nones.
        These tensors may be converted to CPU tensors at the step boundary and
        passed to `run_on_host` to do the host (i.e. CPU) portion of the
        computation.

        The default implementation is just a passthrough where the arguments
        are converted to host tensors as is.

        This method is called for every iteration.

        NOTE: No tensors should be evaluated in this method. This method merely
        defines the operations in the graph that runs on device.

        Returns:
            An instance of `DeviceOutputs`.
        """
        return DeviceOutputs(args=list(args), kwargs=kwargs)

    @abstractmethod
    def run_on_host(self, *args, **kwargs) -> Any:
        """Define the portion of the summary computation that runs on host.

        This methods takes as inputs the outputs of `run_on_device` whose
        tensors have been evaluated and converted to CPU tensors. It can do
        any sort of computation on the host (e.g., applying reduction). Its
        return value is passed to `save_on_host`.

        This method is called only when the summary is to be saved.

        Returns:
            The computed summary value.
        """
        raise NotImplementedError

    @abstractmethod
    def save_on_host(
        self, host_outputs: Any, writer: SummaryWriter, step: int
    ) -> None:
        """Save the computed summary into events files.

        Args:
            host_outputs: The return value of `run_on_host` method. This is
                the computed summary value.
            writer: A writer for writing summaries to events files.
            step: The current global step.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> None:
        """Run the device portion of the computation and store its result.

        The arugments to this method are passed directly to `run_on_device`.
        """
        # The device portion needs to run every step to produce a stable graph
        device_outputs = self.run_on_device(*args, **kwargs)
        assert isinstance(device_outputs, DeviceOutputs), (
            f"Expected device outputs to be of type `DeviceOutputs`, "
            f"but got `{type(device_outputs)}`."
        )

        # Detach and clone device outputs to ensure we use the "current" value
        for idx, tensor in enumerate(device_outputs.args):
            if isinstance(tensor, torch.Tensor):
                device_outputs.args[idx] = tensor.detach().clone()
        for key, tensor in device_outputs.kwargs.items():
            if isinstance(tensor, torch.Tensor):
                device_outputs.kwargs[key] = tensor.detach().clone()

        if cm.use_cs():
            state = cbtorch.state()
            state.track_object(
                {
                    "cb_summary": {
                        self.name: [device_outputs.args, device_outputs.kwargs]
                    }
                },
                force=self._is_appliance,
            )

        if self._is_appliance:

            def _on_activations_received():
                cpu_args = [
                    state.get_activation_for_output(tensor)
                    if isinstance(tensor, torch.Tensor)
                    else tensor
                    for tensor in device_outputs.args
                ]
                cpu_kwargs = {
                    key: state.get_activation_for_output(tensor)
                    if isinstance(tensor, torch.Tensor)
                    else tensor
                    for key, tensor in device_outputs.kwargs.items()
                }

                self._cached_cpu_activations.append(
                    self.run_on_host(*cpu_args, **cpu_kwargs)
                )

            state.register_activation_callback(_on_activations_received)
        else:

            @cm.step_closure
            def _run_on_host_closure(
                device_args: List[Any], device_kwargs: Dict[str, Any],
            ):
                device_args = cm.to_cpu(device_args)
                device_kwargs = cm.to_cpu(device_kwargs)
                self._cached_cpu_activations.append(
                    self.run_on_host(*device_args, **device_kwargs)
                )

            _run_on_host_closure(
                device_outputs.args, device_outputs.kwargs,
            )

    @cm.step_closure
    def save(self, *args, **kwargs) -> None:
        """Saves the results.

        This method is intended to be called inside the training loop whenever
        the summary results need to be saved. The arguments are passed directly
        to the `save_on_host` method.

        NOTE: This method should not be overriden. Instead, `run_on_host` and
        `save_on_host` should be overriden to provide full functionality.
        """
        if not self._cached_cpu_activations:
            raise RuntimeError(
                f"Attempting to save summary {self._name} before it "
                f"was fetched. Please ensure to run the summary's host "
                f"closure to fetch the latest value before calling "
                f"save()."
            )
        for value in self._cached_cpu_activations:
            self.save_on_host(value, *args, **kwargs)
        self._cached_cpu_activations.clear()

    @cm.step_closure
    def discard(self) -> None:
        """Discards cached activations on host."""
        self._cached_cpu_activations.clear()


def get_all_summaries() -> Dict[str, CBSummary]:
    """Returns all registered summaries."""
    return _SUMMARIES


def save_all_summaries(writer: SummaryWriter, step: int) -> None:
    """Calls `save` on all registered summaries.

    Args:
        writer: A writer for writing summaries to events files.
        step: The current global step.
    """
    for summary in get_all_summaries().values():
        summary.save(writer, step)


def discard_cached_summaries() -> None:
    """Discards cached activations for all summaries."""
    for summary in get_all_summaries().values():
        summary.discard()
