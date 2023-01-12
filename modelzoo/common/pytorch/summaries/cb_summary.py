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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch.utils.tensorboard import SummaryWriter

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch


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


class CBSummary(ABC):
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

        # Get the metric name
        if name in get_all_summaries():
            raise ValueError(
                f"Summary names must be unique, but got name `{name}` which "
                f"already exists."
            )
        self._name = name

        # Register the summary in the global list
        global _SUMMARIES
        _SUMMARIES[self._name] = self

        # Variable for storing the device outputs after each __call__
        self._last_device_outputs: Optional[DeviceOutputs] = None
        # Variable for storing the received activations in appliance mode
        self._last_host_outputs: Optional[DeviceOutputs] = None

        self._is_appliance = cm.is_appliance()

    @property
    def name(self):
        """Returns the name of the metric."""
        return self._name

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
        self._last_device_outputs = self.run_on_device(*args, **kwargs)
        assert isinstance(self._last_device_outputs, DeviceOutputs), (
            f"Expected device outputs to be of type `DeviceOutputs`, "
            f"but got `{type(self._last_device_outputs)}`."
        )

        if cm.use_cs():
            state = cbtorch.state()
            state.track_object(
                {
                    "cb_summary": {
                        self.name: [
                            self._last_device_outputs.args,
                            self._last_device_outputs.kwargs,
                        ]
                    }
                }
            )

        if self._is_appliance:

            def _on_activations_received():
                cpu_args = list()
                cpu_kwargs = dict()
                for tensor in self._last_device_outputs.args:
                    cpu_args.append(state.get_activation_for_output(tensor))
                for key, tensor in self._last_device_outputs.kwargs.items():
                    cpu_kwargs[key] = state.get_activation_for_output(tensor)

                self._last_host_outputs = self.run_on_host(
                    *cpu_args, **cpu_kwargs
                )

            state.register_activation_callback(_on_activations_received)

    def save(self, *args, **kwargs) -> None:
        """Evaluates the host portion of the summary and saves the results.

        This method is intended to be called inside the training loop whenever
        the summary results need to be saved. The arguments are passed directly
        to the `save_on_host` method.

        NOTE: This method should not be overriden. Instead, `run_on_host` and
        `save_on_host` should be overriden to provide full functionality.
        """
        if self._is_appliance:
            self.save_on_host(self._last_host_outputs, *args, **kwargs)
            self._last_host_outputs = None
        else:

            @cm.step_closure
            def _run_on_host_closure(
                device_args: List[Any],
                device_kwargs: Dict[str, Any],
                *args,
                **kwargs,
            ):
                device_args = cm.to_cpu(device_args)
                device_kwargs = cm.to_cpu(device_kwargs)
                host_outputs = self.run_on_host(*device_args, **device_kwargs)
                self.save_on_host(host_outputs, *args, **kwargs)

            _run_on_host_closure(
                self._last_device_outputs.args,
                self._last_device_outputs.kwargs,
                *args,
                **kwargs,
            )


# Keeps track of all registered summaries
_SUMMARIES = dict()


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
