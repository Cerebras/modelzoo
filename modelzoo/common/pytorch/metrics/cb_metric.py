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

from modelzoo.common.pytorch import cb_model as cm


@dataclass
class DeviceOutputs:
    """Class for encapsulating the outputs of `CBMetric.update_on_device`.

    Args:
        args: postional arguments which are passed to `CBMetric.update_on_host`
            once they are converted to CPU tensors.
        kwargs: keyword arguments which are passed to `CBMetric.update_on_host`
            once they are converted to CPU tensors.
    """

    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


class CBMetric(ABC):
    """Base class for creating metrics on CS devices.

    Subclasses must override methods to provide the full functionality of the
    metric. These methods are meant to split the computation graph into 2
    portions:
        1. update_on_device: Compiles and runs on the device (i.e., Cerebras).
        2. update_on_host: Runs on the host (i.e., CPU).

    These metrics also support running on CPU and GPU.
    """

    def __init__(self, name: Optional[str] = None):
        """Constructs a `CBMetric` instance.

        This also registers the metric in the global pool of metrics. Therefore,
        it is import for subclasses to call `super().__init__()`. Otherwise,
        the metrics will not run.

        Args:
            name: Name of the metric. If None or empty string, it defaults to
                the name of the class.
        """
        # Keeps track of total number of times the metric was updated
        self._num_updates = 0

        # Get the metric name
        self._name = self._get_unique_name(name)

        # Register the metric in the global list
        global _METRICS
        assert self._name not in _METRICS
        _METRICS[self._name] = self

        self.appliance_enabled = False
        if cm.use_cs():
            from modelzoo.common.pytorch import cbtorch

            self.appliance_enabled = cbtorch.env().appliance

            state = cbtorch.state()
            if state.is_inside_execution_loop:
                raise RuntimeError(
                    "Metrics must be created outside the exeuction loop."
                )
            state.register_metric(self)

        # Stores the state tensors on the device
        self.init_state()

    @property
    def num_updates(self):
        """Returns number of times the metric was updated (i.e., stepped)."""
        return self._num_updates

    @property
    def name(self):
        """Returns the name of the metric."""
        return self._name

    def init_state(self):
        """Sets the initial state of the metric.

        Subclasses should override this method to provide any metric-specific
        states. This method is called once as part of `__init__`.
        """

    def update_on_device(self, *args, **kwargs) -> DeviceOutputs:
        """Define the portion of the metric computation that runs on the device.

        This method must return a `DeviceOutputs` object whose args/kwargs
        can only contain a item/list/tuple/dict of torch tensors or Nones.
        These tensors are converted to CPU tensors at the step boundary and
        passed to `update_on_host` to do the host (i.e. CPU) portion of the
        computation.

        The default implementation is just a passthrough where the arguments
        are converted to host tensors as is.

        This method is called for every iteration.

        NOTE: No tensors should be evaluated in this method. This method merely
        defines the operations in the graph that runs on device.
        """
        return DeviceOutputs(args=list(args), kwargs=kwargs)

    @abstractmethod
    def update_on_host(self, *args, **kwargs) -> None:
        """Define the portion of the metric computation that runs on host.

        This methods takes as inputs the outputs of `update_on_device` whose
        tensors have been evaluated and converted to CPU tensors. It can do
        any sort of computation on the host (e.g., updating the metric state).

        This method is called for every iteration.
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Any:
        """Returns the computed metric value over many iterations.

        This is the "reduction" part of the metric over all steps.
        """
        raise NotImplementedError

    def reset_state(self) -> None:
        """Resets the metric state.

        Subclasses should override this method to clear any metrics-specific
        states.
        """

    def reset(self) -> None:
        """Resets the metric state.

        Instead of overriding this method, subclasses should override
        `reset_state` method which is called internally in this method.
        """
        self._num_updates = 0
        self.reset_state()

    def __call__(self, *args, **kwargs) -> None:
        """Run the metric accumulator over one execution step.

        The arugments to this method are passed directly to `update_on_device`.
        """
        device_outputs = self.update_on_device(*args, **kwargs)
        assert isinstance(device_outputs, DeviceOutputs), (
            f"Expected device outputs to be of type `DeviceOutputs`, "
            f"but got `{type(device_outputs)}`."
        )

        self._update_on_host_closure(device_outputs.args, device_outputs.kwargs)

    @cm.step_closure
    def _update_on_host_closure(self, args, kwargs):
        if not self.appliance_enabled:
            args = cm.to_cpu(args)
            kwargs = cm.to_cpu(kwargs)
        self.update_on_host(*args, **kwargs)

        self._num_updates += 1

    def _get_unique_name(self, name: Optional[str] = None):
        """Returns a unique name for this metric.

        Args:
            name: The default name prefix to use. If None, class name is used.
                Defaults to None.
        """
        idx = 0
        prefix = name or self.__class__.__name__

        unique_name = prefix
        for name in get_all_metrics():
            if name == unique_name:
                idx += 1
                unique_name = f"{prefix}_{idx}"
        return unique_name

    def set_metric_state(self, state_dict):
        assert (
            cm.use_cs()
        ), "set_metric_state should only be called for CSX runs"
        from modelzoo.common.pytorch import cbtorch

        state = cbtorch.state()
        state.track_object(state_dict)
        cm.set_metric_state_names(state_dict, self.name)


# Keeps track of all registered metrics
_METRICS = dict()


def get_all_metrics() -> Dict[str, CBMetric]:
    """Returns all registered metrics."""
    return _METRICS


def compute_all_metrics() -> Dict[str, Any]:
    """Computes all the registered metrics and returns them in a dict."""
    metrics = dict()
    for name, metric in get_all_metrics().items():
        metrics[name] = metric.compute()
    return metrics


def reset_all_metrics() -> None:
    """Resets the internal state of all reistered metrics."""
    for metric in get_all_metrics().values():
        metric.reset()
