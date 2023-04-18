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

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch


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
        it is important for subclasses to call `super().__init__()`. Otherwise,
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

        self._is_appliance = False
        self._ws_enabled = False
        if cm.use_cs():
            self._is_appliance = cm.is_appliance()

            state = cbtorch.state()
            if state.is_inside_execution_loop:
                raise RuntimeError(
                    "Metrics must be created outside the exeuction loop."
                )
            self._ws_enabled = cbtorch.env().weight_streaming_mode

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

    def on_device_state_dict(self) -> Dict[str, torch.Tensor]:
        """A hook for subclasses to inject metric state variables (WS only).

        In constrast to pipeline execution strategy where metrics are executed
        on the host, in weight streaming, metrics are part of the graph and are
        executed on device. As such any metric state variables that are updated
        need to be tracked to create a correct graph. This hook provides a
        mechanism for metric implementations to specify their state variables
        which will come up as outputs in the compile.
        """
        return dict()

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
        self._track_state()
        device_outputs = self.update_on_device(*args, **kwargs)
        assert isinstance(device_outputs, DeviceOutputs), (
            f"Expected device outputs to be of type `DeviceOutputs`, "
            f"but got `{type(device_outputs)}`."
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

                self.update_on_host(*cpu_args, **cpu_kwargs)
                self._num_updates += 1

            state = cbtorch.state()
            state.track_object(
                {
                    "cb_metric": {
                        self.name: [device_outputs.args, device_outputs.kwargs]
                    }
                },
                force=True,
            )

            state.register_activation_callback(_on_activations_received)

        else:

            @cm.step_closure
            def _update_on_host_closure(args, kwargs):
                args = cm.to_cpu(args)
                kwargs = cm.to_cpu(kwargs)
                self.update_on_host(*args, **kwargs)

                self._num_updates += 1

            _update_on_host_closure(device_outputs.args, device_outputs.kwargs)

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

    def _track_state(self):
        """Tracks and names the metric state."""
        state_dict = self.on_device_state_dict()
        if state_dict:
            if not self._ws_enabled:
                raise RuntimeError(
                    "On device metric state variables aren't supported for "
                    "Pipeline mode."
                )
            state = cbtorch.state()
            state.track_object(state_dict)
            cm.set_metric_state_names(state_dict, self.name)

    @classmethod
    def create_metric_impl_factory(
        cls,
        pipeline_metric_cls: Optional["CBMetric"] = None,
        ws_metric_cls: Optional["CBMetric"] = None,
    ) -> "CBMetric":
        """
        Returns a factory for generating a correct instance of a metric

        Args:
            pipeline_metric_cls: Optional `CBMetric` which specifies the compute
                for the pipeline execution strategy. Can be used for, and is the
                default for CPU/GPU
            ws_metric_cls: Optional `CBMetric` which species the compute in weight
                streaming execution strategy. Can be used for CPU/GPU
        Returns:
            metric_factory: (*args, **kwargs) -> `CBMetric` that automatically gives
                an instance of the correct metric given the execution strategy
        Raises:
            AssertionError: if values of `pipeline_metric_cls` or `ws_metric_cls`
                are invalid
        """
        if pipeline_metric_cls is None and ws_metric_cls is None:
            raise ValueError(
                f"At least one metric implementation for either pipeline or weight streaming "
                f"execution strategy must be provided, but both are None."
            )

        if pipeline_metric_cls is not None and not (
            issubclass(pipeline_metric_cls, cls)
        ):
            raise TypeError(
                f"expected provided pipeline metric to be a subclass of {cls}, "
                f"got {type(pipeline_metric_cls)}"
            )
        if ws_metric_cls is not None and not (issubclass(ws_metric_cls, cls)):
            raise TypeError(
                f"expected provided WS metric to be a subclass of {cls}, "
                f"got {type(ws_metric_cls)}"
            )

        # check if update on device was overridden
        if pipeline_metric_cls is not None and ws_metric_cls is not None:
            if (
                pipeline_metric_cls.update_on_device
                == CBMetric.update_on_device
            ):
                pipe_method = pipeline_metric_cls.update_on_host
            else:
                pipe_method = pipeline_metric_cls.update_on_device
            pipe_spec = inspect.getfullargspec(pipe_method)

            if ws_metric_cls.update_on_device == CBMetric.update_on_device:
                ws_method = ws_metric_cls.update_on_host
            else:
                ws_method = ws_metric_cls.update_on_device
            ws_spec = inspect.getfullargspec(ws_method)

            if pipe_spec != ws_spec:
                raise ValueError(
                    f"The signature seen by calling the pipeline metric implementation "
                    f"{pipeline_metric_cls} does not match the one seen by the WS metric "
                    f"implementation {ws_metric_cls}. The methods that define these "
                    f"signatures are {pipe_method} for the pipeline metric and {ws_method} for "
                    f"the weight streaming metric. Please ensure that they take the same "
                    f"args, kwargs, and have the same default values."
                )

        # return a factory for the metric
        def metric_factory(*args, **kwargs):
            """
            Returns an instance of the proper metric for the execution strategy

            Raises:
                TypeError: if the needed metric class was not provided
            """
            if cm.use_cs():
                if cbtorch.env().weight_streaming_mode:
                    if ws_metric_cls:
                        return ws_metric_cls(*args, **kwargs)
                    raise TypeError(
                        f"No weight streaming implementation was provided for this metric, "
                        f"but you are running the weight streaming strategy. "
                        f"The only registered metric is a pipeline implementation "
                        f"{pipeline_metric_cls}. If you'd like to use this metric with the "
                        f"weight streaming execution strategy, please provide an implementation, "
                        f"or change the execution strategy."
                    )
                else:
                    if pipeline_metric_cls:
                        return pipeline_metric_cls(*args, **kwargs)
                    raise TypeError(
                        f"No pipeline implementation was provided for this metric, "
                        f"but you are running the pipeline strategy. "
                        f"The only registered metric is a weight streaming implementation "
                        f"{ws_metric_cls}. If you'd like to use this metric with the "
                        f"pipeline execution strategy, please provide an implementation, "
                        f"or change the execution strategy."
                    )
            else:
                if pipeline_metric_cls:  # pipeline is cpu/gpu default
                    return pipeline_metric_cls(*args, **kwargs)
                return ws_metric_cls(*args, **kwargs)

        return metric_factory


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
