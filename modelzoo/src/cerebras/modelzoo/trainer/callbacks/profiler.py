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

"""Implements a callback to profile the model using the Cerebras Profiler."""

import json
from contextlib import contextmanager
from functools import cached_property
from typing import List, Optional

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback
from cerebras.pytorch.utils.data.utils import infer_batch_size


class Profiler(Callback):
    """Base class for all Profiler callbacks."""

    @property
    def perf_metrics(self) -> dict:
        """Returns the performance metrics collected by the profiler."""
        return {}


class OpProfiler(Profiler):
    """Callback class that profiles the model using the Cerebras Profiler."""

    def __init__(
        self,
        start_step: int = -1,
        end_step: int = -1,
        host_activities: Optional[List[str]] = None,
    ):
        """
        Args:
            start_step: Start step for profiling.
            end_step: End step for profiling.
            host_activities: List of ACT/WGT/CSX numbers to profile
        """
        self.start_step = start_step
        self.end_step = end_step
        self.host_activities = host_activities

    @contextmanager
    def setup_op_profiler(self, trainer):
        """Context manager to profile the model using the Cerebras Op Profiler."""
        with cstorch.profiler.profile(
            schedule=cstorch.profiler.schedule(
                start_step=self.start_step, end_step=self.end_step
            ),
            host_activities=self.host_activities,
        ) as profiler:
            yield

        profiler.export_chrome_trace(trainer.model_dir / "chrome_trace.json")
        if profiler.appliance_response:
            trainer.logger.info(f"{profiler.get_summary()}")

    def on_enter_fit(
        self, trainer, stack, train_dataloader, val_dataloader, loop
    ):
        stack.enter_context(self.setup_op_profiler(trainer))


class RateProfiler(Profiler):
    """Callback that tracks the rate of samples processed by the model measured
    by the client.
    """

    def __init__(self):
        """Sets up the rate tracker."""
        self.rate_tracker = cstorch.utils.tracker.RateTracker()

    @cached_property
    def rate(self) -> float:
        """Smoothed samples/second of all the samples added since last queried.

        This value is cached and recomputed only when the count is updated.
        """
        return self.rate_tracker.rate()

    @cached_property
    def global_rate(self) -> float:
        """
        Non-smoothed samples/second since the beginning of when the rate tracker
        as initialized.

        This value is cached and recomputed only when the count is updated.
        """
        return self.rate_tracker.global_rate()

    @cached_property
    def elapsed_seconds(self) -> float:
        """Time (seconds) elapsed since the last reset.

        This value is cached and recomputed only when the count is updated.
        """
        return self.rate_tracker.elapsed_seconds()

    @property
    def total_count(self) -> int:
        """Total number of samples processed since the last reset."""
        return self.rate_tracker.total_count

    def clear_cache(self):
        """Clear all cached properties."""
        self.__dict__.pop("rate", None)
        self.__dict__.pop("global_rate", None)
        self.__dict__.pop("elapsed_seconds", None)

    @cstorch.step_closure
    def conditional_reset(self, trainer):
        """Reset the rate tracker if on first iteration."""
        # We reset the tracker's start time inside a step closure here so that
        # the time is reset after compile and execute setup is done.
        # TODO: add an offset of 1 so that the time isn't ~0 when the first
        #       rate/global_rate is computed
        if trainer.is_first_iteration:
            self.rate_tracker.reset()
            self.clear_cache()

    @cstorch.step_closure
    def update(self, count):
        """Update the rate tracker with the count of samples processed."""
        # If dataloader returns batches whose tensors have inconsistent batch sizes,
        # we may get a `None` count and skip it here.
        if count is not None:
            self.rate_tracker.add(count)
        self.clear_cache()

    @property
    def perf_metrics(self):
        return {
            "total_samples": self.total_count,
            "total_time": self.elapsed_seconds,
            "rate": self.rate,
            "global_rate": self.global_rate,
            "samples_per_sec": self.global_rate,
        }

    def on_before_forward(self, trainer, model, batch, args, kwargs):
        self.conditional_reset(trainer)
        # Update the rate tracker with the batch size inside a step closure
        self.update(infer_batch_size(batch))


class FlopUtilization(Profiler):
    """Callback that computes the FLOP utilization of the model."""

    def __init__(self):
        """Initializes the FLOP utilization tracker."""
        self.algo_flops = None
        self.supported_flops = None
        self.rate_profiler = None

    def setup(self, trainer):
        self.rate_profiler = trainer.get_callback(RateProfiler)

    @property
    def flop_utilization(self) -> Optional[float]:
        """Returns the FLOP utilization of the model."""
        if self.algo_flops and self.supported_flops and self.rate_profiler:
            global_rate = self.rate_profiler.global_rate
            return round(
                (float(self.algo_flops * global_rate) / self.supported_flops)
                * 100,
                3,
            )
        return None

    @property
    def perf_metrics(self) -> dict:
        flop_utilization = self.flop_utilization
        if flop_utilization is not None:
            return {"flops_utilization": flop_utilization}
        return {}

    @cstorch.step_closure
    def get_flops(self, trainer):
        """Get the FLOPs of the model from the compile response."""
        from cerebras.pytorch.backend import current_backend_impl

        backend = current_backend_impl()
        if (
            backend.is_csx
            and backend.appliance
            and backend.appliance.compile_resp
        ):
            self.algo_flops = (
                backend.appliance.compile_resp.perf_info.algo_flops
            )
            self.supported_flops = (
                backend.appliance.compile_resp.perf_info.supported_flops
            )

    def on_enter_train(self, trainer, stack, train_dataloader, loop, loop_idx):
        self.algo_flops = None
        self.supported_flops = None

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        self.algo_flops = None
        self.supported_flops = None

    def on_after_forward(self, trainer, model, outputs, batch):
        if self.algo_flops is None or self.supported_flops is None:
            self.get_flops(trainer)


class SavePerformanceData(Callback):
    """Callback that saves the performance metrics collected by all Profiler
    callbacks.
    """

    @contextmanager
    def save_perf_json(self, trainer):  # pylint: disable=no-self-use
        """Context manager to save the performance metrics to a JSON file."""
        yield
        perf_metrics = {
            k: v
            for callback in trainer.get_callbacks(Profiler)
            for k, v in callback.perf_metrics.items()
        }
        with open(trainer.executor.artifact_dir / "performance.json", "w") as f:
            json.dump(perf_metrics, f, sort_keys=True, indent=4)

    def on_enter_train(self, trainer, stack, train_dataloader, loop, loop_idx):
        stack.enter_context(self.save_perf_json(trainer))

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        stack.enter_context(self.save_perf_json(trainer))
