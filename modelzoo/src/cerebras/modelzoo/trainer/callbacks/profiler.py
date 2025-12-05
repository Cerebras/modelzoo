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

import copy
import datetime
import json
import math
import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import cached_property, lru_cache
from itertools import tee
from typing import List, Optional, Set

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import Callback
from cerebras.modelzoo.trainer.callbacks.run_schedule import StageType
from cerebras.pytorch.backend.ltc_backend import register_after_tensor_to_cpu
from cerebras.pytorch.utils.data.utils import Schedule, infer_batch_size


class Profiler(Callback):
    """Base class for all Profiler callbacks."""

    @property
    def perf_metrics(self) -> dict:
        """Returns the performance metrics collected by the profiler."""
        return {}

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


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
            host_activities: List of ACT/WGT/CSX numbers to profile.
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

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        setattr(self, loop.on_batch_end_hook, self.on_validate_batch_end)

    def on_before_forward(self, trainer, model, batch, args, kwargs):
        self.conditional_reset(trainer)
        # Update the rate tracker with the batch size inside a step closure
        self.update(infer_batch_size(batch))

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.on_batch_end(trainer, infer_batch_size(batch))

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.on_batch_end(trainer, infer_batch_size(batch))

    @cstorch.step_closure
    def on_batch_end(self, trainer, batch_size):
        """Log the rate metrics at the end of a batch."""
        rate_metrics = {
            "local_samples_per_sec": self.rate,
            "avg_samples_per_sec": self.global_rate,
        }

        if batch_size:
            rate_metrics["avg_steps_per_sec"] = (
                rate_metrics["avg_samples_per_sec"] / batch_size
            )

        trainer.log_metrics(**rate_metrics)


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
                8,
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

    def log_flops(self, trainer):
        """Log the FLOP utilization of the model to managment server."""

        if (ml_flops := self.flop_utilization) is not None:
            trainer.log_metrics_dict(
                {"ml_flops": ml_flops},
                dest="TelemetryLogger",
            )

    @cstorch.step_closure
    def log_flops_in_step_closure(self, trainer):
        self.log_flops(trainer)

    def on_enter_train(self, trainer, stack, train_dataloader, loop, loop_idx):
        self.algo_flops = None
        self.supported_flops = None

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        self.algo_flops = None
        self.supported_flops = None

    def on_after_forward(self, trainer, model, outputs, batch):
        if self.algo_flops is None or self.supported_flops is None:
            self.get_flops(trainer)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.log_flops_in_step_closure(trainer)


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

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass


class TimeRemaining(float):
    """Subclass of float whose string representation is a human-readable time duration."""

    time_remaining: float
    stages_missing_duration: Optional[Set[str]] = None

    def __new__(cls, value, stages_missing_duration=None, **kwargs):
        instance = float.__new__(cls, value, **kwargs)
        instance.stages_missing_duration = stages_missing_duration
        return instance

    def __str__(self):
        return str(datetime.timedelta(seconds=round(self)))


class DurationProfiler(Callback):
    """Callback that tracks process duration and estimates the time remaining in a run."""

    def __init__(self):
        self.enable_act_frequency = False
        self.schedule = None
        self.setup_times = defaultdict(dict)
        self.teardown_times = defaultdict(dict)
        self.batch_timestamps = defaultdict(lambda: deque(maxlen=100))
        self.checkpoint_durations = deque(maxlen=10)
        self.batch_unfreeze_step = None
        self.tensor_hook_handle = None

    def pre_setup(self, trainer):
        self.enable_act_frequency = trainer.logging.enable_act_frequency
        self.schedule = trainer.schedule

    @property
    def _current_batch_timestamps(self):
        return self.batch_timestamps[self.schedule.current_stage.name]

    @property
    def _current_setup_time(self):
        return self.setup_times[self.schedule.current_stage.name]

    @property
    def _current_teardown_time(self):
        return self.teardown_times[self.schedule.current_stage.name]

    @property
    def _frozen_batches_remaining(self):
        """
        After a checkpoint, this returns how many batches remain before we expect
        the rate of processing to return to normal.

        In all other cases, returns 0.
        """
        if not self.batch_unfreeze_step:
            return 0
        return max(0, self.batch_unfreeze_step - self.schedule.batch_idx)

    def on_train_start(self, trainer, model, train_dataloader, loop, loop_idx):
        self.on_stage_start()

    def on_validate_start(self, trainer, model, val_dataloader, loop):
        self.on_stage_start()

    def on_stage_start(self):
        self._current_batch_timestamps.clear()
        self._current_setup_time["start"] = time.time()
        self.batch_unfreeze_step = None
        self.tensor_hook_handle = register_after_tensor_to_cpu(
            self.on_tensor_received
        )
        # The only outside state read by _account_for_checkpoints is the
        # checkpoint schedule, which can only change in on_fit_start.
        # We clear our cache here in on_stage_start to be safe.
        self._account_for_checkpoints.cache_clear()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.on_batch_end(trainer)

    def on_validate_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.on_batch_end(trainer)

    def on_batch_end(self, trainer):
        self._current_teardown_time["start"] = time.time()

        self.log_metrics(trainer)

    @cstorch.step_closure
    def log_metrics(self, trainer):
        trainer.log_metrics(
            loop_time_remaining=self.loop_time_remaining(),
            time_remaining=self.time_remaining(),
        )

    def on_train_step_end(self, trainer, model, outputs, batch):
        self.on_step_end(trainer)

    def on_validate_step_end(self, trainer, model, outputs, batch):
        self.on_step_end(trainer)

    @cstorch.step_closure
    def on_step_end(self, trainer):
        # CSX batches are measured in `self.on_tensor_received`.
        # For non-CSX batches, we don't need to fetch tensors from the
        # appliance so we can just mark the time here in on_step_end.
        if trainer.backend.is_csx or self._frozen_batches_remaining:
            return
        self._current_batch_timestamps.append(time.time())

    def on_train_end(self, trainer, model, loop, loop_idx):
        self.on_stage_end()

    def on_validate_end(self, trainer, model, loop):
        self.on_stage_end()

    def on_stage_end(self):
        self._current_teardown_time["duration"] = (
            time.time() - self._current_teardown_time["start"]
        )
        self._clear_tensor_hook_handle()

    def on_train_exception(self, trainer, exception):
        self._clear_tensor_hook_handle()

    def on_validate_exception(self, trainer, exception):
        self._clear_tensor_hook_handle()

    def _clear_tensor_hook_handle(self):
        self.tensor_hook_handle.remove()
        self.tensor_hook_handle = None

    def on_enter_validate(self, trainer, stack, val_dataloader, loop):
        setattr(self, loop.on_start_hook, self.on_validate_start)
        setattr(self, loop.on_end_hook, self.on_validate_end)
        setattr(self, loop.on_step_end_hook, self.on_validate_step_end)
        setattr(self, loop.on_batch_end_hook, self.on_validate_batch_end)

    def on_before_forward(self, trainer, model, batch, args, kwargs):
        @cstorch.step_closure
        def before_forward():
            if trainer.is_first_iteration:
                now = time.time()
                self._current_setup_time["duration"] = (
                    now - self._current_setup_time["start"]
                )
                self._current_batch_timestamps.append(now)
            if trainer.backend.is_csx and not self._frozen_batches_remaining:
                # This initial value of zero will be updated in our tensor received
                # hook as the output tensors for this batch arrive. Setting the
                # initial value here means the tensor received hook can just update
                # the most recent timestamp instead of having to figure out if a
                # new batch has started.
                self._current_batch_timestamps.append(0)

        before_forward()

    def on_tensor_received(self, tensor, name):
        if (
            not self._frozen_batches_remaining
            and len(self._current_batch_timestamps) >= 2
        ):
            # We expect there to be two timestamps already present before we
            # start updating the time here. The first is from our on_before_forward
            # step closure, and the second is the zero added for every batch on CSX
            # which we update here.
            self._current_batch_timestamps[-1] = time.time()

    def on_save_checkpoint(self, trainer, state_dict):
        self.checkpoint_start = time.time()

    def on_after_save_checkpoint(self, trainer, ckpt_path):
        self.checkpoint_durations.append(time.time() - self.checkpoint_start)

        if not self.schedule.stages:
            return

        self._current_teardown_time["start"] = time.time()

        batch_duration = self._batch_duration()[
            self.schedule.current_stage.name
        ]
        if batch_duration is None:
            # This condition is only met when `save_initial_checkpoint` is
            # enabled. In that case, we save a checkpoint before we ever see a
            # batch.
            return
        batches_per_checkpoint = self._checkpoint_duration() // batch_duration
        self.batch_unfreeze_step = (
            self.schedule.batch_idx + 1 + batches_per_checkpoint
        )

    def on_save_trainer_state(self, trainer, state_dict):
        pass

    def on_load_trainer_state(self, trainer, state_dict):
        pass

    def _batch_duration(self):
        batch_durations = {
            stage_name: [y - x for x, y in _pairwise(batch_timestamps)]
            for stage_name, batch_timestamps in self.batch_timestamps.items()
        }
        return {
            stage_name: statistics.median(durations) if durations else None
            for stage_name, durations in batch_durations.items()
        }

    def _checkpoint_duration(self):
        return (
            statistics.median(self.checkpoint_durations)
            if self.checkpoint_durations
            else None
        )

    def _last_train_stage(self):
        train_stages = [
            stage
            for stage in self.schedule.stages_remaining
            if stage.type == StageType.TRAIN_DATALOADER
        ]
        return train_stages[-1] if train_stages else None

    def loop_time_remaining(self) -> Optional[TimeRemaining]:
        """
        Estimates the seconds remaining in a train/eval epoch.

        Accounts for time processing batches and time in checkpointing.
        """
        if self.enable_act_frequency:
            # When enable_act_frequency is enabled, step closures only get called
            # on log steps which breaks our profiling.
            return None

        stage = self.schedule.current_stage
        if (
            self.schedule.checkpoint_schedule is not None
            and self.checkpoint_durations
            and self.schedule.current_stage.type == StageType.TRAIN_DATALOADER
        ):
            time_remaining = self._account_for_checkpoints(
                stage.start_step,
                stage.steps,
                self._batch_duration()["train"],
                self._checkpoint_duration(),
                current_step=self.schedule.batch_idx + stage.start_step,
                frozen_batches_remaining=self._frozen_batches_remaining,
                is_last_train=stage == self._last_train_stage(),
            )
        else:
            time_remaining = (
                self.schedule.batches_remaining
                * self._batch_duration()[stage.name]
            )

        return TimeRemaining(time_remaining)

    def time_remaining(self) -> Optional[TimeRemaining]:
        """
        Estimates the seconds remaining in a run, including all future train/evals.

        Accounts for setup, teardown, batch processing, and checkpointing using the
        run schedule and progress provided by the RunSchedule callback.
        """
        if (time_remaining := self.loop_time_remaining()) is None:
            return None

        if "duration" in self._current_teardown_time:
            # Account for the teardown time of the current stage which is not
            # included in loop_time_remaining().
            time_remaining += self._current_teardown_time["duration"]

        # Precompute these values before the loop.
        batch_duration = self._batch_duration()
        checkpoint_duration = self._checkpoint_duration()
        last_train_stage = self._last_train_stage()

        stages_missing_duration = set()
        for stage in self.schedule.stages_remaining:
            if (
                not stage.steps
                or not batch_duration.get(stage.name, None)
                or "duration" not in self.setup_times[stage.name]
                or "duration" not in self.teardown_times[stage.name]
            ):
                # We mark a stage as missing duration if it's missing any of:
                # number of steps, batch duration, or setup/teardown duration.
                stages_missing_duration.add(stage.name)
            if not stage.steps or not batch_duration.get(stage.name, None):
                # We can estimate a lower bound time for future stages as long
                # as we have the number of steps and the batch duration. If we
                # don't have those, there's nothing for us to do.
                continue
            if "duration" in (setup_time := self.setup_times[stage.name]):
                time_remaining += setup_time["duration"]
            if "duration" in (teardown_time := self.teardown_times[stage.name]):
                time_remaining += teardown_time["duration"]

            if (
                self.schedule.checkpoint_schedule is not None
                and checkpoint_duration
                and stage.type == StageType.TRAIN_DATALOADER
            ):
                # Here we use the fact that there is repetition in long schedules
                # to make our implementation faster. We perform our checkpoint
                # accounting modulo checkpoint_steps and use an LRU cache so that
                # the only time we actually run the computation is the first time
                # we see a given stage configuration.
                checkpoint_steps = self.schedule.checkpoint_schedule.step
                time_remaining += self._account_for_checkpoints(
                    stage.start_step % checkpoint_steps,
                    stage.steps,
                    batch_duration["train"],
                    checkpoint_duration,
                    current_step=(stage.start_step % checkpoint_steps),
                    frozen_batches_remaining=0,
                    is_last_train=stage == last_train_stage,
                )
            else:
                time_remaining += stage.steps * batch_duration[stage.name]

        if self.schedule.checkpoint_schedule and not self.checkpoint_durations:
            stages_missing_duration.add("checkpoint")

        return TimeRemaining(time_remaining, stages_missing_duration)

    @lru_cache
    def _account_for_checkpoints(
        self,
        stage_start_step: int,
        stage_steps: int,
        train_batch_duration: int,
        checkpoint_duration: int,
        current_step: int,
        frozen_batches_remaining: int,
        is_last_train: bool,
    ) -> float:
        """
        For training stages with checkpointing, we need to account for the fact
        that checkpoints occur while runtime is asynchronously processing batches.
        When the checkpoint is done, if there are batches that have been processed
        in the meantime, we see an increase in batch processing rate as framework
        catches up to runtime.
        """
        checkpoint_schedule = copy.copy(self.schedule.checkpoint_schedule)
        if is_last_train:
            # Since we're using modulo arithmetic to speed things up,
            # we need to account for the checkpoint schedule having
            # `include_last` enabled. Moving the end of the checkpoint
            # schedule to the end of this stage is sufficient, when we
            # know that this stage is the last train loop.
            checkpoint_schedule.end = stage_start_step + stage_steps

        end_step = stage_start_step + stage_steps
        if (
            checkpoint_duration
            < checkpoint_schedule.step * train_batch_duration
        ):
            # In this case, batch processing time dominates. The total time for
            # the stage is batch processing time, plus time for any checkpoints
            # that happen towards the end of the stage (and thus have no batches
            # after them to cover the checkpoint time).
            batches_remaining = max(
                0, end_step - current_step - frozen_batches_remaining
            )
            time_remaining = batches_remaining * train_batch_duration
            time_remaining += _time_after_last_checkpoints(
                current_step,
                end_step,
                train_batch_duration,
                checkpoint_duration,
                checkpoint_schedule,
            )
            return time_remaining

        # In this case, checkpoint time dominates. The total time for the stage
        # is checkpoint time, plus time for any batches prior to the first
        # checkpoint in the stage.
        num_checkpoints_in_stage = _num_checkpoints_in_stage(
            current_step, end_step, checkpoint_schedule
        )
        if not num_checkpoints_in_stage:
            return (end_step - current_step) * train_batch_duration
        time_remaining = num_checkpoints_in_stage * checkpoint_duration
        time_remaining += (
            _steps_before_first_checkpoint(
                stage_start_step, current_step, end_step, checkpoint_schedule
            )
            * train_batch_duration
        )
        return time_remaining


def _time_after_last_checkpoints(
    current_step: int,
    end_step: int,
    train_batch_duration: float,
    checkpoint_duration: float,
    checkpoint_schedule: Schedule.Range,
):
    # This function is only called when batch processing dominates the runtime
    # so we can assume that the batches elapsed during a checkpoint are less
    # than the number of steps between checkpoints. With this assumption the loop
    # below will hit at most two checkpoint steps that fall within the range.
    assert checkpoint_duration < checkpoint_schedule.step * train_batch_duration
    batches_per_checkpoint = math.ceil(
        checkpoint_duration / train_batch_duration
    )
    time_after_last_checkpoints = 0
    start_step = max(current_step, end_step - batches_per_checkpoint - 1)
    for step in checkpoint_schedule:
        if step < start_step:
            continue
        if step >= end_step:
            break
        steps_after_checkpoint = end_step - 1 - step
        time_after_last_checkpoints += checkpoint_duration - (
            steps_after_checkpoint * train_batch_duration
        )
    return time_after_last_checkpoints


def _steps_before_first_checkpoint(
    start_step: int,
    current_step: int,
    end_step: int,
    checkpoint_schedule: Schedule.Range,
):
    for step in checkpoint_schedule:
        if step < start_step:
            continue
        if step >= end_step:
            break
        return max(0, step - current_step + 1)
    return 0


def _num_checkpoints_in_stage(
    start_step: int, end_step: int, checkpoint_schedule: Schedule.Range
):
    checkpoints_in_stage = 0
    for step in checkpoint_schedule:
        if step < start_step:
            continue
        if step >= end_step:
            break
        checkpoints_in_stage += 1
    return checkpoints_in_stage


def _pairwise(iterable):
    """
    Taken directly from the python docs as itertools.pairwise was only moved
    into itertools in python 3.10. We can switch to that when we upgrade python.

    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
