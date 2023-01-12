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

import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.basic_session_run_hooks import (
    LoggingTensorHook,
    SecondOrStepTimer,
)
from tensorflow.python.training.session_run_hook import (
    SessionRunArgs,
    SessionRunHook,
)
from tensorflow.python.training.summary_io import SummaryWriterCache

from modelzoo.common.tf.optimizers.Trainer import Trainer


def get_grad_accum_hooks(
    trainer, runconfig_params, summary_dict=None, logging_dict=None
):
    """
    Initializes a `GradAccumLoggingTensorHook`.

    :param Trainer trainer: common.optimizers.Trainer object
        used for model training with gradient accumulation.
    :param dict runconfig_params: Runtime configs dictionary.
    :param dict summary_dict: Dictionary with values containing
        tensors to be written into summaries and keys containing
        summary names, e.g., {"train/total_loss", total_loss}.
        In case of distributed training, the tensors will be
        mean reduced accross all replicas.
    :param dict logging_dict: Dictionary with values containing
        tensors to be logged and keys containing
        the log names, e.g., {"loss", total_loss} will log
        "loss = <total_loss_value>, step = <global_step>"
    """
    if not trainer.is_grad_accum():
        return None
    hooks = []
    log_step_count_steps = runconfig_params.get("log_step_count_steps", 100)

    if log_step_count_steps:
        hooks.append(
            GradAccumStepCounterHook(
                trainer,
                every_n_steps=log_step_count_steps,
                output_dir=runconfig_params["model_dir"],
            )
        )
        if logging_dict:
            hooks.append(
                GradAccumLoggingTensorHook(
                    trainer, logging_dict, every_n_steps=log_step_count_steps,
                )
            )
    if summary_dict:
        hooks.append(
            GradAccumSummarySaverHook(
                trainer,
                summary_dict,
                save_steps=runconfig_params["save_summary_steps"],
                output_dir=runconfig_params["model_dir"],
            )
        )

    return hooks


class GradAccumSummarySaverHook(SessionRunHook):
    """
    Saves summaries every N steps, where N is the number
    of effective batches seen by an optimizer in the gradient
    accumulation mode.
    """

    def __init__(
        self,
        trainer,
        tensors,
        save_steps=None,
        save_secs=None,
        output_dir=None,
        summary_writer=None,
    ):
        """
        Initializes a `GradAccumSummarySaverHook`.

        :param Trainer trainer: common.optimizers.Trainer object
            used for model training with gradient accumulation.
        :param dict tensors: `dict` that maps string-valued tags to
            tensors/tensor names.
        :param int save_steps: Save summaries every N steps. Exactly one of
            `save_secs` and `save_steps` should be set.
        :param int save_secs: Save summaries every N seconds.
        :param string output_dir: The directory to save the summaries to. Only used if
            no `summary_writer` is supplied.
        :param SummaryWriter summary_writer: If `None` and an `output_dir` was passed,
            one will be created accordingly.
        """

        if not isinstance(trainer, Trainer):
            raise ValueError(
                f"`trainer` should be object of class Trainer. "
                f"Received {type(trainer)} instead."
            )
        self._grad_accum_steps = trainer.grad_accum_steps
        self._log_trainer_summaries = trainer.log_summaries
        self._gradient_global_norm = trainer.gradient_global_norm
        self._loss_scale_value = trainer.loss_scale_value
        self._is_loss_scale_optimizer = trainer.uses_loss_scaling()
        self._lr = trainer.get_learning_rate()
        self._tensors = _aggregate_tensors(tensors)

        if output_dir is None and summary_writer is None:
            raise ValueError(
                "Both output_dir and summary_writer can't be None."
            )
        self._output_dir = output_dir
        self._summary_writer = summary_writer
        self._prefix = "grad_accum_summary_saver_hook"

        if (save_steps is None) == (save_secs is None):
            raise ValueError(
                "Exactly one of save_steps and save_secs should be provided."
            )
        self._timer = SecondOrStepTimer(
            every_secs=save_secs, every_steps=save_steps
        )
        self._summary_collection_name = "grad_accum_summaries"

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._next_step = None

        self._current_accum_step, accum_step_counter = _get_current_accum_step(
            self._grad_accum_steps, self._prefix
        )
        self._current_tensors = _get_current_accum_tensors(
            self._tensors,
            accum_step_counter,
            self._grad_accum_steps,
            self._prefix,
        )
        # Disable existing summaries. All summaries
        # execution will be controlled by this hook.
        # Warn the user that some summaries might be disabled!
        clear_summary_names = [
            tens.op.name
            for tens in tf.compat.v1.get_collection(ops.GraphKeys.SUMMARIES)
        ]
        lost_summaries = []
        for summ_name in clear_summary_names:
            if summ_name not in self._current_tensors.keys():
                lost_summaries.append(summ_name)
        if len(lost_summaries) > 0:
            logging.warn(
                f"The grad accum logging hook is clearing these summaries!\n"
                f"    {lost_summaries}\nYou might need to re-add them through"
                " the `summary_dict` parameter of `GradAccumSummarySaverHook`"
            )
        tf.compat.v1.get_default_graph().clear_collection(
            ops.GraphKeys.SUMMARIES
        )
        # Create list of summaries
        self._summary = []
        for name, cur_tensor in self._current_tensors.items():
            self._summary.append(
                tf.compat.v1.summary.scalar(
                    name, cur_tensor, collections=self._summary_collection_name
                )
            )
        if self._log_trainer_summaries:
            self._summary.append(
                tf.compat.v1.summary.scalar(
                    "train/lr",
                    self._lr,
                    collections=self._summary_collection_name,
                )
            )
            self._summary.append(
                tf.compat.v1.summary.scalar(
                    "train/unclipped_grad_norm",
                    self._gradient_global_norm,
                    collections=self._summary_collection_name,
                )
            )
            if self._is_loss_scale_optimizer:
                self._summary.append(
                    tf.compat.v1.summary.scalar(
                        "train/loss_scale",
                        self._loss_scale_value,
                        collections=self._summary_collection_name,
                    )
                )

    def before_run(self, run_context):
        current_accum_step_val = run_context.session.run(
            self._current_accum_step
        )
        self._request_summary = (
            self._timer.should_trigger_for_step(self._next_step)
            and current_accum_step_val == self._grad_accum_steps
        )
        requests = {
            "current_loss": self._current_tensors,
        }
        if self._request_summary:
            requests["summary"] = self._summary

        return SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        if not self._summary_writer:
            return
        global_step = run_context.session.run(
            tf.compat.v1.train.get_global_step()
        )
        if self._request_summary:
            if global_step == 1:
                self._summary_writer.add_session_log(
                    SessionLog(status=SessionLog.START), global_step
                )
            self._timer.update_last_triggered_step(global_step)
            if "summary" in run_values.results:
                for summary in run_values.results["summary"]:
                    self._summary_writer.add_summary(summary, global_step)
        self._next_step = global_step + 1

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()


class GradAccumStepCounterHook(SessionRunHook):
    """Hook that counts and plots steps per second."""

    def __init__(
        self,
        trainer,
        every_n_steps=100,
        every_n_secs=None,
        output_dir=None,
        summary_writer=None,
    ):
        """
        Initializes a `GradAccumStepCounterHook`.

        :param Trainer trainer: common.optimizers.Trainer object
            used for model training with gradient accumulation.
        :param int every_n_steps:  every N steps. Exactly one of
            `every_n_secs` and `every_n_steps` should be set.
        :param int every_n_secs: Log every N seconds.
        :param string output_dir: The directory to save the summaries to. Only used if
            no `summary_writer` is supplied.
        :param SummaryWriter summary_writer: If `None` and an `output_dir` was passed,
            one will be created accordingly.
        """
        if output_dir is None and summary_writer is None:
            raise ValueError(
                "Both output_dir and summary_writer can't be None."
            )
        self._output_dir = output_dir
        self._summary_writer = summary_writer

        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError(
                "Exactly one of every_n_steps and every_n_secs should be provided."
            )
        self._timer = SecondOrStepTimer(
            every_secs=every_n_secs, every_steps=every_n_steps
        )
        if not isinstance(trainer, Trainer):
            raise ValueError(
                f"`trainer` should be object of class Trainer. "
                f"Received {type(trainer)} instead."
            )
        self._grad_accum_steps = trainer.grad_accum_steps
        self._prefix = "grad_accum_step_counter_hook"

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._current_accum_step, _ = _get_current_accum_step(
            self._grad_accum_steps, self._prefix
        )

    def after_run(self, run_context, run_values):
        if not self._summary_writer:
            return
        current_accum_step_val = run_context.session.run(
            self._current_accum_step
        )
        global_step = run_context.session.run(
            tf.compat.v1.train.get_global_step()
        )
        if (
            self._timer.should_trigger_for_step(global_step)
            and current_accum_step_val == self._grad_accum_steps
        ):
            (
                elapsed_time,
                elapsed_steps,
            ) = self._timer.update_last_triggered_step(global_step)
            if elapsed_time is not None:
                self._log_and_record(elapsed_steps, elapsed_time, global_step)

    def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
        steps_per_sec = elapsed_steps / elapsed_time
        if self._summary_writer is not None:
            summary_tag = "global_step/sec"
            summary = Summary(
                value=[
                    Summary.Value(tag=summary_tag, simple_value=steps_per_sec)
                ]
            )
            self._summary_writer.add_summary(summary, global_step)
        logging.info(f"{summary_tag}: {steps_per_sec:.5f}")

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()


class GradAccumLoggingTensorHook(LoggingTensorHook):
    """
    Prints the given tensors every N steps, every N seconds, or at end.

    The tensors will be printed to the log, with `INFO` severity. If you are not
    seeing the logs, you might want to add the following line after your imports:

    ```python
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    ```

     Note that if `at_end` is True, `tensors` should not include any tensor
     whose evaluation produces a side effect such as consuming additional inputs.
    """

    def __init__(
        self,
        trainer,
        tensors,
        every_n_steps=None,
        every_n_secs=None,
        at_end=False,
        formatter=None,
    ):
        """
        Initializes a `GradAccumLoggingTensorHook`.

        :param Trainer trainer: common.optimizers.Trainer object
            used for model training with gradient accumulation.
        :param dict tensors: `dict` that maps string-valued tags to
            tensors/tensor names.
        :param int every_n_steps: Print the values of `tensors` once every N
            steps taken on the current worker.
        :param int every_n_secs: Print the values of `tensors` once every N
            seconds. Exactly one of `every_n_steps` and `every_n_secs`
            should be provided.
        :param bool at_end: Specify whether to print the values of `tensors` at the
            end of the run.
        :param function formatter: Takes dict of `tag`->`Tensor` and returns a string.
            If `None` uses default printing all tensors.
        """
        tensors = _aggregate_tensors(tensors)
        super(GradAccumLoggingTensorHook, self).__init__(
            tensors, every_n_steps, every_n_secs, at_end, formatter,
        )
        if not isinstance(trainer, Trainer):
            raise ValueError(
                f"`trainer` should be object of class Trainer. "
                f"Received {type(trainer)} instead."
            )
        self._grad_accum_steps = trainer.grad_accum_steps
        self._tag_order.append("step")
        self._tag_order.sort()
        self._prefix = "grad_accum_logging_hook"

    def begin(self):
        self._current_accum_step, accum_step_counter = _get_current_accum_step(
            self._grad_accum_steps, self._prefix
        )
        self._current_tensors = _get_current_accum_tensors(
            self._tensors,
            accum_step_counter,
            self._grad_accum_steps,
            self._prefix,
        )
        self._iter_count = 0

    def before_run(self, run_context):
        self._current_accum_step_val = run_context.session.run(
            self._current_accum_step
        )
        return SessionRunArgs(self._current_tensors)

    def after_run(self, run_context, run_values):
        global_step = run_context.session.run(
            tf.compat.v1.train.get_global_step()
        )
        if (
            self._timer.should_trigger_for_step(global_step)
            and self._current_accum_step_val == self._grad_accum_steps
        ):
            run_values.results["step"] = global_step
            self._iter_count = global_step
            self._log_tensors(run_values.results)

    def end(self, session):
        if self._log_at_end:
            values = session.run(self._current_tensors)
            self._log_tensors(values)


def _get_current_accum_step(grad_accum_steps, prefix=None):
    with tf.name_scope(prefix):
        accum_step_counter = tf.Variable(
            initial_value=0,
            trainable=False,
            name="accum_step_counter",
            dtype=tf.int32,
        )
    current_accum_step = tf.cond(
        tf.math.less(accum_step_counter, grad_accum_steps),
        lambda: accum_step_counter.assign_add(1),
        lambda: accum_step_counter.assign(1),
    )
    return current_accum_step, accum_step_counter


def _get_current_accum_tensors(
    tensors, accum_step_counter, grad_accum_steps, prefix=""
):
    # augment graph with accumulation
    # and summary ops
    with tf.name_scope(prefix):
        accumulated_tensors = {}
        for name, t in tensors.items():
            accumulated_tensors[name] = tf.Variable(
                initial_value=0.0,
                trainable=False,
                name=f"accumulated_{name}",
                dtype=t.dtype,
            )

    current_tensors = {}
    for name, t in accumulated_tensors.items():
        current_tensors[name] = tf.cond(
            tf.math.equal(accum_step_counter, 1),
            lambda: t.assign(tensors[name] / grad_accum_steps),
            lambda: t.assign_add(tensors[name] / grad_accum_steps),
        )
    return current_tensors


def _aggregate_tensors(tensors):
    """
    Aggregate tensors if distributed training.
    """
    if tf.distribute.has_strategy():
        if tf.distribute.in_cross_replica_context():
            raise RuntimeError(
                "`_aggregate` should be called in replica context."
            )

        def _aggregate(distribution, tensors):
            for k, v in tensors.items():
                tensors[k] = distribution.reduce(
                    tf.distribute.ReduceOp.MEAN, v, axis=None
                )
            return tensors

        return tf.distribute.get_replica_context().merge_call(
            _aggregate, args=(tensors,)
        )

    return tensors
