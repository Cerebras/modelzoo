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

"""Modulek containing the Base PyTorch Runner"""
# pylint: disable=no-self-use, attribute-defined-outside-init

import abc
import atexit
import copy
import logging
import math
import os
import warnings
from collections import defaultdict
from contextlib import contextmanager
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import modes
from modelzoo.common.pytorch.loss_utils import LossSaver, extract_loss
from modelzoo.common.pytorch.metrics import (
    compute_all_metrics,
    reset_all_metrics,
)
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.common.pytorch.summaries import save_all_summaries
from modelzoo.common.pytorch.summary_collection import SummaryCollection
from modelzoo.common.pytorch.utils import visit_structure


class PyTorchBaseRunner(metaclass=abc.ABCMeta):
    """The base class for running PyTorch models on any device."""

    def __init__(self, model: PyTorchBaseModel, params: dict):
        """Construct a `PyTorchRunner` instance.

        Args:
            model: The PyTorch model to run.
            param: A dict of params that specify the behavior of the model.
        """
        self._model = model
        self._params = params
        self._optimizer = None
        self._lr_scheduler = None

        self._runconfig = copy.deepcopy(params["runconfig"])

        mode = self._runconfig["mode"]
        if mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
            self._optimizer = model.get_optimizer()
            self._lr_scheduler = model.get_lr_scheduler()

        # The mode that is currently active
        self._active_mode = mode

        # Mandatory config options
        self._mixed_precision = params["model"]["mixed_precision"]

        # Optional config options
        self._grad_accum_steps = params["optimizer"].get("grad_accum_steps", 1)
        self._show_debug_metrics = self._runconfig.get(
            "show_debug_metrics", False
        )
        self._save_losses = self._runconfig.get("save_losses", False)
        self._check_loss_values = self._runconfig.get("check_loss_values", True)
        self._checkpoint_path = self._runconfig.get("checkpoint_path")
        self._model_dir = self._runconfig.get("model_dir", "./")
        self._is_pretrained_checkpoint = self._runconfig.get(
            "is_pretrained_checkpoint", False,
        )
        self._save_initial_checkpoint = self._runconfig.get(
            "save_initial_checkpoint", False
        )
        self._save_stream_size = self._runconfig.get("save_stream_size", 0)

        self._fabric_config_file = self._runconfig.get(
            "fabric_config_file", None
        )

        # summary writer object
        self._writers = {}

        # These are set up at the start of execution loop
        self._global_step = None
        self._initial_step = None
        self._total_steps = None

        if self._runconfig.get("enable_summaries", False):
            if cm.is_wse_device():
                raise ValueError(
                    "Summary collection is not supported in CS workflows"
                )
            summary_ctx = SummaryCollection(
                os.path.join(self._model_dir, "summaries"), self._model.model,
            )
            summary_ctx.__enter__()
            atexit.register(summary_ctx.__exit__)

    @property
    def _writer(self) -> Optional[SummaryWriter]:
        return self._writers.get(self._active_mode)

    @property
    def _run_step(self) -> int:
        """Returns the current execution step.

        This is different from global_step in that it indicates the execution
        step of the current run.
        """
        assert self._global_step is not None
        assert self._initial_step is not None
        return self._global_step - self._initial_step

    @property
    def _loss_dir(self) -> str:
        """Return the directory to use for saving intermediate losses."""
        loss_dir = os.path.join(self._model_dir, "losses")
        os.makedirs(loss_dir, exist_ok=True)
        return loss_dir

    def _validate_config(self):
        """Check that the provided config is valid.

        Raises:
            AssertionError if any of the config options is invalid.
        """
        # num_epochs and num_steps are mutually exclusive. max_steps is optional
        # unless neither num_epochs nor num_steps are provided, in which case
        # max_steps must be provided.
        if self._num_epochs is not None and self._num_steps is not None:
            raise ValueError(
                "Please specify only one of `num_epochs` or `num_steps`."
            )
        elif self._num_epochs is not None:
            assert (
                self._num_epochs > 0
            ), "When provided, `num_epochs` must be greater than zero."
        elif self._num_steps is not None:
            assert (
                self._num_steps > 0
            ), "When provided, `num_steps` must be greater than zero."
        else:
            if self._max_steps is None:
                raise ValueError(
                    "`max_steps` must be specified if neither `num_epochs` "
                    "nor `num_steps` are specified."
                )

        assert self._max_steps is None or self._max_steps > 0
        assert (
            self._train_steps_per_epoch is None
            or self._train_steps_per_epoch > 0
        )
        assert (
            self._eval_steps_per_epoch is None or self._eval_steps_per_epoch > 0
        )
        assert self._grad_accum_steps >= 1
        assert self._checkpoint_steps >= 0
        assert self._log_steps >= 0

        if cm.use_cs() and self._grad_accum_steps > 1:
            raise ValueError(
                "Gradient Accumulation not supported on CS workflow."
            )

    def _should_stop(self, epoch_step: int, mode: str) -> Tuple[bool, bool]:
        """Return a tuple indicating whether to stop epoch/training.
        Args:
            epoch_step: The current step in the epoch loop.
        Returns:
            A tuple of booleans. The first item indicates whether to exit the
            epoch loop. The second item indicates whether to exit training.
        """
        exit_epoch = exit_training = False

        if mode == modes.TRAIN:
            steps_per_epoch = self._train_steps_per_epoch
        elif mode == modes.EVAL:
            steps_per_epoch = self._eval_steps_per_epoch
        else:
            raise ValueError(f"Unhandled mode: {mode}.")

        if steps_per_epoch and epoch_step + 1 >= steps_per_epoch:
            exit_epoch = True

        if mode == modes.TRAIN and self._run_step >= self._total_steps:
            exit_epoch = True
            exit_training = True

        return exit_epoch, exit_training

    def _configure_run_steps(self, dataloader, mode: str):
        """Configure steps which specify training behavior.

        This method sets `self._total_steps`, `self._checkpoint_steps`,
        `self._fetch_steps`, and `self._num_epochs` based on the user-provided
        config. If the current global step exceeds max steps, this method raises
        an error.

        Args:
            data_loader: The data loader instance that is used for training.
        Raises:
            RuntimError if global step exceeds max steps.
        """
        assert self._global_step is not None

        self._log_steps = self._runconfig["log_steps"]

        if mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
            self._num_epochs = self._runconfig.get("num_epochs", None)
            self._num_steps = self._runconfig.get("num_steps", None)
            self._max_steps = self._runconfig.get("max_steps", None)
            self._checkpoint_steps = self._runconfig.get("checkpoint_steps", 0)
        elif mode == modes.EVAL:
            self._num_epochs = 1
            self._num_steps = None
            self._max_steps = None
            self._checkpoint_steps = 0
        else:
            raise ValueError(f"Unhandled mode: {mode}.")

        self._train_steps_per_epoch = self._runconfig.get(
            "steps_per_epoch", None
        )
        self._eval_steps_per_epoch = self._runconfig.get("eval_steps", None)

        self._validate_config()

        if self._max_steps is not None and self._global_step >= self._max_steps:
            raise RuntimeError(
                f"Global step {self._global_step} already exceeds "
                f"max step {self._max_steps}."
            )

        if mode == modes.TRAIN:
            train_dataloader = dataloader
        elif mode == modes.EVAL:
            eval_dataloader = dataloader
        elif mode == modes.TRAIN_AND_EVAL:
            train_dataloader, eval_dataloader = dataloader

        train_steps_per_epoch = eval_steps_per_epoch = 0

        if mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
            try:
                # Dataset length is known
                train_steps_per_epoch = len(train_dataloader)
                assert (
                    train_steps_per_epoch > 0
                ), "Train Dataloader does not generate any batches."
                if self._train_steps_per_epoch is not None:
                    assert (
                        self._train_steps_per_epoch <= train_steps_per_epoch
                    ), (
                        f"The requested steps per epoch of {self._train_steps_per_epoch} "
                        f"exceeds total steps in an epoch, which is "
                        f"{train_steps_per_epoch}."
                    )
                    train_steps_per_epoch = self._train_steps_per_epoch

                # With grad accumulation, the global step is incremented every Nth
                # batch, so our effective steps per epoch needs to be adjusted.
                assert self._grad_accum_steps <= train_steps_per_epoch, (
                    f"Gradient accumulation steps of {self._grad_accum_steps} is "
                    f"greater than batches per epoch of {train_steps_per_epoch}."
                )
                train_steps_per_epoch //= self._grad_accum_steps
            except TypeError:
                # Dataset length is not known
                assert self._num_epochs is None, (
                    "Specifying num_epochs for datasets with unknown length is "
                    "not allowed. Please control training behavior through "
                    "number of steps instead."
                )
                train_steps_per_epoch = 1
        if mode in (modes.EVAL, modes.TRAIN_AND_EVAL):
            try:
                # Dataset length is known
                eval_steps_per_epoch = len(eval_dataloader)
                assert (
                    eval_steps_per_epoch > 0
                ), "Eval Dataloader does not generate any batches."
                if self._eval_steps_per_epoch is not None:
                    assert self._eval_steps_per_epoch <= eval_steps_per_epoch, (
                        f"The requested steps per epoch of {self._eval_steps_per_epoch} "
                        f"exceeds total steps in an epoch, which is "
                        f"{eval_steps_per_epoch}."
                    )
                    eval_steps_per_epoch = self._eval_steps_per_epoch
            except TypeError:
                # Dataset length is not known
                assert self._eval_steps_per_epoch is not None, (
                    "`eval_steps` must be specified for datasets with unknown "
                    "length."
                )
                # We assume the dataloader generates as many steps as the user
                # specified. Otherwise, we may get a stall. There's no way for
                # us to validate this since we can't query the dataset length.
                eval_steps_per_epoch = self._eval_steps_per_epoch

        if mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
            steps_per_epoch = train_steps_per_epoch

            # Calculate total steps
            self._total_steps = math.inf
            if self._num_epochs is not None:
                self._total_steps = min(
                    self._total_steps, self._num_epochs * steps_per_epoch
                )
            if self._num_steps is not None:
                self._total_steps = min(self._total_steps, self._num_steps)
            if self._max_steps is not None:
                remaining_steps = self._max_steps - self._global_step
                assert remaining_steps > 0, (  # This was checked above
                    f"Global step {self._global_step} already exceeds "
                    f"max step {self._max_steps}."
                )
                self._total_steps = min(self._total_steps, remaining_steps)

            # At least one of the above if blocks must have been true.
            # Adding an assert in case someone makes a mistake.
            assert not math.isinf(
                self._total_steps
            ), "One of num_epochs, num_steps, or max_steps must be provided"

            # Override num_epochs based on total steps and steps per epoch
            self._num_epochs = math.ceil(self._total_steps / steps_per_epoch)
            self._checkpoint_steps = min(
                self._checkpoint_steps, self._total_steps
            )
        elif mode == modes.EVAL:
            self._total_steps = eval_steps_per_epoch

        self._fetch_steps = min(self._log_steps, self._total_steps)
        if self._fetch_steps == 0:  # Always fetch the outputs of the last step
            self._fetch_steps = self._total_steps

    def _is_fetch_step(self, step_offset: int = 0):
        """
        Checks whether we are on a step where values are pre-scheduled to
        come off of the Cerebras system.

        Primarily for performance reasons.

        Args:
            step_offset: Used to offset the run step in eval where the global
                step is not incremented.
        """
        step = self._run_step + step_offset
        return step == self._total_steps or (
            self._fetch_steps > 0 and step % self._fetch_steps == 0
        )

    def _is_checkpoint_step(self, step_offset: int = 0):
        """
        Checks whether we are on a step where a checkpoint is pre-scheduled to
        come off of the Cerebras system.

        Primarily for performance reasons.

        Args:
            step_offset: Used to offset the run step in eval where the global
                step is not incremented.
        """
        step = self._run_step + step_offset
        return self._checkpoint_steps > 0 and (
            step == self._total_steps or step % self._checkpoint_steps == 0
        )

    def is_master_ordinal(self):
        """
        Checks if distributed if enabled and if so whether
        it's the main process, most reading and writing should
        only happens on main process.
        """
        return cm.is_master_ordinal()

    @contextmanager
    def _configure_run(
        self, mode: str, dataloader: torch.utils.data.DataLoader
    ):
        """Configure the run for the mode using the provided dataloader

        The setup involves loading a checkpoint if specified
        as well as configuring the run steps for performance.

        Args:
            mode: the mode to configure the run for.
            dataloader: the dataloader used to configure the run.
        """
        if not self._model.supports_mode(mode):
            raise ValueError(
                f"{mode} not supported for model. "
                f"Supported modes include: {self._model.supported_modes}."
            )

        self._maybe_load_checkpoint(self._checkpoint_path, mode)

        if self.is_master_ordinal():
            # Save initial checkpoint
            if self._save_initial_checkpoint:
                self._save_checkpoint(self.world_global_step)

            # Save dataloader streams for testing
            if self._save_stream_size:
                self._save_stream(dataloader, mode)

            # Create tensorboard summary writer for logging
            if mode in (modes.EVAL, modes.TRAIN_AND_EVAL):
                self._writers[modes.EVAL] = SummaryWriter(
                    log_dir=os.path.join(self._model_dir, modes.EVAL)
                )
                self._active_mode = modes.EVAL
            if mode in (modes.TRAIN, modes.TRAIN_AND_EVAL):
                self._writers[modes.TRAIN] = SummaryWriter(
                    log_dir=os.path.join(self._model_dir, modes.TRAIN)
                )
                self._active_mode = modes.TRAIN

        self._loss_saver = LossSaver(self._writer)

        # Configure the number of steps to run based on
        # the size of the dataloader
        self._configure_run_steps(dataloader, mode)

        try:
            yield
        finally:
            if self.is_master_ordinal():
                for writer in self._writers.values():
                    writer.flush()
                    writer.close()

    def on_train_start(self):
        """Function to execute before training starts"""

    def on_train_end(self, early_exit: bool):
        """Function to execute after training ends"""

    def on_eval_start(self):
        """Function to execute before eval starts"""

    def on_eval_end(self, early_exit: bool):
        """Function to execute after eval ends"""

    def on_train_epoch_start(self):
        """Function to execute before the training epoch begins"""

    def on_train_epoch_end(self, early_exit: bool):
        """Function to execute after the training epoch ends"""

    def on_eval_epoch_start(self):
        """Function to execute before the eval epoch begins"""

    def on_eval_epoch_end(self, early_exit: bool):
        """Function to execute after the eval epoch ends"""

    def on_train_batch_start(self, data):
        """Optionally pre-process data before train batch start"""
        return data

    def on_train_batch_end(self, loss, epoch: int = None, step: int = None):
        """Actions to perform after the train batch iteration is complete"""
        # Add step closures
        self._maybe_write_log(loss)
        self._maybe_check_loss_value(loss)
        self._maybe_save_loss(loss)
        self._maybe_save_summaries()
        self._maybe_save_checkpoint()

    def on_eval_batch_start(self, data):
        """Optionally pre-process data before eval batch start"""
        return data

    def on_eval_batch_end(self, loss, epoch: int = None, step: int = None):
        """Actions to perform after the eval batch iteration is complete"""
        # Add step closures
        self._maybe_write_log(loss, step_offset=step + 1)
        self._maybe_check_loss_value(loss, step_offset=step + 1)
        self._maybe_save_loss(loss, epoch=epoch, step_offset=step + 1)
        self._maybe_save_summaries(step_offset=step + 1)
        self._accumulate_loss_value(loss)

    def train_forward(self, data):
        """
        Runs the train forward pass.

        Override this method to provide any additional functionality around
        the eval forward pass call.
        """
        return self._model(data)

    def eval_forward(self, data):
        """
        Runs the eval forward pass.

        Override this method to provide any additional functionality around
        the eval forward pass call.
        """
        return self._model(data)

    def backward(self, loss):
        """
        Runs the backward pass.

        Override this method to provide any additional functionality around
        the backward call.
        """
        loss.backward()

    def optimizer_zero_grad(self):
        """Zeroes out the gradients in the optimizer"""
        self._optimizer.zero_grad()

    def optimizer_step(self):
        """Performs the optimizer step"""
        self._optimizer.step()

    def train(self, train_dataloader: torch.utils.data.DataLoader):
        """Train the model with data generated by the given dataloader.

        Args:
            dataloader: A data loader for generating data to feed to the model.
        """
        self._train_dataloader = train_dataloader

        with self._configure_run(modes.TRAIN, train_dataloader):
            self.on_train_start()

            exit_training = False
            for epoch in range(self._num_epochs):
                exit_training = self.train_epoch(epoch, train_dataloader)
                if exit_training:
                    break

            self.on_train_end(exit_training)

    def train_epoch(
        self, epoch: int, dataloader: torch.utils.data.DataLoader
    ) -> bool:
        """Runs an epoch of training

        Args:
            epoch: The current epoch
            dataloader: The dataloader to iterate through
        """
        self._active_mode = modes.TRAIN

        exit_epoch = False
        exit_training = False
        accum_loss = None
        grad_accum_step = 0

        # Set the appropriate writers
        self._loss_saver.writer = self._writer

        self.on_train_epoch_start()

        # Clear the loss to stop any noise from a previous epoch
        self._loss_saver.clear()

        self._model.train()  # Enable training mode
        for epoch_step, data in enumerate(dataloader):
            data = self.on_train_batch_start(data)

            # Only zero out the gradients if on first step or immediately
            # following an optimizer step
            if grad_accum_step % self._grad_accum_steps == 0:
                self.optimizer_zero_grad()

            loss = self.train_forward(data)
            self.backward(loss)

            # accumulate the losses in a way that doesn't unnecessarily add
            # an addition op to the compute graph
            accum_loss = loss if not accum_loss else accum_loss + loss
            grad_accum_step += 1

            if grad_accum_step % self._grad_accum_steps == 0:
                self.optimizer_step()

                if self._lr_scheduler:
                    self._lr_scheduler.step()

                self._increment_global_step()

                self.on_train_batch_end(accum_loss, epoch, epoch_step)

                accum_loss = None

            # Check for early stopping in epoch and training loop.
            exit_epoch, exit_training = self._should_stop(
                epoch_step, modes.TRAIN
            )
            if exit_epoch:
                break

        assert grad_accum_step >= self._grad_accum_steps, (
            f"There were only {grad_accum_step} batches in epoch, which is "
            f"less than the grad accumulation steps {self._grad_accum_steps}. "
            f"This prevents model training as no optimizer step is taken."
        )
        if grad_accum_step % self._grad_accum_steps != 0:
            warnings.warn(
                "There were leftover gradients in the accumulation step. "
                "They will effectively vanish, which could potentially lead "
                "to different convergence behaviour."
            )

        self.on_train_epoch_end(exit_epoch)
        return exit_training

    def evaluate(self, eval_dataloader: torch.utils.data.DataLoader):
        """Evaluate the model with data generated by the given dataloader.

        Args:
            dataloader: A data loader for generating data to feed to the model.
        """
        self._eval_dataloader = eval_dataloader

        with self._configure_run(modes.EVAL, eval_dataloader):
            self.on_eval_start()
            self.eval_epoch(eval_dataloader)
            self.on_eval_end(early_exit=False)

    @torch.no_grad()
    def eval_epoch(self, dataloader, epoch: int = None):
        """Runs an epoch of training

        Args:
            dataloader: The dataloader to iterate through
            epoch: The current epoch
        """
        self._active_mode = modes.EVAL
        exit_epoch = False

        # Set the appropriate writers
        self._loss_saver.writer = self._writer

        reset_all_metrics()

        self.on_eval_epoch_start()

        # Clear the loss to stop any noise from a previous epoch
        self._loss_saver.clear()

        self._model.eval()
        for step, data in enumerate(dataloader):
            data = self.on_eval_batch_start(data)
            outputs = self.eval_forward(data)
            loss = extract_loss(outputs)

            self.on_eval_batch_end(loss, epoch, step)

            exit_epoch, _ = self._should_stop(step, modes.EVAL)
            if exit_epoch:
                break

        self.on_eval_epoch_end(exit_epoch)

        self.compute_eval_metrics()

    def compute_eval_metrics(self):
        """Compute and log the eval metrics"""
        self.print_eval_metrics(compute_all_metrics())

    def print_eval_metrics(self, eval_metrics):
        """Compute and log the eval metrics"""
        if eval_metrics:
            if self._writer:
                for metric_scope, metric_value in visit_structure(
                    eval_metrics,
                    select_fn=lambda struct: isinstance(struct, (int, float)),
                    strict=True,
                ):
                    key = "/".join(metric_scope)
                    self._writer.add_scalar(
                        key, metric_value, self._global_step
                    )
            logging.info(f"Avg eval_metrics = {eval_metrics}")

        # Normalize total loss
        avg_eval_loss = self._loss_saver.average_loss
        if self._writer:
            self._writer.add_scalar(
                "avg_eval_loss", avg_eval_loss, self._global_step
            )
        logging.info(f"Avg Eval. Loss = {avg_eval_loss}")

    def train_and_eval(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
    ):
        """Train and evaluate the model with data generated by dataloaders.

        In each epoch, this method trains the model first, then runs evaluation
        every epoch.

        Args:
            train_dataloader: A data loader for generating training data to
                feed to the model.
            eval_dataloader: A data loader for generating evaluation data to
                feed to the model.
        """
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader
        with self._configure_run(
            modes.TRAIN_AND_EVAL, (train_dataloader, eval_dataloader),
        ):
            self.on_train_start()

            exit_training = False
            for epoch in range(self._num_epochs):
                exit_training = self.train_epoch(epoch, train_dataloader)

                self.eval_epoch(eval_dataloader)

                if exit_training:
                    break

            self.on_train_end(exit_training)

            logging.info("Training and Evaluation Completed Successfully!")

    @property
    def world_global_step(self):
        """The global step amongst multiple devices"""
        return self._global_step

    @abc.abstractmethod
    def _increment_global_step(self):
        raise NotImplementedError()

    def _maybe_write_log(self, loss, step_offset=0):
        if self._is_fetch_step(step_offset):
            self._write_log(loss, self._global_step + step_offset)

    @abc.abstractmethod
    def _write_log(self, loss, global_step):
        raise NotImplementedError()

    def _maybe_save_loss(self, loss, epoch=None, step_offset=0):
        if self._save_losses and self._is_fetch_step(step_offset):
            self._save_loss(loss, self.world_global_step, epoch, step_offset)

    @cm.step_closure
    def _save_loss(
        self,
        loss: torch.Tensor,
        global_step: int,
        epoch: int = None,
        step_offset: int = 0,
    ):
        """Save the current step's loss

        Args:
            loss: The loss tensor.
            global_step: The global step
            epoch: The current epoch. Used in `train_and_eval` mode to distinguish
                the eval losses on a per epoch basis.
            step_offset: The amount to offset to global step to account for the
                fact that the global step is not incremented in eval mode
        """
        if epoch is not None:
            self._loss_saver.add(loss, step_offset, epoch)
        else:
            self._loss_saver.add(loss, global_step + step_offset)

    def _maybe_save_summaries(self, step_offset=0):
        """Saves summaries calculated at the current step."""
        if self._is_fetch_step(step_offset):
            save_all_summaries(
                self._writer, self.world_global_step + step_offset
            )

    def _maybe_save_checkpoint(self, step_offset=0):
        if self._is_checkpoint_step(step_offset):
            self._save_checkpoint(self.world_global_step)

    @abc.abstractmethod
    def _save_checkpoint(self, *args, **kwargs):
        raise NotImplementedError()

    def _maybe_load_checkpoint(self, checkpoint_path: Optional[str], mode: str):
        """Optionally load checkpoint into the model.

        Args:
            checkpoint_path: Path to a checkpoint file.
        Returns:
            The loaded state dict. If checkpoint path was None, returns None.
        """
        if checkpoint_path:
            state_dict = torch.load(
                checkpoint_path, map_location=torch.device('cpu'),
            )
            self._model.set_state(state_dict)
        else:
            state_dict = None

        if (
            state_dict
            and not self._is_pretrained_checkpoint
            and mode in (modes.TRAIN, modes.TRAIN_AND_EVAL)
        ):
            self._global_step = state_dict.get("global_step", 0)
        else:
            self._global_step = 0

        self._initial_step = self._global_step

        return state_dict

    def _maybe_check_loss_value(self, loss, step_offset=0):
        if self._check_loss_values and self._is_fetch_step(step_offset):
            self._check_loss_value(loss)

    @cm.step_closure
    def _check_loss_value(self, loss: torch.Tensor):
        """Checks to see if loss is Nan/inf.

        Args:
            loss: The loss tensor.
        Raises:
            ValueError if the loss is either NaN or inf.
        """
        loss = cm.to_cpu(loss.detach())
        if torch.isnan(loss).any().item():
            raise ValueError("NaN loss detected.")
        if torch.isinf(loss).any().item():
            raise ValueError("inf loss detected.")

    @cm.step_closure
    def _accumulate_loss_value(self, loss: torch.Tensor):
        """
        Accumulates our loss value to a total_loss

        Args:
            loss: The loss tensor.
        """
        self._loss_saver.accumulate(loss)

    def _save_stream(self, data_loader, mode: str):
        if mode == modes.TRAIN_AND_EVAL:
            train_data_loader, eval_data_loader = data_loader
            self._save_stream(train_data_loader, modes.TRAIN)
            self._save_stream(eval_data_loader, modes.EVAL)
            return

        data = defaultdict(list)

        i = 0
        while i < self._save_stream_size:
            for batch in data_loader:
                for scope, tensor in visit_structure(
                    batch, lambda t: isinstance(t, torch.Tensor), strict=True
                ):
                    data[".".join(map(str, scope))].append(
                        cm.to_cpu(tensor.detach()).numpy()
                    )

                i += 1
                if i >= self._save_stream_size:
                    break

        ordinal = cm.get_ordinal()
        stream_dir = os.path.join(self._model_dir, mode)
        os.makedirs(stream_dir, exist_ok=True)
        np.savez(os.path.join(stream_dir, f"streams.{ordinal}.npz"), **data)

    @staticmethod
    def create(
        model_fn: Callable[[dict, Optional[torch.device]], PyTorchBaseModel],
        params: dict,
    ) -> "PyTorchBaseRunner":
        """
        Creates and returns an instance of PyTorchBaseRunner
        that has been configured based on the hardware specified by the
        provided params dictionary

        Args:
            model_fn: A callable that takes in a 'params' argument
                and optionally a torch.device which it uses to configure
                and return a PyTorchBaseModel
            params: A dictionary containing all the parameters required
                to initialize and configure both the model and the runner
        """
        runconfig_params = params["runconfig"]

        # For k8s flow, cs_ip is determined based on our custom scheduler. When K8S_CS_IP
        # is set, we use its value as cs_ip. We raise an error if cs_ip is set in the runconfig.
        k8s_cs_ip = os.environ.get("K8S_CS_IP", None)
        if k8s_cs_ip:
            if runconfig_params["cs_ip"] is not None:
                raise ValueError(
                    "cs_ip is determined internally and should not be specified"
                )
            cs_ip = k8s_cs_ip
            logging.info(f"Setting cs_ip to: {cs_ip}")
        else:
            cs_ip = runconfig_params["cs_ip"]

        if (
            runconfig_params["compile_only"]
            and runconfig_params["validate_only"]
        ):
            raise ValueError("Please set one of compile_only and validate_only")
        validate_only = runconfig_params["validate_only"]
        compile_only = runconfig_params["compile_only"] or validate_only
        appliance = runconfig_params.get("appliance", False)
        use_cs = bool(cs_ip or compile_only or appliance)

        # Providing a CS ip or providing the compile_only flag indicates that
        # a CS workflow is being run. Thus, the cbtorch backend must be initialized
        # in order for the run to be configured correctly
        if use_cs:
            from modelzoo.common.pytorch import cbtorch

            use_cbfloat16 = params.get("csconfig", {}).get(
                "use_cbfloat16", False
            )

            service_workdir = runconfig_params["service_dir"]
            compile_dir = runconfig_params.get("compile_dir")
            default_dir = service_workdir
            if appliance:
                from cerebras_appliance import DEFAULT_COMPILE_DIR

                default_dir = DEFAULT_COMPILE_DIR
            compile_dir = compile_dir or default_dir

            cbtorch.initialize(
                service_workdir=service_workdir,
                compile_dir=compile_dir,
                cs_ip=cs_ip,
                compile_only=compile_only,
                appliance=appliance,
                use_cbfloat16=use_cbfloat16,
            )

            # Initialize the model and runner
            model: PyTorchBaseModel = model_fn(params)

            if appliance:
                from modelzoo.common.pytorch.pytorch_cs_appliance import (
                    PyTorchCSAppliance,
                )

                return PyTorchCSAppliance(model, params)
            elif compile_only:
                from modelzoo.common.pytorch.pytorch_cs_compiler import (
                    PyTorchCSCompiler,
                )

                return PyTorchCSCompiler(model, params)
            else:
                from modelzoo.common.pytorch.pytorch_cs_runner import (
                    PyTorchCSRunner,
                )

                return PyTorchCSRunner(model, params)
        else:
            from modelzoo.common.pytorch.pytorch_dist_runner import (
                PyTorchDistRunner,
            )
            from modelzoo.common.pytorch.pytorch_runner import PyTorchRunner

            if (
                runconfig_params["cpu"] or not torch.cuda.is_available()
            ):  # use cpu
                device = torch.device("cpu")
                model: PyTorchBaseModel = model_fn(params, device)
                return PyTorchRunner(device, model, params)
            else:  # use gpu
                world_size = torch.cuda.device_count()
                if world_size == 1:  # single gpu
                    device = torch.device("cuda")
                    model: PyTorchBaseModel = model_fn(params, device)
                    return PyTorchRunner(device, model, params)
                else:  # multi gpu

                    # model with no device, used to create optimizer and scheduler
                    # actual models will be created in on_process_start()
                    model: PyTorchBaseModel = model_fn(params, None)
                    return PyTorchDistRunner(model, params)
