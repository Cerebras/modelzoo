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
The ModelZoo Trainer class is the main entry point for training models in ModelZoo.
It is responsible for setting up the training environment, running the training/validation loop,
and saving the checkpoint.
"""

from __future__ import annotations

import os
import uuid
from collections import Counter, OrderedDict
from contextlib import ExitStack, contextmanager, nullcontext
from copy import copy
from functools import wraps
from logging import Logger as PythonLogger
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Union,
    final,
)
from warnings import warn
from weakref import finalize

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.trainer.callbacks import (
    GLOBAL_CALLBACK_REGISTRY,
    ArtifactDirCallback,
    AutoRestart,
    BackendCallback,
    Callback,
    Checkpoint,
    CoreCallback,
    DataLoaderCallback,
    EarlyExit,
    GradientAccumulationCallback,
    Logging,
    LoopCallback,
    LossAccumulationCallback,
    ModelCallback,
    OptimizerCallback,
    Precision,
    Reproducibility,
    RunSchedule,
    SchedulersCallback,
    SchedulersInput,
    SparsityCallback,
    TrainingLoop,
    ValidationCallback,
    ValidationLoop,
)
from cerebras.modelzoo.trainer.loggers import Logger
from cerebras.modelzoo.trainer.utils import convert_output_to_dict
from cerebras.pytorch.backend import Backend
from cerebras.pytorch.optim import Optimizer
from cerebras.pytorch.sparse import SparsityAlgorithm


class Trainer:
    """The Trainer class is the main entry point for training models in ModelZoo."""

    @final
    def __init__(
        self,
        device: Optional[str] = None,
        backend: Optional[Backend] = None,
        model_dir: str = ...,
        model: Union[Callable[[], torch.nn.Module], torch.nn.Module] = ...,
        optimizer: Union[
            Optimizer,
            Callable[[torch.nn.Module], Optimizer],
            None,
        ] = None,
        schedulers: SchedulersInput = None,
        precision: Optional[Precision] = None,
        sparsity: Optional[SparsityAlgorithm] = None,
        # Training args
        loop: Optional[LoopCallback] = None,
        checkpoint: Optional[Checkpoint] = None,
        logging: Optional[Logging] = None,
        # Trainer args
        callbacks: Optional[List[Callback]] = None,
        loggers: Optional[List[Logger]] = None,
        seed: Optional[int] = None,
        autorestart: Optional[AutoRestart] = None,
    ):
        """
        Args:
            device: The device to train the model on. It must be one of "CSX",
                "CPU", or "GPU".

            backend: The backend used to train the model. This argument is mutually
                exclusive with `device`.

            model_dir: The directory where the model artifacts are saved.

            model: The model to train. It must be one of the following:

                - If a callable is passed, it is assumed to be a function that
                  takes in no arguments returns a torch.nn.Module.

                - If a torch.nn.Module is passed, it is used as is.

            optimizer: The optimizer used to optimize the model. It must be one of the following:

                - If a :py:class:`~cerebras.pytorch.optim.Optimizer` is passed, it is used as is.

                - If a callable is passed, it is assumed to be a function that
                  takes in a torch.nn.Module and returns a
                  :py:class:`~cerebras.pytorch.optim.Optimizer`.

                - If not passed, then assume that only validation will be run.

            schedulers: The set of optimizer schedulers to be used. Common schedulers include LR
                schedulers. It must be a list of these items:

                - If a cstorch.optim.scheduler.Scheduler is passed, it is used as is.

                - A callable that is assumed to be a function that takes in a
                  :py:class:`~cerebras.pytorch.optim.Optimizer` and returns a
                  cstorch.optim.scheduler.Scheduler.

                - If None, there is no optimizer param group scheduling.

            precision: The Precision callback used during training

            sparsity: The sparsity algorithm used to sparsify weights during training/validation
                It must be one of the following:

                - If a callable is passed, it is assumed to be a function that
                  takes in no arguments returns a
                  :py:class:`~cerebras.pytorch.sparse.SparsityAlgorithm`.

                - If a :py:class:`~cerebras.pytorch.sparse.SparsityAlgorithm` is
                  passed, it is used as is.

            loop: The loop callback to use for training. It must be an instance of LoopCallback.
                If not provided, the default loop is TrainingLoop(num_epochs=1).

            checkpoint: The checkpoint callback to use for saving/loading checkpoints. It must
                be an instance of Checkpoints. If not provided, then no checkpoints are saved.

            logging: The logging callback used to set up python logging. This callback also
                controls when logs are supposed to be logged. If not provided, the default
                logging settings ``Logging(log_steps=1, log_level="INFO")`` are used.

            callbacks: A list of callbacks to used by the trainer. The order in
                which the callbacks are provided is important as it determines
                the order in which the callback's hooks are executed.

            loggers: A list of loggers to use for logging.

            seed: Initial seed for the torch random number generator.

            autorestart: The autorestart callback to automatically restart the run if it fails.
                NOTE: If the trainer is constructed manually with this callback, the run will not
                be automatically restarted. Rather, this callback provides a way to configure
                autorestart only when specified via the YAML config, since autorestart is
                implemented in wrapper class around that spawns subprocesses to run the trainer.

        """
        super().__init__()

        if model_dir is Ellipsis:
            raise ValueError("model_dir is a required argument")

        self.model_dir = Path(model_dir)

        if model is Ellipsis:
            raise ValueError("model is a required argument")
        if not isinstance(model, torch.nn.Module) and isinstance(
            optimizer, cstorch.optim.Optimizer
        ):
            raise ValueError(
                f"Expected optimizer to be a callable that takes in a torch.nn.Module "
                f"and returns a cstorch.optim.Optimizer. Got: {type(optimizer)}"
            )

        # Attributes set by core callbacks
        self.artifact_dir: Path
        self.summary_dir: Path
        self.backend: "cstorch.backend.Backend"
        self.model: torch.nn.Module
        self.compiled_model: Callable
        self.optimizer: Optional[cstorch.optim.Optimizer]
        self.schedulers: Optional[List[cstorch.optim.scheduler.Scheduler]]
        self.executor: Optional[cstorch.utils.data.DataExecutor] = None
        self.global_step: int

        # Other attributes that callbacks may set
        self.activation_steps: Optional[int] = None

        if precision is not None and not isinstance(precision, Precision):
            raise TypeError(
                f"Expected precision to be an instance of Precision. "
                f"Got: {type(precision)}"
            )
        if loop is None:
            loop = TrainingLoop(num_epochs=1)
        elif not isinstance(loop, LoopCallback):
            raise TypeError(
                f"Expected loop to be an instance of LoopCallback."
                f"Got: {type(loop)}"
            )
        if checkpoint is None:
            checkpoint = Checkpoint()
        elif not isinstance(checkpoint, Checkpoint):
            raise TypeError(
                f"Expected checkpoint to be an instance of Checkpoint. "
                f"Got: {type(checkpoint)}"
            )

        if logging is None:
            logging = Logging(log_steps=1, log_level="INFO")
        elif not isinstance(logging, Logging):
            raise TypeError(
                f"Expected logging to be an instance of Logging. "
                f"Got: {type(logging)}"
            )

        if autorestart is None:
            autorestart = AutoRestart(max_num_restarts=None)
        elif not isinstance(autorestart, AutoRestart):
            raise TypeError(
                f"Expected autorestart to be an instance of AutoRestart. "
                f"Got: {type(autorestart)}"
            )

        # Order of the core callbacks is important and should not be changed
        self.callbacks = OrderedDict(
            {
                "autorestart": autorestart,
                "artifact_dir": ArtifactDirCallback(),
                "reproducibility": Reproducibility(seed),
                "backend": BackendCallback(backend, device),
                "model": ModelCallback(model),
                "dataloader": DataLoaderCallback(),
                "optimizer": OptimizerCallback(optimizer),
                "schedulers": SchedulersCallback(schedulers),
                "precision": precision,
                "sparsity": SparsityCallback(sparsity),
                "loop": loop,
                "logging": logging,
                "loss_accum": LossAccumulationCallback(),
                "grad_accum": GradientAccumulationCallback(),
                "run_schedule": RunSchedule(),
                "checkpoint": checkpoint,
                "early_exit": EarlyExit(),
            }
        )
        # This is a sanity check, all callbacks in the above list should be core
        # callbacks and not user-defined callbacks
        for callback in self.callbacks.values():
            if callback is not None and not isinstance(callback, CoreCallback):
                raise TypeError(
                    f"Found non-core callback in core callback list: {type(callback)}"
                )

        user_callbacks = callbacks or []

        for callback in user_callbacks:
            if isinstance(callback, Logger):
                warn(
                    f"Passed logger {type(callback)} as a `callback`. "
                    f"It will not be used by trainer.log_metrics(). "
                    f"To use for logging metrics, pass it as a `logger` "
                    f"instead."
                )

        self.loggers = loggers or []

        for logger in self.loggers:
            if not isinstance(logger, Logger):
                raise TypeError(
                    f"logger must be an instance of Logger. Got: {type(logger)}"
                )

        # Default set of logger destinations to log to
        self._logger_destinations = {logger.name for logger in self.loggers}

        counter = Counter(self.callbacks.keys())

        def get_name(callback):
            name = type(callback).__name__
            counter[name] += 1
            if counter[name] > 1:
                return f"{name}_{counter[name]}"
            return name

        # Call all user callbacks before loggers in case they need to log metrics
        for callback in user_callbacks + self.loggers:
            if isinstance(callback, CoreCallback):
                raise ValueError(
                    f"Callback {type(callback).__name__} is a core callback "
                    f"and cannot be overridden"
                )

            self.callbacks[get_name(callback)] = callback

        # RunSchedule should run after other callbacks for any given hook
        # so that the schedule can read any state that's been updated by
        # other callbacks (e.g. loop callback updating global step).
        self.callbacks.move_to_end("run_schedule")

        # Checkpoint should be the last callback as saving a checkpoint is slow
        self.callbacks.move_to_end("checkpoint")

        # Whitelist of non-standard hooks that callbacks can implement
        self.non_standard_hooks_whitelist = set()

        # ID map for validation dataloaders
        self._name_scope_stack = []
        self._val_dataloader_id_map = {}
        self._val_dataloader_id = 0

        self.call("pre_setup")
        self.call("setup")
        self.call("post_setup")

        def callback_finalize(callbacks):
            for callback in callbacks:
                if callback is not None:
                    callback.finalize()

        self._finalizer = finalize(
            self, callback_finalize, self.callbacks.values()
        )

    def get_callback_name(self, callback) -> str:
        """Get the unique name of the given callback instance."""
        for name, cb in self.callbacks.items():
            if cb == callback:
                return name

        raise ValueError(
            f"Callback {type(callback).__name__} not found in the callback list"
        )

    @property
    def all_callbacks(self) -> Generator[Callback, None, None]:
        """Get all callback objects available to the trainer."""
        yield from self.callbacks.values()
        yield from GLOBAL_CALLBACK_REGISTRY.values()

    def get_callbacks(
        self, callback_type: type
    ) -> Generator[Callback, None, None]:
        """Get all callbacks of the given type."""
        for callback in self.all_callbacks:
            if isinstance(callback, callback_type):
                yield callback

    def get_callback(self, callback_type: type) -> Optional[Callback]:
        """Get the first callback of the given type."""
        return next(self.get_callbacks(callback_type), None)

    @property
    def validation_callbacks(self) -> List[ValidationCallback]:
        """Returns all validation callbacks in the Trainer's callback list."""
        return list(self.get_callbacks(ValidationCallback))

    def call(self, hook_name: str, *args, **kwargs):
        """
        Call the hook with name hook_name for all callbacks
        in the Trainer's callback list as well as the callbacks
        in the global registry.

        The callback's method is passed in the trainer object itself
        as well as any args and kwargs that are passed into this method. e.g.

        .. code: python

            getattr(callback, hook_name)(self, *args, **kwargs)

        Args:
            hook_name: The name of the hook to call. It must be the name of
                a method in the Callback class.
            args: Other positional arguments to forward
                along to the called hook.
            kwargs: Other keyword arguments to forward
                along to the called hook.
        """
        seen = set()

        for callback in self.all_callbacks:
            if callback is None:
                continue

            if callback in seen:
                warn(f"Duplicate callback found in the list: {callback}")
            else:
                seen.add(callback)

            hook = getattr(callback, hook_name, None)
            if hook:
                try:
                    hook(self, *args, **kwargs)
                except Exception as e:
                    # TODO(SW-128935): Add this to the exception notes once
                    #                  we support python 3.11
                    raise RuntimeError(
                        f"Encountered error when calling "
                        f"{type(callback).__name__}.{hook_name}"
                    ) from e
            elif hook_name not in self.non_standard_hooks_whitelist:
                raise AttributeError(
                    f"Callback {type(callback)} does not implement {hook_name}"
                )

    @property
    def precision(self) -> Optional[Precision]:
        """Returns the precision callback instance if it exists."""
        return self.callbacks["precision"]

    @property
    def grad_accum(self) -> GradientAccumulationCallback:
        """Returns the gradient accumulation callback instance."""
        return self.callbacks["grad_accum"]

    @property
    def should_run_optimizer_step(self) -> bool:
        """Returns True if we should run the optimizer step.

        The gradient accumulation callback may set this to False if we are
        accumulating gradients and have not reached the accumulation steps.
        Note, this only applies to CPU/GPU runs.
        """
        return self.grad_accum.should_run_optimizer_step

    @property
    def autorestart(self) -> AutoRestart:
        """Returns the autorestart callback."""
        return self.callbacks["autorestart"]

    @property
    def early_exit(self) -> EarlyExit:
        """Returns the early exit callback."""
        return self.callbacks["early_exit"]

    @property
    def schedule(self) -> RunSchedule:
        """Returns the run schedule callback."""
        return self.callbacks["run_schedule"]

    @property
    def loop(self) -> LoopCallback:
        """Returns the default loop settings."""
        return self.callbacks["loop"]

    @property
    def checkpoint(self) -> Checkpoint:
        """Returns the checkpoint callback."""
        return self.callbacks["checkpoint"]

    @property
    def logging(self) -> Logging:
        """Returns the logging callback."""
        return self.callbacks["logging"]

    @property
    def logger(self) -> PythonLogger:
        """Returns the Trainer's Python logger object."""
        return self.logging.logger

    @property
    def is_log_step(self) -> bool:
        """Returns True if the current step is a log step."""
        return self.logging.is_log_step(self)

    @property
    def is_first_iteration(self) -> bool:
        """Returns True if the executor is on its first iteration."""
        return self.executor and self.executor.iteration == 0

    @property
    def is_final_iteration(self) -> bool:
        """Returns True if the executor is on its final iteration."""
        return self.executor and self.executor.on_final_iteration

    @final
    @property
    def is_tracing(self) -> bool:
        """Returns True if we are currently tracing the model."""
        return self.backend.is_tracing

    @final
    def remove_logger_destination(self, name: str):
        """
        Remove the given logger destination from the default set.

        This is useful when a logger does need to log all the basic
        metrics such as the loss value.
        """
        self._logger_destinations -= {name}

    @final
    @cstorch.step_closure
    def log_metrics_in_step_closure(self, **kwargs):
        """Log the given kwargs inside a step closure."""
        # This is a sanity check and should never assert
        assert not self.is_tracing
        self.log_metrics_dict(**kwargs)

    @final
    def log_metrics(self, **kwargs):
        """
        Log the given kwargs to all loggers.

        Example usage:

        .. code:: python

            trainer.log_metrics(loss=loss.item())

        Args:
            kwargs: The key-value pairs to log.
        """
        self.log_metrics_dict(metrics=kwargs)

    @final
    def log_metrics_dict(
        self,
        metrics,
        step: Optional[int] = None,
        dest: Union[str, Set[str], None] = None,
    ):
        """
        Log the given metrics to associated loggers in <dest> set,
        else to default loggers within self._logger_destinations.

        Args:
            metrics: The dictionary containing the metrics to log.
            step: The step to log the metrics at. If not provided,
                the global step is used.
            dest: A singular string or set of logger destinations to
                log the metrics to. If not provided, trainer's
                default set of logger destinations is used.
        """
        if self.is_tracing:
            # If we are tracing, log the metrics inside a step closure
            self.log_metrics_in_step_closure(
                metrics=metrics, step=step, dest=dest
            )
            return

        # If we're not inside a logging step and an executor is not
        # active, don't send to loggers. The executor check is to
        # handle the case where a user wants to log something after
        # execution.
        if self.executor is not None and not self.is_log_step:
            return

        if not self.loggers:
            warn(
                "No loggers are attached to the trainer. "
                "Call to trainer.log_metrics() will be a no-op."
            )
            return

        from cerebras.modelzoo.trainer.loggers import TensorBoardLogger

        # Don't add prefixes to metric names if the tensorboard logger
        # is configured to output logs to legacy event directories
        if any(
            logger.legacy_event_dirs
            for logger in self.get_callbacks(TensorBoardLogger)
        ):
            prefix = ""
        else:
            prefix = self.name_scope_path

        step = step if step is not None else self.global_step
        if dest is not None:
            if isinstance(dest, str):
                dest = {dest}
            elif isinstance(dest, (list, tuple)):
                dest = set(dest)
            elif not isinstance(dest, set):
                raise TypeError(
                    f"`dest` must be a string or a set of strings, "
                    f"got {type(dest)}"
                )

            if not all(isinstance(item, str) for item in dest):
                raise TypeError("All elements in `dest` must be strings.")
        else:
            dest = self._logger_destinations

        logger_matched = False
        for logger in self.loggers:
            if logger.name in dest:
                logger_matched = True
                logger.log_metrics(
                    {f"{prefix}{k}": v for k, v in metrics.items()},
                    step=step,
                )

        if not logger_matched:
            warn(f"None of the loggers matched the given destinations: {dest}")

    @final
    @contextmanager
    def name_scope(self, name: str):
        """Append name to the trainer's name scope stack whilst inside the
        context.

        Args:
            name: The name to append to the name scope stack.
        """
        try:
            self._name_scope_stack.append(name)
            yield name
        finally:
            self._name_scope_stack.pop()

    @final
    @property
    def name_scope_path(self) -> str:
        """Returns the current name scope path.

        This is the the name scope stack joined by '/'.
        """
        return os.path.join(*self._name_scope_stack, "")

    @final
    def get_val_dataloader_scope(self, val_dataloader):
        """Get the name scope for the given val dataloader."""
        if val_dataloader.id not in self._val_dataloader_id_map:
            self._val_dataloader_id_map[val_dataloader.id] = (
                f"validate_{self._val_dataloader_id}"
            )
            self._val_dataloader_id += 1

        return self._val_dataloader_id_map[val_dataloader.id]

    @final
    @cstorch.trace
    def training_step(self, batch) -> Dict[str, Any]:
        """Run a single training step on the given batch.

        Note that if retrace is off, content of this method will only run on
        the first iteration. So any inputs to this method must either be
        non-changing or torch tensors.

        Args:
            batch: The batch of data to train on.
            batch_idx: The index of the batch in the dataloader.

        Returns:
            A dictionary containing the loss and any other outputs.
        """

        self.call("on_train_step_start", self.model, batch)

        outputs = self.forward(batch)
        self.backward(outputs)

        # Only run the optimizer step if an optimizer was defined
        # and we should run the optimizer step
        if self.optimizer and self.should_run_optimizer_step:
            self.optimizer_step()
            self.optimizer_zero_grad()
            self.schedulers_step()

        self.call("on_train_step_end", self.model, outputs, batch)

        return outputs

    @final
    def forward(self, batch) -> Dict[str, Any]:
        """
        Run the forward pass on the given batch.

        Args:
            batch: The batch of data to run the forward pass on.

        Returns:
            A dictionary containing the loss and any other outputs.
        """
        if self.precision:
            ctx = self.precision.autocast_context_manager()
        else:
            ctx = nullcontext()

        with ctx:
            args = [batch]
            kwargs = {}

            self.call("on_before_forward", self.model, batch, args, kwargs)

            output = self.compiled_model(*args, **kwargs)
            outputs = convert_output_to_dict(output)

            self.call("on_after_forward", self.model, outputs, batch)

            return outputs

    @final
    def backward(self, outputs: dict):
        """
        Run the backward pass on the given loss.

        Args:
            outputs: The outputs of the model. Expect key 'loss' to be present.
        """
        self.call("on_before_backward", self.model, outputs)

        loss = outputs["loss"]

        if self.precision:
            self.precision.backward(loss)
        else:
            loss.backward()

        self.call("on_after_backward", self.model, outputs)

    @final
    def optimizer_step(self):
        """Run the optimizer step."""
        self.call("on_before_optimizer_step", self.model, self.optimizer)

        if self.precision:
            self.precision.clip_gradients(self.optimizer)
            self.precision.optimizer_step(self.optimizer)
        else:
            self.optimizer.step()

        self.call("on_after_optimizer_step", self.model, self.optimizer)

    @final
    def optimizer_zero_grad(self):
        """Zero the gradients of the optimizer."""
        self.call("on_before_optimizer_zero_grad", self.model, self.optimizer)

        self.optimizer.zero_grad()

        self.call("on_after_optimizer_zero_grad", self.model, self.optimizer)

    @final
    def schedulers_step(self):
        """Step all the schedulers."""
        if not self.schedulers:
            return

        for scheduler in self.schedulers:
            self.call(
                "on_before_scheduler_step",
                self.model,
                self.optimizer,
                scheduler,
            )
            scheduler.step()
            self.call(
                "on_after_scheduler_step",
                self.model,
                self.optimizer,
                scheduler,
            )

    @contextmanager
    def on_exception(self, hook):
        """Context manager to handle exceptions in the given hook.

        Args:
            hook: The hook to handle exceptions for.
        """
        try:
            yield
        except Exception as e:
            try:
                self.call(f"on_{hook}_exception", e)
            except Exception as e2:
                raise e2 from e
            raise

    @final
    def fit(
        self,
        train_dataloader: cstorch.utils.data.DataLoader,
        val_dataloader: Union[
            cstorch.utils.data.DataLoader,
            List[cstorch.utils.data.DataLoader],
            None,
        ] = None,
        ckpt_path: Optional[str] = ...,
    ):
        """Complete a full training run on the given train and validation dataloaders.

        Args:
            train_dataloader: The training dataloader.

            val_dataloader: The validation dataloader.

                If provided, validation is run every `eval_frequency` steps as defined
                in the loop callback.

                If not provided, only training is run.

                If a list of dataloaders is provided, then each dataloader is
                validated in sequence.

            ckpt_path: The path to the checkpoint to load before starting training.
                If not provided and `autoload_last_checkpoint` is True,
                then the latest checkpoint is loaded
        """
        loop = self.loop
        if not isinstance(loop, TrainingLoop):
            raise TypeError(
                f"Expected loop to be an instance of TrainingLoop. "
                f"Got: {type(loop)}"
            )

        schedule = self.schedule
        if not isinstance(schedule, RunSchedule):
            raise TypeError(
                f"Expected schedule to be an instance of RunSchedule. "
                f"Got: {type(schedule)}"
            )

        with ExitStack() as fit_stack:
            self.call(
                "on_enter_fit",
                fit_stack,
                train_dataloader,
                val_dataloader,
                loop,
            )

            self.load_checkpoint(ckpt_path)

            self.call("on_fit_start", train_dataloader, val_dataloader, loop)

            for loop_idx in range(schedule.epochs):
                if self.early_exit.should_exit:
                    break

                self._run_train(train_dataloader, loop, loop_idx)

                if loop.eval_frequency is not None and loop.eval_frequency != 0:
                    # Run upstream and downstream validation after each training iteration
                    self._validate_all(
                        val_dataloader,
                        ckpt_paths=None,
                        loop=None,
                        # pylint: disable=cell-var-from-loop
                        run_validation=lambda: self.call(
                            "run_validation",
                            loop_idx=loop_idx,
                            is_last=loop_idx == schedule.epochs - 1,
                        ),
                    )

            self.call("on_fit_end", loop)

    @final
    def _run_train(self, train_dataloader, loop, loop_idx=0):
        if not isinstance(loop, TrainingLoop):
            raise TypeError(
                f"Expected loop to be an instance of TrainingLoop. "
                f"Got: {type(loop)}"
            )

        with ExitStack() as stack:
            self.call("on_enter_train", stack, train_dataloader, loop, loop_idx)

            self.call(
                "on_train_start", self.model, train_dataloader, loop, loop_idx
            )

            self.executor = cstorch.utils.data.DataExecutor(
                train_dataloader,
                num_steps=self.schedule.train_steps,
                checkpoint_steps=self.schedule.checkpoint_steps,
                activation_steps=self.activation_steps,
                profiler_activities=[],  # Don't use data executor's profiler
            )

            for batch_idx, batch in enumerate(self.executor):
                if self.early_exit.should_exit:
                    break

                self.call("on_train_batch_start", self.model, batch, batch_idx)

                outputs = self.training_step(batch)

                self.call(
                    "on_train_batch_end", self.model, outputs, batch, batch_idx
                )

            self.call("on_train_end", self.model, loop, loop_idx)

        self.executor = None

    @final
    @cstorch.trace
    @torch.no_grad()
    def validation_step(
        self, batch: Any, loop: ValidationLoop
    ) -> Dict[str, Any]:
        """Run a single validation step on the given batch and batch index.

        Note that if retrace is off, content of this method will only run on
        the first iteration. So any inputs to this method must either be
        constant or torch tensors.

        Args:
            batch: The batch of data to validate on.
            loop: The validation loop object.

        Returns:
            A dictionary containing the loss and any other outputs.
        """
        self.call(loop.on_step_start_hook, self.model, batch)
        outputs = self.forward(batch)
        self.call(loop.on_step_end_hook, self.model, outputs, batch)

        return outputs

    @final
    def validate(
        self,
        val_dataloader: Optional[cstorch.utils.data.DataLoader] = None,
        ckpt_path: Optional[str] = ...,
        loop: Optional[ValidationLoop] = None,
    ):
        """Complete a full validation run on the validation dataloader.

        Args:
            val_dataloader: The validation dataloader.
                If a list of dataloaders is provided, then each dataloader is
                    validated in sequence.

            ckpt_path: The path to the checkpoint to load before starting validation.
                If not provided and `autoload_last_checkpoint` is True,
                then the latest checkpoint is loaded.

            loop: The loop callback to use for validation. If not provided, the default
                loop is used. If provided, it must be an instance of ValidationLoop.
                Note, this should only be provided if the loop callback provided in
                the constructor is not sufficient.
        """
        if not isinstance(val_dataloader, cstorch.utils.data.DataLoader):
            raise TypeError(
                f"val_dataloader must be a cstorch.utils.data.DataLoader. "
                f"Got {type(val_dataloader)}"
            )

        if not loop:
            loop = self.loop

            if isinstance(loop, TrainingLoop):
                loop = loop.val_loop

        if not isinstance(loop, ValidationLoop):
            raise TypeError(
                f"Expected loop to be an instance of ValidationLoop. "
                f"Got: {type(loop)}"
            )

        with ExitStack() as stack:
            stack.enter_context(
                self.name_scope(self.get_val_dataloader_scope(val_dataloader))
            )

            if loop not in self.all_callbacks:
                stack.enter_context(loop)

            self.call("on_enter_validate", stack, val_dataloader, loop)

            self.load_checkpoint(ckpt_path)

            self.call(loop.on_start_hook, self.model, val_dataloader, loop)

            # For every validation run we want to iterate the dataloader from
            # the scratch, so we make a shallow copy of the validation dataloader,
            # so the streamer will treat it as new dataloader.
            self.executor = cstorch.utils.data.DataExecutor(
                copy(val_dataloader),
                num_steps=loop.eval_steps,
                profiler_activities=[],  # Don't use data executor's profiler
            )

            for batch_idx, batch in enumerate(self.executor):
                if self.early_exit.should_exit:
                    break

                self.call(
                    loop.on_batch_start_hook, self.model, batch, batch_idx
                )

                outputs = self.validation_step(batch, loop)

                self.call(
                    loop.on_batch_end_hook,
                    self.model,
                    outputs,
                    batch,
                    batch_idx,
                )

            self.call(loop.on_end_hook, self.model, loop)

        self.executor = None

    @final
    def validate_all(
        self,
        val_dataloaders: Union[
            List[cstorch.utils.data.DataLoader],
            cstorch.utils.data.DataLoader,
            None,
        ] = None,
        ckpt_paths: Union[List[str], str, None] = ...,
        loop: Optional[ValidationLoop] = None,
    ):
        """
        Runs all upstream and downstream validation permutations.

        .. code:: python

            for ckpt_path in ckpt_paths:
                 for val_dataloader in val_dataloaders:
                     trainer.validate(val_dataloader, ckpt_path)

                 # run downstream validation
                 run_validation(...)

        Args:
            val_dataloaders: A list of validation dataloaders to run validation on.
            ckpt_paths: A list of checkpoint paths to run validation on. Each checkpoint
                path must be a path to a checkpoint file, or a glob pattern.
            loop: The validation loop to use for validation. If not provided, then the
                default loop is used.
        """
        self._validate_all(
            val_dataloaders,
            ckpt_paths,
            loop,
            run_validation=lambda: self.call(
                "run_validation", loop_idx=None, is_last=True
            ),
        )

    @final
    @wraps(validate_all)
    def _validate_all(self, val_dataloaders, ckpt_paths, loop, run_validation):
        if not callable(run_validation) and run_validation is not None:
            raise RuntimeError(
                f"Expected run_validation to be a callable or None. "
                f"Got: {run_validation}"
            )

        if val_dataloaders is None:
            val_dataloaders = []
        elif not isinstance(val_dataloaders, (list, tuple)):
            val_dataloaders = [val_dataloaders]

        for i, val_dataloader in enumerate(val_dataloaders):
            if not isinstance(val_dataloader, cstorch.utils.data.DataLoader):
                raise TypeError(
                    f"Expected val_dataloader to be an instance of DataLoader. "
                    f"Got {type(val_dataloader)} for element {i}"
                )

        # Early exit if the training loop from the call to fit triggers an early exit
        if self.early_exit.should_exit:
            return

        if ckpt_paths is Ellipsis or ckpt_paths is None:
            ckpt_paths = [ckpt_paths]
        else:
            if not isinstance(ckpt_paths, (list, tuple)):
                ckpt_paths = [ckpt_paths]

            for ckpt_path in ckpt_paths:
                if not isinstance(ckpt_path, (str, Path)):
                    raise ValueError(
                        f"Expected ckpt_path to be a path to a checkpoint file, "
                        f"or a glob pattern. Got: {ckpt_path}"
                    )

                ckpt_path = Path(ckpt_path)
                if not any(ckpt_path.parent.glob(ckpt_path.name)):
                    raise FileNotFoundError(
                        f"Checkpoint file(s) not found at: {ckpt_path}"
                    )

            # Flatten all ckpt paths into a list
            ckpt_paths = [
                checkpoint_file
                for ckpt_path in map(Path, ckpt_paths)
                for checkpoint_file in ckpt_path.parent.glob(ckpt_path.name)
            ]

        with ExitStack() as stack:
            self.call(
                "on_enter_validate_all",
                stack,
                val_dataloaders,
                loop,
                ckpt_paths,
            )

            for ckpt_path in ckpt_paths:
                # Load the checkpoint
                self.load_checkpoint(ckpt_path)

                # Run upstream validation
                for val_dataloader in val_dataloaders:
                    if self.early_exit.should_exit:
                        return

                    self.validate(val_dataloader, ckpt_path=None, loop=loop)

                # Run downstream validation
                if run_validation is not None:
                    run_validation()

    @final
    @cstorch.checkpoint_closure
    def save_checkpoint(self):
        """Save a checkpoint at the current global step.

        The checkpoint state dict is constructed by various callbacks
        that implement the `on_save_checkpoint` method.
        """
        state_dict = {}

        self.call("on_save_checkpoint", state_dict)
        self.call("postprocess_checkpoint", state_dict)

        ckpt_path = self.checkpoint.get_checkpoint_path(self.global_step)

        cstorch.save(state_dict, ckpt_path)

        self.call("on_after_save_checkpoint", ckpt_path)

        # Save trainer state to a checkpoint if autorestart is enabled
        if self.autorestart.enabled:
            state_dict = cstorch.load(ckpt_path)
            state_dict["__trainer_state__"] = {}
            self.call("on_save_trainer_state", state_dict["__trainer_state__"])

            # atomic ckpt saving
            with cstorch.storage.serializers.use_external_link(True):
                tmp_filepath = Path(
                    f"{self.autorestart.trainer_state_file}.{str(uuid.uuid4())[:8]}.tmp"
                )
                cstorch.save(state_dict, tmp_filepath)
                tmp_filepath.rename(self.autorestart.trainer_state_file)

            self.call(
                "on_after_save_trainer_state",
                self.autorestart.trainer_state_file,
            )

    @final
    def load_checkpoint(self, ckpt_path: Optional[str] = None):
        """Load a checkpoint from the given path.

        The checkpoint state dict is loaded and processed by various callbacks
        that implement the `on_load_checkpoint` method.

        Args:
            ckpt_path: The path to the checkpoint to load
                If not provided and `autoload_last_checkpoint` is True,
                then the latest checkpoint is loaded
        """
        # Don't load a checkpoint if compile/validate only
        if not self.backend.is_e2e_execution:
            return
        if ckpt_path is Ellipsis and self.checkpoint.autoload_last_checkpoint:
            ckpt_path = self.checkpoint.get_latest_checkpoint(self)
        if not ckpt_path or ckpt_path is Ellipsis:
            self.call("on_before_load_checkpoint", None)
            return

        self.call("on_before_load_checkpoint", ckpt_path)

        state_dict = cstorch.load(ckpt_path)

        if "__trainer_state__" in state_dict:
            self.call(
                "on_load_trainer_state", state_dict.pop("__trainer_state__")
            )

        self.call("preprocess_checkpoint", state_dict)

        self.call("on_load_checkpoint", state_dict)
