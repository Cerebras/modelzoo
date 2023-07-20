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

"""Contains the CS Appliance mode runner"""

# pylint: disable=attribute-defined-outside-init

import logging
import os
import time
import warnings
from pathlib import Path
from shutil import rmtree
from typing import Optional, Tuple

import torch

from cerebras_appliance.CSConfig import CSConfig
from cerebras_appliance.pb.workflow.appliance.common.common_config_pb2 import (
    DebugArgs,
)
from cerebras_appliance.run_utils import (
    get_debug_args,
    update_debug_args_with_autogen_policy,
    update_debug_args_with_mem_limits,
)
from modelzoo import CSOFT_PACKAGE, CSoftPackage
from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch
from modelzoo.common.pytorch.loss_utils import extract_loss
from modelzoo.common.pytorch.metrics import get_all_metrics
from modelzoo.common.pytorch.perf_utils import save_perf
from modelzoo.common.pytorch.pytorch_base_cs_runner import PyTorchBaseCSRunner
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel
from modelzoo.common.pytorch.sparsity.appliance import (
    build_sparsify_grouper,
    validate_sparsity_params,
)

COMPILE_ONLY_MSG = "Compiling the model. This may take a few minutes."


class PyTorchCSAppliance(PyTorchBaseCSRunner):
    """Class for compiling PyTorch models for Cerebras hardware."""

    def __init__(self, model: PyTorchBaseModel, params: dict):
        super().__init__(model, params)

        if self._save_stream_size:
            raise ValueError(f"Saving input streams on CSX is not supported.")

        self._save_losses = self._runconfig.get("save_losses", True)

        self._validate_only = self._runconfig.get("validate_only", False)
        self._compile_only = (
            self._runconfig.get("compile_only", False) or self._validate_only
        )

        if self._compile_only or self._validate_only:
            # nothing to save if compile only
            self._save_initial_checkpoint = False
        else:
            self._save_initial_state()

        self._initial_state_file = None

        self._num_batches_processed = 0

        debug_args = DebugArgs()
        if self._runconfig.get("debug_args_path"):
            debug_args = get_debug_args(self._runconfig["debug_args_path"])

        update_debug_args_with_autogen_policy(
            debug_args, self._runconfig.get("autogen_policy")
        )
        update_debug_args_with_mem_limits(debug_args, self._runconfig)

        cs_config = CSConfig(
            num_csx=self._runconfig.get("num_csx"),
            max_wgt_servers=self._runconfig["num_wgt_servers"],
            mgmt_address=self._runconfig.get("mgmt_address"),
            mgmt_namespace=self._runconfig.get("mgmt_namespace"),
            credentials_path=self._runconfig.get("credentials_path"),
            debug_args=debug_args,
            mount_dirs=self._runconfig.get("mount_dirs"),
            python_paths=self._runconfig.get("python_paths"),
            transfer_processes=self._runconfig.get("transfer_processes"),
            num_workers_per_csx=self._runconfig["num_workers_per_csx"],
            job_labels=self._runconfig.get("job_labels"),
            max_act_per_csx=self._runconfig["num_act_servers"],
            job_time_sec=self._runconfig["job_time_sec"],
            disable_version_check=self._runconfig["disable_version_check"],
        )
        precision_opt_level = None
        model_pol = self._params.get("model", {}).get("precision_opt_level")
        if model_pol is not None:
            warnings.warn(
                "Passing `precision_opt_level` via `model` params is deprecated."
                "Please use `params[\"runconfig\"][\"precision_opt_level\"]`"
            )
        precision_opt_level = self._runconfig.get(
            "precision_opt_level", model_pol
        )
        if precision_opt_level != model_pol and model_pol is not None:
            logging.warning(
                f"Using `precision_opt_level:{precision_opt_level}` from"
                f"`runconfig` instead of `{model_pol}` from `model`"
            )
        if precision_opt_level is None:
            precision_opt_level = 1

        cs_config.precision_opt_level = precision_opt_level

        use_cs_grad_accum = self._runconfig.get("use_cs_grad_accum", False)
        self.skip_train_recv_activations = self._runconfig.get(
            "skip_train_recv_activations", False
        )

        self._appliance = cbtorch.core.appliance.ApplianceMode(
            os.path.join(cbtorch.env().service_workdir, "cerebras_logs"),
            cbtorch.env().compile_dir,
            cs_config,
            use_cs_grad_accum,
            full_config=cbtorch.state().full_config,
        )
        # Cache the original xla loss tensor for retrieving its value later
        self._loss_tensor = None

        self.train_data_fn = None
        self.eval_data_fn = None
        self.send_weights_grouper = None

        sparsity = self._params.get("sparsity", {})
        if sparsity and sparsity.get("type") == "sideband":
            validate_sparsity_params(sparsity)

    @property
    def _should_log_extra_summaries(self):
        return self._log_summaries and self._num_batches_processed == 0

    def get_loss_value(self) -> torch.Tensor:
        """Fetch all activations and return the loss value."""
        assert self._loss_tensor is not None, "Loss tensor was not found!"

        logging.debug("Receiving activations")
        # This will fetch activations and store them in cbtorch.state()
        self._appliance.receive_activations(self._num_batches_processed)

        return cbtorch.state().get_activation_for_output(self._loss_tensor)

    def maybe_get_loss_value(self, step) -> torch.Tensor:
        """Fetch loss value if its a fetch step otherwise return None."""
        if self._is_fetch_step_helper(step):
            loss = self.get_loss_value()
        else:
            loss = None
        return loss

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def on_train_start(self):

        if not self._compile_only:
            self._start_time = time.time()

        # Losses are fetched (but maybe not displayed) at every step
        if self._model.grad_scaler:
            self._scaler = self._model.grad_scaler
        cm.set_run_config(self._total_steps, self._checkpoint_steps, 1)

        # Now that training has started, no need to store any new tensors
        os.environ["CEREBRAS_APPLIANCE_NO_STORAGE"] = "1"

    def on_train_end(self, early_exit=False):
        if not self._compile_only:
            save_perf(self._perf_dir)
            cm.run_step_closures()
            super().on_train_end(early_exit)

            # Delete appliance data if train was successful
            self._delete_initial_state()

    def on_train_epoch_end(self, early_exit: bool):
        pass

    def on_train_batch_start(self, data):
        if self._num_batches_processed == 0:
            self._appliance.tracker_execute.start("Tracing forward pass")

            sparsity = self._params.get("sparsity", {})
            if sparsity and sparsity.get("type") == "sideband":
                # Build tensor grouper before tracing model so the initial
                # weights can have their sparsity attributes annotated.
                self.send_weights_grouper = build_sparsify_grouper(
                    sparsity, self._model
                )

            return super().on_train_batch_start(data)
        return data

    def on_train_batch_end(self, loss, epoch: int = None, step: int = None):
        if self._num_batches_processed == 0:
            self._appliance.tracker_execute.stop("Tracing forward pass")
            logging.info(COMPILE_ONLY_MSG)

            batch_size = self._train_dataloader.batch_size

            with self._appliance.build_worker_image(
                should_skip=self._compile_only or self._validate_only
            ):
                self._appliance.compile(
                    cbtorch.state().outputs, batch_size, self._validate_only
                )
                logging.info("Compile for training completed successfully!")

            if not self._compile_only:
                assert self._initial_state_file is not None

                self._appliance.execute(
                    self.train_data_fn,
                    self.get_input_fn_params(),
                    batch_size,
                    self._total_steps,
                    self._checkpoint_steps,
                    self._active_mode,
                    self._initial_state_file,
                    cleanup_stack=self._cleanup_stack,
                    send_weights_grouper=self.send_weights_grouper,
                )

                logging.debug("Execute setup complete")

                if self.skip_train_recv_activations:
                    loss = self.maybe_get_loss_value(
                        self._num_batches_processed
                    )
                else:
                    loss = self.get_loss_value()

        if not self._compile_only:
            super().on_train_batch_end(loss, epoch, step)

        self._num_batches_processed += 1

    def train_forward(self, data):
        if self._num_batches_processed == 0:
            # Cache the loss lazy tensor used in compile
            self._loss_tensor = super().train_forward(data)
            # FIXME: Add a no-op to fix output mapping issue with ScopeBoundary
            self._loss_tensor = self._loss_tensor * 1
            return self._loss_tensor
        if self.skip_train_recv_activations:
            return self.maybe_get_loss_value(self._num_batches_processed + 1)
        else:
            return self.get_loss_value()

    def backward(self, loss):
        if self._num_batches_processed == 0:
            return super().backward(loss)
        return None

    def optimizer_zero_grad(self):
        if self._num_batches_processed == 0:
            return super().optimizer_zero_grad()
        return None

    def optimizer_step(self):
        if self._num_batches_processed == 0:
            return super().optimizer_step()
        return None

    def lr_scheduler_step(self):
        if self._num_batches_processed == 0:
            return super().lr_scheduler_step()
        return None

    ##################################################################
    #                        Evaluation Hooks                        #
    ##################################################################

    def on_eval_start(self):
        # Losses are fetched (but maybe not displayed) at every step
        cm.set_run_config(self._total_steps, self._checkpoint_steps, 1)

    def on_eval_end(self, early_exit=False):
        if not self._compile_only:
            save_perf(self._perf_dir)
            cm.run_step_closures()
            super().on_eval_end(early_exit)

            # Delete appliance data if eval was successful
            self._delete_initial_state()

    def eval_forward(self, data):
        if self._num_batches_processed == 0:
            outputs = super().eval_forward(data)

            # Need to track eval model outputs to compile
            loss = extract_loss(outputs)
            # Cache the loss lazy tensor used in compile
            self._loss_tensor = loss

            cbtorch.state().track_object({"loss": loss})
            cbtorch.state().track_object(outputs)

            return outputs
        else:
            return self.get_loss_value()

    def on_eval_epoch_end(self, early_exit: bool):
        if not self._compile_only:
            cm.run_step_closures()

    def on_eval_batch_start(self, data):
        if self._num_batches_processed == 0:
            self._appliance.tracker_execute.start("Tracing forward pass")

            sparsity = self._params.get("sparsity", {})
            if sparsity and sparsity.get("type") == "sideband":
                # Build tensor grouper before tracing model so the initial
                # weights can have their sparsity attributes annotated.
                # We don't actually save/apply this during eval.
                build_sparsify_grouper(sparsity, self._model)

            return super().on_eval_batch_start(data)
        return data

    def on_eval_batch_end(self, loss, epoch: int = None, step: int = None):
        if self._num_batches_processed == 0:
            self._appliance.tracker_execute.stop("Tracing forward pass")
            logging.info(COMPILE_ONLY_MSG)

            batch_size = self._eval_dataloader.batch_size

            with self._appliance.build_worker_image(
                should_skip=self._compile_only or self._validate_only
            ):
                self._appliance.compile(
                    cbtorch.state().outputs, batch_size, self._validate_only
                )
                logging.info("Compile for evaluation completed successfully!")

            if not self._compile_only:
                assert self._initial_state_file is not None

                self._appliance.execute(
                    self.eval_data_fn,
                    self.get_input_fn_params(),
                    batch_size,
                    self._total_steps,
                    0,  # checkpoint_steps
                    self._active_mode,
                    initial_checkpoint_file=self._initial_state_file,
                    cleanup_stack=self._cleanup_stack,
                )

                loss = self.get_loss_value()

        if not self._compile_only:
            super().on_eval_batch_end(loss, epoch, step)

        self._num_batches_processed += 1

    def compute_eval_metrics(self):
        if not self._compile_only:
            super().compute_eval_metrics()

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def train(self, train_dataloader: torch.utils.data.DataLoader) -> None:
        dataloader = cbtorch.dataloader(
            train_dataloader, use_parallel_loader=False
        )
        super().train(dataloader)

    def evaluate(self, eval_dataloader: cbtorch.data.DataLoader):
        dataloader = cbtorch.dataloader(
            eval_dataloader, use_parallel_loader=False
        )
        super().evaluate(dataloader)

    def _should_stop(self, epoch_step: int, mode: str) -> Tuple[bool, bool]:
        if self._compile_only:
            return True, True
        return super()._should_stop(epoch_step, mode)

    def is_master_ordinal(self):
        return True

    def _configure_run_steps(self, dataloader, mode: str):
        if self._compile_only:
            self._num_epochs = 1
            self._total_steps = 1
            self._checkpoint_steps = 0
            self._fetch_steps = 0
        else:
            super()._configure_run_steps(dataloader, mode)

    def _maybe_load_checkpoint(self, checkpoint_path: Optional[str], mode: str):
        """Optionally load checkpoint into the model.

        Args:
            checkpoint_path: Path to a checkpoint file.
        """
        if not checkpoint_path:
            logging.info(
                f"No checkpoint was provided, using randomly initialized model "
                f"parameters."
            )
            self._global_step = 0
            self._initial_step = 0
            return

        if CSOFT_PACKAGE in (CSoftPackage.SRC, CSoftPackage.WHEEL):
            from cerebras_pytorch.saver.pt_h5_saver import PyTorchH5Saver
        else:
            raise ImportError("Cerebras PyTorch package not installed")

        logging.info(f"Loading weights from checkpoint {self._checkpoint_path}")

        with self._appliance.tracker_execute.entry("Load Checkpoint"):
            saver = PyTorchH5Saver()

            if PyTorchH5Saver.is_valid_checkpoint(checkpoint_path):
                assert self._checkpoint_path == checkpoint_path
                tensor_names = saver.tensor_names(checkpoint_path)
                if "global_step" in tensor_names:
                    self._global_step = saver.load_tensor(
                        checkpoint_path, "global_step"
                    )
                else:
                    self._global_step = 0

            else:
                # If we get a normal PyTorch checkpoint we need to convert it into H5 format
                with self._appliance.tracker_execute.entry(
                    "Convert PyTorch Checkpoint"
                ):
                    state_dict = torch.load(
                        checkpoint_path, map_location=torch.device('cpu'),
                    )
                    self._global_step = state_dict.get("global_step", 0)

                    self._checkpoint_path = os.path.join(
                        self._model_dir, f"loaded_checkpoint.mdl"
                    )
                    saver.save(self._checkpoint_path, state_dict)

            if self._is_pretrained_checkpoint:
                self._global_step = 0

            self._initial_step = int(self._global_step)

    @cm.step_closure
    def _save_initial_state(self):
        self._initial_state_file = os.path.join(
            self._model_dir, f"initial_state_{time.time()}.hdf5"
        )

        # construct initial state dict
        state_dict = self._model.get_state()
        state_dict["global_step"] = state_dict.get(
            "global_step", self._initial_step
        )
        initial_metric_state = {}
        for metric in get_all_metrics().values():
            on_device_state_dict = metric.on_device_state_dict()
            if on_device_state_dict:
                metric_name = metric.name.replace("/", "_")
                initial_metric_state[metric_name] = on_device_state_dict

        if initial_metric_state:
            state_dict[cm.METRIC_NAME_PREFIX] = initial_metric_state

        if CSOFT_PACKAGE in (CSoftPackage.SRC, CSoftPackage.WHEEL):
            from cerebras_pytorch.saver.pt_h5_saver import PyTorchH5Saver
        else:
            raise ImportError("Cerebras PyTorch package not installed")

        saver = PyTorchH5Saver(
            loaded_checkpoint=self._checkpoint_path,
            is_pretrained_checkpoint=self._is_pretrained_checkpoint,
        )
        saver.save(self._initial_state_file, state_dict)

    def _delete_initial_state(self):
        """Delete the initial state file and its associated data directory."""
        initial_state_file = Path(self._initial_state_file)
        initial_state_file.unlink(missing_ok=True)

        initial_state_data_dir = Path(f"{self._initial_state_file}.data")
        if initial_state_data_dir.exists():
            rmtree(str(initial_state_data_dir), ignore_errors=True)

    @cm.step_closure
    def _save_checkpoint(self, step):  # pylint: disable=arguments-differ
        logging.info(f"Saving checkpoint at global step {step}")
        file_name = os.path.join(self._model_dir, f"checkpoint_{step}.mdl")

        if os.path.exists(file_name):
            # If checkpoint path already exists, need to come up with a unique
            # name. Appending the current time, should be sufficient
            file_name = os.path.join(
                self._model_dir, f"checkpoint_{step}_{time.time()}.mdl"
            )

        if CSOFT_PACKAGE in (CSoftPackage.SRC, CSoftPackage.WHEEL):
            from cerebras_pytorch.saver.pt_h5_saver import (
                CerebrasStateDict,
                PyTorchH5Saver,
            )
        else:
            raise ImportError("Cerebras PyTorch package not installed")

        saver = PyTorchH5Saver()

        state_dict = self._model.get_state()
        state_dict["global_step"] = state_dict.get("global_step", step)
        flattened, spec = saver.flatten_state_dict(state_dict)
        # save the spec before saving tensors so we know what was
        # intended to be saved, even if something fails
        saver.save_spec(file_name, spec)

        if step == self._initial_step:
            assert self._initial_state_file is not None
            with self._appliance.tracker_execute.entry(
                "Saving Initial Checkpoint"
            ):
                src_tensor_names = PyTorchH5Saver.tensor_names(
                    self._initial_state_file
                )
                for key in flattened:
                    if key in src_tensor_names:
                        val = saver.load_tensor(self._initial_state_file, key)
                        saver.save_tensor(file_name, key, val)
                    else:
                        saver.save_tensor(file_name, key, flattened[key])
        else:
            with self._appliance.tracker_execute.entry("Saving Checkpoint"):
                self._appliance.save_weights(
                    flattened.items(),
                    file_name,
                    step - self._initial_step - 1,
                    self._model.duplicate_params_map,
                )

                # Save dataloader checkpoint via WRK
                self._appliance.save_dataloader_checkpoint(
                    state_dict["global_step"]
                )

        def post_transfer_callback(state_dict):
            if "optimizer" in state_dict:
                state_dict[
                    "optimizer"
                ] = self._optimizer.convert_state_dict_for_checkpoint(
                    state_dict["optimizer"]
                )
            return state_dict

        post_transfer_callback(CerebrasStateDict.create(spec, file_name))
        logging.info(f"Saved checkpoint at global step: {step}")
        logging.debug(f"Checkpoint file: {file_name}")

        self.on_checkpoint_saved(file_name, step)

    def _increment_global_step(self):
        self._global_step += 1
