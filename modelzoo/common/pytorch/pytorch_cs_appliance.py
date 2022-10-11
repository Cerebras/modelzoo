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

"""Contains the CS Compiler"""

# pylint: disable=attribute-defined-outside-init

import logging
import os
from typing import Tuple

import torch

from modelzoo.common.pytorch import amp
from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch import cbtorch
from modelzoo.common.pytorch.metrics import get_all_metrics
from modelzoo.common.pytorch.perf_utils import save_perf
from modelzoo.common.pytorch.pytorch_base_cs_runner import PyTorchBaseCSRunner
from modelzoo.common.pytorch.PyTorchBaseModel import PyTorchBaseModel

COMPILE_ONLY_MSG = "Compiling the model. This may take a few minutes."


class PyTorchCSAppliance(PyTorchBaseCSRunner):
    """Class for compiling PyTorch models for Cerebras hardware."""

    def __init__(self, model: PyTorchBaseModel, params: dict):
        super().__init__(model, params)

        # HACK: Hardcode save_initial_checkpoint to True,
        # since this checkpoint is used in ApplianceMode
        self._save_initial_checkpoint = True
        self._initial_metric_state = None

        self._save_losses = self._runconfig.get("save_losses", False)

        self._validate_only = self._runconfig.get("validate_only", False)
        self._compile_only = (
            self._runconfig.get("compile_only", False) or self._validate_only
        )

        self._num_batches_processed = 0

        self._appliance = cbtorch.core.appliance.ApplianceMode(
            params, self.load_and_return_state,
        )
        self._activations = None

    def load_and_return_state(self):
        file_name = os.path.join(self._model_dir, f"checkpoint_0.mdl")
        state_dict = torch.load(file_name)
        state_dict["metrics"] = self._initial_metric_state
        return state_dict

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def on_train_start(self):
        # Losses are fetched (but maybe not displayed) at every step
        cm.set_run_config(self._total_steps, self._checkpoint_steps, 1)

    def on_train_end(self, early_exit=False):
        if not self._compile_only:
            save_perf(self._perf_dir)
            cm.run_step_closures()
            self._appliance.done()

    def on_train_epoch_end(self, early_exit: bool):
        pass

    def receive_activations(self):
        logging.debug("Receiving loss")
        activations = self._appliance.receive_activations(
            self._num_batches_processed
        )
        self._activations = [
            activations[name] for name in self._appliance.output_names
        ]
        return self._activations

    def on_train_batch_end(self, loss, epoch: int = None, step: int = None):
        if self._num_batches_processed == 0:
            logging.info(COMPILE_ONLY_MSG)

            batch_size = self._train_dataloader.batch_size

            self._appliance.compile(
                batch_size, self._fabric_config_file, self._validate_only
            )
            logging.info("Compile for training completed successfully!")

            if not self._compile_only:
                self._appliance.execute(
                    self.train_data_fn,
                    batch_size,
                    self._total_steps,
                    self._checkpoint_steps,
                )

                logging.debug("Execute setup complete")

                loss = self.receive_activations()[
                    0
                ]  # if loss is present it's first

        if not self._compile_only:
            super().on_train_batch_end(loss, epoch, step)

        self._num_batches_processed += 1

    def train_forward(self, data):
        if self._num_batches_processed == 0:
            return super().train_forward(data)
        else:
            return self.receive_activations()[
                0
            ]  # if loss is present it's first

    def backward(self, loss):
        if self._num_batches_processed == 0:
            return super().backward(loss)

    def optimizer_zero_grad(self):
        if self._num_batches_processed == 0:
            return super().optimizer_zero_grad()

    def optimizer_step(self):
        if self._num_batches_processed == 0:
            return super().optimizer_step()

    ##################################################################
    #                        Evaluation Hooks                        #
    ##################################################################

    def on_eval_start(self):
        # Losses are fetched (but maybe not displayed) at every step
        cm.set_run_config(self._total_steps, self._checkpoint_steps, 1)

    def on_eval_end(self, early_exit=False):
        if not self._compile_only:
            cm.run_step_closures()
            self._appliance.done()

    def eval_forward(self, data):
        if self._num_batches_processed == 0:
            outputs = super().eval_forward(data)

            # Need to track eval model outputs to compile
            cbtorch.state().track_object(outputs)

            return outputs
        else:
            return self.receive_activations()[
                0
            ]  # if loss is present it's first

    def on_eval_epoch_end(self, early_exit: bool):
        if not self._compile_only:
            cm.run_step_closures()

    def on_eval_batch_start(self, data):
        if self._num_batches_processed == 0:
            state = cbtorch.state()

            state_dict = {
                "modules": [module.state_dict() for module in state.modules],
            }
            if state.mixed_precision:
                state_dict["amp"] = amp.state_dict()

            import torch_xla

            for scope, tensor in cbtorch.utils.nest.visit_xla_tensors(
                state_dict
            ):
                torch_xla._XLAC._xla_set_parameter_name(tensor, ".".join(scope))

        return data

    def on_eval_batch_end(self, loss, epoch: int = None, step: int = None):
        if self._num_batches_processed == 0:
            logging.info(COMPILE_ONLY_MSG)

            batch_size = self._eval_dataloader.batch_size

            self._appliance.compile(
                batch_size, self._fabric_config_file, self._validate_only
            )
            logging.info("Compile for evaluation completed successfully!")

            if not self._compile_only:
                self._appliance.execute(
                    self.eval_data_fn,
                    batch_size,
                    self._total_steps,
                    checkpoint_steps=0,
                )

                loss = self.receive_activations()[
                    0
                ]  # if loss is present it's first

        if not self._compile_only:
            super().on_eval_batch_end(loss, epoch, step)

        self._num_batches_processed += 1

    def compute_eval_metrics(self):
        if not self._compile_only:
            assert len(get_all_metrics()) == len(self._activations) - 1
            eval_metrics = {
                metric: float(value)
                for metric, value in zip(
                    get_all_metrics(), self._activations[1:]
                )
            }
            super().print_eval_metrics(eval_metrics)

    ##################################################################
    #                   Override Abstract Methods                    #
    ##################################################################

    def train(self, dataloader: torch.utils.data.DataLoader) -> None:
        dataloader = cbtorch.dataloader(dataloader, use_parallel_loader=False)
        super().train(dataloader)

    def evaluate(self, dataloader: cbtorch.data.DataLoader):
        dataloader = cbtorch.dataloader(dataloader, use_parallel_loader=False)
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

    @cm.step_closure
    def _save_checkpoint(self, step):  # pylint: disable=arguments-differ
        file_name = os.path.join(self._model_dir, f"checkpoint_{step}.mdl")

        state_dict = self._model.get_state()
        state_dict["global_step"] = state_dict.get("global_step", step)

        def post_transfer_callback(state_dict):
            if "optimizer" in state_dict:
                state_dict[
                    "optimizer"
                ] = self._optimizer.convert_state_dict_for_checkpoint(
                    state_dict["optimizer"]
                )
            return state_dict

        if step == 0:
            # HACK to save initial metric state since XRT service
            # gets corrupted tensors
            state = cbtorch.state()
            self._initial_metric_state = {}
            for metric in state.metrics:
                self._initial_metric_state.update(
                    {metric.name: cm.to_cpu(metric.state_dict)}
                )

            cm.save(
                state_dict,
                file_name,
                master_only=True,
                post_transfer_callback=post_transfer_callback,
            )
            return

        weights = self._appliance.receive_weights(state_dict, step - 1)

        def map_fn(scope, tensor):
            weight_name = ".".join(scope[1:])
            tensor_name = self._appliance.weight_names.get(weight_name)
            if not tensor_name:
                # search all duplicates in self._appliance.weight_names
                for duplicate_param in self._model.duplicate_params_map[
                    weight_name
                ]:
                    tensor_name = self._appliance.weight_names.get(
                        duplicate_param, None
                    )
                    if tensor_name:
                        break

            return weights[tensor_name]

        state_dict = cbtorch.utils.nest.map_xla_tensors(map_fn, state_dict)
        state_dict = post_transfer_callback(state_dict)

        logging.info(f"Saved checkpoint at global step: {step}")

        torch.save(state_dict, file_name)

    def _increment_global_step(self):
        self._global_step += 1
