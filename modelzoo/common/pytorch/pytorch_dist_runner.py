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

import logging
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from modelzoo.common.pytorch import cb_model as cm
from modelzoo.common.pytorch.metrics import compute_all_metrics, get_all_metrics
from modelzoo.common.pytorch.pytorch_runner import PyTorchRunner
from modelzoo.common.pytorch.utils import visit_structure


class PyTorchDistRunner(PyTorchRunner):
    """Class for running PyTorch models on multiple GPUs."""

    def __init__(self, model, params):
        self._epoch = 0
        # The main process we use to aggregate results and do most IOs
        self._main_process_id = params["runconfig"].get("main_process_id", 0)
        self._dist_backend = params["runconfig"].get("dist_backend", "nccl")
        self._init_method = params["runconfig"].get("init_method", "env://")
        self._should_sync_batchnorm = params["runconfig"].get(
            "sync_batchnorm", False
        )

        if not dist.is_available():
            raise RuntimeError(f"torch.distributed package is not available")

        dist_addr = params["runconfig"].get("dist_addr", "localhost:8888")
        master_addr, master_port = dist_addr.split(":")
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

        # pass in an instance of the model for housing keep
        super().__init__(device=None, model=model, params=params)

    def is_master_ordinal(self):
        """ 
        Checks if distributed if enabled and if so whether
        it's the main process, most reading and writing should
        only happens on main process.
        """
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank() == self._main_process_id
        else:
            return cm.is_master_ordinal()

    ##################################################################
    #                         Training Hooks                         #
    ##################################################################

    def on_train_batch_start(self, data):
        return self._to_device(data, non_blocking=True)

    def on_train_epoch_start(self):
        if hasattr(self._train_sampler, "set_epoch"):
            self._train_sampler.set_epoch(self._epoch)

    def on_train_epoch_end(self, early_exit: bool):
        # change _epoch for shuffling in dataloader sampler
        self._epoch += 1

    def on_train_batch_end(self, loss, epoch: int = None, step: int = None):
        """Actions to perform after the train batch iteration is complete"""
        self._maybe_check_loss_value(loss)

        if not torch.is_tensor(loss):
            loss = torch.tensor(loss).to(self._device)

        # check _is_fetch_step ahead of time to minimize loss syncing
        if self._is_fetch_step(0):
            # not using AVG since it's only available with NCCL
            dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
            loss /= dist.get_world_size()
            dist.barrier()

        if self.is_master_ordinal():
            self._maybe_write_log(loss)
            self._maybe_save_loss(loss)
            self._maybe_save_summaries()
            self._maybe_save_checkpoint()

    def on_train_end(self, early_exit: bool):
        if self.is_master_ordinal():
            logging.info("Training Completed Successfully!")

    ##################################################################
    #                        Evaluation Hooks                        #
    ##################################################################

    def on_eval_batch_start(self, data):
        return self._to_device(data, non_blocking=True)

    def on_eval_batch_end(self, loss, epoch: int = None, step: int = None):
        """Actions to perform after the eval batch iteration is complete"""
        self._maybe_check_loss_value(loss, step_offset=step + 1)

        if not torch.is_tensor(loss):
            loss = torch.tensor(loss).to(self._device)

        # not using AVG since it's only available with NCCL
        dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
        loss /= dist.get_world_size()
        dist.barrier()

        if self.is_master_ordinal():
            self._maybe_write_log(loss, step_offset=step + 1)
            self._maybe_save_loss(loss, epoch=epoch, step_offset=step + 1)
            self._maybe_save_summaries(step_offset=step + 1)
            self._accumulate_loss_value(loss)

    def on_eval_end(self, early_exit: bool):
        if self.is_master_ordinal():
            logging.info("Evaluation Completed Successfully!")

    def compute_eval_metrics(self):
        """Compute and log the eval metrics"""
        eval_metrics = compute_all_metrics()

        # aggregate eval metrics across processes
        aggregated_eval_metrics = dict()
        for key, metric in eval_metrics.items():
            if not torch.is_tensor(metric):
                metric = torch.tensor(metric).to(self._device)
            dist.reduce(metric, 0, op=dist.ReduceOp.SUM)
            metric /= dist.get_world_size()
            dist.barrier()
            if self.is_master_ordinal():
                aggregated_eval_metrics[key] = metric.detach().cpu().item()

        if self.is_master_ordinal():
            if aggregated_eval_metrics:
                if self._writer:
                    for metric_scope, metric_value in visit_structure(
                        aggregated_eval_metrics,
                        select_fn=lambda struct: isinstance(
                            struct, (int, float)
                        ),
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

        dist.barrier()

    ##################################################################
    #                        Override train/eval functions           #
    ##################################################################

    def on_process_start(self, all_metrics):
        get_all_metrics().update(all_metrics)
        logging.getLogger().setLevel(logging.INFO)

        rank = dist.get_rank()
        self._device = torch.device(rank)
        self._model.model.to(self._device)
        self._optimizer = self._model.get_optimizer()
        self._optimizer.to(self._device)
        self._lr_scheduler = self._model.get_lr_scheduler()
        if self._should_sync_batchnorm:
            self._model.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self._model.model
            )
        self._model.model = DistributedDataParallel(
            self._model.model, device_ids=[rank], output_device=rank
        )

    def _train_dist(self, rank, world_size, train_data_fn, all_metrics):
        dist.init_process_group(
            backend=self._dist_backend,
            init_method=self._init_method,
            world_size=world_size,
            rank=rank,
        )
        self.on_process_start(all_metrics)
        train_dataloader = train_data_fn(self._params)
        self._train_sampler = train_dataloader.sampler
        super().train(train_dataloader)
        dist.destroy_process_group()

    def train(self, train_data_fn):
        all_metrics = get_all_metrics()
        world_size = torch.cuda.device_count()
        mp.spawn(
            self._train_dist,
            nprocs=world_size,
            args=(world_size, train_data_fn, all_metrics),
        )

    def _evaluate_dist(self, rank, world_size, eval_data_fn, all_metrics):
        dist.init_process_group(
            backend=self._dist_backend,
            init_method=self._init_method,
            world_size=world_size,
            rank=rank,
        )
        self.on_process_start(all_metrics)
        eval_dataloader = eval_data_fn(self._params)
        super().evaluate(eval_dataloader)
        dist.destroy_process_group()

    def evaluate(self, eval_data_fn):
        """Evaluate the model with data generated by the given dataloader.

        Args:
            dataloader: A data loader for generating data to feed to the model.
        """
        all_metrics = get_all_metrics()
        world_size = torch.cuda.device_count()
        mp.spawn(
            self._evaluate_dist,
            nprocs=world_size,
            args=(world_size, eval_data_fn, all_metrics),
        )

    def _train_and_eval_dist(
        self, rank, world_size, train_data_fn, eval_data_fn, all_metrics
    ):
        dist.init_process_group(
            backend=self._dist_backend,
            init_method=self._init_method,
            world_size=world_size,
            rank=rank,
        )
        self.on_process_start(all_metrics)
        train_dataloader = train_data_fn(self._params)
        self._train_sampler = train_dataloader.sampler
        eval_dataloader = eval_data_fn(self._params)
        super().train_and_eval(train_dataloader, eval_dataloader)
        dist.destroy_process_group()

    def train_and_eval(self, train_data_fn, eval_data_fn):
        """Train the model with data generated by the given dataloader.

        Args:
            dataloader: A data loader for generating data to feed to the model.
        """
        all_metrics = get_all_metrics()
        world_size = torch.cuda.device_count()
        mp.spawn(
            self._train_and_eval_dist,
            nprocs=world_size,
            args=(world_size, train_data_fn, eval_data_fn, all_metrics),
        )
