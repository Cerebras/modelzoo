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
This module contains utility functions for configuring a Trainer object from a params dictionary.
"""

import fnmatch
import functools
import types
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from math import prod
from typing import List, Union
from warnings import warn

import torch
import yaml
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.classes import retrieve_all_subclasses
from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    get_class_type,
)
from cerebras.modelzoo.config_manager.config_loader import flatten_data_params
from cerebras.pytorch.backend import get_backend_args


def run_trainer(mode, params, model_fn, train_data_fn, eval_data_fn):
    """
    Runs training and/or validation using the Trainer with the given params.

    Args:
        mode: The mode to run the Trainer in. Can be one of:
            - "train": Train the model.
            - "eval": Evaluate the model.
            - "train_and_eval": Train the model and then evaluate it.
            - "eval_all": Evaluate the model on all available data.
        params: A dictionary/object containing the configuration for the Trainer.
            If legacy keys are detected, they will be automatically converted
            to the new format.
        model_fn: A function that returns an instance of the model to train.
        train_data_fn: A function that returns the training dataloader.
        eval_data_fn: A function that returns the validation dataloader.
    """
    if isinstance(params, dict) and is_legacy_params(params):
        warn(
            f"Detected that legacy params are being used. "
            f"Automatically converting params to new format."
        )

        params = convert_legacy_params_to_trainer_params(
            params,
            # Allow None values in the params
            obj_filter=lambda obj: obj is None,
        )

    if "trainer" not in params:
        raise KeyError(
            "Trainer configuration not found in params. "
            "Please ensure that the params contain a 'trainer' key."
        )

    trainer_params = params["trainer"]
    if isinstance(trainer_params, (list, tuple)):
        for p in trainer_params:
            run_trainer(mode, p, model_fn, train_data_fn, eval_data_fn)
        return

    trainer = configure_trainer_from_params(params, model_fn)

    if mode == "eval":
        validate = deepcopy(trainer_params.get("validate", None))
        if validate is None:
            raise KeyError(
                "Missing 'validate' key in params needed to run validation"
            )

        if "val_dataloader" not in validate:
            raise KeyError(
                "Missing 'val_dataloader' key in validate params "
                "needed to run validation."
            )

        validate["val_dataloader"] = cstorch.utils.data.DataLoader(
            eval_data_fn, {"eval_input": validate["val_dataloader"]}
        )

        trainer.validate(**validate)
        return

    if mode == "eval_all":
        validate_all = deepcopy(trainer_params.get("validate_all", None))
        if validate_all is None:
            raise KeyError(
                "Missing 'validate_all' key in params needed to run eval_all"
            )

        if "val_dataloaders" in validate_all:
            val_dataloaders = validate_all["val_dataloaders"]
            if not isinstance(val_dataloaders, (list, tuple)):
                val_dataloaders = [val_dataloaders]

            validate_all["val_dataloaders"] = [
                cstorch.utils.data.DataLoader(
                    eval_data_fn, {"eval_input": val_dataloader}
                )
                for val_dataloader in val_dataloaders
            ]

        if "ckpt_paths" not in validate_all:
            all_ckpts = []
            if trainer.checkpoint.autoload_last_checkpoint:
                all_ckpts = trainer.checkpoint.get_all_checkpoints(
                    trainer.model_dir
                )

            if all_ckpts:
                validate_all["ckpt_paths"] = all_ckpts
            else:
                raise FileNotFoundError(
                    f"No checkpoints were found for evaluation. "
                    f"Please pass in at least one checkpoint via ckpt_paths or "
                    f"set `autoload_last_checkpoint` to True and ensure that the model "
                    f"directory \"{trainer.model_dir}\" contains at least one "
                    f"checkpoint whose name matches the expected format of: "
                    f"{trainer.checkpoint.checkpoint_name}"
                )

        trainer.validate_all(**validate_all)
        return

    fit = deepcopy(trainer_params.get("fit", None))
    if fit is None:
        raise KeyError("Missing 'fit' key in params needed to run training.")

    if "train_dataloader" not in fit:
        raise KeyError(
            "Missing 'train_dataloader' key in fit params "
            "needed to run training and validation"
        )

    fit["train_dataloader"] = cstorch.utils.data.DataLoader(
        train_data_fn, {"train_input": fit["train_dataloader"]}
    )

    if mode == "train_and_eval" and "val_dataloader" in fit:
        val_dataloaders = fit["val_dataloader"]
        if not isinstance(val_dataloaders, (list, tuple)):
            val_dataloaders = [val_dataloaders]

        fit["val_dataloader"] = [
            cstorch.utils.data.DataLoader(
                eval_data_fn, {"eval_input": val_dataloader}
            )
            for val_dataloader in val_dataloaders
        ]

    if mode == "train":
        # Disable all validation during training including eval harness
        trainer.loop.eval_frequency = None

    trainer.fit(**fit)


@lru_cache(maxsize=1)
def cached_cstorch_backend(backend_type, **kwargs):
    """
    Cached version of the cstorch.backend function.

    If the user provides the exact same arguments for different trainer configs
    we want to use the same backend instance across trainer instances.
    """
    from cerebras.pytorch.backend import BackendType, current_backend_impl

    if isinstance(backend_type, str):
        backend_type = BackendType.from_str(backend_type)

    backend = current_backend_impl(raise_exception=False)
    if backend is not None and backend.is_csx and backend_type.is_csx:
        if "cluster_config" in kwargs:
            backend.cluster_config = kwargs.pop("cluster_config")

        mismatching = "\n".join(
            f"\t{name}: {getattr(backend, name)} != {kwargs[name]}"
            for name in get_backend_args(backend_type)
            if name in kwargs
            and hasattr(backend, name)
            and getattr(backend, name) != kwargs[name]
        )
        if mismatching:
            raise RuntimeError(
                f"Detected mismatching arguments for {backend_type.name} backend:"
                f"\n{mismatching}\n"
                f"Currently, only cluster_config may be modified across different trainers."
            )

        return cstorch.backend()

    if "artifact_dir" in kwargs:
        warn(
            "artifact_dir was specified for backend, but will not be used. "
            "The Trainer's artifact directory will be used instead."
        )

    return cstorch.backend(backend_type, **kwargs)


def configure_trainer_from_params(
    params, model_fn_or_name: Union[str, torch.nn.Module]
):
    """
    Configure a Trainer object from a params dictionary.

    If the params dictionary contains legacy keys, then it will be converted
    to the new format before configuring the Trainer object.

    Args:
        params (dict): A dictionary containing the configuration for the Trainer.
        model_fn_or_name: The model class or name of the model to query from the registry.
    """
    # pylint: disable=unused-import
    import cerebras.modelzoo.trainer.extensions  # noqa
    from cerebras.modelzoo.trainer import Trainer
    from cerebras.modelzoo.trainer.callbacks import (
        Callback,
        CheckLoss,
        Checkpoint,
        ComputeNorm,
        FlopUtilization,
        Logging,
        LogOptimizerParamGroup,
        MixedPrecision,
        ModelEvalMetrics,
        ModelZooParamsMetadata,
        RateProfiler,
        SavePerformanceData,
        TrainingLoop,
    )
    from cerebras.modelzoo.trainer.loggers import (
        Logger,
        ProgressLogger,
        TensorBoardLogger,
    )

    if "trainer" not in params:
        raise KeyError(
            "Trainer configuration not found in params. "
            "Please ensure that the params contain a 'trainer' key."
        )

    trainer_params = params["trainer"]

    if "init" not in trainer_params:
        raise KeyError(
            "Trainer configuration must contain an 'init' key. "
            "Please ensure that the trainer params contain an 'init' key."
        )
    init_params = trainer_params["init"]

    metadata_params = deepcopy(params)

    # If we get a model that has a config class, convert it here, otherwise
    # pass the params as is.
    if isinstance(model_fn_or_name, str):
        _validate_trainer_params(trainer_params, model_fn_or_name)
        model_fn_or_name = registry.get_model_class(model_fn_or_name)
        trainer_params["init"]["model"].cls = model_fn_or_name
    else:
        # set model cls in params post-conversion
        trainer_params["init"]["model"]["cls"] = model_fn_or_name

    metadata_params["trainer"]["init"]["model"][
        "cls"
    ] = model_fn_or_name.__name__

    def backend_fn():
        backend_params = init_params.pop("backend", {})
        if device := init_params.pop("device", None):
            backend_params.setdefault("backend_type", device)

        backend_type = backend_params.pop("backend_type", None)
        if backend_type is None:
            raise ValueError(
                f"No device specified. Please specify a device using the 'device' key "
                f"inside the 'init' key of the trainer params"
            )

        backend_type = backend_type.upper()

        backend_args = {
            name: backend_params.pop(name)
            for name, _ in get_backend_args(backend_type).items()
            if name != "self" and name in backend_params
        }

        if backend_type == "CSX":
            # Special handling for cluster config as dicts are not hashable
            cluster_config = backend_args.pop("cluster_config", {})
            if isinstance(cluster_config, dict):
                cluster_config = cstorch.distributed.ClusterConfig(
                    **cluster_config
                )
            backend_args["cluster_config"] = cluster_config

        return cached_cstorch_backend(backend_type, **backend_args)

    def model_fn():  # pylint: disable=function-redefined
        model_params = init_params.pop("model")

        if is_dataclass(model_params):
            lora_params = getattr(model_params, "lora_params", None)
            model_init_params = types.SimpleNamespace(model=model_params)
            model = model_params.cls(model_init_params)
        else:
            lora_params = model_params.pop("lora_params", None)
            model_init_params = {"model": model_params}
            model = model_params.pop("cls")(model_init_params)
        if lora_params:
            from cerebras.modelzoo.common.utils.model.lora import (
                make_model_lora,
            )

            if is_dataclass(lora_params):
                lora_params = asdict(lora_params)

            model = make_model_lora(model, lora_params)

        return model

    def optimizer_fn(model):
        from cerebras.modelzoo.common.optim_utils import (
            configure_param_groups,
            flatten_optimizer_params,
        )

        optimizer_params = init_params.pop("optimizer", None)
        if optimizer_params is None:
            return None

        optimizer_type = next(iter(optimizer_params))
        optimizer_params = optimizer_params[optimizer_type]

        params = configure_param_groups(
            model,
            optimizer_params,
        )
        optimizer_params.pop("adjust_learning_rate", None)

        return cstorch.optim.configure_optimizer(
            optimizer_type=optimizer_type,
            params=params,
            **flatten_optimizer_params(optimizer_params),
        )

    def scheduler_fn():
        scheduler_params = init_params.pop("schedulers", None)
        if not scheduler_params:
            return None

        def _create_scheduler(optimizer, params):
            scheduler = cstorch.optim.configure_scheduler(optimizer, params)

            # The new muP interface injects adjust_learning_rate into param groups
            # if muP is enabled. So even if the params doesn't have a ScalePerParamLR,
            # we need to wrap the LR scheduler here so the scaling is applied.
            # In cases where params explicitly specify the top-level scheduler to be
            # ScalePerParamLR, we don't need to re-wrap again.
            if (
                isinstance(scheduler, cstorch.optim.lr_scheduler.LRScheduler)
                and not isinstance(
                    scheduler, cstorch.optim.lr_scheduler.ScalePerParamLR
                )
                and any(
                    "adjust_learning_rate" in g for g in optimizer.param_groups
                )
            ):
                scheduler = cstorch.optim.lr_scheduler.ScalePerParamLR(
                    optimizer, scheduler
                )

            return scheduler

        if not isinstance(scheduler_params, list):
            scheduler_params = [scheduler_params]

        return [
            functools.partial(_create_scheduler, params=p)
            for p in scheduler_params
        ]

    def sparsity_fn():
        if sparsity_params := init_params.pop("sparsity", None):
            return cstorch.sparse.configure(sparsity_params)
        return None

    def precision_fn():
        precision_params = init_params.pop("precision", None)
        if not precision_params:
            return None

        return MixedPrecision(**precision_params)

    def loop_fn():
        loop_params = init_params.pop("loop", None)
        if not loop_params:
            return None

        return TrainingLoop(**loop_params)

    def checkpoint_fn():
        checkpoint_params = init_params.pop("checkpoint", None)
        if not checkpoint_params:
            return None

        return Checkpoint(**checkpoint_params)

    def logging_fn():
        logging_params = init_params.pop("logging", None)
        if not logging_params:
            return None

        return Logging(**logging_params)

    def callbacks_fn():
        # Callbacks to enable even if not explicitly specified
        default_callbacks = {
            # pylint: disable=unnecessary-lambda
            "CheckLoss": lambda: CheckLoss(),
            "LogOptimizerParamGroup": lambda: LogOptimizerParamGroup("lr"),
            "ComputeNorm": lambda: ComputeNorm(),
            "ModelEvalMetrics": lambda: ModelEvalMetrics(),
            "RateProfiler": lambda: RateProfiler(),
            "FlopUtilization": lambda: FlopUtilization(),
            "SavePerformanceData": lambda: SavePerformanceData(),
            "ModelZooParamsMetadata": lambda: ModelZooParamsMetadata(
                metadata_params
            ),
        }

        callback_map = {
            cls.__name__: cls for cls in retrieve_all_subclasses(Callback)
        }

        callbacks = []
        for idx, callback_params in enumerate(init_params.pop("callbacks", [])):
            if isinstance(callback_params, dict):
                if len(callback_params) > 1:
                    raise ValueError(
                        f"Expected each callback to be a dictionary with the name "
                        f"of the callback being the only key, but callback at "
                        f"position {idx} has more keys: {list(callback_params.keys())}."
                    )
                name = next(iter(callback_params))

                if name not in callback_map:
                    raise ValueError(
                        f"Invalid callback. Expected one of: {sorted(callback_map)}. "
                        f"Got: {name}"
                    )
                callback_cls = callback_map[name]

                default_callbacks.pop(name, None)

                args = callback_params[name]
                if args is None:
                    # None as args means don't include
                    continue

                # TODO: Inspect callback_cls's constructor arguments for generality
                if isinstance(args, dict):
                    callbacks.append(callback_cls(**args))
                else:
                    raise ValueError(
                        f"Expected each callback argument to be a dict of input arguments, "
                        f"but \"{name}\" key has argument of type {type(args)}."
                    )
            else:
                raise ValueError(
                    f"Expected each callback to be a dict with a single key representing "
                    f"the callback name, but callback at position {idx} is of type "
                    f"{type(callback_params)}."
                )

        for callback_fn in default_callbacks.values():
            callbacks.append(callback_fn())

        return callbacks

    def loggers_fn():
        # Loggers to enable even if not explicitly specified
        default_loggers = {
            # pylint: disable=unnecessary-lambda
            "ProgressLogger": lambda: ProgressLogger(),
            "TensorBoardLogger": lambda: TensorBoardLogger(),
        }

        logger_map = {
            cls.__name__: cls for cls in retrieve_all_subclasses(Logger)
        }

        loggers = []
        for logger_params in init_params.pop("loggers", []):
            if isinstance(logger_params, str):
                logger_params = {logger_params: {}}
            if isinstance(logger_params, dict):
                if len(logger_params) > 1:
                    raise ValueError(
                        "Expected logger to be a dictionary with the name "
                        "of the logger being the only key"
                    )
                name = next(iter(logger_params))

                # TODO: Should we support automatic snake case to Pascal case conversion?
                if name not in logger_map:
                    raise ValueError(
                        f"Invalid logger. Expected one of: {sorted(logger_map)}. "
                        f"Got: {name}"
                    )
                logger_cls = logger_map[name]

                default_loggers.pop(name, None)

                args = logger_params[name]
                if args is None:
                    # None as args means don't include
                    continue

                # TODO: Inspect logger_cls's constructor arguments for generality
                if isinstance(args, dict):
                    loggers.append(logger_cls(**args))
                else:
                    loggers.append(logger_cls(args))

        for logger_fn in default_loggers.values():
            loggers.append(logger_fn())

        return loggers

    class SaveTrainerParams(Callback):
        """Save the Trainer params to the artifact directory."""

        def pre_setup(self, trainer):
            # Save a full copy of the params to the artifact directory
            with (trainer.artifact_dir / "trainer_params.yaml").open("w") as f:
                yaml.dump(metadata_params, f, sort_keys=False)
            # Save a full copy of the params to the summary directory as well
            with (trainer.summary_dir / "trainer_params.yaml").open("w") as f:
                yaml.dump(metadata_params, f, sort_keys=False)

    with SaveTrainerParams():
        return Trainer(
            backend=backend_fn(),
            model_dir=init_params.pop("model_dir", "./model_dir"),
            model=model_fn,
            optimizer=optimizer_fn,
            schedulers=scheduler_fn(),
            precision=precision_fn(),
            sparsity=sparsity_fn,
            loop=loop_fn(),
            checkpoint=checkpoint_fn(),
            logging=logging_fn(),
            callbacks=callbacks_fn(),
            loggers=loggers_fn(),
            seed=init_params.get("seed"),
        )


def is_legacy_params(params):
    """Returns True if the params dictionary contains legacy keys."""
    return "trainer" not in params


TRAINER_PARAMS_TO_LEGACY = {
    "init": {
        "model_dir": "runconfig.model_dir",
        "seed": "runconfig.seed",
        "backend": {
            "backend_type": "runconfig.target_device",
            # CSX Args
            "compile_dir": "runconfig.compile_dir",
            "compile_only": "runconfig.compile_only",
            "validate_only": "runconfig.validate_only",
            "cluster_config": {
                "num_csx": "runconfig.num_csx",
                "max_wgt_servers": "runconfig.num_wgt_servers",
                "mgmt_address": "runconfig.mgmt_address",
                "mgmt_namespace": "runconfig.mgmt_namespace",
                "credentials_path": "runconfig.credentials_path",
                "mount_dirs": "runconfig.mount_dirs.*",
                "python_paths": "runconfig.python_paths.*",
                "num_workers_per_csx": "runconfig.num_workers_per_csx",
                "max_act_per_csx": "runconfig.num_act_servers",
                "job_labels": "runconfig.job_labels.*",
                "job_priority": "runconfig.job_priority",
                "job_time_sec": "runconfig.job_time_sec",
                "disable_version_check": "runconfig.disable_version_check",
            },
            # GPU Args
            "enable_distributed": "runconfig.enable_distributed",
            "main_process_id": "runconfig.main_process_id",
            "dist_backend": "runconfig.dist_backend",
            "init_method": "runconfig.init_method",
            "sync_batchnorm": "runconfig.sync_batchnorm",
        },
        "model": {"*": "model.*"},
        "optimizer": "optimizer.*",
        "schedulers": "optimizer.learning_rate.*",
        "precision": {
            "enabled": "model.mixed_precision",
            "fp16_type": "model.fp16_type",
            "precision_opt_level": "runconfig.precision_opt_level",
            "loss_scaling_factor": "optimizer.loss_scaling_factor",
            "initial_loss_scale": "optimizer.initial_loss_scale",
            "steps_per_increase": "optimizer.steps_per_increase",
            "min_loss_scale": "optimizer.min_loss_scale",
            "max_loss_scale": "optimizer.max_loss_scale",
            "max_gradient_norm": "optimizer.max_gradient_norm",
            "max_gradient_value": "optimizer.max_gradient_value",
            "log_loss_scale": "optimizer.log_summaries",
        },
        "sparsity": {
            "*": "sparsity.*",
            "sparsity": "sparsity",
        },
        "loop": {
            "num_steps": "runconfig.num_steps",
            "max_steps": "runconfig.max_steps",
            "num_epochs": "runconfig.num_epochs",
            "steps_per_epoch": "runconfig.steps_per_epoch",
            "eval_frequency": "runconfig.eval_frequency",
            "eval_steps": "runconfig.eval_steps",
            "grad_accum_steps": "optimizer.grad_accum_steps",
        },
        "checkpoint": {
            "steps": "runconfig.checkpoint_steps",
            "disable_strict_checkpoint_loading": "runconfig.disable_strict_checkpoint_loading",
            "autoload_last_checkpoint": "runconfig.autoload_last_checkpoint",
            "save_initial_checkpoint": "runconfig.save_initial_checkpoint",
        },
        "logging": {
            "log_steps": "runconfig.log_steps",
            "log_level": "runconfig.logging",
            "wsc_log_level": "runconfig.wsc_log_level.*",
            "enable_act_frequency": "runconfig.enable_act_frequency",
        },
        "callbacks": [
            {
                "GlobalFlags": {
                    "csx.performance.transfer_processes": "runconfig.transfer_processes",
                    "csx.debug.ini": "runconfig.ini.*",
                    "csx.debug.debug_args": "runconfig.debug_args.*",
                    "csx.debug.retrace_every_iteration": "runconfig.retrace_every_iteration",
                    "csx.debug.lazy_initialization": "runconfig.lazy_initialization",
                    "csx.debug.log_initialization": "runconfig.log_initialization",
                    "csx.debug.drop_data": "runconfig.drop_data",
                    "csx.debug.fabric_type_blacklist": "runconfig.fabric_type_blacklist.*",
                    "csx.debug.compile_crd_memory_gi": "runconfig.compile_crd_memory_gi",
                    "csx.debug.execute_crd_memory_gi": "runconfig.execute_crd_memory_gi",
                    "csx.debug.wrk_memory_gi": "runconfig.wrk_memory_gi",
                    "csx.debug.act_memory_gi": "runconfig.act_memory_gi",
                    "csx.debug.cmd_memory_gi": "runconfig.cmd_memory_gi",
                    "csx.debug.wgt_memory_gi": "runconfig.wgt_memory_gi",
                }
            },
            {
                "ScopedTrainFlags": {
                    "csx.performance.micro_batch_size": "train_input.micro_batch_size",
                }
            },
            {
                "ScopedValidateFlags": {
                    "csx.performance.micro_batch_size": "eval_input.micro_batch_size",
                }
            },
            {
                "DebugArgsPath": {
                    "debug_args_path": "runconfig.debug_args_path",
                },
            },
            {"LogInputSummaries": "runconfig.log_input_summaries"},
            {"CheckLoss": "runconfig.check_loss_values"},
            {"ComputeNorm": "optimizer.log_summaries"},
            {
                "LoadCheckpointStates": {
                    "load_checkpoint_states": "runconfig.load_checkpoint_states",
                }
            },
            {
                "KeepNCheckpoints": {
                    "n": "runconfig.max_checkpoints",
                }
            },
            {"LogSparsity": "sparsity.add_summaries"},
            {"WeightCompression": {"compressions": "model.compression.*"}},
            {"SelectiveGrad": {"selective_grads": "model.selective_grad.*"}},
            {"Listener": {"listeners": "runconfig.experimental.listeners.*"}},
            {"OpProfiler": "runconfig.op_profiler_config.*"},
            {"DumpActivations": "runconfig.dump_activations"},
            {"HFCacheDir": {"cache_dir": "runconfig.hf_cache_dir"}},
        ],
        "loggers": [
            {
                "TensorBoardLogger": {
                    "summary_dir": "runconfig.summary_dir",
                    "legacy_event_dirs": "runconfig.legacy_event_dirs",
                },
            },
            {
                "WandbLogger": {
                    "project": "wandb.project",
                    "group": "wandb.group",
                    "run_id": "wandb.run_id",
                    "run_name": "wandb.run_name",
                    "job_type": "wandb.job_type",
                    "tags": "wandb.tags",
                    "resume": "wandb.resume",
                }
            },
        ],
    },
    "fit": {
        "train_dataloader": {"*": "train_input.*"},
        "val_dataloader": {"*": "eval_input.*"},
        "ckpt_path": "runconfig.checkpoint_path",
    },
    "validate": {
        "val_dataloader": {"*": "eval_input.*"},
        "ckpt_path": "runconfig.checkpoint_path",
    },
    "validate_all": {
        "val_dataloaders": {"*": "eval_input.*"},
        "ckpt_paths": "runconfig.checkpoint_path",
    },
}


# List of V2 keys to pop after conversion from V1 to V2
TRAINER_POST_CONVERSION_PRUNE_LIST = [
    "init.model.compression",
    "init.model.selective_grad",
    "init.optimizer.loss_scaling_factor",
    "init.optimizer.initial_loss_scale",
    "init.optimizer.steps_per_increase",
    "init.optimizer.min_loss_scale",
    "init.optimizer.max_loss_scale",
    "init.optimizer.max_gradient_norm",
    "init.optimizer.max_gradient_value",
    "init.optimizer.log_summaries",
    "init.optimizer.grad_accum_steps",
    "fit.train_dataloader.micro_batch_size",
    "fit.val_dataloader.micro_batch_size",
    "validate.val_dataloader.micro_batch_size",
    "validate_all.val_dataloader.micro_batch_size",
]


# List of keys to ignore during conversion from V1 to V2 if they are extra
IGNORE_LEGACY_KEYS = [
    "runconfig.mode",
]


def convert_legacy_params_to_trainer_params(params, obj_filter=None):
    """Converts params from the V1 structure to the V2 structure.

    If params are already in V2 structure, return as-is.

    Legacy params will have the format:

    .. code:: yaml

        train_input:
            ...
        eval_input:
            ...
        model:
            ...
        optimizer:
            ...
        sparsity:
            ...
        runconfig:
            ...

    We want to convert this to the format expected by the Trainer:

    .. code:: yaml

        trainer:
            init:
                device: "csx"
                backend:  # optional
                    ...
                model:
                    ...
                optimizer:
                    ...
                schedulers:
                    ...
                precision:
                    ...
                sparsity:
                    ...
                loop:
                    ...
                checkpoint:
                    ...
                callbacks:
                    ...
                loggers:
                    ...
            fit:
                train_dataloader:
                    ...
                val_dataloader:
                    ...
    """
    if not is_legacy_params(params):
        return params

    params = deepcopy(params)

    # Flatten the mapping of trainer params to legacy params
    legacy_keys, trainer_spec = tree_flatten(TRAINER_PARAMS_TO_LEGACY)

    # Flatten the params dictionary
    params_values, params_spec = tree_flatten(params)
    params_keys = [
        ".".join(s) for s in cstorch.utils.nest.recurse_spec(params_spec)
    ]
    flattened_params = dict(zip(params_keys, params_values))

    # Prune legacy keys from the provided params
    for key in IGNORE_LEGACY_KEYS:
        flattened_params.pop(key, None)

    # Collect fully matching keys
    legacy_values = {
        key: flattened_params.pop(key)
        for key in legacy_keys
        if key in flattened_params
    }

    def get_spec(scope: List[str], spec: TreeSpec):
        """Recursively get the spec given a scope."""
        if len(scope) == 0:
            return spec
        i = spec.context.index(scope[0])
        return get_spec(scope[1:], spec.children_specs[i])

    # Get all keys that are globs, i.e. model.*, optimizer.*, etc.
    glob_keys = (
        key
        for key in (set(legacy_keys) - set(legacy_values))
        if key.endswith(".*")
    )

    MISSING = object()

    # iterate through the glob keys in order of longest scope to shortest scope
    for glob_key in sorted(glob_keys, key=lambda key: -len(key.split("."))):
        try:
            # Extract the spec given the scope
            scope = glob_key.split(".")[:-1]
            spec = get_spec(scope, params_spec)
        except ValueError:
            # If we encounter a ValueError, then it means that the spec doesn't
            # contain the scope. We can just ignore it and move onto the next one.
            continue

        # Get all the values in the original params that matches the glob
        values = [
            flattened_params.pop(key, legacy_values.get(key, MISSING))
            for key in fnmatch.filter(params_keys, glob_key)
        ]
        if not values:
            # No params keys match the glob key, so we can skip
            # this glob key
            continue

        # unflatten the values using the extracted spec. We need to do this
        # to capture the underlying nested structure of these glob'd keys
        legacy_values[glob_key] = tree_unflatten(values, spec)

    if len(flattened_params) > 0:
        warn(f"Found extra keys in the params: {sorted(flattened_params)}")

    # Construct a list of values pertaining to the trainer spec
    # and then unflatten them to get the full trainer params dict
    trainer_values = [legacy_values.get(key, MISSING) for key in legacy_keys]
    trainer_params = tree_unflatten(trainer_values, trainer_spec)

    def extract_key(config, key, default):
        if isinstance(config, dict):
            yield config.pop(key, default)
            for v in config.values():
                yield from extract_key(v, key, default)
        elif isinstance(config, (list, tuple)):
            for c in config:
                yield from extract_key(c, key, default)
        else:
            yield default

    def is_not_empty(obj):
        return (obj is not MISSING) and (
            obj
            or isinstance(obj, (int, float, bool, str))
            or (obj_filter is not None and obj_filter(obj))
        )

    # Extract add_summaries from sparsity config lists
    if any(
        # Explicitly construct a list first to avoid short circuiting
        list(
            filter(
                is_not_empty,
                extract_key(
                    trainer_params["init"]["sparsity"], "add_summaries", False
                ),
            )
        )
    ):
        for callback in trainer_params["init"]["callbacks"]:
            if "LogSparsity" in callback:
                callback["LogSparsity"] = True
                break

    # Remove all MISSING objects from the dictionary, effectively
    # removing keys in the trainer params that werent' in the original
    def cleanup(params):
        if isinstance(params, dict):
            cleaned_dict = {}
            for k, v in params.items():
                if v is MISSING:
                    continue

                # Collapse glob entries
                if k == "*":
                    if isinstance(v, dict):
                        # Don't clean up globs. Just copy them as is.
                        # This is because the model may be expecting
                        # things like an empty list.
                        cleaned_dict.update(**v)
                    else:
                        return cleanup(v)
                else:
                    v = cleanup(v)
                    # Only include non-empty and scalar values
                    if is_not_empty(v):
                        cleaned_dict[k] = v

            return cleaned_dict
        elif isinstance(params, (tuple, list)):
            return type(params)(
                cp for p in params if is_not_empty(cp := cleanup(p))
            )
        else:
            return params

    trainer_params = cleanup(trainer_params)

    # Use the same object for fit and validate's val_dataloader
    if "fit" in trainer_params and "val_dataloader" in trainer_params["fit"]:
        if (
            "validate" in trainer_params
            and "val_dataloader" in trainer_params["validate"]
        ):
            trainer_params["validate"]["val_dataloader"] = trainer_params[
                "fit"
            ]["val_dataloader"]
        if (
            "validate_all" in trainer_params
            and "val_dataloaders" in trainer_params["validate_all"]
        ):
            trainer_params["validate_all"]["val_dataloaders"] = trainer_params[
                "fit"
            ]["val_dataloader"]

    # Read adjust_learning_rate before convert_optimizer_params since
    # that changes the structure.
    adjust_learning_rate = None
    if "optimizer" in trainer_params["init"]:
        adjust_learning_rate = trainer_params["init"]["optimizer"].get(
            "adjust_learning_rate", None
        )

    if "schedulers" in trainer_params["init"]:
        trainer_params["init"]["schedulers"] = [
            convert_lr_scheduler(
                trainer_params["init"]["schedulers"],
                adjust_learning_rate,
            ),
        ]
    elif "optimizer" in trainer_params["init"] and (
        "learning_rate" in trainer_params["init"]["optimizer"]
    ):
        # For constant LR, we want to create an explicit ConstantLR so that
        # ScalePerParam can be applied if needed.
        trainer_params["init"]["schedulers"] = [
            convert_lr_scheduler(
                {
                    "scheduler": "ConstantLR",
                    "learning_rate": float(
                        trainer_params["init"]["optimizer"]["learning_rate"]
                    ),
                },
                adjust_learning_rate,
            ),
        ]

    for callback in trainer_params["init"].get("callbacks", []):
        class_name = next(iter(callback))
        if callback[class_name] is True:
            callback[class_name] = {}
        if callback[class_name] is False:
            callback[class_name] = None

    # Continue to allow mixed_precision in model params
    # TODO: Deprecate this
    if (
        mixed_precision := trainer_params["init"]
        .get("precision", {})
        .get("enabled")
    ) is not None:
        trainer_params["init"]["model"]["mixed_precision"] = mixed_precision

    # Continue to allow fp16_type in model params
    # TODO: Deprecate this
    if (
        fp16_type := trainer_params["init"]
        .get("precision", {})
        .get("fp16_type")
    ) is not None:
        trainer_params["init"]["model"]["fp16_type"] = fp16_type

    for full_key in TRAINER_POST_CONVERSION_PRUNE_LIST:
        keys = full_key.split(".")
        section = trainer_params
        for idx, key in enumerate(keys):
            if not isinstance(section, dict) or key not in section:
                break

            if idx < len(keys) - 1:
                section = section[key]
            else:
                section.pop(key)

    if "optimizer" in trainer_params["init"]:
        trainer_params["init"]["optimizer"] = convert_optimizer_params(
            trainer_params["init"]["optimizer"]
        )

    return {"trainer": trainer_params}


def merge_trainer_params(params1: dict, params2: dict):
    """
    Recursively merges two trainer param dictionaries.

    Callbacks are merged specially with a call to merge_callbacks.
    """
    from cerebras.modelzoo.common.utils.utils import merge_recursively

    params1 = deepcopy(params1)
    params2 = deepcopy(params2)

    init1 = params1["trainer"]["init"]
    init2 = params2["trainer"]["init"]

    for key in ("callbacks", "loggers"):
        init1[key] = merge_callbacks(
            init1.get(key, []),
            init2.pop(key, []),
        )

    return merge_recursively(params1, params2)


def merge_callbacks(callbacks1: list, callbacks2: list):
    """Merges two callbacks lists.

    We do this by collapsing the list of callbacks into a dictionary,
    recursively merging the dictionaries, and then reconstructing a
    list of callbacks.
    """
    from cerebras.modelzoo.common.utils.utils import merge_recursively

    def collapse(callbacks):
        counter = Counter()

        def increment(k):
            counter[k] += 1
            return counter[k]

        return {
            (k, increment(k)): v
            for callback in callbacks
            for k, v in callback.items()
        }

    return [
        {k: v}
        for (k, i), v in merge_recursively(
            collapse(callbacks1), collapse(callbacks2)
        ).items()
    ]


def _validate_trainer_params(trainer_params: dict, model_name: str):
    """Runs validation on the model and inputs."""

    if (config_class := registry.get_config_class(model_name)) is None:
        raise ValueError(
            f"Could not find a config class registered for model with name "
            f"{model_name}. Available models are: {', '.join(registry.list_models())}"
        )

    model_cfg_cls = get_class_type(config_class, "model")
    train_cfg_cls = get_class_type(config_class, "train_input")
    eval_cfg_cls = get_class_type(config_class, "eval_input")

    if any(x is None for x in [model_cfg_cls, train_cfg_cls, eval_cfg_cls]):
        raise ValueError(
            f"Expected config class {config_class} to have a field for each "
            f"section named \"model\", \"train_input\", \"eval_input\"."
        )

    model_cfg = (model_cfg_cls(**trainer_params["init"]["model"]), None)
    trainer_params["init"]["model"] = model_cfg[0]

    # We need to special case this for backwards compatibility reasons
    # We can get rid of it once we deprecate and remove these keys from
    # the model config.
    trainer_params["init"].setdefault("precision", {}).update(
        {
            v2_key: getattr(trainer_params["init"]["model"], v1_key)
            for v1_key, v2_key in {
                "mixed_precision": "enabled",
                "fp16_type": "fp16_type",
            }.items()
            if hasattr(trainer_params["init"]["model"], v1_key)
        }
    )

    train_cfg = None
    if "fit" in trainer_params:
        train_cfg = (
            train_cfg_cls(**trainer_params["fit"]["train_dataloader"]),
            functools.partial(
                trainer_params["fit"].__setitem__, "train_dataloader"
            ),
        )

    eval_cfgs = []

    # Check "fit" method
    if (key := "fit") in trainer_params and "val_dataloader" in trainer_params[
        key
    ]:
        val_dataloaders = trainer_params[key]["val_dataloader"]
        if not isinstance(val_dataloaders, (list, tuple)):
            val_dataloaders = [val_dataloaders]

        trainer_params[key]["val_dataloader"] = []
        for params in val_dataloaders:
            eval_cfgs.append(
                (
                    eval_cfg_cls(**params),
                    functools.partial(
                        trainer_params[key]["val_dataloader"].append
                    ),
                )
            )

    # check "validate" method
    if (key := "validate") in trainer_params:
        validate = trainer_params[key]
        if "val_dataloader" in validate:
            eval_cfgs.append(
                (
                    eval_cfg_cls(**validate["val_dataloader"]),
                    functools.partial(validate.__setitem__, "val_dataloader"),
                )
            )

    # check "validate_all" method
    if (key := "validate_all") in trainer_params:
        val_dataloaders = trainer_params[key].get("val_dataloaders", [])
        if not isinstance(val_dataloaders, (list, tuple)):
            val_dataloaders = [val_dataloaders]

        trainer_params[key]["val_dataloaders"] = []
        for val_dataloader in val_dataloaders:
            eval_cfgs.append(
                (
                    eval_cfg_cls(**val_dataloader),
                    functools.partial(
                        trainer_params[key]["val_dataloaders"].append
                    ),
                )
            )

    # Add dataloader defaults for eval harness callbacks
    callbacks = trainer_params["init"].get("callbacks", [])
    for callback_dict in callbacks:
        for eh in ("EleutherEvalHarness", "BigCodeEvalHarness"):
            if eh in callback_dict:
                eh_callback = callback_dict[eh]
                eval_cfgs.append(
                    (
                        eval_cfg_cls(**eh_callback),
                        functools.partial(eh_callback.update),
                    )
                )

    if (fn := getattr(config_class, "set_config_defaults", None)) and callable(
        fn
    ):
        # pylint: disable=not-callable
        fn(
            model_cfg[0],
            train_cfg[0].params if train_cfg is not None else None,
            [x[0].params if x is not None else None for x in eval_cfgs],
        )

    for item in [model_cfg, train_cfg, *eval_cfgs]:
        if item is None:
            continue

        cfg, setter = item
        if cfg is not None:
            cfg.__validate__()

    for item in [train_cfg, *eval_cfgs]:
        if item is None:
            continue

        cfg, setter = item
        if cfg is not None:
            setter(flatten_data_params(asdict(cfg)))


def convert_output_to_dict(output):
    """
    Converts the output of a model to a dictionary containing the
    loss and logits.
    """
    if isinstance(output, torch.Tensor):
        # Assume that if output is scalar Tensor that it is the loss
        if prod(output.shape) == 1:
            outputs = {"loss": output}
        else:
            outputs = {"output": output}
    # TODO: This tuple is arbitrary. We should change all the models to return a dict
    elif isinstance(output, (list, tuple)):
        loss, logits = output
        outputs = {"loss": loss, "logits": logits}
    elif isinstance(output, dict):
        outputs = output
    else:
        raise TypeError(
            f"Output must be a torch.Tensor or dict. Got: {type(output)}"
        )

    return outputs


def convert_optimizer_params(optimizer_params):
    """Converts the older optimizer params to the newer format.

    The older format specified the optimizer in the following way

    ..code:: yaml

        optimizer_type: SGD
        momentum: 0.9
        ...

    The newer format refactors the params into the following format

    ..code:: yaml

        SGD:
          momentum: 0.9
        ...

    Args:
        optimizer_params: The params to convert.
    """
    new_optimizer_params = {}
    optimizer_type = optimizer_params.pop("optimizer_type")
    new_optimizer_params[optimizer_type] = optimizer_params
    return new_optimizer_params


def convert_lr_scheduler(scheduler_params, adjust_learning_rate=None):
    """Converts the older scheduler params to the newer format.

    The older format specified the scheduler in the following way

    ..code:: yaml

        - scheduler: LinearLR
          main_scheduler: SequentialLR
          total_iters: 100
          ...
        - scheduler: LinearLR
          main_scheduler: SequentialLR
          total_iters: 100
          ...

    The newer format refactors the params into the following format

    ..code:: yaml

        SequentialLR:
          schedulers:
          - LinearLR:
            total_iters: 100
            ...
          - LinearLR:
            total_iters: 100
            ...
          milestones:
            ...

    Args:
        scheduler_params: The params to convert.
        adjust_learning_rate: The factor to scale the LR by for mUP.
    """

    new_scheduler_params = {}
    if adjust_learning_rate:
        new_scheduler_params["ScalePerParamLR"] = {
            "scheduler": convert_lr_scheduler(scheduler_params)
        }
    elif isinstance(scheduler_params, (list, tuple)):
        scheduler_params_list = []
        main_scheduler = set()
        for sub_scheduler_params in scheduler_params:
            if sched := sub_scheduler_params.pop("main_scheduler", None):
                main_scheduler.add(sched)
            scheduler_params_list.append(
                convert_lr_scheduler(sub_scheduler_params)
            )
        if len(main_scheduler) > 1:
            raise ValueError(
                f"Got conflicting `main_scheduler` values: {main_scheduler}. "
                f"Please make sure to specify the same main scheduler."
            )
        main_scheduler = next(iter(main_scheduler)) if main_scheduler else None
        if main_scheduler is not None and main_scheduler.lower() in (
            "chained",
            "chainedlr",
        ):
            key = "ChainedScheduler"
        else:
            key = "SequentialLR"
        if len(scheduler_params_list) == 1 and key == "SequentialLR":
            new_scheduler_params = scheduler_params_list[0]
        else:
            new_scheduler_params[key] = {"schedulers": scheduler_params_list}
    else:
        lr_scheduler_map = {
            cls.__name__.lower(): cls.__name__
            for cls in retrieve_all_subclasses(
                cstorch.optim.lr_scheduler.LRScheduler
            )
        }
        scheduler_name = scheduler_params.pop("scheduler").lower()
        for name in (scheduler_name, f"{scheduler_name}lr"):
            if name in lr_scheduler_map:
                name = lr_scheduler_map[name]
                break
        else:
            raise ValueError(
                f"Invalid lr_scheduler type. Expected one of "
                f"{list(lr_scheduler_map.keys())}. Got: {scheduler_name}"
            )
        new_scheduler_params[name] = scheduler_params
    return new_scheduler_params
