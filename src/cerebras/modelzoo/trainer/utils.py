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
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from functools import lru_cache
from math import prod
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from warnings import warn

import torch
import yaml
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.classes import retrieve_all_subclasses
from cerebras.modelzoo.config import BaseConfig
from cerebras.modelzoo.trainer.validate import validate_trainer_params
from cerebras.pytorch.backend import get_backend_args

ModeT = Literal["train", "train_and_eval", "eval", "eval_all"]


def mode_to_cmd(mode: ModeT):
    """Convert from ModeT semantics to Trainer semantics."""
    if mode in ("train", "train_and_eval"):
        return "fit"
    elif mode == "eval":
        return "validate"
    elif mode == "eval_all":
        return "validate_all"
    else:
        raise ValueError(f"Invalid mode {mode}.")


def run_trainer(mode: ModeT, params: Union[Dict[str, Any], BaseConfig]):
    """Runs training and/or validation using the Trainer with the given params.

    Args:
        mode: The mode to run the Trainer in. Can be one of:
            - "train": Train the model.
            - "eval": Evaluate the model.
            - "train_and_eval": Train the model and then evaluate it.
            - "eval_all": Evaluate the model on all available checkpoints and dataloaders.
        params: A dictionary/object containing the configuration for the Trainer.
            If legacy keys are detected, they will be automatically converted
            to the new format.
    """
    if isinstance(params, dict) and is_legacy_params(params):
        warn(
            f"Detected that legacy params are being used. "
            f"Automatically converting params to new format. "
            f"To see how the legacy params map to the new format, see: "
            f"https://docs.cerebras.net/en/latest/wsc/Model-zoo/yaml/table.html"
        )

        params = convert_legacy_params_to_trainer_params(
            params,
            # Allow None values in the params
            obj_filter=lambda obj: obj is None,
        )

    if isinstance(params, BaseConfig):
        config = params
        try:
            trainer = configure_trainer_from_config(config, mode)
        except:
            import json

            warn(
                f"Failed to configure trainer from config:\n"
                f"{json.dumps(config.model_dump(), sort_keys=False, indent=4)}"
            )
            raise

        if mode == "eval":
            if not config.validate:
                raise RuntimeError(
                    "Validation requested but config is missing `validate` section. "
                    "Please add a `validate` section to your trainer configuration."
                )

            trainer.validate(
                val_dataloader=create_dataloader_from_config(
                    config.validate.val_dataloader
                ),
                ckpt_path=config.validate.ckpt_path,
            )

        elif mode == "eval_all":
            if not config.validate_all:
                raise RuntimeError(
                    "Validation requested but config is missing `validate_all` section. "
                    "Please add a `validate_all` section to your trainer configuration."
                )

            val_dataloaders = list(
                map(
                    create_dataloader_from_config,
                    config.validate_all.val_dataloaders,
                )
            )

            ckpt_paths = config.validate_all.ckpt_paths
            if ckpt_paths is Ellipsis:
                all_ckpts = []
                if trainer.checkpoint.autoload_last_checkpoint:
                    all_ckpts = trainer.checkpoint.get_all_checkpoints(
                        trainer.model_dir
                    )
                if all_ckpts:
                    ckpt_paths = all_ckpts
                else:
                    raise FileNotFoundError(
                        f"No checkpoints were found for evaluation. "
                        f"Please pass in at least one checkpoint via ckpt_paths or "
                        f"set `autoload_last_checkpoint` to True and ensure that the model "
                        f"directory \"{trainer.model_dir}\" contains at least one "
                        f"checkpoint whose name matches the expected format of: "
                        f"{trainer.checkpoint.checkpoint_name}"
                    )

            trainer.validate_all(
                val_dataloaders=val_dataloaders,
                ckpt_paths=ckpt_paths,
            )

        elif mode in ("train", "train_and_eval"):
            if not config.fit:
                raise RuntimeError(
                    "Fit requested but config is missing `fit` section. "
                    "Please add a `fit` section to your trainer configuration."
                )

            train_dataloader = create_dataloader_from_config(
                config.fit.train_dataloader
            )
            val_dataloader = None

            if mode == "train":
                # Disable all validation during training including eval harness
                trainer.loop.eval_frequency = None
            else:
                if config.fit.val_dataloader is not None:
                    val_dataloader = list(
                        map(
                            create_dataloader_from_config,
                            config.fit.val_dataloader,
                        )
                    )

            trainer.fit(train_dataloader, val_dataloader, config.fit.ckpt_path)

        else:
            raise ValueError(
                f"Invalid mode \"{mode}\". "
                f"Expected one of: train, train_and_eval, eval, eval_all."
            )

    else:
        configs = validate_trainer_params(params)
        for config in configs:
            run_trainer(mode, config)


def create_dataloader_from_config(data_processor_config):
    data_processor_cls = data_processor_config.get_orig_class()

    def data_processor_fn(**kwargs):
        data_processor = data_processor_cls(**kwargs)
        if isinstance(data_processor, torch.utils.data.DataLoader):
            return data_processor
        elif hasattr(data_processor, "create_dataloader"):
            return data_processor.create_dataloader()
        elif isinstance(data_processor, Iterable):
            return data_processor

        raise TypeError(
            "Expected dataprocessor to be an iterable (e.g., torch dataloader) "
            "or have a `create_dataloader()` method that returns "
            "an iterable for generating input data."
        )

    return cstorch.utils.data.DataLoader(
        data_processor_fn, **data_processor_config.get_orig_class_args()
    )


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
            backend.cluster.config = kwargs.pop("cluster_config")

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


def create_backend_from_config(init_config):
    backend_params = init_config.backend
    if device := init_config.device:
        backend_params.setdefault("backend_type", device)

    backend_type = backend_params.get("backend_type", None)
    if backend_type is None:
        raise ValueError(
            "No device specified. Please specify a device using the 'device' key "
            "inside the 'init' key of the trainer params or through the cszoo CLI "
            "with the target_device flag (e.g. --target_device=CSX)."
        )

    backend_type = backend_type.upper()

    backend_args = {
        name: backend_params.get(name)
        for name, _ in get_backend_args(backend_type).items()
        if name not in ("self", "backend_type") and name in backend_params
    }

    if backend_type == "CSX":
        # Special handling for cluster config as dicts are not hashable
        cluster_config = backend_args.get("cluster_config", {})
        if isinstance(cluster_config, dict):
            cluster_config = cstorch.distributed.ClusterConfig(**cluster_config)
        backend_args["cluster_config"] = cluster_config

    return cached_cstorch_backend(backend_type, **backend_args)


def configure_trainer_from_config(
    trainer_config: BaseConfig, mode: Optional[ModeT] = None
):
    """Configure a Trainer object from a trainer config object.

    Args:
        trainer_config: The trainer config object used to configure the Trainer.
        mode: The mode that the trainer is being configured for. If None, no
            mode-specific modifications are applied.
    """
    # pylint: disable=unused-import
    import cerebras.modelzoo.trainer.extensions  # noqa
    from cerebras.modelzoo.trainer import Trainer
    from cerebras.modelzoo.trainer.callbacks import Callback

    init_config = trainer_config.init

    def backend_fn():
        return create_backend_from_config(init_config)

    def model_fn():
        return init_config.model()

    def optimizer_fn(model):
        # No need for optimizer in eval.
        if mode in ("eval", "eval_all"):
            return None

        optimizer_config = init_config.optimizer
        if optimizer_config is None:
            return None

        from cerebras.modelzoo.common.optim_utils import configure_param_groups

        params = configure_param_groups(
            model,
            optimizer_config.dict(),
        )

        return optimizer_config(params=params)

    def scheduler_fn():
        # No need for schedulers in eval.
        if mode in ("eval", "eval_all"):
            return None

        if init_config.schedulers is None:
            return None

        def _create_scheduler(optimizer, scheduler_config):
            scheduler = scheduler_config(optimizer=optimizer)

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

        return [
            functools.partial(
                _create_scheduler, scheduler_config=scheduler_config
            )
            for scheduler_config in init_config.schedulers
        ]

    def sparsity_fn():
        if init_config.sparsity is None:
            return None

        return cstorch.sparse.configure(
            [config.model_dump() for config in init_config.sparsity]
        )

    def construct_callback(c: Optional[Callable]):
        if c is None:
            return None
        return c()

    def callbacks_fn():
        return [callback() for callback in init_config.callbacks]

    def loggers_fn():
        return [logger() for logger in init_config.loggers]

    class SaveTrainerParams(Callback):
        """Save the Trainer params to the artifact directory."""

        def pre_setup(self, trainer):
            # Save a full copy of the params to the artifact directory
            with open(trainer.artifact_dir / "trainer_params.yaml", "w") as f:
                yaml.dump(
                    {"trainer": trainer_config.model_dump(exclude_unset=False)},
                    f,
                    sort_keys=False,
                )

            # Save the original params to the summary directory
            with open(trainer.summary_dir / "trainer_params.yaml", "w") as f:
                try:
                    # We want to first try to dump a JSON serializable version
                    # of the trainer first to mimic the params.yaml that they
                    # provided.
                    params = trainer_config.model_dump(
                        mode="json", exclude_unset=True, warnings=False
                    )
                except Exception as e:  # pylint: disable=broad-except
                    # If the trainer is not JSON serializable, then we just dump
                    # the trainer as is as YAML can handle non JSON serializable types.
                    # The only caveat being that the user may not be able to use
                    # the saved YAML as is.
                    msg = (
                        f"Trainer is not JSON serializable due to:\n{e}\n"
                        f"You may encounter issues when loading the saved yaml file:\n"
                        f"{trainer.summary_dir / 'trainer_params.yaml'}"
                    )
                    trainer.logger.warning(msg)

                    params = trainer_config.model_dump(
                        exclude_unset=True, warnings=False
                    )

                yaml.dump({"trainer": params}, f, sort_keys=False)

        def on_save_trainer_state(self, trainer, state_dict):
            pass

        def on_load_trainer_state(self, trainer, state_dict):
            pass

    with SaveTrainerParams():
        return Trainer(
            backend=backend_fn(),
            model_dir=init_config.model_dir,
            model=model_fn,
            optimizer=optimizer_fn,
            schedulers=scheduler_fn(),
            precision=construct_callback(init_config.precision),
            sparsity=sparsity_fn(),
            loop=construct_callback(init_config.loop),
            checkpoint=construct_callback(init_config.checkpoint),
            logging=construct_callback(init_config.logging),
            callbacks=callbacks_fn(),
            loggers=loggers_fn(),
            seed=init_config.seed,
            autorestart=construct_callback(init_config.autorestart),
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
                "cbcore_image": "runconfig.cbcore_image",
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
                    "csx.debug.chf_memory_gi": "runconfig.chf_memory_gi",
                }
            },
            {
                "ScopedTrainFlags": {
                    "csx.performance.micro_batch_size": {
                        "*": "train_input.micro_batch_size.*",
                        "csx.performance.micro_batch_size": "train_input.micro_batch_size",
                    },
                }
            },
            {
                "ScopedValidateFlags": {
                    "csx.performance.micro_batch_size": {
                        "*": "eval_input.micro_batch_size.*",
                        "csx.performance.micro_batch_size": "eval_input.micro_batch_size",
                    },
                }
            },
            {
                "DebugArgsPath": {
                    "debug_args_path": "runconfig.debug_args_path",
                },
            },
            {
                "Lora": {
                    "lora_params": "model.lora_params.*",
                }
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
                    "entity": "wandb.entity",
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


EEH_TRAINER_PARAMS_TO_LEGACY = {
    "EleutherEvalHarness": {
        "eeh_args": {
            "tasks": "runconfig.tasks",
            "num_fewshot": "runconfig.num_fewshot",
            "output_path": "runconfig.output_path",
            "limit": "runconfig.limit",
            "use_cache": "runconfig.use_cache",
            "cache_requests": "runconfig.cache_requests",
            "check_integrity": "runconfig.check_integrity",
            "write_out": "runconfig.write_out",
            "log_samples": "runconfig.log_samples",
            "system_instruction": "runconfig.system_instruction",
            "apply_chat_template": "runconfig.apply_chat_template",
            "fewshot_as_multiturn": "runconfig.fewshot_as_multiturn",
            "show_config": "runconfig.show_config",
            "include_path": "runconfig.include_path",
            "predict_only": "runconfig.predict_only",
            "trust_remote_code": "runconfig.trust_remote_code",
            "max_tokens": "runconfig.max_tokens",
            "temperature": "runconfig.temperature",
            "top_k": "runconfig.top_k",
            "top_p": "runconfig.top_p",
        },
        "keep_data_dir": "runconfig.keep_data_dir",
    }
}


BCEH_TRAINER_PARAMS_TO_LEGACY = {
    "BigCodeEvalHarness": {
        "bigcode_args": {
            "prefix": "runconfig.prefix",
            "n_samples": "runconfig.n_samples",
            "tasks": "runconfig.tasks",
            "instruction_tokens": "runconfig.instruction_tokens",
            "limit": "runconfig.limit",
            "limit_start": "runconfig.limit_start",
            "save_every_k_tasks": "runconfig.save_every_k_tasks",
            "load_generations_path": "runconfig.load_generations_path",
            "load_data_path": "runconfig.load_data_path",
            "metric_output_path": "runconfig.metric_output_path",
            "load_generations_intermediate_paths": "runconfig.load_generations_intermediate_paths",
            "save_generations_path": "runconfig.save_generations_path",
            "save_references_path": "runconfig.save_references_path",
            "prompt": "runconfig.prompt",
            "check_references": "runconfig.check_references",
            "max_tokens": "runconfig.max_tokens",
            "temperature": "runconfig.temperature",
            "top_k": "runconfig.top_k",
            "top_p": "runconfig.top_p",
        },
        "keep_data_dir": "runconfig.keep_data_dir",
    }
}


# List of V2 keys to pop after conversion from V1 to V2
TRAINER_POST_CONVERSION_PRUNE_LIST = [
    "init.model.compression",
    "init.model.lora_params",
    "init.model.mixed_precision",
    "init.model.fp16_type",
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


def convert_legacy_params_to_trainer_params(
    params, obj_filter=None, extra_legacy_mapping_fn=None
):
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

    if extra_legacy_mapping_fn is not None:
        extra_legacy_mapping_fn(TRAINER_PARAMS_TO_LEGACY)

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

    # Disallow fit without train dataloader
    if (
        "fit" in trainer_params
        and "train_dataloader" not in trainer_params["fit"]
    ):
        trainer_params.pop("fit")

    # Disallow validate without val dataloader
    if (
        "validate" in trainer_params
        and "val_dataloader" not in trainer_params["validate"]
    ):
        trainer_params.pop("validate")

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

    # Parse through callbacks and adjust as necessary
    callbacks = []
    for callback in trainer_params["init"].get("callbacks", []):
        class_name = next(iter(callback))
        if callback[class_name] is True:
            callback[class_name] = {}
        if callback[class_name] is False:
            callback[class_name] = None

        if class_name == "Listener":
            # Unpack listeners into individual callbacks
            for listener_config in callback[class_name]["listeners"]:
                if "listener_type" not in listener_config:
                    raise ValueError(
                        f"Expected to find a key named `listener_type` in all listeners. "
                        f"But found a listener config that doesn't conform to this requirement: "
                        f"\n{listener_config}"
                    )
                listener_type = listener_config.pop("listener_type")
                callbacks.append({listener_type: listener_config})
        elif class_name in ["ScopedTrainFlags", "ScopedValidateFlags"]:
            k = "csx.performance.micro_batch_size"
            if k in callback[class_name] and list(
                callback[class_name][k].keys()
            ) == [k]:
                callbacks.append({class_name: {k: callback[class_name][k][k]}})
            else:
                callbacks.append(callback)
        else:
            callbacks.append(callback)

    # Assign back the new callbacks to the params
    if callbacks:
        trainer_params["init"]["callbacks"] = callbacks

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


def inject_cli_args_to_trainer_params(
    runconfig, params, extra_legacy_mapping_fn=None
):
    """Inject CLI arguments into a trainer config."""
    runconfig = deepcopy(runconfig)
    params = deepcopy(params)

    # Recursively update the params with the runconfig.
    # `checkpoint_path` shows up in fit/validate/validate_all, but
    # if there are no input sections, legacy->trainer conversion pops
    # them before returnng, so handle it specially.
    if "checkpoint_path" in runconfig:
        ckpt_path = runconfig.pop("checkpoint_path")

        def add_ckpt_path(p):
            for method, key in [
                ("fit", "ckpt_path"),
                ("validate", "ckpt_path"),
                ("validate_all", "ckpt_paths"),
            ]:
                if method in p["trainer"] and p["trainer"][method] is not None:
                    p["trainer"][method][key] = ckpt_path

    else:

        def add_ckpt_path(p):
            return None

    # Add a dummy model input so we always have an "init" key, but pop it later
    cli_args = convert_legacy_params_to_trainer_params(
        {"runconfig": runconfig, "model": {"dummy": 1}},
        extra_legacy_mapping_fn=extra_legacy_mapping_fn,
    )
    del cli_args["trainer"]["init"]["model"]

    trainers = params["trainer"]
    if isinstance(params["trainer"], (list, tuple)):
        params["trainer"] = []
        for trainer in trainers:
            merged = merge_trainer_params(trainer, cli_args)
            add_ckpt_path(merged)
            params["trainer"].append(merged)
    else:
        params = merge_trainer_params(params, cli_args)
        add_ckpt_path(params)

    return params


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
