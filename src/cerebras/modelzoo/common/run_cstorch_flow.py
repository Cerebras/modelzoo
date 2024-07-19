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

"""Generic run scripts build using the cstorch API."""

import copy
import dataclasses
import logging
import numbers
import os
import re
import shlex
import sys
import warnings
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, Optional, Union
from warnings import warn

import torch
from typing_extensions import Self

import cerebras.pytorch as cstorch
from cerebras.appliance.utils.debug_args import (
    get_debug_args,
    update_debug_args_from_keys,
)
from cerebras.appliance.utils.ini import set_ini
from cerebras.modelzoo.common.checkpoint_utils import (
    CkptInfo,
    get_all_checkpoints,
    get_latest_checkpoint,
)
from cerebras.modelzoo.common.dump_context import DumpContext
from cerebras.modelzoo.common.half_dtype import set_half_dtype_from_params
from cerebras.modelzoo.common.input_utils import (
    validate_streaming_and_micro_batch_size,
)
from cerebras.modelzoo.common.mlops import mlops_run
from cerebras.modelzoo.common.optim_utils import (
    configure_param_groups,
    flatten_optimizer_params,
)
from cerebras.modelzoo.common.pytorch_utils import (
    load_from_checkpoint_file,
    setup_artifact_dir,
    setup_logging,
)
from cerebras.modelzoo.common.utils.model.mup_utils import is_mup
from cerebras.modelzoo.common.utils.run.utils import DeviceType
from cerebras.modelzoo.common.utils.utils import (
    configure_compression,
    configure_selective_gradient,
    format_rate,
    update_debug_args_with_mem_limits,
)
from cerebras.modelzoo.trainer import summarize_scalar, summarize_tensor
from cerebras.pytorch.core import modes
from cerebras.pytorch.utils.nest import visit_torch_tensors


def setup_hf_env_vars(hf_cache_dir=None):
    from cerebras.appliance.environment import appliance_environ

    if hf_cache_dir is not None:
        appliance_environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        appliance_environ["HF_HOME"] = hf_cache_dir
        appliance_environ["HF_DATASETS_CACHE"] = hf_cache_dir


def get_cluster_config(params: dict) -> cstorch.distributed.ClusterConfig:
    """Sets up CS cluster config for the run."""
    runconfig = params["runconfig"]

    debug_args = get_debug_args(runconfig.get("debug_args_path"))
    update_debug_args_with_mem_limits(debug_args, runconfig)
    if extra_debug_args := runconfig.get("debug_args"):
        update_debug_args_from_keys(debug_args, extra_debug_args)
    if ini := runconfig.get("ini"):
        set_ini(debug_args, **ini)

    cluster_config = cstorch.distributed.ClusterConfig(
        mgmt_address=runconfig.get("mgmt_address"),
        mgmt_namespace=runconfig.get("mgmt_namespace"),
        credentials_path=runconfig.get("credentials_path"),
        num_csx=runconfig.get("num_csx"),
        max_wgt_servers=runconfig.get("num_wgt_servers"),
        max_act_per_csx=runconfig.get("num_act_servers"),
        num_workers_per_csx=runconfig.get("num_workers_per_csx"),
        job_labels=runconfig.get("job_labels"),
        job_time_sec=runconfig.get("job_time_sec"),
        mount_dirs=runconfig.get("mount_dirs"),
        python_paths=runconfig.get("python_paths"),
        disable_version_check=runconfig.get("disable_version_check"),
    )

    job_priority = runconfig.get("job_priority")
    if job_priority:
        cluster_config.job_priority = job_priority

    transfer_processes = runconfig.get("transfer_processes")
    if transfer_processes:
        cstorch.backends.csx.performance.transfer_processes = transfer_processes

    fabric_type_blacklist = runconfig.get("fabric_type_blacklist")
    if fabric_type_blacklist:
        cstorch.backends.csx.debug.fabric_type_blacklist = fabric_type_blacklist

    cstorch.backends.csx.debug.debug_args = debug_args

    if "precision_opt_level" in params["model"]:
        raise ValueError(
            "Passing `precision_opt_level` via `model` params is no longer supported. "
            "Please use `params[\"runconfig\"][\"precision_opt_level\"]` instead."
        )
    precision_opt_level = runconfig.get("precision_opt_level")
    if precision_opt_level is None:
        precision_opt_level = 1

    cstorch.backends.csx.precision.optimization_level = precision_opt_level

    return cluster_config


def run_cstorch_flow(params, params_obj, model_fn, train_data_fn, eval_data_fn):
    """
    Set up the cstorch run and call the appropriate helper based on the mode.

    Args:
        params: the params dictionary extracted from the params.yaml used
        params_obj: Config object based on the params dict
        model_fn: A callable that takes in the params dictionary and returns
            a torch.nn.Module
        train_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader
        eval_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader
    """
    runconfig = params["runconfig"]

    if "seed" in runconfig and runconfig["seed"] is not None:
        # Ensure we set seed before any model initialization
        torch.manual_seed(runconfig["seed"])

    # Configure the Cerebras Wafer Scale cluster
    cluster_config = get_cluster_config(params)

    artifact_dir = setup_artifact_dir(
        runconfig.get("model_dir"), runconfig["mode"]
    )

    # Set up logging level
    setup_logging(
        runconfig.get("logging"),
        runconfig.get("streamer_logging"),
        logging_dir=artifact_dir,
        model_dir=runconfig.get("model_dir"),
    )
    setup_hf_env_vars(hf_cache_dir=runconfig.get("hf_cache_dir"))

    # log the command used to run
    logging.info(
        f"Modelzoo Command Executed: {shlex.join(sys.argv)}",
        extra={"block": "console"},
    )

    # Check if current run is configured with muP
    if is_mup(params.get("model", {})):
        logging.info("This is a muP configured run")

    with mlops_run(params):
        if runconfig["mode"] == modes.TRAIN:
            run_cstorch_train(
                params,
                params_obj,
                model_fn,
                train_data_fn,
                cluster_config,
                artifact_dir,
            )
        elif runconfig["mode"] in [modes.EVAL, modes.EVAL_ALL]:
            run_cstorch_eval(
                params,
                params_obj,
                model_fn,
                eval_data_fn,
                cluster_config,
                artifact_dir,
            )
        elif runconfig["mode"] == modes.TRAIN_AND_EVAL:
            from cerebras.modelzoo.common.train_and_eval import train_and_eval

            train_and_eval(
                params,
                params_obj,
                model_fn,
                train_data_fn,
                eval_data_fn,
                cluster_config,
                artifact_dir,
            )
        else:
            raise ValueError(
                f"Unsupported mode: {runconfig['mode']}. "
                f"Supported modes are: {modes.TRAIN}, {modes.EVAL}, "
                f"{modes.TRAIN_AND_EVAL}"
            )


def run_cstorch_train(
    params, params_obj, model_fn, input_fn, cluster_config, artifact_dir
):
    """
    Runs the training workflow built using the cstorch API.

    Args:
        params: the params dictionary extracted from the params.yaml used
        model_fn: A callable that takes in the params dictionary and returns
            a torch.nn.Module
        input_data: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader
    """
    from cerebras.appliance.errors import ApplianceNanError

    params_copy = copy.deepcopy(params)

    if not isinstance(params.get("optimizer"), dict):
        raise ValueError(
            "An `optimizer` configuration is required for training"
        )

    input_params = params.get("train_input", {})
    runconfig = params["runconfig"]

    target_device = runconfig["target_device"]

    model_dir = runconfig["model_dir"]
    compile_dir = runconfig.get("compile_dir")
    log_steps = runconfig.get("log_steps")
    checkpoint_steps = runconfig.get("checkpoint_steps")
    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)
    log_summaries = params["optimizer"].pop("log_summaries", False)
    check_loss_values = runconfig.get("check_loss_values", True)
    log_input_summaries = runconfig.get("log_input_summaries", False)
    activation_steps = (
        log_steps if runconfig.get("enable_act_frequency", False) else None
    )
    experimental_configs = runconfig.get("experimental", {})
    listener_configs = experimental_configs.get("listeners", [])

    optimizer_params = params["optimizer"]

    # Parse grad scaler params and pop them from optimizer params
    grad_scaler = None
    grad_scaler_params = GradScalerParams.from_dict(optimizer_params)

    mixed_precision = params["model"].get("mixed_precision", False)
    mp_dtype = set_half_dtype_from_params(params["model"])

    if params["model"].get("fp16_type") == "cbfloat16":
        if target_device != DeviceType.CSX:
            mp_dtype = cstorch.amp.set_half_dtype(torch.bfloat16)
            msg = f"cbfloat16 is only supported on CSX. Setting half dtype to bfloat16."
            if grad_scaler_params.loss_scaling_factor == "dynamic":
                grad_scaler_params.loss_scaling_factor = 1.0
                msg = (
                    f"{msg} DLS is not needed in bfloat16 either, so setting "
                    f"`loss_scaling_factor` to `1.0`."
                )
            warnings.warn(msg)
        elif grad_scaler_params.loss_scaling_factor != "dynamic":
            raise ValueError(
                f"In order to use cbfloat16, dynamic loss scaling must be enabled. "
                f"Otherwise, gradients might underflow/overflow in the middle of "
                f"training and cause NaNs. Please set `loss_scaling_factor` to "
                f"`dynamic` to use cbfloat16."
            )

    if (
        grad_scaler_params.loss_scaling_factor == "dynamic"
        and params["model"].get("fp16_type") == "bfloat16"
    ):
        grad_scaler_params.loss_scaling_factor = 1.0
        logging.info(
            f"No need to use DLS for loss when half dtype is bfloat16. "
            f"Setting `loss_scaling_factor ` to `1.0`."
        )

    if (
        mixed_precision
        and mp_dtype != torch.bfloat16
        and target_device == DeviceType.CPU
    ):
        warnings.warn(
            "Mixed precision on CPU is only supported with bfloat16. "
            "Setting half dtype to bfloat16."
        )
        mp_dtype = cstorch.amp.set_half_dtype(torch.bfloat16)

    use_cstorch_optimizer_step = runconfig.get(
        "use_cstorch_optimizer_step", False
    )
    # Default to keeping all checkpoints
    max_checkpoints = runconfig.get("max_checkpoints", None)

    grad_accum_steps = optimizer_params.pop("grad_accum_steps", 1)
    if target_device == DeviceType.CSX:
        if grad_accum_steps != 1:
            logging.info(
                "`grad_accum_steps` param has no effect when running on the CSX. "
                "Consider setting `micro_batch_size` to \"auto\" or \"disable\" to enable or "
                "disable micro batch tiling on CSX."
            )
    else:
        logging.info(f"Gradient accumulation steps is {grad_accum_steps}")

    ckpt_info = CkptInfo(ckpt_dir=model_dir)

    if target_device == DeviceType.CSX:
        backend = cstorch.backend(
            "CSX",
            artifact_dir=artifact_dir,
            compile_dir=compile_dir,
            compile_only=compile_only,
            validate_only=validate_only,
            retrace_every_iteration=runconfig.get(
                "retrace_every_iteration", False
            ),
            cluster_config=cluster_config,
        )

        backend.device.config.lazy_initialization = runconfig.get(
            "lazy_initialization", True
        )
        if not backend.device.config.lazy_initialization:
            # Drop data only has any effect if we're not tracing the
            # initialization
            backend.device.config.drop_data = runconfig.get(
                "drop_data", compile_only or validate_only
            )

    elif target_device == DeviceType.CPU:
        backend = cstorch.backend(
            "CPU",
            artifact_dir=artifact_dir,
        )
    elif target_device == DeviceType.GPU:
        backend = cstorch.backend(
            "GPU",
            artifact_dir=artifact_dir,
            enable_distributed=runconfig.get("enable_distributed", False),
            main_process_id=runconfig.get("main_process_id", 0),
            dist_backend=runconfig.get("dist_backend", "nccl"),
            init_method=runconfig.get("init_method", "env://"),
            sync_batchnorm=runconfig.get("sync_batchnorm", False),
        )

    # Pop lora_params before model_fn is run to prevent it from showing up
    # in the unused params warning
    lora_params = None
    if "lora_params" in params["model"]:
        lora_params = params["model"].pop("lora_params")

    with backend.device:
        model = model_fn(params_obj if params_obj else params)

        if lora_params:
            # After model init, we need to convert it into a LoRA model
            from cerebras.modelzoo.common.utils.model.lora import (
                make_model_lora,
            )

            model = make_model_lora(model, lora_params)

    compiled_model = cstorch.compile(model, backend)
    compiled_model.train()

    # Register tensor listeners.
    listeners = [
        cstorch.experimental.listener.create_listener(**listener_params)
        for listener_params in listener_configs
    ]

    # group optimizer params
    param_optimizer_grouped = configure_param_groups(
        model,
        optimizer_params,
    )

    optimizer = cstorch.optim.configure_optimizer(
        optimizer_type=optimizer_params.pop("optimizer_type"),
        params=param_optimizer_grouped,
        **flatten_optimizer_params(
            {
                k: v
                for k, v in optimizer_params.items()
                if k != "adjust_learning_rate"
            }
        ),
    )

    sparsity = None
    sparsity_config = params.pop("sparsity", {})
    if sparsity_config:

        def extract_key(config, key, default):
            if isinstance(config, dict):
                yield config.pop(key, default)
            elif isinstance(config, (list, tuple)):
                for c in config:
                    yield from extract_key(c, key, default)
            else:
                yield default

        add_summaries = any(
            extract_key(sparsity_config, "add_summaries", False)
        )

        sparsity = cstorch.sparse.configure(sparsity_config)

        # Sparsify model parameters and optimizer state
        model.apply(sparsity)
        optimizer.apply(sparsity)

        if add_summaries:
            sparsity.register_target_sparsity_hook(
                lambda _, name, target: summarize_scalar(
                    f"sparsity/{name}/target", target
                )
            )
            sparsity.register_computed_sparsity_hook(
                lambda _, name, actual: summarize_scalar(
                    f"sparsity/{name}/actual", actual
                )
            )

    # apply weight compression
    compression_config = params["model"].pop("compression", {})
    if compression_config:
        compressions = configure_compression(compression_config)
        for compression in compressions:
            model.apply(compression)

    # apply selective gradients
    selective_grad_config = params["model"].pop("selective_grad", {})
    if selective_grad_config:
        selective_grads = configure_selective_gradient(selective_grad_config)
        for selective_grad in selective_grads:
            model.apply(selective_grad)

    lr_scheduler = cstorch.optim.configure_lr_scheduler(
        optimizer,
        optimizer_params.get("learning_rate"),
        any("adjust_learning_rate" in g for g in optimizer.param_groups),
    )

    dataloader = cstorch.utils.data.DataLoader(input_fn, params)

    if grad_scaler_params.loss_scaling_factor is not None:
        if backend.is_csx:
            grad_scaler = cstorch.amp.GradScaler(
                loss_scale=grad_scaler_params.loss_scaling_factor,
                init_scale=grad_scaler_params.initial_loss_scale,
                steps_per_increase=grad_scaler_params.steps_per_increase,
                min_loss_scale=grad_scaler_params.min_loss_scale,
                max_loss_scale=grad_scaler_params.max_loss_scale,
            )
        elif backend.is_gpu:
            if grad_scaler_params.loss_scaling_factor == "dynamic":
                grad_scaler = torch.cuda.amp.GradScaler(
                    init_scale=grad_scaler_params.initial_loss_scale,
                    growth_interval=grad_scaler_params.steps_per_increase,
                )
            else:
                grad_scaler = torch.cuda.amp.GradScaler(
                    init_scale=grad_scaler_params.loss_scaling_factor,
                    growth_interval=2**63 - 1,
                )
        else:
            logging.warning(
                f"Gradient scaling is not supported on "
                f"{backend.backend_type.name}. "
                f"Disabling gradient scaling for this run"
            )

    @cstorch.checkpoint_closure
    def save_checkpoint(step):
        logging.info(f"Saving checkpoint at step {step}")

        checkpoint_file = os.path.join(model_dir, f"checkpoint_{step}.mdl")

        if os.path.exists(checkpoint_file):
            # If checkpoint path already exists, need to come up with a unique
            # name. Appending the current time, should be sufficient.
            # When changing this, keep in mind to also reflect the changes in
            # `get_latest_checkpoint` function.
            checkpoint_file = os.path.join(
                model_dir,
                f"checkpoint_{step}_{datetime.now():%Y%m%d_%H%M%S}.mdl",
            )

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if sparsity:
            state_dict["sparsity"] = sparsity.state_dict()
        if lr_scheduler:
            state_dict["lr_scheduler"] = lr_scheduler.state_dict()
        if grad_scaler:
            state_dict["grad_scaler"] = grad_scaler.state_dict()

        if dataloader.is_restartable:
            dl_state = dataloader.state_dict()
            state_dict["dataloader"] = dl_state

        state_dict["global_step"] = step

        # save modelzoo metadata
        state_dict["__metadata__"] = [
            {
                "version": cstorch.__version__,
                "model_name": model.__class__.__name__,
                "params": params_copy,
            }
        ]

        cstorch.save(state_dict, checkpoint_file)
        ckpt_info.update(checkpoint_file, max_checkpoints)

        logging.info(f"Saved checkpoint {checkpoint_file}")

    def load_checkpoint(checkpoint_path):
        nonlocal global_step

        state_dict = load_from_checkpoint_file(checkpoint_path)

        disable_strict_checkpoint_loading = runconfig.get(
            "disable_strict_checkpoint_loading", False
        )
        checkpoint_states_to_load = {
            "model",
            "optimizer",
            "sparsity",
            "dataloader",
            "grad_scaler",
            "lr_scheduler",
            "global_step",
        }
        if runconfig.get("load_checkpoint_states", "all") != "all":
            states_to_include = set(
                runconfig["load_checkpoint_states"].split(",")
            )
            if not states_to_include.issubset(checkpoint_states_to_load):
                raise KeyError(
                    "Unexpected keys specified via `load_checkpoint_states`: "
                    f"{', '.join(states_to_include - checkpoint_states_to_load)} "
                    "Only the keys in the following list are accepted: "
                    f"{', '.join(checkpoint_states_to_load)}"
                )
            checkpoint_states_to_load = states_to_include

        if "model" not in checkpoint_states_to_load:
            warn(
                "Explicitly opted-out of loading model state dict "
                "via not including \"model\" in runconfig param "
                "`load_checkpoint_states`. Model state will not be loaded."
            )
        else:
            if "model" not in state_dict:
                warn(
                    f"Checkpoint does not contain a model state dict. "
                    f"Model state was not loaded"
                )

            # This check is required for backward compatibility with checkpoints
            # saved with older versions of ModelZoo (pre rel-2.0.0)
            # We check that the model state dict keys start with "model."
            # and if they don't, we load the state dict into the model's model
            elif hasattr(model, "model") and not all(
                k.startswith("model.") for k in state_dict["model"].keys()
            ):
                model.model.load_state_dict(
                    state_dict["model"],
                    strict=not disable_strict_checkpoint_loading,
                )

            # This should be the case that is used for all checkpoints saved
            # post rel-2.0.0
            else:
                model.load_state_dict(
                    state_dict["model"],
                    strict=not disable_strict_checkpoint_loading,
                )

        if "global_step" in checkpoint_states_to_load:
            if "global_step" in state_dict:
                global_step = state_dict["global_step"]
                logging.info(
                    f"Global step {global_step} found in the checkpoint and loaded."
                )
            else:
                global_step = 0
                logging.info(
                    f"Global step not found in the checkpoint and not loaded. "
                    f"Using default initialized global step of {global_step}."
                )
        else:
            global_step = 0
            logging.info(
                f"Opting out of loading global step. Using default initialized global step of "
                f"{global_step}."
            )

        def maybe_load_state(
            component,
            name,
            not_load_log_msg="Using default initialized state.",
        ):
            key = name.lower().replace(' ', '_')
            log_fmt = "{component} state {found} in the checkpoint and {loaded}"
            if key in checkpoint_states_to_load:
                if key in state_dict:
                    component.load_state_dict(state_dict[key])
                    log = log_fmt.format(
                        component=name, found="found", loaded="loaded"
                    )
                else:
                    log = log_fmt.format(
                        component=name,
                        found="not found",
                        loaded=f"not loaded. {not_load_log_msg}",
                    )
            else:
                log = f"Opting out of loading {name} state. {not_load_log_msg}"

            logging.info(log)

        maybe_load_state(
            optimizer,
            "Optimizer",
            not_load_log_msg="Using default preinitialized state.",
        )
        maybe_load_state(
            sparsity,
            "Sparsity",
            not_load_log_msg="Using default preinitialized state.",
        )
        maybe_load_state(
            dataloader,
            "DataLoader",
            not_load_log_msg="DataLoaders will yield samples from the beginning.",
        )
        if lr_scheduler:
            maybe_load_state(lr_scheduler, "LR scheduler")

        if grad_scaler:
            maybe_load_state(grad_scaler, "Grad Scaler")

    global_step = 0

    # Load checkpoint if provided and not compiling or validating
    if (
        not compile_only
        and not validate_only
        and (checkpoint_path := get_model_checkpoint(runconfig))
    ):
        load_checkpoint(checkpoint_path)

    if runconfig.get("save_initial_checkpoint", False) and not compile_only:
        save_checkpoint(global_step)

    summary_dir = (
        runconfig["summary_dir"]
        if ("summary_dir" in runconfig and runconfig["summary_dir"] is not None)
        else os.path.join(model_dir, "train")
    )

    from cerebras.pytorch.utils import tensorboard

    writer = tensorboard.SummaryWriter(
        log_dir=summary_dir,
        # The post_training_step summaries are written after global_step is
        # incremented, so +1.
        base_step=global_step + 1,
    )

    # parameters for numeric debugging
    numeric_debug = lambda f: f
    numeric_debug.flush = lambda: None
    if runconfig.get("dump_activations", False):
        if not backend.is_csx:
            numeric_debug = DumpContext(
                outdir=os.path.join(model_dir, "act_dumps"),
                model=model,
            )
        else:
            warn(
                "Got `dump_activations=True` but activation dumping is not "
                "supported on CSX. Disabling activation dumping for this run."
            )

    accum_loss = 0

    @cstorch.trace
    @numeric_debug
    def training_step(batch, step):
        nonlocal accum_loss

        if log_input_summaries:
            log_input_summary(batch)

        if mixed_precision and target_device != DeviceType.CSX:
            ctx = cstorch.amp.autocast()
        else:
            ctx = nullcontext()

        with ctx:
            loss = compiled_model(batch)

        if log_summaries:
            with cstorch.name_scope("params_norm"):
                compute_params_norm(compiled_model)

        if grad_accum_steps != 1:
            # Normalize loss to account for gradient accumulation
            with cstorch.amp.autocast():
                loss = loss / grad_accum_steps

            if step % grad_accum_steps != 0:
                # if not on a grad accumulation step, call backwards and return
                # the loss
                if grad_scaler:
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_loss += loss
                return loss, None

        if not grad_scaler:
            loss.backward()
            if log_summaries:
                with cstorch.name_scope("grad_norm"):
                    compute_grad_norm(compiled_model)

            optimizer.step()
            optimizer.zero_grad()
        elif use_cstorch_optimizer_step:
            cstorch.amp.optimizer_step(
                loss,
                optimizer,
                grad_scaler,
                max_gradient_norm=grad_scaler_params.max_gradient_norm,
                max_gradient_value=grad_scaler_params.max_gradient_value,
            )
        else:
            optimizer_step_with_summaries(
                loss,
                optimizer,
                grad_scaler,
                max_gradient_norm=grad_scaler_params.max_gradient_norm,
                max_gradient_value=grad_scaler_params.max_gradient_value,
                log_summaries=log_summaries,
                model=compiled_model,
            )

        if lr_scheduler:
            with cstorch.name_scope("lr"):
                log_learning_rate_pre_step(lr_scheduler.get_last_lr())
                lr_scheduler.step()

        if grad_accum_steps != 1:
            loss = accum_loss + loss
            accum_loss = 0

        # Extract the loss scale value from the grad scaler
        loss_scale = None
        if grad_scaler and log_summaries:
            loss_scale = grad_scaler.get_scale()

        # return final values
        return loss, loss_scale

    def is_log_step():
        return executor.on_final_iteration or (
            log_steps and executor.user_iteration % log_steps == 0
        )

    @cstorch.step_closure
    def log_learning_rate_pre_step(last_lr):
        if is_log_step():
            for group, lr in enumerate(last_lr):
                writer.add_scalar(f"lr/{group}", lr, global_step)

    @cstorch.step_closure
    def post_training_step(loss, loss_scale):
        # extract the loss scalar
        if is_log_step():
            rate = executor.profiler.rate_tracker.rate
            global_rate = executor.profiler.rate_tracker.global_rate

            # Print some logs to provide an update to the client
            logging.info(
                f"| Train Device={backend.device}, "
                f"Step={global_step}, "
                f"Loss={loss.item():.5f}, "
                f"Rate={format_rate(rate)} samples/sec, "
                f"GlobalRate={format_rate(global_rate)} samples/sec"
            )

            # record rates in tensorboard for future reference
            writer.add_scalar("local_samples_per_sec", rate, global_step)
            writer.add_scalar("avg_samples_per_sec", global_rate, global_step)
            if dataloader.batch_size:
                writer.add_scalar(
                    "avg_steps_per_sec",
                    global_rate / dataloader.batch_size,
                    global_step,
                )

            # Save the loss value to be able to plot the loss curve
            writer.add_scalar("loss", loss.item(), global_step)

            if loss_scale is not None:
                if isinstance(loss_scale, torch.Tensor):
                    loss_scale = loss_scale.item()

                # Save the loss_scale value to be able to inspect it for
                # debugging purposes
                writer.add_scalar("loss_scale", loss_scale, global_step)

        if check_loss_values:
            msg_postfix = (
                "This could potentially be due to selected hyperparameters "
                "such as the learning rate, batch size, etc. or it could due "
                "an internal error. Please try with different set of "
                "hyperparameters and contact Cerebras Support if the issue "
                "persists."
            )
            if torch.isnan(loss).any().item():
                raise ApplianceNanError(f"NaN loss detected. {msg_postfix}")
            if torch.isinf(loss).any().item():
                raise ApplianceNanError(f"inf loss detected. {msg_postfix}")

    if compile_only or validate_only:
        num_steps = None
    else:
        num_steps = cstorch.utils.data.compute_num_steps(
            dataloader,
            initial_step=global_step,
            num_steps=runconfig.get("num_steps"),
            max_steps=runconfig.get("max_steps"),
            num_epochs=runconfig.get("num_epochs"),
            steps_per_epoch=runconfig.get("steps_per_epoch"),
            grad_accum_steps=grad_accum_steps,
        )

    micro_batch_size = None
    if backend.is_csx:
        micro_batch_size = input_params.get("micro_batch_size", "auto")
        if "batch_size" in input_params:
            # Checks for invalid setting of num_csx, micro_batch_size and batch_size
            validate_streaming_and_micro_batch_size(
                input_params["batch_size"],
                micro_batch_size,
                cluster_config.num_csx,
            )

    op_profiler_config = runconfig.get("op_profiler_config", {})
    schedule_obj = (
        cstorch.profiler.schedule(
            start_step=op_profiler_config.get("start_step", -1),
            end_step=op_profiler_config.get("end_step", -1),
        )
        if op_profiler_config
        else cstorch.profiler.schedule(start_step=-1, end_step=-1)
    )
    host_activities = (
        op_profiler_config.get("host_activities", None)
        if op_profiler_config
        else None
    )
    with cstorch.profiler.profile(
        schedule=schedule_obj,
        on_trace_ready=(
            cstorch.profiler.tensorboard_trace_handler(summary_dir)
            if (
                op_profiler_config and op_profiler_config.get("tb_trace", False)
            )
            else None
        ),
        host_activities=host_activities,
    ) as prof:
        executor = cstorch.utils.data.DataExecutor(
            dataloader,
            num_steps=num_steps,
            checkpoint_steps=checkpoint_steps,
            activation_steps=activation_steps,
            writer=writer,
            listeners=listeners,
            micro_batch_size=micro_batch_size,
        )

        try:
            for step, batch in enumerate(executor, start=1):
                loss, loss_scale = training_step(batch, step)

                if step % grad_accum_steps == 0:
                    global_step += 1

                    # Wait for outputs to become available to fetch from the CS system(s)
                    post_training_step(loss, loss_scale)

                # only saves checkpoint if current step is a checkpoint step
                save_checkpoint(global_step)

            # pylint: disable=undefined-loop-variable
            assert step >= grad_accum_steps, (
                f"There were only {step} batches in epoch, which is "
                f"less than the grad accumulation steps {grad_accum_steps}. "
                f"This prevents model training as no optimizer step is taken."
            )
            if step % grad_accum_steps != 0:
                warnings.warn(
                    "There were leftover gradients in the accumulation step. "
                    "They will effectively vanish, which could potentially lead "
                    "to different convergence behaviour."
                )

            if not (compile_only or validate_only):
                logging.info("Training completed successfully!")
        finally:
            numeric_debug.flush()
            if not (compile_only or validate_only) and executor.profiler:
                # compute the total samples processed based on the number of steps
                # and the number of Cerebras systems in the cluster
                total_samples = int(
                    executor.profiler.rate_tracker.total_samples
                )
                total_time = executor.profiler.rate_tracker.total_time

                logging.info(
                    f"Processed {total_samples} sample(s) "
                    f"in {total_time} seconds."
                )
    prof.export_chrome_trace(f"{artifact_dir}/chrome_trace.json")
    if prof.appliance_response:
        logging.info(f"{prof.get_summary()}")


def run_cstorch_eval(
    params, params_obj, model_fn, input_fn, cluster_config, artifact_dir
):
    """
    Runs the evaluation workflow built using the cstorch API.

    Args:
        params: the params dictionary extracted from the params.yaml used
        model_fn: A callable that takes in the params dictionary and returns
            a torch.nn.Module
        input_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader
    """
    import cerebras.pytorch.metrics as metrics
    from cerebras.appliance.errors import ApplianceNanError

    input_params = params.get("eval_input", {})
    runconfig = params["runconfig"]

    target_device = runconfig["target_device"]

    mode = runconfig["mode"]
    model_dir = runconfig["model_dir"]
    compile_dir = runconfig.get("compile_dir")
    log_steps = runconfig.get("log_steps")
    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)
    check_loss_values = runconfig.get("check_loss_values", True)
    log_input_summaries = runconfig.get("log_input_summaries", False)
    experimental_configs = runconfig.get("experimental", {})
    listener_configs = experimental_configs.get("listeners", [])
    if runconfig.get("enable_act_frequency", False):
        logging.warning("Activation frequency is not supported for eval mode.")

    mixed_precision = params["model"].get("mixed_precision", False)
    mp_dtype = set_half_dtype_from_params(params["model"])

    if (
        params["model"].get("fp16_type") == "cbfloat16"
        and target_device != DeviceType.CSX
    ):
        mp_dtype = cstorch.amp.set_half_dtype(torch.bfloat16)
        warnings.warn(
            f"cbfloat16 is only supported on CSX. Setting half dtype to bfloat16."
        )

    if (
        mixed_precision
        and mp_dtype != torch.bfloat16
        and target_device == DeviceType.CPU
    ):
        warnings.warn(
            "Mixed precision on CPU is only supported with bfloat16. "
            "Setting half dtype to bfloat16."
        )
        mp_dtype = cstorch.amp.set_half_dtype(torch.bfloat16)

    if target_device == DeviceType.CSX:
        backend = cstorch.backend(
            "CSX",
            artifact_dir=artifact_dir,
            compile_dir=compile_dir,
            compile_only=compile_only,
            validate_only=validate_only,
            retrace_every_iteration=runconfig.get(
                "retrace_every_iteration", False
            ),
            cluster_config=cluster_config,
        )

        backend.device.config.lazy_initialization = runconfig.get(
            "lazy_initialization", True
        )
        if not backend.device.config.lazy_initialization:
            # Drop data only has any effect if we're not tracing the
            # initialization
            backend.device.config.drop_data = runconfig.get(
                "drop_data", compile_only or validate_only
            )

    elif target_device == DeviceType.CPU:
        backend = cstorch.backend("CPU", artifact_dir=artifact_dir)
    elif target_device == DeviceType.GPU:
        backend = cstorch.backend(
            "GPU",
            artifact_dir=artifact_dir,
            enable_distributed=runconfig.get("enable_distributed", False),
            main_process_id=runconfig.get("main_process_id", 0),
            dist_backend=runconfig.get("dist_backend", "nccl"),
            init_method=runconfig.get("init_method", "env://"),
            sync_batchnorm=runconfig.get("sync_batchnorm", False),
        )

    # Pop lora_params before model_fn is run to prevent it from showing up
    # in the unused params warning
    lora_params = None
    if "lora_params" in params["model"]:
        lora_params = params["model"].pop("lora_params")

    with backend.device:
        model = model_fn(params_obj if params_obj else params)

        if lora_params:
            # After model init, we need to convert it into a LoRA model
            from cerebras.modelzoo.common.utils.model.lora import (
                disable_lora_merge_weights,
                make_model_lora,
            )

            disable_lora_merge_weights(lora_params)
            model = make_model_lora(model, lora_params)

    compiled_model = cstorch.compile(model, backend)
    compiled_model.eval()

    sparsity_config = params.pop("sparsity", {})
    if sparsity_config:
        sparsity = cstorch.sparse.configure(sparsity_config)

        # Sparsify model parameters
        model.apply(sparsity)

    # apply weight compression
    compression_config = params["model"].pop("compression", {})
    if compression_config:
        compressions = configure_compression(compression_config)
        for compression in compressions:
            model.apply(compression)

    # Register tensor listeners.
    listeners = [
        cstorch.experimental.listener.create_listener(**listener_params)
        for listener_params in listener_configs
    ]

    def load_checkpoint(checkpoint_path):
        nonlocal global_step

        state_dict = load_from_checkpoint_file(checkpoint_path)

        if "model" not in state_dict:
            warn(
                f"Checkpoint does not contain a model state dict. "
                f"Model state was not loaded"
            )

        # This check is required for backward compatibility with checkpoints
        # saved with older versions of ModelZoo (pre rel-2.0.0)
        # We check that the model state dict keys start with "model."
        # and if they don't, we load the state dict into the model's model
        elif hasattr(model, "model") and not any(
            k.startswith("model.") for k in state_dict["model"].keys()
        ):
            model.model.load_state_dict(state_dict["model"])

        # This should be the case that is used for all checkpoints saved
        # post rel-2.0.0
        else:
            if sparsity_config and not any(
                k.endswith("_mask")
                and k[: -len("_mask")] in state_dict["model"]
                for k in state_dict["model"].keys()
            ):
                raise RuntimeError(
                    "Did not find any sparsity masks in the model checkpoint. "
                )

            model.load_state_dict(state_dict["model"])

        global_step = state_dict.get("global_step", 0)

    global_step = 0

    # Load checkpoint if provided and not compiling or validating
    checkpoint_paths = []
    if not (compile_only or validate_only):
        if mode == modes.EVAL_ALL:
            checkpoint_paths = get_all_checkpoints(model_dir)
            if not checkpoint_paths:
                raise ValueError(
                    f"No checkpoints were found for evaluation. Please ensure that the "
                    f"model directory \"{model_dir}\" contains at least one valid checkpoint."
                )
        elif checkpoint_path := get_model_checkpoint(runconfig):
            checkpoint_paths = [checkpoint_path]

    checkpoint_paths = checkpoint_paths or [None]

    summary_dir = (
        runconfig["summary_dir"]
        if ("summary_dir" in runconfig and runconfig["summary_dir"] is not None)
        else os.path.join(model_dir, "eval")
    )

    from cerebras.pytorch.utils import tensorboard

    writer = tensorboard.SummaryWriter(log_dir=summary_dir)

    # parameters for numeric debugging
    numeric_debug = lambda f: f
    numeric_debug.flush = lambda: None
    if runconfig.get("dump_activations", False):
        if not backend.is_csx:
            numeric_debug = DumpContext(
                outdir=os.path.join(model_dir, "act_dumps"),
                model=model,
            )
        else:
            warn(
                "Got `dump_activations=True` but activation dumping is not "
                "supported on CSX. Disabling activation dumping for this run."
            )

    @cstorch.trace
    @numeric_debug
    @torch.no_grad()
    def eval_step(batch):
        if log_input_summaries:
            log_input_summary(batch)

        if mixed_precision and target_device != DeviceType.CSX:
            ctx = cstorch.amp.autocast()
        else:
            ctx = nullcontext()

        with ctx:
            loss = compiled_model(batch)

        return loss

    total_loss = 0
    total_steps = 0

    @cstorch.step_closure
    def post_eval_step(loss):
        nonlocal total_loss
        nonlocal total_steps

        is_log_step = executor.on_final_iteration or (
            log_steps and executor.user_iteration % log_steps == 0
        )

        rate = executor.profiler.rate_tracker.rate
        global_rate = executor.profiler.rate_tracker.global_rate

        if is_log_step:
            # Print some logs to provide an update to the client
            logging.info(
                f"| Eval Device={backend.device}, "
                f"Step={executor.user_iteration}, "
                f"Loss={loss.item():.5f}, "
                f"Rate={format_rate(rate)} samples/sec, "
                f"GlobalRate={format_rate(global_rate)} samples/sec"
            )

        if executor.on_final_iteration:
            # log the throughput of the eval run to tensorboard on the last step
            writer.add_scalar("local_samples_per_sec", rate, global_step)
            writer.add_scalar("avg_samples_per_sec", global_rate, global_step)
            if dataloader.batch_size:
                writer.add_scalar(
                    "avg_steps_per_sec",
                    global_rate / dataloader.batch_size,
                    global_step,
                )

        if check_loss_values:
            if torch.isnan(loss).any().item():
                raise ApplianceNanError("NaN loss detected.")
            if torch.isinf(loss).any().item():
                raise ApplianceNanError("inf loss detected.")

        total_loss += loss.item()
        total_steps += 1

    dataloader = cstorch.utils.data.DataLoader(input_fn, params)

    if compile_only or validate_only:
        num_steps = None
    else:
        eval_steps = runconfig.get("eval_steps")
        num_epochs = 1 if eval_steps is None else None
        num_steps = cstorch.utils.data.compute_num_steps(
            dataloader, num_steps=eval_steps, num_epochs=num_epochs
        )

    micro_batch_size = None
    if backend.is_csx:
        micro_batch_size = input_params.get("micro_batch_size", "auto")
        if "batch_size" in input_params:
            # Checks for invalid setting of num_csx, micro_batch_size and batch_size
            validate_streaming_and_micro_batch_size(
                input_params["batch_size"],
                micro_batch_size,
                cluster_config.num_csx,
            )

    for eval_id, checkpoint_path in enumerate(checkpoint_paths):
        global_step = 0
        total_steps = 0
        total_loss = 0
        try:
            if eval_id > 0:
                dataloader = cstorch.utils.data.DataLoader(input_fn, params)

            executor = cstorch.utils.data.DataExecutor(
                dataloader,
                num_steps=num_steps,
                writer=writer,
                listeners=listeners,
                micro_batch_size=micro_batch_size,
            )

            if checkpoint_path:
                load_checkpoint(checkpoint_path)

            for batch in executor:
                loss = eval_step(batch)
                post_eval_step(loss)

            if not (compile_only or validate_only):
                for name, metric in metrics.get_all_metrics():
                    value = float(metric)
                    writer.add_scalar(name, value, global_step)
                    logging.info(f"Metric: {name} = {value}")

                avg_eval_loss = total_loss / total_steps
                writer.add_scalar("loss", avg_eval_loss, global_step)
                logging.info(f"Avg Eval Loss: {avg_eval_loss}")

                logging.info("Evaluation completed successfully!")

        finally:
            numeric_debug.flush()
            if not (compile_only or validate_only) and executor.profiler:
                # compute the total samples processed based on the number of steps
                # and the number of Cerebras systems in the cluster
                total_samples = int(
                    executor.profiler.rate_tracker.total_samples
                )
                total_time = executor.profiler.rate_tracker.total_time

                logging.info(
                    f"Processed {total_samples} sample(s) "
                    f"in {total_time} seconds."
                )


def get_model_checkpoint(runconfig: Dict[str, Any]) -> Union[str, None]:
    """Get the path to the model checkpoint, if any."""
    model_dir = runconfig["model_dir"]
    ckpt_path = None

    # if a checkpoint path is provided, use that
    if runconfig.get("checkpoint_path"):
        ckpt_path = runconfig["checkpoint_path"]
    elif runconfig.get("autoload_last_checkpoint", True):
        logging.info(
            f"Checkpoint autoloading is enabled. Looking for latest checkpoint "
            f"in \"{model_dir}\" directory with the following naming "
            f"convention: `checkpoint_(step)(_timestamp)?.mdl`."
        )
        ckpt_path = get_latest_checkpoint(model_dir)
        if ckpt_path:
            logging.info(f"Found latest checkpoint at \"{ckpt_path}\".")
        else:
            logging.info(f"No checkpoints were found in \"{model_dir}\".")

    if not ckpt_path:
        logging.info(
            f"No checkpoint was provided. Using randomly initialized model "
            f"parameters."
        )

    return ckpt_path


def log_input_summary(batch):
    """Log the input tensors to tensorboard."""
    for scope, tensor in visit_torch_tensors(batch, scope=["input"]):
        summarize_tensor(".".join(map(str, scope)), tensor)


def compute_params_norm(model):
    """Compute the model wise norm of the parameters."""
    param_norm = torch.tensor(0.0).to(model.device)
    for _, param in model.named_parameters():
        if param.requires_grad:
            # simply add if we want to include all params
            param_norm += torch.pow(torch.norm(param), 2.0)
    summarize_scalar("model_wise_params_norm", torch.sqrt(param_norm))


def compute_grad_norm(model):
    """Compute the model wise and per layer norm of the gradients."""
    params_grad_norm = torch.tensor(0.0).to(model.device)
    for _, param in model.named_parameters():
        if param.grad is not None:
            params_grad_norm += torch.pow(torch.norm(param.grad), 2.0)
    params_grad_norm = torch.sqrt(params_grad_norm)

    summarize_scalar("model_wise_grad_norm", params_grad_norm)

    per_layer_grad_norm = {}
    layer_pattern = re.compile(r".*(layers\.)(\d+)(\.).*")
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        # get a match if module name contains `layers.i.0` where i is layer num
        match = layer_pattern.match(name)
        if match:
            layer_id = match.group(2)
            if layer_id not in per_layer_grad_norm:
                per_layer_grad_norm[layer_id] = torch.tensor(0.0).to(
                    model.device
                )
            per_layer_grad_norm[layer_id] += torch.pow(
                torch.norm(param.grad), 2.0
            )

    for layer_id in per_layer_grad_norm:
        summarize_scalar(
            f"per_layer_grad_norm/layer_{layer_id}",
            torch.sqrt(per_layer_grad_norm[layer_id]),
        )


def optimizer_step_with_summaries(
    loss: torch.Tensor,
    optimizer: "cstorch.optim.Optimizer",
    grad_scaler: "cstorch.amp.GradScaler",
    max_gradient_norm: float = None,
    max_gradient_value: float = None,
    log_summaries: bool = False,
    model: torch.nn.Module = None,
):
    """
    Customized equivalent to cstorch.amp.optimizer_step
    additionally featuring grad norm summaries.
    """
    grad_scaler.scale(loss).backward()

    # Unscales the gradients of optimizer's assigned params in-place
    grad_scaler.unscale_(optimizer)

    if log_summaries:
        assert model is not None
        compute_grad_norm(model)

    # gradient clipping
    if max_gradient_norm is not None and max_gradient_norm <= 0.0:
        raise ValueError(
            f"max_gradient_norm has to be a positive float. Got "
            f"{max_gradient_norm}"
        )
    if max_gradient_value is not None and max_gradient_value <= 0.0:
        raise ValueError(
            f"max_gradient_value has to be a positive float. Got "
            f"{max_gradient_value}"
        )
    if max_gradient_norm is not None and max_gradient_value is not None:
        raise ValueError(
            f"Gradients can be clipped by norm(={max_gradient_norm}) or by "
            f"value(={max_gradient_value}), but not both. "
            f"Do not set both `max_gradient_norm` and `max_gradient_value`."
        )

    params = (
        p
        for param_group in optimizer.param_groups
        for p in param_group["params"]
    )
    if max_gradient_norm is not None:
        torch.nn.utils.clip_grad_norm_(list(params), max_gradient_norm)
    elif max_gradient_value is not None:
        torch.nn.utils.clip_grad_value_(list(params), max_gradient_value)

    grad_scaler.step(optimizer)
    grad_scaler.update()

    optimizer.zero_grad()


@dataclasses.dataclass
class GradScalerParams:
    """Dataclass for parsing grad scaler params from optimizer params."""

    loss_scaling_factor: Union[float, str, None] = 1.0
    initial_loss_scale: Optional[float] = None  # default set in GradScaler
    steps_per_increase: Optional[int] = 2000
    min_loss_scale: Optional[float] = None  # default set in GradScaler
    max_loss_scale: Optional[float] = None  # default set in GradScaler
    max_gradient_norm: Optional[float] = None
    max_gradient_value: Optional[float] = None

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> Self:
        """Returns an instance of GradScalerParams from a dictionary.

        Note that matching keys are popped from the dictionary.
        """
        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name in params:
                kwargs[field.name] = params.pop(field.name)
        return cls(**kwargs)

    def __post_init__(self):
        """validate loss_scaling_factor."""
        if not (
            isinstance(self.loss_scaling_factor, numbers.Number)
            or self.loss_scaling_factor is None
            or self.loss_scaling_factor == "dynamic"
        ):
            raise ValueError(
                f"Optimizer param 'loss_scaling_factor' must be a float, None, or 'dynamic'."
                f"Got {self.loss_scaling_factor} instead."
            )
