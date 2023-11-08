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

"""Generic run scripts build using the cstorch API"""
import dataclasses
import logging
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from warnings import warn

import torch
from typing_extensions import Self

import cerebras_pytorch as cstorch
from cerebras_pytorch.utils.nest import visit_torch_tensors
from modelzoo.common.input.utils import update_debug_args_with_mem_limits
from modelzoo.common.pytorch.dump_context import DumpContext
from modelzoo.common.pytorch.half_dtype import half_dtype_instance
from modelzoo.common.pytorch.input_utils import (
    validate_streaming_and_micro_batch_size,
)
from modelzoo.common.pytorch.utils import is_mup_run, setup_logging
from modelzoo.common.run_utils.utils import DeviceType


def run_cstorch_flow(params, model_fn, train_data_fn, eval_data_fn):
    """
    Set up the cstorch run and call the appropriate helper based on the mode

    Args:
        params: the params dictionary extracted from the params.yaml used
        model_fn: A callable that takes in the params dictionary and returns
            a torch.nn.Module
        train_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader
        eval_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader
    """
    from cerebras_appliance.run_utils import get_debug_args

    runconfig = params["runconfig"]

    if "seed" in runconfig:
        # Ensure we set seed before any model initialization
        cstorch.manual_seed(runconfig["seed"])

    debug_args = get_debug_args(runconfig.get("debug_args_path"))
    update_debug_args_with_mem_limits(debug_args, runconfig)

    # Configure the Cerebras Wafer Scale cluster
    cs_config = cstorch.utils.CSConfig(
        num_csx=runconfig.get("num_csx"),
        max_wgt_servers=runconfig.get("num_wgt_servers"),
        mgmt_address=runconfig.get("mgmt_address"),
        mgmt_namespace=runconfig.get("mgmt_namespace"),
        credentials_path=runconfig.get("credentials_path"),
        debug_args=debug_args,
        mount_dirs=runconfig.get("mount_dirs"),
        python_paths=runconfig.get("python_paths"),
        transfer_processes=runconfig.get("transfer_processes"),
        num_workers_per_csx=runconfig.get("num_workers_per_csx"),
        job_labels=runconfig.get("job_labels"),
        max_act_per_csx=runconfig.get("num_act_servers"),
        job_time_sec=runconfig.get("job_time_sec"),
        disable_version_check=runconfig["disable_version_check"],
    )

    # Set up logging level
    setup_logging(
        runconfig.get("logging"),
        runconfig.get("streamer_logging"),
        logging_dir=runconfig.get("model_dir"),
    )

    # check if current run is configured with muP
    if is_mup_run(params):
        logging.info(f'This is a muP configured run')

    if runconfig["mode"] == "train":
        run_cstorch_train(params, model_fn, train_data_fn, cs_config)
    elif runconfig["mode"] == "eval":
        run_cstorch_eval(params, model_fn, eval_data_fn, cs_config)


def run_cstorch_train(params, model_fn, input_fn, cs_config):
    """
    Runs the training workflow built using the cstorch API

    Args:
        params: the params dictionary extracted from the params.yaml used
        model_fn: A callable that takes in the params dictionary and returns
            a torch.nn.Module
        input_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader
    """
    from cerebras_appliance.errors import ApplianceNanError

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
    num_csx = runconfig.get("num_csx")
    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)
    drop_data = runconfig.get("drop_data", False)
    log_summaries = params["optimizer"].pop("log_summaries", False)
    check_loss_values = runconfig.get("check_loss_values", True)
    log_input_summaries = runconfig.get("log_input_summaries", False)
    precision_opt_level = None
    model_pol = params["model"].get("precision_opt_level")
    if model_pol is not None:
        warnings.warn(
            "Passing `precision_opt_level` via `model` params is deprecated. "
            "Please use `params[\"runconfig\"][\"precision_opt_level\"]`"
        )
    precision_opt_level = runconfig.get("precision_opt_level", model_pol)
    if precision_opt_level != model_pol and model_pol is not None:
        logging.warning(
            f"Using `precision_opt_level:{precision_opt_level}` from `runconfig` "
            f"instead of `{model_pol}` from `model`"
        )
    if precision_opt_level is None:
        precision_opt_level = 1

    cs_config.precision_opt_level = precision_opt_level

    mixed_precision = params["model"].get("mixed_precision", False)
    use_bfloat16 = params["model"].get("use_bfloat16", False)

    if mixed_precision and not use_bfloat16 and target_device == DeviceType.CPU:
        warnings.warn(
            "Mixed precision on CPU is only supported with bfloat16. "
            "Setting use_bfloat16 in model config to True."
        )
        params["model"]["use_bfloat16"] = True
        use_bfloat16 = True

    half_dtype_instance.use_bfloat16 = use_bfloat16
    if use_bfloat16:
        cstorch.amp.use_bfloat16(True)

    optimizer_params = params["optimizer"]

    # Parse grad scaler params and pop them from optimizer params
    grad_scaler = None
    grad_scaler_params = GradScalerParams.from_dict(optimizer_params)
    if grad_scaler_params.loss_scaling_factor == "dynamic" and use_bfloat16:
        grad_scaler_params.loss_scaling_factor = 1.0
        logging.info(
            f"No need to use DLS for loss when `use_bfloat16` is set to"
            " `True`. Setting `loss_scaling_factor ` to `1.0`."
        )

    use_cstorch_optimizer_step = runconfig.get(
        "use_cstorch_optimizer_step", False
    )
    # Default to only keeping the 5 latest checkpoints.
    max_checkpoints = runconfig.get("max_checkpoints", 5)

    if target_device == DeviceType.CSX:
        use_cs_grad_accum = runconfig.get("use_cs_grad_accum", False)
        micro_batch_size = input_params.get("micro_batch_size")

        # Checks for invalid setting of num_csx, micro_batch_size and batch_size
        if use_cs_grad_accum and micro_batch_size is not None:
            batch_size = input_params.get("batch_size")
            validate_streaming_and_micro_batch_size(
                batch_size, micro_batch_size, num_csx
            )

        grad_accum_steps = 1
        if "grad_accum_steps" in optimizer_params:
            logging.info(
                "`grad_accum_steps` param has no effect when running on the CSX. "
                "Consider setting `use_cs_grad_accum` to enable or "
                "disable grad accumulation on CSX."
            )
            optimizer_params.pop("grad_accum_steps")
    else:
        grad_accum_steps = optimizer_params.pop("grad_accum_steps", 1)
        logging.info(f"Gradient accumulation steps is {grad_accum_steps}")

    if target_device == DeviceType.CSX:
        retrace_every_iteration = runconfig.get(
            "retrace_every_iteration", False
        )

        backend = cstorch.backend(
            "CSX",
            artifact_dir=os.path.join(model_dir, "cerebras_logs"),
            compile_dir=compile_dir,
            compile_only=compile_only,
            validate_only=validate_only,
            drop_data=drop_data,
            max_checkpoints=max_checkpoints,
            use_cs_grad_accum=use_cs_grad_accum,
            micro_batch_size=micro_batch_size,
            retrace_every_iteration=retrace_every_iteration,
        )
    elif target_device == DeviceType.CPU:
        backend = cstorch.backend(
            "CPU",
            max_checkpoints=max_checkpoints,
            mixed_precision=mixed_precision,
        )
    elif target_device == DeviceType.GPU:
        backend = cstorch.backend(
            "GPU",
            enable_distributed=runconfig.get("enable_distributed", False),
            main_process_id=runconfig.get("main_process_id", 0),
            dist_backend=runconfig.get("dist_backend", "nccl"),
            init_method=runconfig.get("init_method", "env://"),
            sync_batchnorm=runconfig.get("sync_batchnorm", False),
        )

    with backend.device:
        model = model_fn(params)

    compiled_model = cstorch.compile(model, backend)
    compiled_model.train()

    # group optimizer params
    param_optimizer_grouped = cstorch.optim.configure_param_groups(
        model, optimizer_params,
    )

    optimizer = cstorch.optim.configure_optimizer(
        optimizer_type=optimizer_params.pop("optimizer_type"),
        params=param_optimizer_grouped,
        **{
            k: v
            for k, v in optimizer_params.items()
            if k != "adjust_learning_rate"
        },
    )

    sparsity_optimizer = None
    sparsity_config = params.pop("sparsity", {})
    if sparsity_config:
        sparsity_type = sparsity_config.pop("type", None)
        if sparsity_type in (None, "sideband"):
            raise ValueError(
                "In this version of cerebras_pytorch, multiple types of "
                "sparsity (static and dynamic) are supported and thus we "
                "require configuring which type of sparsity to apply. Please "
                "update your configuration to indicate `type: \"static\"` "
                "if updating from a previous software release."
            )

        # Construct a SparsityOptimizer from the yaml config block. This will
        # also select which parameters are configured for sparsity (and
        # possibly group parameters with differing configurations).
        sparsity_optimizer = cstorch.sparse.configure_sparsity_optimizer(
            sparsity_type, model.named_parameters(), **sparsity_config
        )

        # Install the "apply_sparsity" hooks so that the model's .forward()
        # always computes using the sparse view of the parameters and the
        # gradients are sparsifed before any use.
        sparsity_optimizer.hook_module(model)

        # Replace the optimizer with a wrapper so that sparsity state is saved,
        # sparsity masks are updated by .step(), and conditional execution of
        # dynamic gradient scaling handles skipping sparsity update too.
        optimizer = cstorch.sparse.SparsityWrapperOptimizer(
            optimizer, sparsity_optimizer
        )

        # cstorch.sparse.configure_sparsity_wrapper is a helper method to
        # perform these 3 steps in one function call.

    lr_scheduler = cstorch.optim.configure_lr_scheduler(
        optimizer,
        optimizer_params.get("learning_rate"),
        optimizer_params.get("adjust_learning_rate"),
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
                max_gradient_norm=grad_scaler_params.max_gradient_norm,
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
                    growth_interval=2 ** 63 - 1,
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
        if lr_scheduler:
            state_dict["lr_scheduler"] = lr_scheduler.state_dict()
        if grad_scaler:
            state_dict["grad_scaler"] = grad_scaler.state_dict()

        if dataloader.is_restartable:
            logging.debug("Saving dataloader state")
            dl_state = dataloader.state_dict()
            state_dict["dataloader"] = dl_state
            logging.debug(f"Saved dataloader state: {dl_state}")

        state_dict["global_step"] = step

        cstorch.save(state_dict, checkpoint_file)

        logging.info(f"Saved checkpoint {checkpoint_file}")

    def load_checkpoint(checkpoint_path):
        nonlocal global_step

        logging.info(f"Loading weights from checkpoint {checkpoint_path}")
        state_dict = cstorch.load(checkpoint_path)

        disable_strict_checkpoint_loading = runconfig.get(
            "disable_strict_checkpoint_loading", False
        )
        checkpoint_states_to_load = {
            "model",
            "optimizer",
            "dataloader",
            "grad_scaler",
            "lr_scheduler",
            "global_step",
        }
        if runconfig.get("load_checkpoint_states"):
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
            dataloader,
            "DataLoader",
            not_load_log_msg="DataLoaders will yield samples from the beginning.",
        )
        if lr_scheduler:
            maybe_load_state(lr_scheduler, "LR scheduler")
        if grad_scaler:
            maybe_load_state(grad_scaler, "Grad Scaler")

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

    global_step = 0

    # Load checkpoint if provided and not compiling or validating
    if (
        not compile_only
        and not validate_only
        and (checkpoint_path := get_model_checkpoint(runconfig))
    ):
        load_checkpoint(checkpoint_path)

    # Initialize sparsity after maybe loading checkpoint.
    if sparsity_optimizer:
        sparsity_optimizer.initialize_sparsity()

    if runconfig.get("save_initial_checkpoint", False) and not compile_only:
        save_checkpoint(global_step)

    writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=runconfig.get("summary_dir", os.path.join(model_dir, "train")),
        # The post_training_step summaries are written after global_step is
        # incremented, so +1.
        base_step=global_step + 1,
    )

    # parameters for numeric debugging
    numeric_debug = lambda f: f
    if runconfig.get("dump_activations", False):
        if not backend.is_csx:
            numeric_debug = DumpContext(
                outdir=os.path.join(model_dir, "act_dumps"), model=model,
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

        loss = compiled_model(batch)

        if log_summaries:
            compute_params_norm(compiled_model)

        if grad_accum_steps != 1:
            # Normalize loss to account for gradient accumulation
            with cstorch.amp.autocast():
                loss /= grad_accum_steps

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
            lr_scheduler.step()

        if grad_accum_steps != 1:
            loss = accum_loss + loss
            accum_loss = 0

        # Extract the loss scale value from the grad scaler
        loss_scale = None

        # return final values
        return loss, loss_scale

    # Counts the total number of steps actually executed
    total_steps = 0

    @cstorch.step_closure
    def post_training_step(loss, loss_scale):
        nonlocal total_steps

        is_log_step = executor.on_final_iteration or (
            log_steps and global_step % log_steps == 0
        )

        # extract the loss scalar
        if is_log_step:
            rate = executor.profiler.rate()
            global_rate = executor.profiler.global_rate()

            # Print some logs to provide an update to the client
            logging.info(
                f"| Train Device={backend.device}, "
                f"Step={global_step}, "
                f"Loss={loss.item():.5f}, "
                f"Rate={rate:.2f} samples/sec, "
                f"GlobalRate={global_rate:.2f} samples/sec"
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

        if is_log_step and lr_scheduler:
            for group, lr in enumerate(lr_scheduler.get_last_lr()):
                writer.add_scalar(f"lr.{group}", lr, global_step)

        total_steps += 1

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

    executor = cstorch.utils.data.DataExecutor(
        dataloader,
        num_steps=num_steps,
        checkpoint_steps=checkpoint_steps,
        cs_config=cs_config,
        writer=writer,
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
        if not (compile_only or validate_only) and executor.profiler:
            # compute the total samples processed based on the number of steps
            # and the number of Cerebras systems in the cluster
            total_samples = int(executor.profiler.total_samples())
            total_time = executor.profiler.total_time()

            logging.info(
                f"Processed {total_samples} sample(s) "
                f"in {total_time} seconds."
            )


def run_cstorch_eval(params, model_fn, input_fn, cs_config):
    """
    Runs the evaluatiion workflow built using the cstorch API

    Args:
        params: the params dictionary extracted from the params.yaml used
        model_fn: A callable that takes in the params dictionary and returns
            a torch.nn.Module
        input_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader
    """
    import cerebras_pytorch.metrics as metrics
    from cerebras_appliance.errors import ApplianceNanError

    input_params = params.get("eval_input", {})
    runconfig = params["runconfig"]

    target_device = runconfig["target_device"]

    model_dir = runconfig["model_dir"]
    compile_dir = runconfig.get("compile_dir")
    log_steps = runconfig.get("log_steps")
    num_csx = runconfig.get("num_csx")
    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)
    drop_data = runconfig.get("drop_data", False)
    check_loss_values = runconfig.get("check_loss_values", True)
    log_input_summaries = runconfig.get("log_input_summaries", False)

    precision_opt_level = None
    model_pol = params["model"].get("precision_opt_level")
    if model_pol is not None:
        warnings.warn(
            "Passing `precision_opt_level` via `model` params is deprecated. "
            "Please use `params[\"runconfig\"][\"precision_opt_level\"]`"
        )
    precision_opt_level = runconfig.get("precision_opt_level", model_pol)
    if precision_opt_level != model_pol and model_pol is not None:
        logging.warning(
            f"Using `precision_opt_level:{precision_opt_level}` from `runconfig` "
            f"instead of `{model_pol}` from `model`"
        )
    if precision_opt_level is None:
        precision_opt_level = 1

    cs_config.precision_opt_level = precision_opt_level

    mixed_precision = params["model"].get("mixed_precision", False)
    use_bfloat16 = params["model"].get("use_bfloat16", False)

    if mixed_precision and not use_bfloat16 and target_device == DeviceType.CPU:
        warnings.warn(
            "Mixed precision on CPU is only supported with bfloat16. "
            "Setting use_bfloat16 in model config to True."
        )
        params["model"]["use_bfloat16"] = True
        use_bfloat16 = True

    half_dtype_instance.use_bfloat16 = use_bfloat16
    if use_bfloat16:
        cstorch.amp.use_bfloat16(True)

    if target_device == DeviceType.CSX:
        use_cs_grad_accum = runconfig.get("use_cs_grad_accum", False)
        micro_batch_size = input_params.get("micro_batch_size")

        # Checks for invalid setting of num_csx, micro_batch_size and batch_size
        if use_cs_grad_accum and micro_batch_size is not None:
            batch_size = input_params.get("batch_size")
            validate_streaming_and_micro_batch_size(
                batch_size, micro_batch_size, num_csx
            )

        backend = cstorch.backend(
            "CSX",
            artifact_dir=os.path.join(model_dir, "cerebras_logs"),
            compile_dir=compile_dir,
            compile_only=compile_only,
            validate_only=validate_only,
            drop_data=drop_data,
            use_cs_grad_accum=use_cs_grad_accum,
            micro_batch_size=micro_batch_size,
        )
    elif target_device == DeviceType.CPU:
        backend = cstorch.backend("CPU", mixed_precision=mixed_precision)
    elif target_device == DeviceType.GPU:
        backend = cstorch.backend(
            "GPU",
            enable_distributed=runconfig.get("enable_distributed", False),
            main_process_id=runconfig.get("main_process_id", 0),
            dist_backend=runconfig.get("dist_backend", "nccl"),
            init_method=runconfig.get("init_method", "env://"),
            sync_batchnorm=runconfig.get("sync_batchnorm", False),
        )

    with backend.device:
        model = model_fn(params)

    compiled_model = cstorch.compile(model, backend)
    compiled_model.eval()

    def configure_eval_sparsity(sparsity_config, state_dict):
        # The sparsity_optimizer.hook_module() is needed to inform the compile
        # for the Cerebras Stack that weights will be sparse. During eval, we
        # won't call .step() on the sparsity optimizer to update the sparsity
        # patterns, but it should still be constructed and loaded from the
        # state_dict for performance reasons on CSX. The weights should already
        # have zeros in pruned positions, so constructing one and installing
        # the module hook isn't necessary on non-CSX devices.

        # Construct a SparsityOptimizer from the yaml config block. This will
        # also select which parameters are configured for sparsity.
        sparsity_type = sparsity_config.pop("type")
        sparsity_optimizer = cstorch.sparse.configure_sparsity_optimizer(
            sparsity_type, model.named_parameters(), **sparsity_config
        )

        # Load the sparsity patterns.
        sparsity_optimizer.load_state_dict(state_dict)

        # Install the "apply_sparsity" hooks so that the model's .forward()
        # always computes using the sparse view of the parameters.
        sparsity_optimizer.hook_module(model)

    def load_checkpoint(checkpoint_path):
        nonlocal global_step

        logging.info(f"Loading weights from checkpoint {checkpoint_path}")
        state_dict = cstorch.load(checkpoint_path)

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
            model.load_state_dict(state_dict["model"])

        global_step = state_dict.get("global_step", 0)

        sparsity_config = params.pop("sparsity", {})
        if sparsity_config and target_device == DeviceType.CSX:
            # SparsityWrapperOptimizer save its state (masks) into
            # state_dict["optimizer"]["sparsity"]
            if (
                "optimizer" in state_dict
                and "sparsity" in state_dict["optimizer"]
            ):
                configure_eval_sparsity(
                    sparsity_config, state_dict["optimizer"]["sparsity"]
                )
            else:
                warn(
                    "Sparsity masks not available, so evaluation will "
                    "use dense computation at lower performance."
                )

    global_step = 0

    # Load checkpoint if provided and not compiling or validating
    if (
        not compile_only
        and not validate_only
        and (checkpoint_path := get_model_checkpoint(runconfig))
    ):
        load_checkpoint(checkpoint_path)

    writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=runconfig.get("summary_dir", os.path.join(model_dir, "eval"))
    )

    # parameters for numeric debugging
    if runconfig.get("dump_activations", False):
        if not backend.is_csx:
            numeric_debug = DumpContext(
                outdir=os.path.join(model_dir, "act_dumps"), model=model,
            )
        else:
            warn(
                "Got `dump_activations=True` but activation dumping is not "
                "supported on CSX. Disabling activation dumping for this run."
            )
    else:
        numeric_debug = lambda f: f

    @cstorch.trace
    @numeric_debug
    @torch.no_grad()
    def eval_step(batch):
        if log_input_summaries:
            log_input_summary(batch)

        loss = compiled_model(batch)
        return loss

    total_loss = 0
    total_steps = 0

    @cstorch.step_closure
    def post_eval_step(loss, step):
        nonlocal total_loss
        nonlocal total_steps

        is_log_step = executor.on_final_iteration or (
            log_steps and step % log_steps == 0
        )

        rate = executor.profiler.rate()
        global_rate = executor.profiler.global_rate()

        if is_log_step:
            # Print some logs to provide an update to the client
            logging.info(
                f"| Eval Device={backend.device}, "
                f"Step={step}, "
                f"Loss={loss.item():.5f}, "
                f"Rate={rate:.2f} samples/sec, "
                f"GlobalRate={global_rate:.2f} samples/sec"
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

    executor = cstorch.utils.data.DataExecutor(
        dataloader, num_steps=num_steps, cs_config=cs_config, writer=writer,
    )

    try:
        for step, batch in enumerate(executor, start=1):
            loss = eval_step(batch)

            post_eval_step(loss, step)

        if not (compile_only or validate_only):
            for name, value in metrics.compute_all_metrics().items():
                writer.add_scalar(name, value, global_step)
                logging.info(f"Metric: {name} = {value}")

            avg_eval_loss = total_loss / total_steps
            writer.add_scalar("loss", avg_eval_loss, global_step)
            logging.info(f"Avg Eval Loss: {avg_eval_loss}")

            logging.info("Evaluation completed successfully!")

    finally:
        if not (compile_only or validate_only) and executor.profiler:
            # compute the total samples processed based on the number of steps
            # and the number of Cerebras systems in the cluster
            total_samples = int(executor.profiler.total_samples())
            total_time = executor.profiler.total_time()

            logging.info(
                f"Processed {total_samples} sample(s) "
                f"in {total_time} seconds."
            )


def get_latest_checkpoint(model_dir: str) -> Union[str, None]:
    """Get the path to the checkpoint with the highest global step"""
    ckpts = []
    for checkpoint in Path(model_dir).glob("checkpoint_*.mdl"):
        match = re.match(
            r"checkpoint_(?P<step>\d+)(?:_(?P<timestamp>\d{8}_\d{6}))?.mdl",
            checkpoint.name,
        )
        if not match:
            continue

        step = int(match.group("step"))
        timestamp = match.group("timestamp")
        if timestamp is not None:
            try:
                date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
        else:
            date = datetime.min

        ckpts.append((checkpoint, step, date))

    # sort by step and then by timestamp
    return max(ckpts, key=lambda x: (x[1], x[2]))[0] if ckpts else None


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
    """Log the input tensors to tensorboard"""
    for scope, tensor in visit_torch_tensors(batch, scope=["input"]):
        cstorch.summarize_tensor(".".join(map(str, scope)), tensor)


def compute_params_norm(model):
    """Compute the model wise norm of the parameters"""
    param_norm = torch.tensor(0.0).to(model.device)
    for _, param in model.named_parameters():
        if param.requires_grad:
            # simply add if we want to include all params
            param_norm += torch.pow(torch.norm(param), 2.0)
    cstorch.summarize_scalar("model_wise_params_norm", torch.sqrt(param_norm))


def compute_grad_norm(model):
    """Compute the model wise and per layer norm of the gradients"""
    params_grad_norm = torch.tensor(0.0).to(model.device)
    for _, param in model.named_parameters():
        if param.grad is not None:
            params_grad_norm += torch.pow(torch.norm(param.grad), 2.0)
    params_grad_norm = torch.sqrt(params_grad_norm)

    cstorch.summarize_scalar("model_wise_grad_norm", params_grad_norm)

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
        cstorch.summarize_scalar(
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
    additionally featuring grad norm summaries
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
    initial_loss_scale: Optional[float] = 2.0 ** 15
    steps_per_increase: Optional[int] = 2000
    min_loss_scale: Optional[float] = None
    max_loss_scale: Optional[float] = None
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
