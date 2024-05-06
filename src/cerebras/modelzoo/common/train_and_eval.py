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
import copy
import logging
import math
import os
import warnings
from datetime import datetime
from warnings import warn

import torch

import cerebras.pytorch as cstorch
import cerebras.pytorch.metrics as metrics
from cerebras.modelzoo.common.checkpoint_utils import CkptInfo
from cerebras.modelzoo.common.dump_context import DumpContext
from cerebras.modelzoo.common.half_dtype import set_half_dtype_from_params
from cerebras.modelzoo.common.input_utils import (
    validate_streaming_and_micro_batch_size,
)
from cerebras.modelzoo.common.pytorch_utils import load_from_checkpoint_file
from cerebras.modelzoo.common.run_cstorch_flow import (
    GradScalerParams,
    compute_grad_norm,
    compute_params_norm,
    get_model_checkpoint,
    log_input_summary,
    optimizer_step_with_summaries,
)
from cerebras.modelzoo.common.utils.run.utils import DeviceType
from cerebras.modelzoo.common.utils.utils import format_rate


def train_and_eval(
    params, model_fn, train_data_fn, eval_data_fn, cs_config, artifact_dir
):
    """
    Runs the train and eval workflow built using the cstorch API

    Args:
        params: the params dictionary extracted from the params.yaml used
        model_fn: A callable that takes in the params dictionary and returns
            a torch.nn.Module
        train_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader meant for training
        eval_data_fn: A callable that takes in the param dictionary and
            returns a torch.utils.data.DataLoader meant for eval
    """
    from cerebras.appliance.errors import ApplianceNanError

    params_copy = copy.deepcopy(params)

    if not isinstance(params.get("optimizer"), dict):
        raise ValueError(
            "An `optimizer` configuration is required for training"
        )

    runconfig = params["runconfig"]

    target_device = runconfig["target_device"]

    model_dir = runconfig["model_dir"]
    compile_dir = runconfig.get("compile_dir")
    log_steps = runconfig.get("log_steps")
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

    if activation_steps != 1 and activation_steps is not None:
        raise ValueError(
            "Enabling activation frequency in train_and_eval is not yet supported. "
            "Please set `enable_act_frequency` to False."
        )

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

    train_micro_batch_size, eval_micro_batch_size = None, None
    if target_device == DeviceType.CSX:
        (
            train_micro_batch_size,
            eval_micro_batch_size,
        ) = validate_train_and_eval_micro_batch_size(params, cs_config.num_csx)

        backend = cstorch.backend(
            "CSX",
            artifact_dir=artifact_dir,
            compile_dir=compile_dir,
            compile_only=compile_only,
            validate_only=validate_only,
            retrace_every_iteration=runconfig.get(
                "retrace_every_iteration", False
            ),
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
            mixed_precision=mixed_precision,
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
        model = model_fn(params)

        if lora_params:
            # After model init, we need to convert it into a LoRA model
            from modelzoo.common.utils.model.lora import make_model_lora

            model = make_model_lora(model, lora_params)

    compiled_model = cstorch.compile(model, backend)
    compiled_model.train()

    # Register tensor listeners.
    listeners = [
        cstorch.experimental.listener.create_listener(**listener_params)
        for listener_params in listener_configs
    ]

    # group optimizer params
    param_optimizer_grouped = cstorch.optim.configure_param_groups(
        model,
        optimizer_params,
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

    sparsity = None
    sparsity_config = params.pop("sparsity", {})
    if sparsity_config:
        sparsity = cstorch.sparse.configure(sparsity_config)

        # Sparsify model parameters and optimizer state
        model.apply(sparsity)
        optimizer.apply(sparsity)

    lr_scheduler = cstorch.optim.configure_lr_scheduler(
        optimizer,
        optimizer_params.get("learning_rate"),
        optimizer_params.get("adjust_learning_rate"),
    )

    train_dataloader = cstorch.utils.data.DataLoader(train_data_fn, params)

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

        if train_dataloader.is_restartable:
            logging.debug("Saving dataloader state")
            dl_state = train_dataloader.state_dict()
            state_dict["dataloader"] = dl_state
            logging.debug(f"Saved dataloader state: {dl_state}")

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
            train_dataloader,
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

    train_writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join(model_dir, "train"),
        # The post_training_step summaries are written after global_step is
        # incremented, so +1.
        base_step=global_step + 1,
    )
    eval_writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join(model_dir, "eval"),
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
                executor.writer.add_scalar(f"lr/{group}", lr, global_step)

    def check_loss(loss, msg=""):
        if torch.isnan(loss).any().item():
            raise ApplianceNanError(f"NaN loss detected. {msg}")
        if torch.isinf(loss).any().item():
            raise ApplianceNanError(f"inf loss detected. {msg}")

    total_samples = cstorch.utils.tracker.RateTracker()

    @cstorch.step_closure
    def post_training_step(loss, loss_scale):
        # extract the loss scalar
        if is_log_step():
            rate = executor.profiler.rate_tracker.rate
            global_rate = executor.profiler.rate_tracker.global_rate

            # Print some logs to provide an update to the client
            logging.info(
                f"| Train Device={backend.device}, "
                f"GlobalStep={global_step}, "
                f"Loss={loss.item():.5f}, "
                f"Rate={format_rate(rate)} samples/sec, "
                f"GlobalRate={format_rate(global_rate)} samples/sec"
            )

            # record rates in tensorboard for future reference
            executor.writer.add_scalar(
                "local_samples_per_sec", rate, global_step
            )
            executor.writer.add_scalar(
                "avg_samples_per_sec", global_rate, global_step
            )
            if train_dataloader.batch_size:
                executor.writer.add_scalar(
                    "avg_steps_per_sec",
                    global_rate / train_dataloader.batch_size,
                    global_step,
                )

            # Save the loss value to be able to plot the loss curve
            executor.writer.add_scalar("loss", loss.item(), global_step)

            if loss_scale is not None:
                if isinstance(loss_scale, torch.Tensor):
                    loss_scale = loss_scale.item()

                # Save the loss_scale value to be able to inspect it for
                # debugging purposes
                executor.writer.add_scalar(
                    "loss_scale", loss_scale, global_step
                )

        if check_loss_values:
            msg_postfix = (
                "This could potentially be due to selected hyperparameters "
                "such as the learning rate, batch size, etc. or it could due "
                "an internal error. Please try with different set of "
                "hyperparameters and contact Cerebras Support if the issue "
                "persists."
            )
            check_loss(loss, msg_postfix)

    @cstorch.trace
    @numeric_debug
    @torch.no_grad()
    def eval_step(batch):
        if log_input_summaries:
            log_input_summary(batch)

        loss = compiled_model(batch)
        return loss

    total_eval_loss = 0
    total_eval_steps = 0

    @cstorch.step_closure
    def post_eval_step(loss):
        nonlocal total_eval_loss
        nonlocal total_eval_steps

        rate = executor.profiler.rate_tracker.rate
        global_rate = executor.profiler.rate_tracker.global_rate

        if is_log_step():
            # Print some logs to provide an update to the client
            logging.info(
                f"| Eval Device={backend.device}, "
                f"GlobalStep={global_step}, "
                f"Batch={executor.user_iteration}, "
                f"Loss={loss.item():.5f}, "
                f"Rate={format_rate(rate)} samples/sec, "
                f"GlobalRate={format_rate(global_rate)} samples/sec"
            )

        if executor.on_final_iteration:
            # log the throughput of the eval run to tensorboard on the last step
            executor.writer.add_scalar(
                "eval/local_samples_per_sec", rate, global_step
            )
            executor.writer.add_scalar(
                "eval/avg_samples_per_sec", global_rate, global_step
            )
            if eval_dataloader.batch_size:
                executor.writer.add_scalar(
                    "eval/avg_steps_per_sec",
                    global_rate / eval_dataloader.batch_size,
                    global_step,
                )

        if check_loss_values:
            check_loss(loss)

        total_eval_loss += loss.item()
        total_eval_steps += 1

    if compile_only or validate_only:
        total_steps = 1
        train_steps = 1
        eval_steps = 1
        num_trains = 1
        checkpoint_steps = None
    else:
        total_steps = cstorch.utils.data.compute_num_steps(
            train_dataloader,
            initial_step=global_step,
            num_steps=runconfig.get("num_steps"),
            max_steps=runconfig.get("max_steps"),
            num_epochs=runconfig.get("num_epochs"),
            steps_per_epoch=runconfig.get("steps_per_epoch"),
            grad_accum_steps=grad_accum_steps,
        )
        eval_steps = runconfig.get("eval_steps")

        eval_frequency = runconfig.get("eval_frequency")
        if eval_frequency is not None:
            if not isinstance(eval_frequency, int) or eval_frequency <= 0:
                raise ValueError(
                    f"`eval_frequency` must be a positive integer, got {eval_frequency}"
                )
            train_steps = eval_frequency
        elif runconfig.get("num_epochs"):
            if runconfig.get("steps_per_epoch"):
                train_steps = runconfig["steps_per_epoch"]
            else:
                train_steps = len(train_dataloader)
        else:
            raise ValueError(
                "eval_frequency or num_epochs must be specified for train_and_eval mode"
            )

        num_trains = math.ceil(total_steps / train_steps)

        checkpoint_steps = runconfig.get("checkpoint_steps")
        if (
            checkpoint_steps
            and checkpoint_steps % train_steps != 0
            and train_steps % checkpoint_steps != 0
        ):
            raise ValueError(
                f"checkpoint_steps ({checkpoint_steps}) and frequency of evaluation steps "
                f"({train_steps}) must be multiples of each other."
            )

        logging.info(
            f"Training and evaluating with the following settings:"
            f"\n\tTotal training steps: {total_steps}"
            f"\n\tNumber of training loops: {num_trains}"
            f"\n\tNumber of training steps between each checkpoint: {checkpoint_steps}"
            f"\n\tNumber of training steps between each round of evaluation: {train_steps}"
        )

    try:
        for i in range(num_trains):
            logging.info(
                f"Starting training loop {i + 1}, from global step {global_step} to "
                f"{global_step + train_steps}"
            )

            local_checkpoint_steps = None
            if checkpoint_steps:
                if checkpoint_steps <= train_steps:
                    local_checkpoint_steps = checkpoint_steps
                elif (train_steps * (i + 1)) % checkpoint_steps == 0:
                    local_checkpoint_steps = train_steps

            if i == num_trains - 1:
                train_steps = total_steps - train_steps * i
                if checkpoint_steps:
                    local_checkpoint_steps = min(checkpoint_steps, train_steps)

            executor = cstorch.utils.data.DataExecutor(
                train_dataloader,
                num_steps=train_steps,
                checkpoint_steps=local_checkpoint_steps,
                activation_steps=activation_steps,
                cs_config=cs_config,
                writer=train_writer,
                listeners=listeners,
                micro_batch_size=train_micro_batch_size,
            )

            model.train()
            # Run training loop
            for step, batch in enumerate(executor, start=1):
                loss, loss_scale = training_step(batch, step)

                if step % grad_accum_steps == 0:
                    global_step += 1

                    # Wait for outputs to become available to fetch from the CS system(s)
                    post_training_step(loss, loss_scale)

                # only saves checkpoint if current step is a checkpoint step
                save_checkpoint(global_step)

                # Track how many total samples
                total_samples.add(train_dataloader.batch_size)

            # pylint: disable=undefined-loop-variable
            assert step >= grad_accum_steps, (
                f"There were only {step} batches in training run, which is "
                f"less than the grad accumulation steps {grad_accum_steps}. "
                f"This prevents model training as no optimizer step is taken."
            )
            if step % grad_accum_steps != 0:
                warnings.warn(
                    "There were leftover gradients in the accumulation step. "
                    "They will effectively vanish, which could potentially lead "
                    "to different convergence behaviour."
                )

            eval_dataloader = cstorch.utils.data.DataLoader(
                eval_data_fn, params
            )

            executor = cstorch.utils.data.DataExecutor(
                eval_dataloader,
                num_steps=cstorch.utils.data.compute_num_steps(
                    eval_dataloader,
                    num_steps=eval_steps,
                    num_epochs=1 if eval_steps is None else None,
                ),
                cs_config=cs_config,
                writer=eval_writer,
                listeners=listeners,
                micro_batch_size=eval_micro_batch_size,
            )

            model.eval()
            for batch in executor:
                loss = eval_step(batch)
                post_eval_step(loss)

            if not (compile_only or validate_only):
                for name, metric in metrics.get_all_metrics():
                    value = float(metric)
                    eval_writer.add_scalar(name, value, global_step)
                    logging.info(f"Metric: {name} = {value}")

                    # Reset the metric for the next eval run
                    metric.reset()

                avg_eval_loss = total_eval_loss / total_eval_steps
                eval_writer.add_scalar("loss", avg_eval_loss, global_step)
                logging.info(f"Avg Eval Loss: {avg_eval_loss}")

                total_eval_loss = 0
                total_eval_steps = 0

        if not (compile_only or validate_only):
            logging.info("Training and Eval runs completed successfully!")
    finally:
        numeric_debug.flush()
        if not (compile_only or validate_only):
            logging.info(
                f"Processed {total_samples.total_count} training sample(s) "
                f"in {total_samples.elapsed_seconds()} seconds."
            )


def validate_train_and_eval_micro_batch_size(params, num_csx):
    """Validate training and evaluation batch_size/micro_batch_size settings."""
    train_input = params["train_input"]
    eval_input = params["eval_input"]

    train_bs = train_input["batch_size"]
    train_mbs = train_input.get("micro_batch_size", "auto")
    validate_streaming_and_micro_batch_size(train_bs, train_mbs, num_csx)

    eval_bs = eval_input["batch_size"]
    eval_mbs = eval_input.get("micro_batch_size", "auto")
    validate_streaming_and_micro_batch_size(eval_bs, eval_mbs, num_csx)

    def _is_explore(mbs):
        return mbs == "explore" or isinstance(mbs, dict)

    if any(_is_explore(mbs) for mbs in [train_mbs, eval_mbs]):
        # If any is explore, all must be explore
        if not all(_is_explore(mbs) for mbs in [train_mbs, eval_mbs]):
            raise ValueError(
                f"Expected either both or none of train and eval graphs to be "
                f"using \"explore\". Got {train_mbs} for train and {eval_mbs} for eval."
            )

    effective_train_mbs = train_bs if train_mbs is None else train_mbs
    effective_eval_mbs = eval_bs if eval_mbs is None else eval_mbs
    if (
        isinstance(effective_eval_mbs, int)
        and isinstance(effective_train_mbs, int)
        and effective_eval_mbs > effective_train_mbs
    ):
        raise ValueError(
            f"In train and eval mode, the eval micro batch size must be less than or equal "
            f"to the train micro batch size, but got {effective_train_mbs} for train and "
            f"{effective_eval_mbs} for eval. Either set the `micro_batch_size` settings "
            f"explicitly to honor this requirement or set the eval `micro_batch_size` to "
            f"\"auto\" for the compiler to choose a valid `micro_batch_size`."
        )

    return (train_mbs, eval_mbs)
