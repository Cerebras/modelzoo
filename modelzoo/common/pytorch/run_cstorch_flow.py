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
import logging
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch

from modelzoo.common.pytorch.half_dtype import half_dtype_instance
from modelzoo.common.pytorch.utils import (
    is_mup_run,
    named_parameters_requiring_grad,
    partition_params_groups_with_adjusted_lr,
    partition_params_groups_with_weight_decay,
    setup_logging,
)
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
    import cerebras_pytorch.experimental as cstorch
    from cerebras_appliance.run_utils import get_debug_args

    runconfig = params["runconfig"]

    if "seed" in runconfig:
        # Ensure we set seed before any model initialization
        torch.manual_seed(runconfig["seed"])

    debug_args = None
    if runconfig.get("debug_args_path"):
        debug_args = get_debug_args(runconfig["debug_args_path"])

    # Configure the Cerebras Wafer Scale cluster
    cs_config = cstorch.utils.CSConfig(
        num_csx=runconfig.get("num_csx"),
        max_wgt_servers=runconfig.get("num_wgt_servers"),
        mgmt_address=runconfig.get("mgmt_address"),
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
    import cerebras_pytorch.experimental as cstorch

    runconfig = params["runconfig"]

    model_dir = runconfig["model_dir"]
    compile_dir = runconfig.get("compile_dir")
    log_steps = runconfig.get("log_steps")
    checkpoint_steps = runconfig.get("checkpoint_steps")
    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)
    drop_data = runconfig.get("drop_data", False)
    log_summaries = params["optimizer"].get("log_summaries", False)
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

    use_bfloat16 = params["model"].get("use_bfloat16", False)
    half_dtype_instance.use_bfloat16 = use_bfloat16
    if use_bfloat16:
        cstorch.amp.use_bfloat16(True)

    optimizer_params = params["optimizer"]

    grad_scaler = None
    loss_scale = params["optimizer"].get("loss_scaling_factor", 1.0)
    if loss_scale == "dynamic" and use_bfloat16:
        optimizer_params["loss_scaling_factor"] = 1.0
        logging.info(
            f"No need to use DLS for loss when `use_bfloat16` is set to"
            " `True`. Setting `loss_scaling_factor ` to `1.0`."
        )

    use_cstorch_optimizer_step = runconfig.get(
        "use_cstorch_optimizer_step", False
    )
    # Default to only keeping the 5 latest checkpoints.
    max_checkpoints = runconfig.get("max_checkpoints", 5)

    target_device = runconfig["target_device"]
    if target_device == DeviceType.CSX:
        backend = cstorch.backend(
            "CSX",
            artifact_dir=os.path.join(model_dir, "cerebras_logs"),
            compile_dir=compile_dir,
            compile_only=compile_only,
            validate_only=validate_only,
            drop_data=drop_data,
            max_checkpoints=max_checkpoints,
        )
    elif target_device == DeviceType.CPU:
        backend = cstorch.backend("CPU", max_checkpoints=max_checkpoints,)

    with backend.device:
        model = model_fn(params)

    compiled_model = cstorch.compile(model, backend)
    compiled_model.train()

    # learning rate scaling params
    lr_adjustment_scalars = []
    lr_adjustment_layers = []
    if optimizer_params.get("adjust_learning_rate"):
        for layer_type, adjustment_scalar in optimizer_params.get(
            "adjust_learning_rate"
        ).items():
            lr_adjustment_layers.append(layer_type)
            lr_adjustment_scalars.append(adjustment_scalar)
    assert len(lr_adjustment_scalars) == len(
        lr_adjustment_layers
    ), "number of keys for layer types should match the number of scalars"
    param_optimizer = list(named_parameters_requiring_grad(model))
    # default: assemble all params in 1 group
    param_optimizer_grouped = [{"params": list(param_optimizer)}]
    # split param_groups in 2 groups: with and without weight decay
    param_optimizer_grouped = partition_params_groups_with_weight_decay(
        model,
        param_optimizer_grouped,
        optimizer_params.get("weight_decay_rate", 0.0),
    )
    # create additional param groups for each layer type with lr adjustment scalar
    param_optimizer_grouped = partition_params_groups_with_adjusted_lr(
        model,
        param_optimizer_grouped,
        lr_adjustment_layers,
        lr_adjustment_scalars,
    )
    # remove param name from the (name, param) tuple as the name was only used for referencing
    # while grouping params
    for group_idx in range(len(param_optimizer_grouped)):
        param_list = []
        for _, param in param_optimizer_grouped[group_idx]["params"]:
            param_list.append(param)
        param_optimizer_grouped[group_idx].pop("params")
        param_optimizer_grouped[group_idx]["params"] = param_list

    optimizer = cstorch.optim.configure_optimizer(
        optimizer_type=optimizer_params.pop("optimizer_type"),
        params=param_optimizer_grouped,
        **optimizer_params,
    )

    lr_scheduler = cstorch.optim.configure_lr_scheduler(
        optimizer, optimizer_params.get("learning_rate"),
    )

    if loss_scale is not None:
        if backend.is_csx:
            grad_scaler = cstorch.amp.GradScaler(
                loss_scale=optimizer_params.get("loss_scaling_factor"),
                init_scale=optimizer_params.get("initial_loss_scale"),
                steps_per_increase=optimizer_params.get("steps_per_increase"),
                min_loss_scale=optimizer_params.get("min_loss_scale"),
                max_loss_scale=optimizer_params.get("max_loss_scale"),
                max_gradient_norm=optimizer_params.get("max_gradient_norm"),
            )
        elif backend.is_gpu:
            grad_scaler = torch.cuda.amp.GradScaler()
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
            # name. Appending the current time, should be sufficient
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

        state_dict["global_step"] = step

        cstorch.save(state_dict, checkpoint_file)

        logging.info(f"Saved checkpoint {checkpoint_file}")

    def load_checkpoint(checkpoint_path):
        logging.info(f"Loading weights from checkpoint {checkpoint_path}")
        state_dict = cstorch.load(checkpoint_path)
        model.load_state_dict(state_dict["model"])
        if not runconfig.get("is_pretrained_checkpoint", False):
            optimizer.load_state_dict(state_dict["optimizer"])

            if lr_scheduler and "lr_scheduler" in state_dict:
                lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

            if grad_scaler and "grad_scaler" in state_dict:
                grad_scaler.load_state_dict(state_dict["grad_scaler"])

        global_step = state_dict.get("global_step", 0)

        return global_step

    global_step = 0

    if compile_only or validate_only:
        # Don't bother loading a checkpoint if only compiling/validating
        pass
    elif runconfig.get("checkpoint_path") is not None:
        global_step = load_checkpoint(runconfig["checkpoint_path"])
    elif runconfig.get("autoload_last_checkpoint", True):
        # get the path to the checkpoint with the highest global step
        last_checkpoint = get_latest_checkpoint(model_dir)
        if last_checkpoint:
            logging.info(f"Found latest checkpoint at {last_checkpoint}")
            global_step = load_checkpoint(last_checkpoint)
        else:
            logging.info(
                f"Expected checkpoints named `checkpoint_(\\d+).mdl` "
                f"in {model_dir} but found None. "
                f"Using randomly initialized model parameters."
            )
    else:
        logging.info(
            f"No checkpoint was provided, using randomly initialized model "
            f"parameters."
        )

    if runconfig.get("save_initial_checkpoint", False) and not compile_only:
        save_checkpoint(global_step)

    writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join(model_dir, "train")
    )

    @cstorch.compile_step
    def training_step(*args, **kwargs):
        loss = compiled_model(*args, **kwargs)

        if log_summaries:
            compute_params_norm(compiled_model)

        if not grad_scaler:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif use_cstorch_optimizer_step:
            cstorch.amp.optimizer_step(
                loss,
                optimizer,
                grad_scaler,
                max_gradient_norm=optimizer_params.get("max_gradient_norm"),
                max_gradient_value=optimizer_params.get("max_gradient_value"),
            )
        else:
            optimizer_step_with_summaries(
                loss,
                optimizer,
                grad_scaler,
                max_gradient_norm=optimizer_params.get("max_gradient_norm"),
                max_gradient_value=optimizer_params.get("max_gradient_value"),
                log_summaries=log_summaries,
                model=compiled_model,
            )

        if lr_scheduler:
            lr_scheduler.step()

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
            writer.add_scalar(
                "avg_steps_per_sec",
                global_rate / dataloader.batch_size,
                global_step,
            )

            # Save the loss value to be able to plot the loss curve
            writer.add_scalar("loss", loss.item(), global_step)

        msg_postfix = (
            "This could potentially be due to selected hyperparameters such as "
            "the learning rate, batch size, etc. or it could due an internal "
            "error. Please try with different set of hyperparameters and "
            "contact Cerebras Support if the issue persists."
        )
        if torch.isnan(loss).any().item():
            raise ValueError(f"NaN loss detected. {msg_postfix}")
        if torch.isinf(loss).any().item():
            raise ValueError(f"inf loss detected. {msg_postfix}")

        if lr_scheduler:
            for group, lr in enumerate(lr_scheduler.get_last_lr()):
                writer.add_scalar(f"lr.{group}", lr, global_step)

        total_steps += 1

    dataloader = cstorch.utils.data.DataLoader(input_fn, params)

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
        )

    executor = cstorch.utils.data.DataExecutor(
        dataloader,
        num_steps=num_steps,
        checkpoint_steps=checkpoint_steps,
        cs_config=cs_config,
        writer=writer,
    )

    start_time = time.time()

    # The main training loop
    try:
        for batch in executor:
            loss, loss_scale = training_step(batch)

            global_step += 1

            # Wait for outputs to become available to fetch from the CS system(s)
            post_training_step(loss, loss_scale)

            # only saves checkpoint if current step is a checkpoint step
            save_checkpoint(global_step)

        if not (compile_only or validate_only):
            logging.info("Training completed successfully!")
    finally:
        if not (compile_only or validate_only):
            # compute the total samples processed based on the number of steps
            # and the number of Cerebras systems in the cluster
            total_samples = total_steps * dataloader.batch_size
            end_time = time.time()

            logging.info(
                f"Processed {total_samples} sample(s) "
                f"in {end_time - start_time} seconds."
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
    import cerebras_pytorch.experimental as cstorch
    import cerebras_pytorch.experimental.metrics as metrics

    runconfig = params["runconfig"]

    model_dir = runconfig["model_dir"]
    compile_dir = runconfig.get("compile_dir")
    log_steps = runconfig.get("log_steps")
    compile_only = runconfig.get("compile_only", False)
    validate_only = runconfig.get("validate_only", False)
    drop_data = runconfig.get("drop_data", False)
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

    use_bfloat16 = params["model"].get("use_bfloat16", False)
    half_dtype_instance.use_bfloat16 = use_bfloat16
    if use_bfloat16:
        cstorch.amp.use_bfloat16(True)

    target_device = runconfig["target_device"]
    if target_device == DeviceType.CSX:
        backend = cstorch.backend(
            "CSX",
            artifact_dir=os.path.join(model_dir, "cerebras_logs"),
            compile_dir=compile_dir,
            compile_only=compile_only,
            validate_only=validate_only,
            drop_data=drop_data,
        )
    elif target_device == DeviceType.CPU:
        backend = cstorch.backend("CPU")

    with backend.device:
        model = model_fn(params)

    compiled_model = cstorch.compile(model, backend)

    def load_checkpoint(checkpoint_path):
        logging.info(f"Loading weights from checkpoint {checkpoint_path}")
        state_dict = cstorch.load(checkpoint_path)
        model.load_state_dict(state_dict["model"])

        global_step = state_dict.get("global_step", 0)
        return global_step

    global_step = 0

    if compile_only or validate_only:
        # Don't bother loading a checkpoint if only compiling/validating
        pass
    elif runconfig.get("checkpoint_path") is not None:
        global_step = load_checkpoint(runconfig["checkpoint_path"])
    elif runconfig.get("autoload_last_checkpoint", True):
        # get the path to the checkpoint with the highest global step
        last_checkpoint = get_latest_checkpoint(model_dir)
        if last_checkpoint:
            logging.info(f"Found latest checkpoint at {last_checkpoint}")
            global_step = load_checkpoint(last_checkpoint)
        else:
            logging.info(
                f"Expected checkpoints named `checkpoint_(\\d+).mdl` "
                f"in {model_dir} but found None. "
                f"Using randomly initialized model parameters."
            )
    else:
        logging.info(
            f"No checkpoint was provided, using randomly initialized model "
            f"parameters."
        )

    writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join(model_dir, "eval")
    )

    @cstorch.compile_step
    def eval_step(*args, **kwargs):
        loss = compiled_model(*args, **kwargs)
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

        is_log_step = executor.on_final_iteration or (
            log_steps and global_step % log_steps == 0
        )

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
            writer.add_scalar(
                "avg_steps_per_sec",
                global_rate / dataloader.batch_size,
                global_step,
            )

        if torch.isnan(loss).any().item():
            raise ValueError("NaN loss detected.")
        if torch.isinf(loss).any().item():
            raise ValueError("inf loss detected.")

        total_loss += loss.item()
        total_steps += 1

    dataloader = cstorch.utils.data.DataLoader(input_fn, params)

    if compile_only or validate_only:
        num_steps = None
    else:
        num_steps = cstorch.utils.data.compute_num_steps(
            dataloader, num_steps=runconfig.get("eval_steps"), num_epochs=1
        )

    executor = cstorch.utils.data.DataExecutor(
        dataloader, num_steps=num_steps, cs_config=cs_config, writer=writer,
    )

    start_time = time.time()

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
        if not (compile_only or validate_only):
            # compute the total samples processed based on the number of steps
            # and the number of Cerebras systems in the cluster
            end_time = time.time()
            total_samples = total_steps * dataloader.batch_size
            logging.info(
                f"Processed {total_samples} sample(s) "
                f"in {end_time - start_time} seconds."
            )


def get_latest_checkpoint(model_dir):
    """Get the path to the checkpoint with the highest global step"""
    checkpoints = sorted(
        Path(model_dir).glob("checkpoint_*.mdl"),
        key=lambda p: int(re.match(r"checkpoint_(\d+).mdl", p.name).group(1)),
    )
    if len(checkpoints) > 0:
        return checkpoints[-1]
    else:
        return None


def compute_params_norm(model):
    """Compute the model wise norm of the parameters"""
    import cerebras_pytorch.experimental as cstorch

    param_norm = torch.tensor(0.0).to(model.device)
    for _, param in model.named_parameters():
        if param.requires_grad:
            # simply add if we want to include all params
            param_norm += torch.pow(torch.norm(param), 2.0)
    cstorch.summarize_scalar("model_wise_params_norm", torch.sqrt(param_norm))


def compute_grad_norm(model):
    """Compute the model wise and per layer norm of the gradients"""
    import cerebras_pytorch.experimental as cstorch

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
    optimizer.zero_grad()
    grad_scaler.scale(loss).backward()

    # Unscales the gradients of optimizer's assigned params in-place
    grad_scaler.unscale_(optimizer)

    if log_summaries:
        assert model is not None
        compute_grad_norm(model)

    # gradient clipping
    if max_gradient_norm is not None and max_gradient_norm < 0.0:
        raise ValueError(
            f"max_gradient_norm has to be a non-negative float. Got "
            f"{max_gradient_norm}"
        )
    if max_gradient_value is not None and max_gradient_value < 0.0:
        raise ValueError(
            f"max_gradient_value has to be a non-negative float. Got "
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
