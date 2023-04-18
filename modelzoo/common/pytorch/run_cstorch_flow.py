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

import torch
from torch.utils.tensorboard import SummaryWriter

from modelzoo.common.pytorch.utils import (
    RunConfigParamsValidator,
    group_optimizer_params,
    setup_logging,
    trainable_named_parameters,
)


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

    from cerebras_appliance.pb.workflow.appliance.common.common_config_pb2 import (
        ExecutionStrategy,
    )
    from cerebras_appliance.run_utils import get_debug_args

    runconfig = params["runconfig"]

    RunConfigParamsValidator().validate(runconfig)

    if "seed" in runconfig:
        # Ensure we set seed before any model initialization
        torch.manual_seed(runconfig["seed"])

    debug_args = None
    if runconfig.get("debug_args_path"):
        debug_args = get_debug_args(runconfig["debug_args_path"])

    # Configure the Cerebras Wafer Scale cluster
    cstorch.configure(
        model_dir=runconfig["model_dir"],
        compile_dir=runconfig["compile_dir"],
        compile_only=runconfig["compile_only"],
        validate_only=runconfig["validate_only"],
        checkpoint_steps=runconfig["checkpoint_steps"],
        seed=runconfig.get("seed"),
        # CSConfig params
        num_csx=runconfig["num_csx"],
        max_wgt_servers=runconfig["num_wgt_servers"],
        mgmt_address=runconfig["mgmt_address"],
        credentials_path=runconfig["credentials_path"],
        debug_args=debug_args,
        mount_dirs=runconfig.get("mount_dirs"),
        python_paths=runconfig.get("python_paths"),
        transfer_processes=runconfig["transfer_processes"],
        num_workers_per_csx=runconfig["num_workers_per_csx"],
        execution_strategy=ExecutionStrategy.ES_WEIGHT_STREAMING,
        job_labels=runconfig.get("job_labels"),
        max_act_per_csx=runconfig["num_act_servers"],
        job_time_sec=runconfig["job_time_sec"],
        disable_version_check=runconfig["disable_version_check"],
    )

    # Set up logging level
    setup_logging(
        runconfig.get("logging"), runconfig.get("streamer_logging"),
    )

    if runconfig["mode"] == "train":
        run_cstorch_train(params, model_fn, train_data_fn)
    elif runconfig["mode"] == "eval":
        run_cstorch_eval(params, model_fn, eval_data_fn)


def run_cstorch_train(params, model_fn, input_fn):
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
    log_steps = runconfig.get("log_steps")
    checkpoint_steps = runconfig.get("checkpoint_steps")
    compile_only = runconfig.get("compile_only") or runconfig.get(
        "validate_only"
    )
    log_summaries = params["optimizer"].get("log_summaries", False)

    optimizer_params = params["optimizer"]

    use_cstorch_optimizer_step = runconfig.get(
        "use_cstorch_optimizer_step", False
    )

    model = model_fn(params)
    compiled_model = cstorch.compile(model, backend="WSE_WS",)
    compiled_model.train()

    # Sanity check
    assert all(p.device.type == "xla" for p in model.parameters())

    no_decay_layers, param_optimizer = trainable_named_parameters(model)
    param_optimizer_grouped = group_optimizer_params(
        param_optimizer,
        no_decay_layers,
        optimizer_params.get("weight_decay_rate", 0.0),
    )
    optimizer = cstorch.optim.configure_optimizer(
        optimizer_type=optimizer_params.pop("optimizer_type"),
        params=param_optimizer_grouped,
        **optimizer_params,
    )
    lr_scheduler = cstorch.optim.configure_lr_scheduler(
        optimizer, optimizer_params.get("learning_rate"),
    )

    grad_scaler = None
    loss_scale = params["optimizer"].get("loss_scale")
    if loss_scale is not None:
        if compiled_model.backend.is_wse():
            grad_scaler = cstorch.amp.GradScaler()
        elif compiled_model.backend.is_gpu():
            grad_scaler = torch.cuda.amp.GradScaler()
        else:
            raise RuntimeError(
                f"Gradient scaling not supported on {compiled_model.backend}"
            )

    @cstorch.step_closure
    def save_checkpoint(step):
        logging.info(f"Saving checkpoint at step {step}")

        checkpoint_file = os.path.join(model_dir, f"checkpoint_{step}.mdl")

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if grad_scaler:
            state_dict["grad_scaler"] = grad_scaler.state_dict()

        state_dict["global_step"] = step

        cstorch.save(state_dict, checkpoint_file)

        logging.info(f"Saved checkpoint {checkpoint_file}")

    def load_checkpoint(checkpoint_path):
        state_dict = cstorch.load(checkpoint_path)
        model.load_state_dict(state_dict["model"])
        if not runconfig.get("is_pretrained_checkpoint", False):
            optimizer.load_state_dict(state_dict["optimizer"])
            if grad_scaler:
                grad_scaler.load_state_dict(state_dict["grad_scaler"])

        global_step = state_dict.get("global_step", 0)

        return global_step

    global_step = 0
    if runconfig.get("checkpoint_path") is not None:
        global_step = load_checkpoint(runconfig["checkpoint_path"])

    if runconfig.get("save_initial_checkpoint", False) and not compile_only:
        save_checkpoint(global_step)

    writer = SummaryWriter(log_dir=os.path.join(model_dir, "train"))

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
        if grad_scaler:
            loss_scale = grad_scaler.get_scale()

        # return final values
        return loss, loss_scale

    @cstorch.step_closure
    def post_training_step(loss, loss_scale):
        # extract the loss scalar
        if log_steps and global_step % log_steps == 0:
            # Print some logs to provide an update to the client
            logging.info(
                f"| Train: {compiled_model.backend.name} "
                f"Step={global_step}, "
                f"Loss={loss.item():.5f}"
            )

        if torch.isnan(loss).any().item():
            raise ValueError(
                "NaN loss detected. "
                "Please try different hyperparameters "
                "such as the learning rate, batch size, etc."
            )
        if torch.isinf(loss).any().item():
            raise ValueError("inf loss detected.")

        # Save the loss value to be able to plot the loss curve
        writer.add_scalar("loss", loss.item(), global_step)

        cstorch.save_summaries(writer, global_step)

        if lr_scheduler:
            for group, lr in enumerate(lr_scheduler.get_last_lr()):
                writer.add_scalar(f"lr.{group}", lr, global_step)

    dataloader = cstorch.utils.data.DataLoader(
        input_fn,
        params,
        initial_step=global_step,
        num_steps=runconfig.get("num_steps"),
        max_steps=runconfig.get("max_steps"),
        num_epochs=runconfig.get("num_epochs"),
        steps_per_epoch=runconfig.get("steps_per_epoch"),
    )

    total_steps = 0
    start_time = time.time()

    # The main training loop
    try:
        for _epoch in range(dataloader.num_epochs):
            for _i, batch in enumerate(dataloader):
                loss, loss_scale = training_step(batch)

                global_step += 1
                total_steps += 1

                # Wait for outputs to become available to fetch from the CS system(s)
                post_training_step(loss, loss_scale)

                if checkpoint_steps and global_step % checkpoint_steps == 0:
                    save_checkpoint(global_step)

        logging.info("Training Completed Successfully!")
    finally:
        # compute the total samples processed based on the number of steps
        # and the number of Cerebras systems in the cluster
        total_samples = (
            total_steps * dataloader.batch_size * runconfig["num_csx"]
        )
        end_time = time.time()

        logging.info(
            f"Processed {total_samples} sample(s)"
            f" in {end_time - start_time} seconds."
        )


def run_cstorch_eval(params, model_fn, input_fn):
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

    model = model_fn(params)
    compiled_model = cstorch.compile(model, backend="WSE_WS",)

    def load_checkpoint(checkpoint_path):
        state_dict = cstorch.load(checkpoint_path)
        model.load_state_dict(state_dict["model"])

        global_step = state_dict.get("global_step", 0)
        return global_step

    global_step = 0

    if runconfig.get("checkpoint_path") is not None:
        global_step = load_checkpoint(runconfig["checkpoint_path"])

    writer = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))

    @cstorch.compile_step
    def eval_step(*args, **kwargs):
        loss = compiled_model(*args, **kwargs)
        return loss

    total_loss = 0
    total_samples = 0

    @cstorch.step_closure
    def post_eval_step(loss):
        nonlocal total_loss
        nonlocal total_samples

        # Print some logs to provide an update to the client
        logging.info(
            f"| Eval: {compiled_model.backend.name} "
            f"Step={global_step}, "
            f"Loss={loss.item():.5f}"
        )

        if torch.isnan(loss).any().item():
            raise ValueError("NaN loss detected.")
        if torch.isinf(loss).any().item():
            raise ValueError("inf loss detected.")

        total_loss += loss.item()
        total_samples += 1

        cstorch.save_summaries(writer, global_step)

    dataloader = cstorch.utils.data.DataLoader(
        input_fn,
        params,
        initial_step=global_step,
        num_steps=runconfig.get("num_steps"),
        max_steps=runconfig.get("max_steps"),
        num_epochs=runconfig.get("num_epochs"),
        steps_per_epoch=runconfig.get("steps_per_epoch"),
    )
    assert (
        dataloader.num_epochs == 1
    ), "Cannot specify more than 1 epoch for eval"

    start_time = time.time()

    try:
        for _epoch in range(dataloader.num_epochs):
            for _i, batch in enumerate(dataloader):
                loss = eval_step(batch)

                post_eval_step(loss)

        for name, value in metrics.compute_all_metrics():
            writer.add_scalar(name, value, global_step)
            logging.info(f"Metric: {name} = {value}")

        avg_eval_loss = total_loss / total_samples
        writer.add_scalar("loss", avg_eval_loss, global_step)
        logging.info(f"Avg Eval Loss: {avg_eval_loss}")

        logging.info("Evaluation completed successfully!")
    finally:
        # compute the total samples processed based on the number of steps
        # and the number of Cerebras systems in the cluster
        end_time = time.time()

        logging.info(
            f"Process {total_samples} sample(s)"
            f" in {end_time - start_time} seconds."
        )


def compute_params_norm(model):
    """Compute the model wise norm of the parameters"""
    import cerebras_pytorch.experimental as cstorch

    param_norm = torch.tensor(0.0).to(model.device)
    for _, param in model.named_parameters():
        if param.requires_grad:
            # simply add if we want to include all params
            param_norm += torch.pow(torch.norm(param), 2.0)
    cstorch.scalar_summary("model_wise_params_norm", torch.sqrt(param_norm))


def compute_grad_norm(model):
    """Compute the model wise and per layer norm of the gradients"""
    import cerebras_pytorch.experimental as cstorch

    params_grad_norm = torch.tensor(0.0).to(model.device)
    for _, param in model.named_parameters():
        if param.grad is not None:
            params_grad_norm += torch.pow(torch.norm(param.grad), 2.0)
    params_grad_norm = torch.sqrt(params_grad_norm)

    cstorch.scalar_summary("model_wise_grad_norm", params_grad_norm)

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
        cstorch.scalar_summary(
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
