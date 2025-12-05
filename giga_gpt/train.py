# Copyright 2023 Cerebras Systems.
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
from pathlib import Path
from dataclasses import asdict

import torch

import cerebras_pytorch as cstorch
from configuration import parse_args
from data import get_dataloader
from model import GPTModel

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main(model_config, config, cs_config):
    if config.backend == "CSX":
        backend = cstorch.backend(config.backend, use_cs_grad_accum=True)
    else:
        backend = cstorch.backend(config.backend)

    out_dir = Path(config.out_dir)

    if not backend.is_cpu:
        cstorch.amp.use_bfloat16(True)

    with backend.device:
        model = GPTModel(model_config)

    compiled_model = cstorch.compile(model, backend)

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = cstorch.optim.AdamW(
        param_groups,
        lr=0.1,  # just a placeholder as we are using learning rate scheduling
        weight_decay=config.weight_decay,
        correct_bias=True,
        betas=(0.9, 0.95),
        eps=config.adam_epsilon,
    )
    lr_scheduler = cstorch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            cstorch.optim.lr_scheduler.LinearLR(
                optimizer,
                initial_learning_rate=0.0,
                end_learning_rate=config.learning_rate,
                total_iters=config.warmup_steps,
            ),
            cstorch.optim.lr_scheduler.CosineDecayLR(
                optimizer,
                initial_learning_rate=config.learning_rate,
                end_learning_rate=0.1 * config.learning_rate,
                total_iters=config.decay_steps,
            ),
        ],
        milestones=[config.warmup_steps],
    )
    all_params = (
        p
        for param_group in optimizer.param_groups
        for p in param_group["params"]
    )

    if config.checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {config.checkpoint_path}")

        state_dict = cstorch.load(config.checkpoint_path)

        model.load_state_dict(state_dict["model"])
        if "optimizer" in state_dict:
            optimizer.load_state_dict(state_dict["optimizer"])
        if "lr_scheduler" in state_dict:
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        global_step = state_dict.get("global_step", 0)
    else:
        global_step = 0

    @cstorch.checkpoint_closure
    def save_checkpoint(step):
        checkpoint_path = out_dir.joinpath(f"checkpoint_{step}.mdl")
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "global_step": step,
            "model_config": asdict(model_config),
        }
        cstorch.save(state_dict, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    @cstorch.trace
    def training_step(batch):
        input_ids, labels = batch
        loss = compiled_model(input_ids, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(all_params), config.max_gradient_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        return loss

    writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=out_dir.joinpath("train")
    )

    @cstorch.step_closure
    def log_loss(loss, step):
        rate = executor.profiler.rate()
        global_rate = executor.profiler.global_rate()

        logger.info(
            f"| Step={step}, "
            f"Loss={loss.item():.5f}, "
            f"Rate={rate:.2f} samples/sec, "
            f"GlobalRate={global_rate:.2f} samples/sec"
        )
        writer.add_scalar("loss", loss.item(), step)
        writer.add_scalar("samples_per_second", global_rate, step)

    data_path = os.path.join(config.dataset, "train.bin")
    dataloader = cstorch.utils.data.DataLoader(
        get_dataloader,
        data_path,
        config.sequence_length,
        config.batch_size,
        config.seed,
    )
    executor = cstorch.utils.data.DataExecutor(
        dataloader,
        num_steps=config.num_steps - global_step,
        checkpoint_steps=config.checkpoint_steps,
        cs_config=cs_config,
        writer=writer,
    )

    for step, batch in enumerate(executor, start=global_step + 1):
        if step > config.num_steps:
            break
        loss = training_step(batch)
        log_loss(loss, step)
        save_checkpoint(step)

    logger.info("Training completed successfully!")


if __name__ == '__main__':
    model_config, run_config, cs_config = parse_args()
    main(model_config, run_config, cs_config)
