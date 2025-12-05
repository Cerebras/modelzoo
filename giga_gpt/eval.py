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

import torch
from tqdm import tqdm

import cerebras_pytorch as cstorch
from configuration import parse_args
from data import get_dataloader
from model import GPTModel

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def main(model_config, config, cs_config):
    backend = cstorch.backend(config.backend)
    out_dir = Path(config.out_dir)
    state_dict = cstorch.load(config.checkpoint_path)

    if not backend.is_cpu:
        cstorch.amp.use_bfloat16(True)

    with backend.device:
        model = GPTModel(model_config)
    compiled_model = cstorch.compile(model, backend)

    logger.info(f"Loading checkpoint from {config.checkpoint_path}")
    state_dict = cstorch.load(config.checkpoint_path)
    model.load_state_dict(state_dict["model"])
    global_step = state_dict.get("global_step", 0)

    @cstorch.trace
    @torch.no_grad()
    def eval_step(batch):
        input_ids, labels = batch
        loss = compiled_model(input_ids, labels)
        return loss

    total_loss = 0
    total_steps = 0

    @cstorch.step_closure
    def post_eval_step(loss, step):
        nonlocal total_loss
        nonlocal total_steps
        total_loss += loss
        total_steps += 1

    writer = cstorch.utils.tensorboard.SummaryWriter(
        log_dir=out_dir.joinpath("eval")
    )

    data_path = os.path.join(config.dataset, "val.bin")
    dataloader = cstorch.utils.data.DataLoader(
        get_dataloader,
        data_path,
        config.sequence_length,
        config.batch_size,
    )
    num_steps = len(dataloader)
    executor = cstorch.utils.data.DataExecutor(
        dataloader, num_steps=num_steps, cs_config=cs_config, writer=writer,
    )

    logger.info(f"Total eval steps: {num_steps}")
    for step, batch in tqdm(enumerate(executor, start=1), total=num_steps):
        if step > num_steps:
            break
        loss = eval_step(batch)
        post_eval_step(loss, step)

    avg_loss = total_loss / total_steps
    writer.add_scalar("loss", avg_loss, global_step)
    logger.info(f"Average eval loss: {avg_loss}")


if __name__ == "__main__":
    model_config, run_config, cs_config = parse_args()
    if run_config.checkpoint_path is None:
        raise ValueError(f"You must specify a checkpoint path for model eval")
    main(model_config, run_config, cs_config)
