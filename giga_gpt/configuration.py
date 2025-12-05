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

import argparse
import dataclasses
from typing import Union

from transformers import HfArgumentParser

from cerebras_pytorch.utils import CSConfig
from model import GPTConfig


@dataclasses.dataclass
class RunConfig:
    out_dir: str = "out"
    dataset: str = "openwebtext"
    batch_size: int = 120
    sequence_length: int = 2048
    num_steps: int = 10000
    checkpoint_steps: int = 1000
    learning_rate: float = 6e-4
    warmup_steps: int = 1500
    decay_steps: int = None
    weight_decay: float = 0.1
    max_gradient_norm: float = 1.0
    adam_epsilon: float = 1e-8
    backend: str = "CSX"
    checkpoint_path: str = None
    seed: int = 0
    max_gradient_norm: float = 1.0

    def __post_init__(self):
        assert self.backend in ["CSX", "CPU", "GPU"]
        assert 0 < self.warmup_steps
        assert 0 < self.num_steps
        if self.decay_steps is None:
            self.decay_steps = max(self.num_steps - self.warmup_steps, 1)


def convert_optional_types(t):
    if t == Union[int,  None]:
        return int
    if t == Union[str, None]:
        return str
    if t == Union[float, None]:
        return float
    if t == Union[list, None]:
        return list
    return t


def parse_args():
    config_classes = (GPTConfig, RunConfig, CSConfig)

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    for config_class in config_classes[:-1]:
        class_name = config_class.__name__
        fields = dataclasses.fields(config_class)
        for f in fields:
            parser.add_argument(
                f"--{f.name}",
                type=convert_optional_types(f.type),
                dest=f"{class_name}.{f.name}",
            )
    args = parser.parse_args()

    hf_parser = HfArgumentParser(config_classes)
    configs = hf_parser.parse_yaml_file(args.config_file)

    new_configs = []
    args = vars(args)
    for config, config_class in zip(configs, config_classes):
        class_name = config_class.__name__
        kws = {}
        for k in args:
            if k.startswith(class_name) and args[k] is not None:
                field = k[len(class_name) + 1 :]
                kws[field] = args[k]
        if kws:
            config = dataclasses.replace(config, **kws)
        new_configs.append(config)

    return new_configs
