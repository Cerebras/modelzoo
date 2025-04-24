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
Config classes of T5 data Configs

"""

from typing import List, Literal, Optional, Union

from pydantic import PositiveInt, field_validator

from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.config.types import ValidatedPath


class DiffusionBaseProcessorConfig(DataConfig):
    data_dir: Union[ValidatedPath, List[ValidatedPath]] = ...
    use_worker_cache: bool = False
    num_classes: int = ...
    noaugment: bool = False
    transforms: List[dict] = []
    vae_scaling_factor: Optional[float] = None
    label_dropout_rate: Optional[float] = None
    latent_size: Optional[List[int]] = None
    latent_channels: Optional[PositiveInt] = None
    num_diffusion_steps: Optional[PositiveInt] = None
    schedule_name: Optional[str] = None
    drop_last: bool = True
    var_loss: Optional[bool] = None
    image_channels: int = ...
    split: str = "train"
    shuffle: Optional[bool] = None
    shuffle_seed: Optional[PositiveInt] = None
    prefetch_factor: Optional[PositiveInt] = None
    persistent_workers: Optional[bool] = None
    num_workers: Optional[PositiveInt] = None
    image_size: Optional[List[int]] = None
    batch_size: Optional[PositiveInt] = None

    @field_validator(
        "vae_scaling_factor",
        "label_dropout_rate",
        "latent_size",
        "latent_channels",
        "schedule_name",
        "var_loss",
        mode="after",
    )
    @classmethod
    def validate_fields(cls, v, info):
        if info.context:
            model_config = info.context.get("model", {}).get("config")
            if model_config:
                if info.field_name == "vae_scaling_factor":
                    field_value = model_config.vae.scaling_factor
                else:
                    field_value = getattr(model_config, info.field_name, None)

                if v is None:
                    v = field_value

                if v != field_value:
                    raise ValueError(
                        f"Found different {info.field_name} in "
                        f"data processor config ({v}) vs. model ({field_value})"
                    )

        return v

    @field_validator(
        "num_diffusion_steps", "num_classes", "image_channels", mode="after"
    )
    @classmethod
    def validate_image_size(cls, v, info):
        if v is not None:
            model_config = info.context.get("model", {}).get("config")
            if model_config:
                if info.field_name == "image_channels":
                    if v != model_config.vae.in_channels:
                        raise ValueError(
                            f"Found different {info.field_name} in "
                            f"data processor config ({v}) vs. model ({model_config.vae.in_channels})"
                        )
                    if v != model_config.vae.out_channels:
                        raise ValueError(
                            f"Found different {info.field_name} in "
                            f"data processor config ({v}) vs. model ({model_config.vae.out_channels})"
                        )
                else:
                    field_value = getattr(model_config, info.field_name, None)
                    if v != field_value:
                        raise ValueError(
                            f"Found different {info.field_name} in "
                            f"data processor config ({v}) vs. model ({field_value})"
                        )

        return v


class DiffusionImageNet1KProcessorConfig(DiffusionBaseProcessorConfig):
    data_processor: Literal["DiffusionImageNet1KProcessor"]


class DiffusionLatentImageNet1KProcessorConfig(DiffusionBaseProcessorConfig):
    data_processor: Literal["DiffusionLatentImageNet1KProcessor"]


class DiffusionSyntheticDataProcessorConfig(DiffusionBaseProcessorConfig):
    data_processor: Literal["DiffusionSyntheticDataProcessor"]
    data_dir: Optional[ValidatedPath] = None
    num_samples: Optional[PositiveInt] = None
    num_examples: Optional[PositiveInt] = None
    use_vae_encoder: Optional[bool] = None
