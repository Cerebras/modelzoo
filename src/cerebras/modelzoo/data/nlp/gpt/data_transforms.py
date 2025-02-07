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

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Literal, Union

import torch
from annotated_types import Ge
from pydantic import Field, TypeAdapter, field_validator
from typing_extensions import Annotated

from cerebras.modelzoo.config import NamedConfig


class DataTransformConfigError(ValueError):
    pass


class BaseTransform(NamedConfig, ABC):
    use_vsl: bool = False

    @abstractmethod
    def __call__(
        self, sample_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Returns a transformed sample dictionary with
        the same keys and value types."""
        raise NotImplementedError


class AddSinkTokensTransform(BaseTransform):
    """Adds a prefix of several sink tokens to the input_ids,
    labels and attention_mask."""

    name: Literal["add_sink_tokens"] = "add_sink_tokens"

    sink_token_id: Annotated[int, Ge(0)] = 0
    num_sink_tokens: Annotated[int, Ge(1)] = 1
    sink_token_start_pos: Annotated[int, Ge(0)] = 0
    """Insert `num_sink_tokens` tokens with `sink_token_id` at position
    `sink_token_start_pos` for all samples in the batch."""
    truncate_to_original_length: bool = False
    """Truncate the batch to have the same sequece length
    as before transformation."""

    @field_validator("use_vsl")
    @classmethod
    def validate_use_vsl(cls, use_vsl):
        if use_vsl:
            raise NotImplementedError(
                "Dynamically inserting sink tokens is not "
                "currently supported with Variable sequence length (VSL) training."
            )
        return use_vsl

    def __call__(
        self, sample_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Zeros in attention_mask are used to mask out sink tokens
        in loss calculation."""
        batch_size, seq_len = sample_dict["input_ids"].shape
        if self.sink_token_start_pos > seq_len:
            raise DataTransformConfigError(
                f"Invalid sink_token_start_pos: {self.sink_token_start_pos}. "
                f"Expected a value less than the sequence length: {seq_len}"
            )

        seq_prefix = torch.full(
            (batch_size, self.num_sink_tokens),
            self.sink_token_id,
            dtype=sample_dict["input_ids"].dtype,
        )
        mask_prefix = torch.full(
            (batch_size, self.num_sink_tokens),
            0,
            dtype=sample_dict["attention_mask"].dtype,
        )

        insertion_fn = lambda values, prefix, pos: torch.cat(
            (values[:, :pos], prefix, values[:, pos:]), dim=-1
        )
        if self.truncate_to_original_length:
            orig_insertion_fn = insertion_fn
            insertion_fn = lambda values, prefix, pos: orig_insertion_fn(
                values, prefix, pos
            )[:, :seq_len]

        mapped_values = {
            "input_ids": insertion_fn(
                sample_dict["input_ids"], seq_prefix, self.sink_token_start_pos
            ),
            "labels": insertion_fn(
                sample_dict["labels"], seq_prefix, self.sink_token_start_pos
            ),
            "attention_mask": insertion_fn(
                sample_dict["attention_mask"],
                mask_prefix,
                self.sink_token_start_pos,
            ),
        }
        sample_dict.update(mapped_values)
        return sample_dict


class AddMemoryTokensTransform(BaseTransform):
    name: Literal["add_memory_tokens"]

    attn_memory_chunk_size: Annotated[int, Ge(1)] = ...
    num_memory_tokens_per_chunk: Annotated[int, Ge(1)] = ...

    @field_validator("use_vsl")
    @classmethod
    def validate_use_vsl(cls, use_vsl):
        if use_vsl:
            raise NotImplementedError(
                "Dynamically inserting sink tokens is not "
                "currently supported with Variable sequence length (VSL) training."
            )
        return use_vsl

    def __call__(
        self, sample_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        batch_size, seq_len = sample_dict["input_ids"].shape
        chunk_size = self.attn_memory_chunk_size
        num_memory_toks = self.num_memory_tokens_per_chunk

        num_blocks = (seq_len + chunk_size - 1) // chunk_size
        block_size = chunk_size + num_memory_toks
        total_length = num_blocks * block_size

        # 'chunk_size' regular tokens are followed by
        # 'num_memory_toks' memory tokens
        mask_pattern = torch.tensor(
            [False] * chunk_size + [True] * num_memory_toks, dtype=torch.bool
        )
        mask = mask_pattern.repeat(num_blocks)
        regular_token_count = (~mask).cumsum(dim=0)
        mask = mask[: total_length - (regular_token_count > seq_len).sum()]
        memory_token_idx = torch.stack(mask.nonzero(as_tuple=True)).squeeze()
        regular_token_idx = torch.stack(
            (~mask).nonzero(as_tuple=True)
        ).squeeze()
        mapped_values = {
            "special_token_indices": {
                "memory_tokens": memory_token_idx.broadcast_to(
                    (batch_size, -1)
                ),
                "regular_tokens": regular_token_idx.broadcast_to(
                    (batch_size, -1)
                ),
            },
        }
        sample_dict.update(mapped_values)
        return sample_dict


DataTransform = Annotated[
    Union[
        AddSinkTokensTransform,
        AddMemoryTokensTransform,
    ],
    Field(discriminator="name"),
]


class ComposeTransform(BaseTransform):
    """Chains multiple transforms together."""

    name: Literal["ComposeTransform"]

    transforms: List[DataTransform]

    def __call__(
        self, sample_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for transform in self.transforms:
            sample_dict = transform(sample_dict)
        return sample_dict


def build_transforms(
    transforms_config: List[Dict], use_vsl: bool
) -> ComposeTransform:
    """Builds a chain of data transforms from a configuration dictionary."""
    return ComposeTransform(
        transforms=[
            transform.copy(update=dict(use_vsl=use_vsl))
            for transform in TypeAdapter(List[DataTransform]).validate_python(
                transforms_config
            )
        ]
    )


def wrap_collate_fn(
    collate_fn: Callable, transforms_config: List[Dict], use_vsl: bool
) -> Callable:
    """Wrap the collate function with a chain of data transforms."""
    if not isinstance(transforms_config, list) or not transforms_config:
        raise DataTransformConfigError(
            f"Invalid configuration for data transforms. "
            f"Expected a list, got {type(transforms_config)}"
        )
    transforms = build_transforms(transforms_config, use_vsl)

    def wrapped_map_fn(sample):
        sample = collate_fn(sample)
        return transforms(sample)

    return wrapped_map_fn
