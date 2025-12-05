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
from annotated_types import Ge, Gt, Le
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


class BaseTransformNoVSL(BaseTransform):
    use_vsl: bool = False

    @abstractmethod
    def __call__(
        self, sample_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Returns a transformed sample dictionary with
        the same keys and value types."""
        raise NotImplementedError

    @field_validator("use_vsl")
    @classmethod
    def validate_use_vsl(cls, use_vsl):
        if use_vsl:
            raise NotImplementedError(
                f"Data transform {cls.__name__} is not "
                "currently supported with Variable sequence length (VSL) training."
            )
        return use_vsl


class AddSinkTokensTransform(BaseTransformNoVSL):
    """Insert auxiliary tokens into the input sequence
    that act as attention sinks for KV cache during inference,
    according to the StreamingLLM paper: https://arxiv.org/abs/2309.17453"""

    name: Literal["add_sink_tokens"]

    sink_token_id: Annotated[int, Ge(0)] = 0
    num_sink_tokens: Annotated[int, Ge(1)] = 1
    sink_token_start_pos: Annotated[int, Ge(0)] = 0
    """Insert `num_sink_tokens` tokens with `sink_token_id` at position
    `sink_token_start_pos` for all samples in the batch."""
    truncate_to_original_length: bool = False
    """Truncate the batch to have the same sequece length
    as before transformation."""

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


class AddMemoryTokensTransform(BaseTransformNoVSL):
    """Insert auxiliary memory tokens into the input sequence
    that are used to compress chunks of input context,
    according to the Activation Beacon paper: https://arxiv.org/abs/2401.03462
    """

    name: Literal["add_memory_tokens"]

    attn_memory_chunk_size: Annotated[int, Ge(1)] = ...
    num_memory_tokens_per_chunk: Annotated[int, Ge(1)] = ...
    memory_token_id: Annotated[int, Ge(0)] = 0
    return_masks: bool = False

    def __call__(
        self, sample_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        input_ids = sample_dict["input_ids"]
        labels = sample_dict["labels"]
        batch_size, seq_len = input_ids.shape
        chunk_size = self.attn_memory_chunk_size
        num_memory_toks = self.num_memory_tokens_per_chunk

        subsequence = [self.memory_token_id] * num_memory_toks
        batch_size, seq_len = input_ids.shape
        memtok_ids_tensor = torch.tensor(
            subsequence, dtype=input_ids.dtype, device=input_ids.device
        )

        output_chunks = []
        output_chunks_labels = []
        i = 0
        while i < seq_len:
            chunk = input_ids[:, i : i + chunk_size]
            chunk_labels = labels[:, i : i + chunk_size]
            output_chunks.append(chunk)
            output_chunks_labels.append(chunk_labels)

            i += chunk_size
            if i < seq_len:
                memtok_ids_batch = memtok_ids_tensor.unsqueeze(0).expand(
                    batch_size, -1
                )
                output_chunks.append(memtok_ids_batch)
                output_chunks_labels.append(memtok_ids_batch)

        new_input_ids = torch.cat(output_chunks, dim=1)
        new_labels = torch.cat(output_chunks_labels, dim=1)

        num_blocks = (seq_len + chunk_size - 1) // chunk_size

        # 'chunk_size' regular tokens are followed by
        # 'num_memory_toks' memory tokens
        mask_pattern = torch.tensor(
            [False] * chunk_size + [True] * num_memory_toks, dtype=torch.bool
        )
        mask = mask_pattern.repeat(num_blocks)
        mask = mask[: new_input_ids.shape[-1]].broadcast_to((batch_size, -1))

        mapped_values = {
            "input_ids": new_input_ids,
            "labels": new_labels,
            "attention_mask": (~mask).to(dtype=input_ids.dtype),
        }
        if self.return_masks:
            mapped_values["special_token_meta"] = {"memory_token_mask": mask}
        sample_dict.update(mapped_values)
        return sample_dict


class AddPoSETransform(BaseTransformNoVSL):
    """Position ID shifting to simulate longer sequences,
    based on the PoSE paper https://arxiv.org/pdf/2309.10400"""

    name: Literal["add_pose_shift"]
    segment_length: Annotated[int, Ge(1)] = ...
    target_seq_len: Annotated[int, Ge(1)] = ...
    # Only `ratio_shifted_samples` of samples in the batch are position-shifted
    ratio_shifted_samples: Annotated[float, Gt(0), Le(1)] = 0.5

    def __call__(
        self, sample_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = sample_dict["input_ids"].shape
        position_ids = torch.arange(seq_len).broadcast_to((batch_size, -1))

        if self.target_seq_len <= seq_len:
            raise DataTransformConfigError(
                f"Invalid target_seq_len: {self.target_seq_len}. "
                f"Expected a value greater than the sequence length: {seq_len}"
            )

        num_segments = (
            seq_len + self.segment_length - 1
        ) // self.segment_length
        biases = torch.zeros((batch_size, seq_len), dtype=torch.int64)
        segment_biases = torch.zeros(
            (batch_size, num_segments), dtype=torch.int64
        )

        max_bias = self.target_seq_len - seq_len
        segment_biases[:, 0] = 0
        for i in range(1, num_segments):
            for b in range(batch_size):
                segment_biases[b, i] = torch.randint(
                    segment_biases[b, i - 1] + 1, max_bias, (1,)
                ).item()

        num_shifted_samples = int(self.ratio_shifted_samples * batch_size)
        shifted_samples = torch.randperm(batch_size)[:num_shifted_samples]
        for i in range(num_segments):
            start = i * self.segment_length
            end = min((i + 1) * self.segment_length, seq_len)
            for b in range(batch_size):
                # Only a subset of samples are position-shifted
                if b in shifted_samples:
                    biases[b, start:end] = segment_biases[b, i]
                else:
                    biases[b, start:end] = 0

        # Shift the original tensor by the biases
        shifted_position_ids = position_ids + biases

        mapped_values = {
            "position_ids": shifted_position_ids,
        }

        sample_dict.update(mapped_values)
        return sample_dict


DataTransform = Annotated[
    Union[
        AddSinkTokensTransform,
        AddMemoryTokensTransform,
        AddPoSETransform,
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
