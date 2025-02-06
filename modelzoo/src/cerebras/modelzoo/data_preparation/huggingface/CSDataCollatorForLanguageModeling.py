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

# Based on https://github.com/huggingface/transformers/blob/04ab5605fbb4ef207b10bf2772d88c53fc242e83/src/transformers/data/data_collator.py#L607
# Cerebras LM models expect the labels to be shifted in the dataloader,
# so we need to customize the implementation of DataCollatorForLanguageModeling

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling


def _torch_collate_batch(
    examples, tokenizer, pad_to_multiple_of: Optional[int] = None
):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(
        x.size(0) == length_of_first for x in examples
    )
    if are_tensors_same_length and (
        pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
    ):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (
        max_length % pad_to_multiple_of != 0
    ):
        max_length = (
            (max_length // pad_to_multiple_of) + 1
        ) * pad_to_multiple_of
    result = examples[0].new_full(
        [len(examples), max_length], tokenizer.pad_token_id
    )
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


class CSDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    """
    Overrides DataCollatorForLanguageModeling from HF to shift the inputs/labels in the dataloader
    """

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples,
                    self.tokenizer,
                    pad_to_multiple_of=self.pad_to_multiple_of,
                )
            }

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

            ####### Cerebras LM models expect the labels to be shifted in the dataloader #####
            batch_size = batch["input_ids"].shape[0]
            batch["input_ids"] = torch.cat(
                (
                    torch.full(
                        [batch_size, 1],
                        self.tokenizer.bos_token_id,
                        dtype=batch["input_ids"].dtype,
                    ),
                    batch["input_ids"][:, :-1],
                ),
                dim=1,
            )
            # Cerebras kernels accept torch.int32 inputs
            for key in batch.keys():
                batch[key] = batch[key].to(dtype=torch.int32)

        if not isinstance(batch, dict):
            batch = dict(batch.items())
        return batch
