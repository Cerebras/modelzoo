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

"""HuggingFace Eli5 Dataset"""

import os

from datasets import load_dataset
from transformers import AutoTokenizer

# Suppress warnings about using fast tokenizers
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def HuggingFace_Eli5(split="train", num_workers=8, sequence_length=128):
    from cerebras.modelzoo.data_preparation.huggingface.CSDataCollatorForLanguageModeling import (
        CSDataCollatorForLanguageModeling,
    )

    # based on https://huggingface.co/docs/transformers/tasks/language_modeling
    eli5_dataset_path = None
    if eli5_dataset_path is None:
        raise ValueError(
            "Please set `eli5_dataset_path` to a location containing the Eli5 dataset."
        )
    eli5 = load_dataset(eli5_dataset_path, split="train[:5000]")

    eli5 = eli5.train_test_split(test_size=0.2, seed=0)
    eli5 = eli5[split]  # Select dataset split
    eli5 = eli5.flatten()

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
    tokenizer.add_bos_token = (
        False  # BOS token added in CSDataCollatorForLanguageModeling
    )

    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=eli5.column_names,
    )

    block_size = sequence_length

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_eli5.map(
        group_texts, batched=True, num_proc=num_workers
    )

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = CSDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    return dataset, data_collator
