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

"""HuggingFace BookCorpus Dataset"""

import os

from datasets import load_dataset
from transformers import AutoTokenizer

from cerebras.modelzoo.data_preparation.huggingface.CSDataCollatorForLanguageModeling import (
    CSDataCollatorForLanguageModeling,
)

# Suppress warnings about using fast tokenizers
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def HuggingFace_BookCorpus(split="train", num_workers=8, sequence_length=2048):
    bookcorpus = load_dataset("bookcorpus", split=split)

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.add_bos_token = (
        False  # BOS token added in CSDataCollatorForLanguageModeling
    )

    def preprocess_function(examples):
        return tokenizer(
            [" ".join(x) for x in examples["text"]],
            truncation=True,
            max_length=2048,
        )

    tokenized_bookcorpus = bookcorpus.map(
        preprocess_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"],
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: sum(examples[k], []) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length:
            total_length = (total_length // sequence_length) * sequence_length
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length]
                for i in range(0, total_length, sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_bookcorpus.map(
        group_texts, batched=True, num_proc=num_workers
    )

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = CSDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    return dataset, data_collator
