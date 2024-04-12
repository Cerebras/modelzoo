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

import argparse

import torch
from datasets import load_dataset
from transformers import DPRContextEncoderTokenizer

from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.convert_dataset_to_HDF5 import (
    convert_dataset_to_HDF5,
)


def HuggingFace_WikiDPR(cache_dir, hf_model_dir, sequence_length, is_toy):
    split = "train"  # there is only train set

    wiki_dpr_dataset = load_dataset(
        "wiki_dpr",
        "psgs_w100.multiset.no_index.no_embeddings",
        split=split,
        cache_dir=cache_dir,
        download_mode="force_redownload",
        streaming=True,
    )
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(hf_model_dir)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=sequence_length,
        )

    dataset = wiki_dpr_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "title"],
    )
    if is_toy:
        dataset = dataset.take(50)

    def collate_fn(batch):
        def dict_to_tensor(dict, key, cast_fn=None):
            return torch.tensor(
                [cast_fn(x[key]) if cast_fn else x[key] for x in batch]
            )

        # we don't include position-ids as they will be inferred in the model
        out = {
            "input_ids": dict_to_tensor(batch, "input_ids"),
            "attention_mask": dict_to_tensor(batch, "attention_mask"),
            "segment_ids": dict_to_tensor(batch, "token_type_ids"),
            "id": dict_to_tensor(batch, "id", int).unsqueeze(
                1
            )  # out["id"] is shape [BS], and convert_dataset_to_HDF5 strips away
            # batch-dim. Thus we need to add another dim
        }
        return out

    return dataset, collate_fn


def main():
    """
    Sample usage: 
    python convert_dpr_dataset.py \
      --dataset_cache_dir /path/to/hf/cache/ \
      --hf_model_dir /path/to/tokenizer/ \
      --output_dir /path/to/output/dir/ \
      --toy_dataset
    """
    parser = argparse.ArgumentParser(
        description="Process Huggingface Wiki-DPR data into HDF5 format for CS."
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        required=True,
        help="Cache directory for storing or downloading HuggingFace datasets.",
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Directory containing HF model and tokenizer checkpoint.",
    )
    parser.add_argument(
        "--seq_len", type=int, default=512, help="Input sequence length."
    )
    parser.add_argument(
        "--toy_dataset",
        action="store_true",
        help="If flag is set, script will create only 50 samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./hdf5_wiki_dpr/",
        help="Output directory path.",
    )
    args = parser.parse_args()

    dataset, collate_fn = HuggingFace_WikiDPR(
        cache_dir=args.dataset_cache_dir,
        hf_model_dir=args.hf_model_dir,
        sequence_length=args.seq_len,
        is_toy=args.toy_dataset,
    )
    samples_per_file = 5 if args.toy_dataset else 2000
    batch_size = 5 if args.toy_dataset else 64
    convert_dataset_to_HDF5(
        dataset=dataset,
        data_collator=collate_fn,
        output_dir=args.output_dir,
        num_workers=1,  # this dataset is not sharded and only supports 1 worker
        samples_per_file=samples_per_file,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
