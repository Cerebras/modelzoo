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
import os

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# Maximum sequence length
MSL = 1026


def preprocess_data(examples, tokenizer, max_length=MSL):
    """
    Preprocesses the data by tokenizing the sequences and adding the labels.

    Args:
        examples (dict): A dictionary containing the sequences and labels.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding.
        max_length (int): The maximum length of the sequences. Default is MSL.

    Returns:
        dict: A dictionary containing the tokenized sequences and labels.
    """
    text = examples["Sequence"]
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    encoding["labels"] = examples["Membrane"]
    return encoding


def convert_to_comma_separated_strings(examples):
    """
    Converts ID lists to comma-separated strings to comply with `BertCSVDataProcessor`.

    Args:
        examples (dict): A dictionary containing the input IDs and attention masks.

    Returns:
        dict: A dictionary with input IDs and attention masks as comma-separated strings.
    """
    examples["input_ids"] = [
        f"[{', '.join(map(str, ids))}]" for ids in examples["input_ids"]
    ]
    examples["attention_mask"] = [
        f"[{', '.join(map(str, mask))}]" for mask in examples["attention_mask"]
    ]
    return examples


def main(train_data_path, test_data_path):
    """
    Main function to load, preprocess, and save the dataset.

    Args:
        train_data_path (str): Path to save the processed training data.
        test_data_path (str): Path to save the processed test data.
    """
    # Load and preprocess dataset
    df = pd.read_csv(
        "https://services.healthtech.dtu.dk/services/DeepLoc-2.0/data/Swissprot_Train_Validation_dataset.csv"
    ).drop(["Unnamed: 0", "Partition"], axis=1)
    df["Membrane"] = df["Membrane"].astype("int32")

    # Filter for sequences between 100 and 1026 amino acids
    df = df[df["Sequence"].apply(lambda x: len(x)).between(100, MSL)]

    # Remove unnecessary features
    df = df[["Sequence", "Kingdom", "Membrane"]]

    # Convert pandas DataFrame to Hugging Face Dataset and split into training and test sets
    dataset = Dataset.from_pandas(df).train_test_split(
        test_size=0.2, shuffle=True
    )

    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Apply preprocessing to the dataset
    encoded_dataset = dataset.map(
        lambda examples: preprocess_data(examples, tokenizer),
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset["train"].column_names,
    )

    # Convert tokenized sequences to comma-separated strings
    encoded_dataset = encoded_dataset.map(
        convert_to_comma_separated_strings, batched=True
    )

    # Save the processed datasets to CSV files
    encoded_dataset["train"].to_csv(train_data_path)
    encoded_dataset["test"].to_csv(test_data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and save DeepLoc-2 dataset."
    )
    parser.add_argument(
        "--output_train_csv_path",
        type=str,
        help="Output training data path (in CSV format).",
    )
    parser.add_argument(
        "--output_test_csv_path",
        type=str,
        help="Output test data path (in CSV format).",
    )

    args = parser.parse_args()
    main(args.output_train_csv_path, args.output_test_csv_path)
