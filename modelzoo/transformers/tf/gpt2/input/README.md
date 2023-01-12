# GPT-2/3 pre-training data

This document describes the process of preparing the input data that will be used for pre-training the GPT-2 and GPT-3 models. In order to speed up training, we process the dataset into TFRecords. During training, the generated TFRecords are read by the `GptTfRecordsProcessor` and fed to the model. The steps shown here are for OpenWebText dataset, but you can follow the same steps for preparing other datasets.

We provide the data preparation scripts for Pile and OpenWebText datasets.

# Pile dataset

For pile dataset, refer to the [Pile Dataset README](../../../data_processing/scripts/pile/README.md).

# OpenWebText dataset

## Download OpenWebText dataset

1. Download and extract data for the [OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/) as described in [Data download and extraction](../../../data_processing/scripts/owt/README.md#data-download-and-extraction).
2. Download encoder and vocab files by running the next commands:
    ```bash
    mkdir -p bpe
    wget -O bpe/encoder.json https://openaipublic.blob.core.windows.net/gpt-2/models/774M/encoder.json
    wget -O bpe/vocab.bpe https://openaipublic.blob.core.windows.net/gpt-2/models/774M/vocab.bpe
    ```

## Allocate subsets for training and validation

In the next step, you will create two subsets of extracted txt files, one for training and the second one for validation. These two subsets are then used to create TFRecords that will be used for pre-training.

> **IMPORTANT**: The training and validation subsets must contain mutually exclusive txt files.

Proceed as follows:

Define metadata files that contain paths to subsets of documents in the `openwebtext` folder to be used for training or validation.

For training, in this tutorial we use a subset of 512,000 documents. The associated metadata file can be found in `metadata/train_512k.txt`.

For validation, we choose 5,000 documents that are not in the training set. The metadata file for validation can be found in `metadata/val_files.txt`.

>**NOTE**: You are free to create your own metadata files that define your train and validation data subsets, with your preferred content and sizes. You can also create a data subset for test.

The following plain text files need to be prepared before generating the TFRecords:

-   Metadata file
    -   Contains file paths for flat text cleaned data files (one data file per line).
    -   Examples of Metadata files can be found in the folder [metadata](../../data_processing/scripts/owt/metadata/).
-   Data files
    -   Contains cleaned plain text. Each file has one paragraph per line and are separated by an empty line as follows:
    ```
    <paragraph-1>

    <paragraph-2>
    ```
-   Vocab file
    -   Contains a pair of symbols per line (symbols being variable-length strings). The two symbols are separated by a white-space.
    -   Needs to be compatible with the raw text data (e.g. same language).
    -   An example of vocab file can be downloaded from [vocab.bpe](https://openaipublic.blob.core.windows.net/gpt-2/models/774M/vocab.bpe).
-   Encoder file
    -   Is a json file that has string symbols as keys and their IDs as values. It is used to map from BPE tokens to token IDs.
    -   Should have an ID for each string symbol in the vocab file.
    -   An example of encoder file can be downloaded from [encoder.josn](https://openaipublic.blob.core.windows.net/gpt-2/models/774M/encoder.json).


## Generate TFRecords

This script `create_owt_tfrecords.py` generates TFRecords for pre-training a GPT-2/3 model.

### Description

A high level overview of the imlpementation is as follows:

1. Given a list of raw text documents, generate the tokens using BPE and create an input sequence of tokens based on `max_seq_length` and `overlap_size`. If `add_special_tokens` is set `True`, then we add the special token `<|endoftext|>` in the beginning, between, and at the end of documents as follows:

        <|endoftext|> <doc-1> <|endoftext|> <doc-2> <|endoftext|>

2. Map tokens to their IDs based on the encoder file.
4. Pad the IDs input sequence to `max_seq_length` with the ID of `<|endoftext|>` (if less than `max_seq_length`).
5. Create the feature dictionary that will be serialized into TFRecords with the features described in [Table 1](#table-1-data-features-in-the-generated-tfrecords).

#### Table 1: Data features in the generated TFRecords
Feature name | Data type | Sequence length | Description
--- | --- | --- | ---
`input_ids` | `tf.int64` | `max_seq_length` | Input token IDs.
`input_mask` | `tf.int64` | `max_seq_length` | Mask for padded positions (has values `0` on the padded positions, and `1` elsewhere).
`labels` | `tf.int64` | `max_seq_length` | Labels for Language Modeling (LM) pre-training task.

The TFRecords generated from this script are used during pre-training by the `GptTfRecordsProcessor` class in [GptTfRecordsProcessor.py](./GptTfRecordsProcessor.py). For more details, refer to [create_owt_tfrecords.py](./create_owt_tfrecords.py).


### Running the generation

Run `create_owt_tfrecords.py` with this command:

```bash
python create_owt_tfrecords.py --metadata_files ../../../data_processing/scripts/owt/metadata/train_512k.txt --vocab_file bpe/vocab.bpe --encoder_file bpe/encoder.json --max_seq_length 128 --output_dir train_512k_msl128
```

Full usage:

```bash
usage: create_owt_tfrecords.py [-h] --metadata_files METADATA_FILES --vocab_file
                           VOCAB_FILE --encoder_file ENCODER_FILE
                           [--max_seq_length MAX_SEQ_LENGTH]
                           [--short_seq_prob SHORT_SEQ_PROB]
                           [--add_special_tokens ADD_SPECIAL_TOKENS]
                           [--overlap_size OVERLAP_SIZE]
                           [--output_dir OUTPUT_DIR]
                           [--num_output_files NUM_OUTPUT_FILES] [--name NAME]
                           [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit.
  --metadata_files METADATA_FILES
                        Path to the text file containing a list of file names
                        corresponding to the raw input documents to be
                        processed and stored; Multiple metadata
                        files must be separated by a comma.
  --vocab_file VOCAB_FILE
                        Path to the vocabulary file.
  --encoder_file ENCODER_FILE
                        Path to the BPE encoder file.
  --max_seq_length MAX_SEQ_LENGTH
                        Maximum sequence length (default: 128).
  --short_seq_prob SHORT_SEQ_PROB
                        Probability of creating sequences that are
                        shorter than the maximum sequence length
                        (default: 0.1).
  --add_special_tokens ADD_SPECIAL_TOKENS
                        Add '<|endoftext|>' token at the end, beginning, and between documents.
                        (default: True).
  --overlap_size OVERLAP_SIZE
                        The overlap size between tokens of the current and previous
                        example. Defaults to None, which sets the overlap to
                        max_seq_len / 4 (default: None).
  --output_dir OUTPUT_DIR
                        Directory where TFRecords will be stored.
                        (default: ./tfrecords/).
  --num_output_files NUM_OUTPUT_FILES
                        TFRecords will be separated into the
                        specified number of files on disk. The larger the number of
                        files, the easier it becomes to parallelize writing/reading of
                        the TFRecords (default: 10).
  --name NAME           Name of the dataset, i.e., prefix to use
                        for TFRecord names (default: "examples").
  --seed SEED           Seed for the random number generators (default: 0).
```
