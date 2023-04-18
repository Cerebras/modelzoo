# Introduction

[The Pile](https://arxiv.org/abs/2101.00027) [1] is a dataset of diverse text for language modeling. It is constructed from 22 diverse high-quality subsets, both existing and newly constructed, many of which derive from academic or professional sources.

This directory contains scripts that provide a workflow from downloading the raw data to processing them as TFRecords or HDF5, to be used by the data pipelines of GPT style (auto-regressive) models for training and validation.

The output dataset can be used by models with an autoregressive language modeling task like GPT-Style models (GPT2, GPT3, GPTJ, GPT-Neox).

## Prerequisites

This script requires at least python version `3.7`.

### Environment Setup

The following pre-requisites are needed to enable a clean run of the script. Below is a setup for a conda environment:

```bash
conda create --name data_env python=3.7.4 pip -y
conda activate data_env

conda install -c conda-forge cudatoolkit=10.1 pyyaml regex -y
conda install -c anaconda cudnn=7.6.4 tqdm -y
pip install tensorflow-gpu==2.2.0
pip install lm-dataformat ftfy
pip install tokenizers h5py
pip install protobuf==3.20.3
```

During the environment setup, if you encounter errors like "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...", please ignore the message as it shouldn't affect the rest of the steps.

## Input data

### Data download

The raw data for Pile is available at the [eye.ai website](https://mystic.the-eye.eu/public/AI/pile/). To download the data locally, you can use `download.py`, the arguments for which are detailed below:

```bash
usage: download.py [-h] --data_dir DATA_DIR [--name NAME] [--debug]

Download the raw Pile data and associated vocabulary for pre-processing.

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Base directory where raw data is to be downloaded.
  --name NAME          Sub-directory where raw data is to be downloaded.
                       Defaults to `pile`.
  --debug              Checks if a given split exists in remote location.
```

This file automatically loops over the train, validation and test splits and downloads them to the given output folder. It also downloads the vocabulary files for two different tokenization mechanisms (one based on GPT2[4] and another based on GPT-NeoX[2, 3]).

### Download Notes

- The `train` split in particular is very large, and it takes approximately 15 hours @ 10MB/s to download. It also needs at least 500GB of space for storage.
- There is an additional `debug` flag, which lets you test if the remote files to download exist or not. To use this, pass an additional argument `--debug`.

### Some Metrics of Downloaded Files

All files are of `jsonl.zst` format.

- train: 30 compressed files, each with ~15GB.
- val: 1 compressed file with ~450MB.
- test: 1 compressed file with ~450MB.

### Input files format

If you want to use your own dataset, you need to process your dataset into jsonl (or jsonl.zst) format for the `create_dataset.py` script to consume.

ex.
```
{"text": "Text of the first doc"}
{"text": "Text of the second doc"}
...
```

**Note**: Based on the setup of lm_dataformat package, you will have to put the text portion under a json record with the key `"text"` (content under other keys is ignored), where each line contains a single record. If the file is large, you can compress it into a zst (.jsonl.zst) or tar (.jsonl.tar) format.

### Definition of vocab_file and encoder_file

We support two different tokenizers with this script, 1. `GPT2Tokenizer`, 2. `NeoXTokenizer`. We need to supply correct `vocab_file` and `encoder_file` when using the desired tokenizer.

- For `GPT2Tokenizer`, `vocab_file=gpt2-vocab.bpe` and `encoder_file=gpt2-encoder.json`.
- For `NeoXTokenizer`, `encoder_file=/neox-encoder.json`.

These files can be found [here](../../../vocab/).

**Note:** For `GPT2Tokenizer` we follow the nomenclature used by OpenAI in their [implementation](https://github.com/openai/gpt-2/blob/master/src/encoder.py#L109-L112) which is slightly different from Hugging Face's nomenclature where they call the `vocab_file` as `merges_file` and `encoder_file` as `vocab_file`. However, the content of the files are the same. For `NeoXTokenizer`, we use the same nomenclature to avoid confusion.

## Running the script

### Generating TFRecords/HDF5 files

Once the dataset is downloaded, you can generate TFRecords or HDF5 files using the `create_dataset.py` file, the arguments for this file are detailed below:

```bash
usage: create_dataset.py [-h] --file_format {tfrecords,HDF5}
                         --input_dir INPUT_DIR 
                         --tokenizer_type {GPT2Tokenizer,NeoXTokenizer} 
                         --vocab_file VOCAB_FILE
                         [--encoder_file ENCODER_FILE]
                         [--max_seq_length MAX_SEQ_LENGTH]
                         [--short_seq_prob SHORT_SEQ_PROB]
                         [--output_dir OUTPUT_DIR] [--output_name OUTPUT_NAME]
                         [--seed SEED] [--processes PROCESSES] [--ftfy]
                         [--ftfy_normalizer {NFC,NFKC,None}]
                         [--wikitext-detokenize] [--write_remainder]
                         [--resume_from_checkpoint] [--display_pbar]
                         [--eos_id EOS_ID] [--pad_id PAD_ID]
                         [--files_per_record FILES_PER_RECORD]
                         [--write_in_batch]

Process the raw Pile dataset as TfRecords/HDF5.

optional arguments:
  -h, --help            show this help message and exit
  --file_format {tfrecords,HDF5}         
                        What format to use for the file, currently only 
                        `tfrecords` or `HDF5` are supported.
  --input_dir INPUT_DIR
                        directory where raw data is stored in the Download phase.
  --tokenizer_type {GPT2Tokenizer,NeoXTokenizer}
                        type of tokenizer to use for tfrecord/HDF5 dataset generation.
                        Can be one of `GPT2Tokenizer` or `NeoXTokenizer`.
  --vocab_file VOCAB_FILE
                        path to the vocabulary file. Defaults to None.
  --encoder_file ENCODER_FILE
                        Path to the encoder file. Defaults to None.
  --max_seq_length MAX_SEQ_LENGTH
                        maximum sequence length. Defaults to 2048.
  --short_seq_prob SHORT_SEQ_PROB
                        probability of creating sequences which are shorter
                        than the maximum sequence length. Defaults to 0.0
  --output_dir OUTPUT_DIR
                        directory where TFRecords/HDF5 files will be stored. 
                        Defaults to `./data_dir/`.
  --output_name OUTPUT_NAME
                        name of the dataset; i.e. prefix to use for 
                        TFRecord/HDF5 file names. Defaults to `examples`.
  --seed SEED           random seed. Defaults to `0`.
  --processes PROCESSES
                        Number of processes to use. Default to cpu count.
                        Note: this number has to be <= number of (compressed) input files,
                        otherwise it returns an error because it cannot divide files across
                        processes properly.
  --ftfy                Fix text with ftfy. Defaults to False.
  --ftfy_normalizer {NFC,NFKC,None}
                        choose what kind of unicode normalization is applied.
                        Usually, we apply `NFC` normalization, so that letters
                        followed by combining characters become single
                        combined characters. Changing this to `NFKC` applies
                        more compatibility conversions. Using `None` applies
                        no normalization while fixing text. This argument only
                        works when `--ftfy` is set to True. Defaults to `NFC`.
  --wikitext-detokenize
                        use wikitext detokenizer to fix text. Defaults to False.
  --write_remainder     write the remainder files when data is left over from
                        processing. Defaults to False.
  --resume_from_checkpoint
                        resume record writing from a given checkpoint. Defaults to False.
  --display_pbar        display progress while runs. Defaults to False.
  --eos_id EOS_ID       id for padding out shorter sequences. Defaults to
                        50256, which is `<|endoftext|>` in tokens.
  --pad_id PAD_ID       id for padding out shorter sequences. Defaults to
                        50256, which is `<|endoftext|>` in tokens.
  --files_per_record FILES_PER_RECORD
                        Number of samples with max_sequence_length to write per output tfrecord/HDF5 file. Defaults to 50000.
  --write_in_batch      Whether to write data samples in batch for the HDF5 format.
                        Batch writing is slightly faster but not memory efficient.
                        It is recommended to set this to False when files_per_record is large.
                        Defaults to False.
```

### Generation Notes

- Since the PILE dataset contains a lot of diverse datasets, it is recommended to use the ftfy module to fix the datasets. This can be enabled by the `--ftfy` argument.
- The NeoXTokenizer uses the HuggingFace library's inbuilt tokenizer and handles NFC normalization on its own. When using this tokenizer_type, it is recommended to set the `--ftfy_normalizer` argument to `None`. For the `GPT2Tokenizer`, use the default `NFC` value for the normalizer.
- Using `NFKC` normalization is not suggested, since it does compatibility conversions, such as replacing the 'micro sign' with a standard Greek lowercase mu, which looks identical. Some normalizations here change the meaning of text as well, such as converting '10<sup>3</sup>' to '103'. We are providing the option for this, since the original GPT-Neo model used this type of normalization.
- To process TFRecords or HDF5 for training, we recommend using multi-processing. Since there are 30 files, one can execute the command in a single shot, using 30 parallel processes. However, this requires a high-spec CPU server, which can handle not only the concurrent running processes in RAM but also the I/O for reads and writes. If the I/O of the server is slow, the processes can appear to be hung for a very long while.
- Another suggestion on how to run preprocessing on train set is proposed under the following section `Generation Examples and Guidance`. The recommendation is to split the data into smaller subsets and write out each subset. One can then mix all TFRecords/HDF5 in a common folder for use by the data pipeline, or just provide the locations of each subset in a list. The overall time to write out TFRecords/HDF5 can depend on the CPU server used.

### Generation Examples and Guidance
This is a sample command we used internally to process the first split (the first 3 jsonl.zst files) of the train dataset of PILE into hdf5 formatted outputs of max_sequence_length=2048 and 50000 samples per file,

```
python create_dataset.py 
  --file_format HDF5
  --input_dir <INPUT_FILE_PATH>
  --tokenizer_type NeoXTokenizer
  --vocab_file <NEOX_VOCAB_FILE_PATH>
  --encoder_file None # neox-tokenizer doesn't require encoder_file
  --max_seq_length 2048
  --output_dir <OUTPUT_PATH>
  --seed 1
  --processes 3
  --ftfy
  --ftfy_normalizer None
  --write_remainder
  --display_pbar
  --eos_id 0
  --pad_id 0
  --files_per_record 50000
```
Note that in the above example, we used 3 processes because the train set is splited into 10 directories (3 compressed files each). We ended up with `324` examples files so less than `324 * 50000` = `16.2` million samples. \
If we want to process validation or test set, we can choose to use only 1 process because the validation/test set is relatively small (~`450MB`). We ended up with `4` examples files so less than `4 * 50000` = `200000` samples.
**Note**: To process the train set, we recommend to split the 30 training input files into ~10 folders, each with 3 files. Then start preprocessing by running 10 times the above command with 10 `--input_dir` and `--output_dir`, each with `--processes 3` (can use multiple machines if possible). \
This is because the train set is relatively large (15GB * 30 ~= 450-500GB in total) and will take a very long to process. By using the above method, it took about 15-20hours to preprocess one split depends on device and I/O speed.

## Output data

### Some Metrics of the generated hdf5/tfrecords files

We use max_sequence_length=2048 and files_per_record=50000 (50000 samples per output file), by modifying arguments like --max_sequence_length, --files_per_record, --ftfy, --write_remainder, etc. you may get different metrics for output files.

- train set: processed set contains ~3240 files. Processed files in hdf5 format each has ~160MB with a total ~530GB, tfrecords each has has ~480M with a total of ~1.5TB.
- val set: set contains 4 files with a total of ~550MB.
- test set: set contains 4 file with a total of ~550MB.

### Output directory structure

There are 2 types of files in the output directory:
1. the output data files with the name prefix same as the argument `--output_name`, defaulted to `examples`.
2. the data_params.json file where information on the arugments used during preprocessing including number of total examples, tokenizer used, etc. are logged.

A tree command on the generated directory of the first split of the preprocessed train dataset:
```
<preprocessed directory path>
├── data_params.json
├── examples_0_0.h5
├── examples_0_1.h5
├── examples_0_2.h5
├── ...
├── ...
├── examples_99_0.h5
├── examples_99_1.h5
└── examples_99_2.h5

0 directories, 325 files
```

### Output file data structure

`input_ids`:
Input token ids, padded with 0's to max_sequence_length.
Shape: [batch_size, max_sequence_length].
Type: int32

`input_mask`:
Mask for padded positions. Has 0's on padded positions and 1's elsewhere.
Shape: [batch_size, max_sequence_length]
Type: int32

`labels`:
Input token ids shifted to the left by 1 position.
Shape: [batch_size, max_sequence_length].
Type: int32

- hdf5: each file contains a key "data" whose value has shape (files_per_record, 3, max_sequence_length), where the above `input_ids`, `input_mask` and `labels` are stacked on dim=1. 
- tfrecords: each file contains `files_per_record` records, each record contains 3 feature keys: `input_ids`, `input_mask` and `labels`.

## References

1. The Pile: An 800GB Dataset of Diverse Text for Language Modeling, [arXiv 2021](https://arxiv.org/abs/2101.00027)
2. GPT-NeoX-20B: An Open-Source Autoregressive Language Model, [arXiv 2022](https://arxiv.org/abs/2204.06745)
3. [GPT-NeoX Github Repository](https://github.com/EleutherAI/gpt-neox)
4. Language Models are Unsupervised Multitask Learners, [2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
