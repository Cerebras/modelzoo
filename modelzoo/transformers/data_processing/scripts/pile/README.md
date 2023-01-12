# Pile Pre-processing for GPT Models

[The Pile](https://arxiv.org/abs/2101.00027) [1] is a dataset of diverse text for language modeling. It is constructed from 22 diverse high-quality subsets, both existing and newly constructed, many of which derive from academic or professional sources.

This directory contains scripts that provide a workflow from downloading the raw data to processing them as TFRecords or HDF5, to be used by the data pipelines of GPT style (auto-regressive) models for training and validation.

## Environment Setup

The following pre-requisites are needed to enable a clean run of the script. Below is a setup for a conda environment:

```bash
conda create --name data_env python=3.7.4 -y
conda activate data_env

conda install -c conda-forge cudatoolkit=10.1 pyyaml regex -y
conda install -c anaconda cudnn=7.6.4 tqdm -y
conda install pip -y
pip install tensorflow-gpu==2.2.0
pip install lm-dataformat ftfy
pip install protobuf==3.20.3
```

During the environment setup, if you encounter errors like "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...", please ignore the message as it shouldn't affect the rest of the steps.

## Downloading the Data

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

## Generating TFRecords/HDF5 files

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
                        directory where raw data is stored.
  --tokenizer_type {GPT2Tokenizer,NeoXTokenizer}
                        type of tokenizer to use for tfrecord/HDF5 dataset generation.
                        Can be one of `GPT2Tokenizer` or `NeoXTokenizer`.
  --vocab_file VOCAB_FILE
                        path to vocabulary.
  --encoder_file ENCODER_FILE
                        path to BPE encoder.
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
  --ftfy                Fix text with ftfy.
  --ftfy_normalizer {NFC,NFKC,None}
                        choose what kind of unicode normalization is applied.
                        Usually, we apply `NFC` normalization, so that letters
                        followed by combining characters become single
                        combined characters. Changing this to `NFKC` applies
                        more compatibility conversions. Using `None` applies
                        no normalization while fixing text.
  --wikitext-detokenize
                        use wikitext detokenizer to fix text.
  --write_remainder     write the remainder files when data is left over from
                        processing.
  --resume_from_checkpoint
                        resume record writing from a given checkpoint
  --display_pbar        display progress while runs.
  --eos_id EOS_ID       id for padding out shorter sequnces. Defaults to
                        50256, which is `<|endoftext|>` in tokens.
  --pad_id PAD_ID       id for padding out shorter sequnces. Defaults to
                        50256, which is `<|endoftext|>` in tokens.
  --files_per_record FILES_PER_RECORD
                        Text files to write per tfrecord/HDF5 file.
  --write_in_batch      Whether to write data samples in batch for the HDF5 format.
                        Batch writing is slightly faster but not memory efficient.
```

### Generation Notes

- Since the PILE dataset contains a lot of diverse datasets, it is recommended to use the ftfy module to fix the datasets. This can be enabled by the `--ftfy` argument.
- The NeoXTokenizer uses the HuggingFace libarary's inbuilt tokenizer and handles NFC normalization on its own. When using this tokenizer_type, it is recommended to set the `--ftfy_normalizer` argument to `None`. For the `GPT2Tokenizer`, use the default `NFC` value for the nomalizer.
- Using `NFKC` normalization is not suggested, since it does compatibility conversions, such as replacing the 'micro sign' with a standard Greek lowercase mu, which looks identical. Some normalizations here change the meaning of text as well, such as converting '10<sup>3</sup>' to '103'. We are providing the option for this, since the original GPT-Neo model used this type of normalization.
- To process TFRecords or HDF5 for training, we recommend using multi-processing. Since there are 30 files, one can execute the command in a single shot, using 30 parallel processes. However, this requires a high-spec CPU server, which can handle not only the concurrent running processes in RAM but also the I/O for reads and writes. If the I/O of the server is slow, the processes can appear to be hung for a very long while.
- The recommendation is to split the data into smaller subsets and write out each subset. One can then mix all TFRecords/HDF5 in a common folder for use by the data pipeline, or just provide the locations of each subset in a list. The overall time to write out TFRecords/HDF5 can depend on the CPU server used.

## References

1. The Pile: An 800GB Dataset of Diverse Text for Language Modeling, [arXiv 2021](https://arxiv.org/abs/2101.00027)
2. GPT-NeoX-20B: An Open-Source Autoregressive Language Model, [arXiv 2022](https://arxiv.org/abs/2204.06745)
3. [GPT-NeoX Github Repository](https://github.com/EleutherAI/gpt-neox)
4. Language Models are Unsupervised Multitask Learners, [2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
