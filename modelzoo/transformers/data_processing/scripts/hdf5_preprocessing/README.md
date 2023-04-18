# Creating HDF5 dataset for GPT Models

For efficient implementation of data loader for GPT style models, we provide a script to generate `.h5` files which can then be used in the input pipeline for GPT style models.

We provide two flavors of generating HDF5 files:

1. Store your text input stored in various extensions into `.h5` files with the raw text documents.
2. Convert yor text input stored in various extensions into `.h5` files with the documents tokenized and converted into token ids, along with labels and attention masks.

Each of the above conversion can be done by running the provided script `create_hdf5_dataset.py`. The script has two sub commands, `raw_text` and `preprocessed_text` to generate the `.h5` files in one of the above two described flavors. Each sub-commands takes in a set of arguments which are described below in [Generating HDF5 files](#generating-hdf5-files) section.

Before doing that, you need to setup a python virtual environment through [conda](https://www.anaconda.com/) as described below.

## Environment Setup

The following pre-requisites are needed to enable a clean run of the script. Below is a setup for a conda environment:

```bash
conda create --name data_env python=3.7.4 pip -y
conda activate data_env

conda install -c conda-forge cudatoolkit=10.1 pyyaml regex -y
conda install -c anaconda cudnn=7.6.4 tqdm -y
pip install lm-dataformat ftfy tokenizers h5py
```

During the environment setup, if you encounter errors like `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed...`, please ignore the message as it shouldn't affect the rest of the steps.

## Input files format

The input text documents need to be in s a specific file format before utilizing the provided script. The acceptable file formats are `'.jsonl', '.jsonl.zst', '.jsonl.zst.tar', '.txt'`. These files should have the data in a specific structure as described in [data format](#input-data-format) section.

To optimally process the files, it is recommended that all files with any of the above file formats other than `.txt` contain enough text in a single file. Recommended size for each file is in the order of GB.

On the contrary, if processing with smaller files with `txt` format, please input a `metadata` file containing a list of paths to these files to better leverage multi processing.

### Input data format

As mentioned above,  there are two primary types on input files accepted into the preprocessing script. They are either `json` based or `txt` based. For each type, the input data needs to follow certain structure in order to be converted into `hdf5` files accurately.

#### Format for `JSONL` files

The raw text and meta data for generation should be represented in the `jsonl` based files as:

```json
{"text": "Any text excerpt from the dataset of interest...\nThe new lines should be represented as a newline character.", 
"meta": {"info": "any other metadata dict"}}
```

For the `jsonl` files, as shown above, by default, the raw text is extracted from `key=text` in the input files. If your input files do not contain `text` key then you should know the `key` corresponding to the text you need to extract. Then, you can use the command line argument `--jsonl_key=<your key name>` to extract the text.

For example, if your jsonl files have the content as below:

```json
{"idea": "Any text excerpt from the dataset of interest, with custom key: 'idea'...\nThe new lines should be represented as a newline character.", 
"meta": {"info": "any other metadata dict"}}
```

then you'd need to pass `--jsonl_key=idea` in the command line arguments.

#### Format for `TXT` based files

The raw text for generation should be represented in the `txt` based files as:

```txt
Any text excerpt from the dataset of interest...
The new lines may not be represented as a newline character.
```

Note that in the above, there are no special tags or any anchors. If they exist, all of these will be treated as a single document and may not represent the natural language.

For example, the below text will be entirely tokenized as is:

```txt
<DOC>
<DOCNO>TRC2-2008-01-01-0000</DOCNO>
<BLOGS08DAY>-13</BLOGS08DAY>
<CONTENT>
Example content in the format that may be outside of the 
</DOCS>
```

If your input files do not follow the above structure, then the input script may generate erroneous data into the hdf5 files, affecting the quality of the fine-tuning.

### Definition of vocab_file and encoder_file

We support two different tokenizers with this script, 1. `GPT2Tokenizer`, 2. `NeoXTokenizer`. We need to supply correct `vocab_file` and `encoder_file` when using the desired tokenizer.

- For `GPT2Tokenizer`, `vocab_file=gpt2-vocab.bpe` and `encoder_file=gpt2-encoder.json`.
- For `NeoXTokenizer`, `encoder_file=/neox-encoder.json`.

These files can be found [here](../../../vocab/).

**Note:** For `GPT2Tokenizer` we follow the nomenclature used by OpenAI in their [implementation](https://github.com/openai/gpt-2/blob/master/src/encoder.py#L109-L112) which is slightly different from Hugging Face's nomenclature where they call the `vocab_file` as `merges_file` and `encoder_file` as `vocab_file`. However, the content of the files are the same. For `NeoXTokenizer`, we use the same nomenclature to avoid confusion.

## Generating HDF5 files

Once you have the text dataset that meets above requirement, you can generate HDF5 files using the `create_hdf5_dataset.py` file and providing one of the `raw_text` and `preprocessed_text` sub-commands.

The arguments for `raw_text` subcommand are detailed below:

```bash
usage: create_hdf5_dataset.py raw_text [-h] [--input_dir INPUT_DIR]
                                       [--metadata_files METADATA_FILES]
                                       [--jsonl_key JSONL_KEY]
                                       [--output_dir OUTPUT_DIR]
                                       [--output_name OUTPUT_NAME]
                                       [--seed SEED] [--processes PROCESSES]
                                       [--write_remainder]
                                       [--resume_from_checkpoint]
                                       [--display_pbar]
                                       [--files_per_record FILES_PER_RECORD]
                                       [--write_in_batch]

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Directory where raw data is stored.
  --metadata_files METADATA_FILES
                        Path to text file containing a list of file names
                        corresponding to the raw input documents to be
                        processed and stored; can handle multiple metadata
                        files separated by comma.
  --jsonl_key JSONL_KEY
                        The key name in input jsonl files from which the raw
                        text will be extracted in order to further process it. Default: "text".
  --output_dir OUTPUT_DIR
                        Directory where HDF5 files will be stored. Defaults to
                        `./data_dir/`.
  --output_name OUTPUT_NAME
                        Name of the dataset; i.e. prefix to use for HDF5 file
                        names.Defaults to `examples`.
  --seed SEED           Random seed. Defaults to `0`.
  --processes PROCESSES
                        Number of processes to use. Default to cpu count.
  --write_remainder     Write the remainder files when data is left over from
                        processing.
  --resume_from_checkpoint
                        Resume record writing from a given checkpoint.
  --display_pbar        Display progress while runs.
  --files_per_record FILES_PER_RECORD
                        Text files to write per HDF5 file.
  --write_in_batch      Whether to write the samples in batch for the HDF5
                        format, setting to false will save memory but a bit
                        slower.
```

The arguments for `preprocessed_text` subcommand are detailed below:

```bash
usage: create_hdf5_dataset.py preprocessed_text [-h] [--input_dir INPUT_DIR]
                                                [--metadata_files METADATA_FILES]
                                                [--jsonl_key JSONL_KEY]
                                                [--output_dir OUTPUT_DIR]
                                                [--output_name OUTPUT_NAME]
                                                [--seed SEED]
                                                [--processes PROCESSES]
                                                [--write_remainder]
                                                [--resume_from_checkpoint]
                                                [--display_pbar]
                                                [--files_per_record FILES_PER_RECORD]
                                                [--write_in_batch]
                                                --tokenizer_type
                                                {GPT2Tokenizer,NeoXTokenizer}
                                                --vocab_file VOCAB_FILE
                                                [--encoder_file ENCODER_FILE]
                                                [--max_seq_length MAX_SEQ_LENGTH]
                                                [--short_seq_prob SHORT_SEQ_PROB]
                                                [--ftfy]
                                                [--ftfy_normalizer {NFC,None}]
                                                [--wikitext-detokenize]
                                                [--eos_id EOS_ID]
                                                [--pad_id PAD_ID]

Required arguments:
  --tokenizer_type {GPT2Tokenizer,NeoXTokenizer}
                        Type of tokenizer to use for tfrecord/HDF5 dataset
                        generation. Can be one of `GPT2Tokenizer` or
                        `NeoXTokenizer`.

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Directory where raw data is stored.
  --metadata_files METADATA_FILES
                        Path to text file containing a list of file names
                        corresponding to the raw input documents to be
                        processed and stored; can handle multiple metadata
                        files separated by comma.
  --jsonl_key JSONL_KEY
                        The key name in input jsonl files from which the raw
                        text will be extracted in order to further process it. Default: "text".
  --output_dir OUTPUT_DIR
                        Directory where HDF5 files will be stored. Defaults to
                        `./data_dir/`.
  --output_name OUTPUT_NAME
                        Name of the dataset; i.e. prefix to use for HDF5 file
                        names.Defaults to `examples`.
  --seed SEED           Random seed. Defaults to `0`.
  --processes PROCESSES
                        Number of processes to use. Default to cpu count.
  --write_remainder     Write the remainder files when data is left over from
                        processing.
  --resume_from_checkpoint
                        Resume record writing from a given checkpoint.
  --display_pbar        Display progress while runs.
  --files_per_record FILES_PER_RECORD
                        Text files to write per HDF5 file.
  --write_in_batch      Whether to write the samples in batch for the HDF5
                        format, setting to false will save memory but a bit
                        slower.
  --vocab_file VOCAB_FILE
                        path to the vocabulary file. Defaults to None.
  --encoder_file ENCODER_FILE
                        Path to the encoder file. Defaults to None.
  --max_seq_length MAX_SEQ_LENGTH
                        Maximum sequence length. Defaults to `2048`.
  --short_seq_prob SHORT_SEQ_PROB
                        Probability of creating sequences which are shorter
                        than the maximum sequence length. Defaults to `0.0`.
  --ftfy                Fix text with ftfy.
  --ftfy_normalizer {NFC,None}
                        Choose what kind of unicode normalization is applied.
                        Usually, we apply `NFC` normalization, so that letters
                        followed by combining characters become single
                        combined characters. Using `None` applies
                        no normalization while fixing text.
  --wikitext-detokenize
                        Use wikitext detokenizer to fix text.
  --eos_id EOS_ID       Id for padding out shorter sequences. Defaults to
                        50256, which is `<|endoftext|>` in tokens.
  --pad_id PAD_ID       Id for padding out shorter sequences. Defaults to
                        50256, which is `<|endoftext|>` in tokens.
```

> **Note**: You have to provide either the `input_dir` or `metadata_files` argument. If you provided both, only files referenced in the `metadata_files` will be processed.

### Generation Notes

- It is recommended to use the ftfy module to fix the datasets. This can be enabled by the `--ftfy` argument.
- The NeoXTokenizer uses the HuggingFace library's inbuilt tokenizer and handles NFC normalization on its own. When using this tokenizer_type, it is recommended to set the `--ftfy_normalizer` argument to `None`. For the `GPT2Tokenizer`, use the default `NFC` value for the normalizer.
- To process HDF5 for training, we recommend using multi-processing. Moreover, we suggest using several input files such that the totalnum,ber of input files are greater than or equal to the number of processes provided by `--processes`. Note that this requires a high-spec CPU server, which can handle not only the concurrent running processes in RAM but also the I/O for reads and writes. If the I/O of the server is slow, the processes can appear to be hung for a very long while.
- For very large dataset (with several files with each file in the order of GBs) the recommendation is to split the data into smaller subsets and write out each subset. You can then mix all HDF5 in a common folder for use by the data pipeline, or just provide the locations of each subset in a list. The overall time to write out HDF5 can depend on the CPU server used.
- It is better to split the input dataset into multiple files, with similar size to leverage the full potential of parallel processing.
- For [CodeGen](https://arxiv.org/pdf/2203.13474.pdf) models processing please use `GPT2Tokenizer` along with the updated vocab files such that vocabulary of GPT-2 is extended by special tokens representing repeating tokens of tabs and white spaces.

### Output files structure

The output directory will contain a bunch of `h5` files as shown below:

```bash
<path/to/output_dir>
├── checkpoint.txt
├── data_params.json
├── examples_0_0.h5
├── examples_0_1.h5
├── examples_1_0.h5
├── examples_1_1.h5
├── examples_2_0.h5
├── examples_2_1.h5
├── examples_3_0.h5
├── examples_3_1.h5
├── examples_4_0.h5
├── examples_4_1.h5
├── examples_5_0.h5
├── examples_6_0.h5
├── examples_7_0.h5
└── examples_8_0.h5
```

Here `data_params.json` is the file which stores the parameters used for generating this set of files. `checkpoint.txt` can be used for resuming the processing in case the run script gets killed for some reason. To use this file, simply resume the previous command that you ran along with additional command line argument `--resume_from_checkpoint <path/to/output_dir>/checkpoint.txt`
