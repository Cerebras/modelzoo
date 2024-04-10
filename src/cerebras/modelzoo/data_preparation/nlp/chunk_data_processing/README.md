## Generating HDF5 data from raw data

We currently offer four modes for generating HDF5 files:

1. `LMData`: for processing language modelling datasets that are in `.jsonl`, `.jsonl.zst`, `.json.gz` , `.jsonl.zst.tar`, `.parquet` or `.txt` format.
2. `Summarization`: for processing fine-tuning datasets that are in `.jsonl`, `.jsonl.zst`, `.jsonl.zst.tar`, `.parquet` format.
3. `FIM`: for processing language modeling data to support Fill-in-the-Middle objective, also requires datasets that are in `.jsonl`, `.jsonl.zst`, `.json.gz` , `.jsonl.zst.tar`, `.parquet` or `.txt` format.
4. `LMData_VSL`: for processing language modelling datasets that are in `.jsonl`, `.jsonl.zst`, `.json.gz` , `.jsonl.zst.tar`, `.parquet` or `.txt` format with packing of sequences.
5. `Summarization_VSL`: for processing fine-tuning datasets that are in `.jsonl`, `.jsonl.zst`, `.jsonl.zst.tar`, `.parquet` format with packing of prompt and completion pairs into a list.
6. `DPO`: Direct preference optimization (DPO) dataset preprocessing. Supported formats are `.jsonl`, `.jsonl.zst`, `.jsonl.zst.tar` and `.parquet`.

Each of the above data processing can be done by running the provided script [`create_hdf5_dataset.py`](./create_hdf5_dataset.py) with the appropriate sub command (modes) to generate the `.h5` files for GPT style models. Each sub-commands takes in a set of arguments which are described below in [Generating HDF5 files](#generating-hdf5-files) section. The [`VSL: Variable Sequence Length for LM and Summarization Tasks`] section describes VSL implementation of `LMData` and `Summarization` data preprocessing and [`Direct preference optimzation (DPO) data prepration`] gives an overview of DPO data generation.

## VSL: Variable Sequence Length for LM and Summarization Tasks

The concept of Variable Sequence Length (VSL) plays a crucial role in optimizing the efficiency and effectiveness of language models (LM) and summarization tasks. This approach focuses on dynamically adjusting and merging tokenized sequences to better fit within the constraints of maximum sequence length. The underlying principle of VSL is to maximize the utilization of available sequence space, reducing the need for padding and ensuring more meaningful information is presented to the model within its input limitations.

In the context of language modeling and summarization, the VSL methodology involves a strategic packing of sequences. This is done by first generating the tokenized data suited for each task where language modeling might deal with texts and summarization with distinct pairs of prompts and completions. Once tokenized, the VSL algorithm assesses the potential for merging these sequences in a way that closely approaches but does not exceed the model's maximum sequence length.

The innovation in VSL lies in its approach to sequence merging. Instead of a straightforward sequential packing, VSL examines the available space in tokenized sequences in reverse order. By doing so, VSL can more effectively identify pairs of sequences that, when combined, optimally utilize the model's capacity. This not only improves the density of information within each input but also reduces the computational waste associated with processing excessive padding. To get more packing, we can increase the chunk size by changine `max_chunk_size`. The default value is 1 MB.

For language modeling, this means more coherent and extended passages of text can be processed together, enhancing the model's ability to learn from broader contexts. In summarization tasks, the efficient packing allows for more examples to be evaluated in tandem, potentially improving the model's comprehension and generation of summaries. `Update the dataset section in the model's config to include the line ``use_vsl: True`` if the dataset is constructed using VSL.`

Overall, the VSL strategy represents a sophisticated advancement in preparing tokenized data for neural network models. By optimizing how sequences are merged and managed within the constraints of sequence length, VSL contributes to more efficient training cycles and improved model performance.

## Direct preference optimzation (DPO) data prepration

In the realm of Direct Preference Optimization (DPO) for conversational models, a distinct approach to tokenizing and structuring data is undertaken. This process meticulously handles the tokenization of prompts and their responses, ensuring the data is optimally prepared for model training. A pivotal aspect of this approach is its nuanced handling of 'chosen' and 'rejected' responses to prompts, an essential factor for models learning preference in dialogue.

Initially, the process involves tokenizing the full response to a prompt without adding any special tokens. This is critical in maintaining the integrity and continuity of the conversational flow. The prompt itself is also tokenized separately to create a distinct set of input IDs. These steps are foundational, ensuring that both elements of the conversation are accurately represented and can be analyzed in conjunction.

A key innovation in this process addresses the challenge of concatenating encoded strings not equating to the encoding of the concatenated string—a common issue with complex tokenizers. By meticulously extracting and concatenating token IDs and attention masks for the prompt and response, the process ensures that the final tokenized sequence represents the natural progression of the conversation. This includes adjusting the start index for the response's token IDs based on the length of the prompt's input IDs, ensuring a seamless transition between the prompt and the response in the tokenized form.

Moreover, the application of chat templates emerges as a crucial step, especially for conversations that involve multiple turns or require a specific format. This step adapts the tokenized data to reflect the conversational model's expectations, aligning with its training on dialogue structures. Whether through predefined templates or dynamic generation, this ensures that the model can interpret the context and nuances of the dialogue accurately.

The ultimate goal of this tokenization and encoding process for DPO is to craft a dataset where each entry meticulously reflects the dynamics of human conversation. This includes distinguishing between 'chosen' and 'rejected' responses based on their context within the prompt, a fundamental aspect for models tasked with understanding preferences in dialogue. By achieving this, the approach sets the stage for training conversational models that can generate responses not just with high relevance and coherence but also aligned with nuanced human preferences.

## Environment Setup

**NOTE**: Modelzoo environment setup is required for a clean run of the data preprocessing script.

Setup the model zoo environment as described in [PYTHON-SETUP.md](../../../../../../PYTHON-SETUP.md).

## Input files format

The input text documents need to be in a specific file format before utilizing the provided script. The acceptable file formats are `'.jsonl', '.json.gz', '.jsonl.zst', '.jsonl.zst.tar', '.parquet', '.txt'`. These files should have the data in a specific structure as described in [data format](#input-data-format) section.

To optimally process the files, it is recommended that all files with any of the above file formats other than `.txt` contain enough text in a single file. Recommended size for each file is in the order of GB.

On the contrary, if processing with smaller files with `txt` format, please input a `metadata` file containing a list of paths to these files to better leverage multi processing.

### Input data format

As mentioned above,  there are three primary types on input files accepted into the preprocessing script. They are either `json` based, `txt` based or `parquet` based. For each type, the input data needs to follow certain structure in order to be converted into `hdf5` files accurately.

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

#### Format for `TXT` based files in `LMData` mode

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

#### Format for `PARQUET` based files in `LMData` mode

Column Name = "text": "Any text excerpt from the dataset of interest...\nThe new lines should be represented as a newline character.",
Column Name = "abc": "...."
etc
```

For the `parquet` files, as shown above, by default, the raw text is extracted from the column with name `text` in the input files. If your input files do not contain a column with name  `text` as key then you should know the `key` corresponding to the text you need to extract. Then, you can use the command line argument `--jsonl_key=<your key name>` to extract the text.

For example, if your parquet files have the content as below and you want to extract the value from column name `idea`:

```parquet
Column Name = "idea": "Any text excerpt from the dataset of interest, with custom key: 'idea'...\nThe new lines should be represented as a newline character.",
Column Name = "abc": "...."
etc
```

then you'd need to pass `--jsonl_key=idea` in the command line arguments.

### Definition of vocab_file and encoder_file

We support three different tokenizers with this script, 1. `GPT2Tokenizer`, 2. `NeoXTokenizer` and 3. `HuggingFaceTokenizer`. We need to supply the following parameters when using a specific tokenizer:

- For `GPT2Tokenizer`, `vocab_file=gpt2-vocab.bpe` and `encoder_file=gpt2-encoder.json`.
- For `NeoXTokenizer`, `encoder_file=neox-encoder.json`.
- For `HuggingFaceTokenizer`, `huggingface_tokenizer` should be specified, for example `huggingface_tokenizer=tiiuae/falcon-7b` .

These files can be found [here](../../../vocab/).

**Note:** For `GPT2Tokenizer` we follow the nomenclature used by OpenAI in their [implementation](https://github.com/openai/gpt-2/blob/master/src/encoder.py#L109-L112) which is slightly different from Hugging Face's nomenclature where they call the `vocab_file` as `merges_file` and `encoder_file` as `vocab_file`. However, the content of the files are the same. For `NeoXTokenizer`, we use the same nomenclature to avoid confusion.

## Generating HDF5 files

Once you have the text dataset that meets above requirement, you can generate HDF5 files using the `create_hdf5_dataset.py` script:

```bash
python create_hdf5_dataset.py [mode] [--arguments]
```

The mode as we mentioned before can be one of {`LMData`, `Summarization`,}. The four modes share the same setup and processing arguments, but differ in their dataset arguments as detailed below:

##### Table 2: Setup Arguments
Argument | Default Value | Description
--- | --- | ---
`params` | N/A | Path to YAML config file for setting dataset preprocessing parameters. Optional alternative for providing command line arguments.
`input_dir` | N/A | Directory where raw data is stored. Supports only the formats: [`'.jsonl', '.jsonl.zst', '.jsonl.zst.tar', '.txt'`].
`metadata_files` | N/A | Path to text file containing a list of file names corresponding to the raw input documents to be processed and stored; can handle multiple metadata files separated by comma.
`output_dir` | `./data_dir/` | Directory where HDF5 files will be stored.
`processes` | cpu count | Number of processes to use.

> **Note**: You have to provide either the `input_dir` or `metadata_files` argument. If you provided both, only files referenced in the `metadata_files` will be processed.

##### Table 3: Processing Arguments
Argument | Default Value | Description
--- | --- | ---
`tokenizer_type` | **required arg** | Type of tokenizer to use for HDF5 dataset generation. Can be one of `GPT2Tokenizer`, `NeoXTokenizer` or `HuggingFaceTokenizer`.
`vocab_file` | N/A | Path to the vocabulary file.
`encoder_file` | N/A | Path to the encoder file.
`eos_id` | `None` | Token id of the end of sentence token. Will be used if tokenizer doesn't have a default eos_id.
`pad_id` | `None` | Token id of the padding token. Will be used if tokenizer doesn't have a default pad_id.
`max_seq_length` | `2048` | Maximum sequence length.
`short_seq_prob` | `0.0` | Probability of creating sequences which are shorter than the maximum sequence length.
`output_name` | `examples` | Name of the dataset; i.e. prefix to use for HDF5 file names.
`files_per_record` | `50000` | Text files to write per HDF5 file.
`write_in_batch` | `False` | Whether to write the samples in batch for the HDF5 format, setting to false will save memory but a bit slower.
`write_remainder` | `True` | Write the remainder files when data is left over from processing.
`resume_from_checkpoint` | `False` | Resume record writing from a given checkpoint.
`display_pbar` | `True` | Display progress while runs.
`seed` | `0` | Random seed.
`auth_token` | `None` | Authentication token to access restricted HuggingFace tokenizers
`max_chunk_size` | 1024 kB | Maximum chunk size for preprocessing. The value is provided in kB.
`shuffle` | `False` | Enable shuffling while preprocessing
`shuffle_seed` | `0` | Shuffle seed value


##### Table 4: Dataset Arguments (`LMData` mode)
Argument | Default Value | Description
--- | --- | ---
`use_ftfy` | `False` | Fix text with ftfy.
`ftfy_normalizer` | `NFC` | Choose what kind of unicode normalization is applied. Usually, we apply `NFC` normalization, so that letters followed by combining characters become single combined characters. Using `None` applies no normalization while fixing text.
`wikitext_detokenize` | `False` | Use wikitext detokenizer to fix text.
`jsonl_key` | `text` | The key name in input jsonl files from which the raw text will be extracted in order to further process it.
`pack_sequences` | `True` | Concatenate a document smaller than maximum sequence length with other documents, instead of filling it with Padding token.
`min_sequence_len` | `10` | Minimum token length to skip the sample.
`input_ids_dtype` | `int32` | dtype of processed input_ids.
`input_mask_dtype` | `int32` | dtype of processed input loss masks.
`inverted_mask` | `False` | If False, 0 represents masked positions. If True 1 represents masked positions.
`split_text_to_tokenize` | `False` | Whether to split the text into smaller chunks before tokenization. This is helpful for very long documents with tokenizers such as Llama tokenizer which performs quadratically in the text length.
`chunk_len_to_split` | `2000` | Length of the text chunks to split the text into before tokenization for slower tokenizers. Could be optionally used with the above flag `split_text_to_tokenize`. Without the previous flag, this argument will be ignored.
`remove_bos_in_chunks` | `False` | Whether to remove the BOS token from the beginning of the chunks. Set this to `True` when using `split_test_to_tokenize` and `chunk_len_to_split` to avoid having multiple BOS tokens in the middle of the text. Not applicable to all tokenizers.

##### Table 5: Dataset Arguments (`Summarization` mode)
Argument | Default Value | Description
--- | --- | ---
`use_ftfy` | `False` | Fix text with ftfy.
`ftfy_normalizer` | `NFC` | Choose what kind of unicode normalization is applied. Usually, we apply `NFC` normalization, so that letters followed by combining characters become single combined characters. Using `None` applies no normalization while fixing text.
`wikitext_detokenize` | `False` | Use wikitext detokenizer to fix text.
`min_sequence_len` | `10` | Minimum token length to skip the sample.
`sep_token` | `None` | Token added between prompt and completion in preprocessed sequences. If supplied with a non-None value, the tokenizer will add the token to the vocab size and modify the vocab size. This may not be advisable for doing fine tuning on a pre-trained model on the types of models that do not provision for extra tokens.
`prompt_key` | **required arg** | Json key for the prompt.
`completion_key` | **required arg** | Json key for the completion.
`input_ids_dtype` | `int32` | dtype of processed input_ids.
`input_mask_dtype` | `int32` | dtype of processed input loss masks.
`inverted_mask` | `False` | If False, 0 represents masked positions. If True 1 represents masked positions.

##### Table 6: Dataset Arguments (`FIM` mode)
Argument | Default Value | Description
--- | --- | ---
`fim_rate` | `0.90` | Float specifying percentage of data to apply FIM transformation, instead of leaving as auto-regressive.
`spm_rate` | `0.50` | Float specifying percentage of FIM transformation to convert to prefix-suffix-middle (PSM) vs suffix-prefix-middle (SPM) formats.
`fim_prefix_tok` | Special token denoting prefix section in a FIM'ed context
`fim_middle_tok` | special token denoting middle section in a FIM'ed context
`fim_suffix_tok` | Special token denoting suffix section in a FIM'ed context

The `FIM` mode is very similar to the `LMData` mode, and uses all the same other arguments as listed in the `LMData` table. These additional parameters determine whether what percentage of samples have the FIM transformation applied, and what percent of these end up in PSM (prefix, suffix, middle) or SPM format.

> **Note**: For CodeLlama, to follow the note [here](https://huggingface.co/docs/transformers/main/model_doc/code_llama#transformers.CodeLlamaTokenizer.eos_token) specify the EOT token as the EOS token in the config.

You can provide the above arguments either as command line arguments, for example:
```bash
python create_hdf5_dataset.py LMData --input_dir /path/to/data --tokenizer_type NeoXTokenizer --encoder_file /path/to/encoder --max_seq_length 4096 --use_ftfy True --pack_sequences False
```

or as Yaml config file:
```bash
python create_hdf5_dataset.py LMData --params ./configs/autoregressive_lm_preprocessing.yaml
```
example yaml files for LMData and Summarization are located under [./configs](./configs).

> **Note**: You can also provide both, but command line arguments will override any common arguments with the yaml configuration file.

> **Note**: The behavior of `eos` and `pad` ids is dependent on the tokenizer used.
For `GPT2Tokenizer`, the `eos` and `pad` ids are the same.
For `NeoXTokenizer` and `HuggingFaceTokenizer`, the `eos` and `pad` ids are the same as the `eos` and `pad` ids in the tokenizer. If the tokenizer does not have a default `pad_id` then the `pad_id` argument will be used. If `pad_id` is not provided, then the default `pad_id` will be set to same as `eos_id`.

### Generation Notes

- It is recommended to use the ftfy module to fix the datasets. This can be enabled by the `--use_ftfy` argument.
- The NeoXTokenizer uses the HuggingFace library's inbuilt tokenizer and handles NFC normalization on its own. When using this tokenizer_type, it is recommended to set the `--ftfy_normalizer` argument to `None`. For the `GPT2Tokenizer`, use the default `NFC` value for the normalizer.
- For [CodeGen](https://arxiv.org/pdf/2203.13474.pdf) models processing please use `GPT2Tokenizer` along with the updated vocab files such that vocabulary of GPT-2 is extended by special tokens representing repeating tokens of tabs and white spaces.

# Data pre-processing pipeline

To process HDF5 for training, we recommend using multi-processing. There are 2 approaches to doing multiprocessing. If the
number of processes that the user wants to run is more than 2, then we do task based mulitprocessing. If it is less than 2,
then we use file based multiprocessing. The system is designed to handle large datasets by reading raw
data, tokenizing it, and writing the tokenized data to HDF5 files efficiently. The following sections explain these 2 approaches:

## Task-Based Multi-Processing in Data Pipeline

This pipeline consists of three main types of processes:

1. **Reader Process**: Responsible for reading raw data in chunks and
   distributing it across multiple tokenizer queues.

2. **Tokenizer Processes**: Each process takes chunks of raw data from a queue, tokenizes
   them, and places the tokenized data onto a writer queue.

3. **Writer Process**: Collects the tokenized data and writes it to disk in
   HDF5 format, tracking progress.

### Process Responsibilities

#### Reader Process

- Breaks down input files into manageable chunks. This can be provided as
  a parameter in the config yaml file. By default, the chunk size is 64kB.
- Distributes chunks in a round-robin fashion to tokenizer processes.
- Ensures balanced load across tokenizers.
- Emits a sentinel value to indicate the end of input.

#### Tokenizer Process

- Retrieves data chunks from its queue.
- Performs tokenization using a predefined `token_generator`.
- Forwards tokenized data to the corresponding writer process.
- Handles sentinel value by signaling the writer process of completion.

#### Writer Process

- Writes tokenized data to HDF5 files.
- Maintains statistics like the number of discarded and successful chunks,
  character and byte counts.
- Manages a checkpoint system for robustness and recovery.
- Sends cumulative data stats to the main control process.

## Pipeline Flow

1. The `task_split_process_dataset` function calculates total dataset size and
   chunks.
2. It initializes all tokenizer and writer processes.
3. The reader process starts and reads data, pushing it to the tokenizer queues.
4. Tokenizer processes tokenize the data and pass it on to the writer process.
5. The writer process manages output files and writes tokenized data.
6. Progress is tracked and logged, with a progress bar displayed in the console.
7. Upon completion, the writer process gathers and returns final data
   statistics.

## Key Features

- **Concurrency**: Utilizes multiple processes for different stages in the
  pipeline to ensure maximum CPU utilization.
- **Fault Tolerance**: Implements a checkpoint system allowing for recovery from
  the last saved state.
- **Progress Tracking**: Includes a real-time progress bar and logging to
  monitor the pipeline's performance.

## File Split Parallel Data Processing

This section outlines the `file_split_process_dataset` method within our data
processing framework, which utilizes file-based parallelism to process large
datasets efficiently.

### Overview

The `file_split_process_dataset` function orchestrates the distribution of data
files across multiple processes, enabling parallel processing. This approach
ensures that each process works on a separate chunk of the dataset to maximize
utilization of CPU resources.

### How It Works

The method executes the following steps:

1. **Initialization**: It calculates the total data size and the number of
   chunks to process.
2. **Checkpointing**: Reads checkpoints to resume processing if previously
   interrupted, keeping track of files already written.
3. **File Distribution**: Assigns files to processes evenly to ensure a balanced
   workload.
4. **Progress Tracking**: Implements a progress bar using `tqdm` for real-time
   visual feedback on the processing progress.
5. **Parallel Processing**: Starts multiple processes, each handling its
   assigned list of files.
6. **Statistics Aggregation**: Collects and aggregates data processing
   statistics from all processes, such as the number of processed and discarded
   sequences, as well as the counts of raw and normalized characters and bytes.

#### Notes:

- The task based processing is more efficient because it always ensures that load is balanced
  across processes.
- It is specially useful in case of large files or large entries.
- As is does fixed data size processing, we can estimate the time to complete data pre-processing
  which is very helpful for the users.
- Therefore, we recommend using task based multi-processing. The framework will automatically switch
  to this mode is the `--processes` provided are 3 and above.

# Online Shuffling in HDF5 File Storage

## Overview
Our data processing pipeline includes an innovative online shuffling feature that
integrates seamlessly with HDF5 file storage. This functionality is crucial for
machine learning models to prevent learning the order of the data and to ensure
a randomized distribution of data sequences.

## How Online Shuffling Works
The online shuffling mechanism is designed to shuffle data sequences as they
are being written to the storage files. Instead of a post-process shuffling
which requires the entire dataset to be loaded into memory, our method
interleaves shuffling with the data serialization process, conserving memory
and reducing overall processing time. This can be enabled using a flag
`shuffle` and passing a seed with the flag `shuffle_seed`. These can be
provided in the config file for data preprocessing.

## Implementation Details
- During the HDF5 file writing operation, each sequence of tokenized data is
  assigned a random index that determines its placement in the output files.
- This randomization ensures that upon reading the data back for training
  purposes, the sequences are already shuffled.
- The shuffling operation is handled efficiently, allowing the system to
  process and shuffle large datasets without excessive memory usage.

## Advantages of Online Shuffling
- **Efficiency:** By eliminating the need for a separate shuffling step, we save
  on both processing time and memory.
- **Scalability:** This approach scales elegantly with the size of the dataset,
  suitable for large-scale machine learning tasks.
- **Simplicity:** Reduces the complexity of the data preparation pipeline by
  integrating shuffling into the data writing process.

For detailed implementation, please refer to the `append_to_hdf5` function in
our codebase.

# Output files structure

The output directory will contain a bunch of `h5` files as shown below (with 2 processes):

```bash
<path/to/output_dir>
├── checkpoint_process_0.txt
├── checkpoint_process_1.txt
├── data_params.json
├── output_chunk_0_0_0_0.h5
├── output_chunk_1_0_0_0.h5
├── output_chunk_1_0_16_1.h5
├── output_chunk_0_0_28_1.h5
├── output_chunk_0_0_51_2.h5
├── output_chunk_1_0_22_2.h5
├── output_chunk_0_1_0_3.h5
├── ...
```

Here `data_params.json` is the file which stores the parameters used for generating this set of files. `checkpoint_*.txt` can be used for resuming the processing in case the run script gets killed for some reason. There is one `checkpoint_*.txt` file for each process. To use this file, simply resume the previous command that you ran along with additional command line argument `--resume_from_checkpoint`

For 3 processes, the structure will be different as we use task based splitting as explained above:

```bash
<path/to/output_dir>
├── checkpoint.txt
├── data_params.json
├── output_chunk_0_0_0_0.h5
├── output_chunk_1_0_0_0.h5
├── output_chunk_1_0_16_1.h5
├── output_chunk_0_0_28_1.h5
├── output_chunk_0_0_51_2.h5
├── output_chunk_1_0_22_2.h5
├── output_chunk_0_1_0_3.h5
├── ...
├── output_chunk_0_2_5_10.h5
├── output_chunk_0_3_0_16.h5
```

Note that we have only 1 checkpoint file because we have 1 reader process only.

We are collecting data statistics during and after data preprocessing which are stored in `data_params.json`

The `h5_dataset_stats` section in the generated `data_params.json` file contains the following statistics:

```bash
"h5_dataset_stats": {
  "detokenized_bytes": # total bytes in the detokenized text using the same tokenizer as used for tokenization,
  "detokenized_chars": # total characters in the detokenized text using the same tokenizer as used for tokenization,
  "loss_valid_tokens": # total number of tokens that are not padding tokens or prompt tokens,
  "non_pad_tokens": # total number of tokens that are not padding tokens,
  "num_sequences": # total number of sequences in the dataset,
  "num_tokens": # total number of tokens in the dataset,
}
```

There are additional statistics generated in `post-process` section:

```bash
"post-process": {
    "average_bytes_per_sequence": # the average number of bytes per sequence after processing
    "average_chars_per_sequence": # the average number of characters per sequence after processing
    "discarded_files": # the number of files that were discarded during processing due to errors or being filtered out
    "eos_id": # the token ID used to signify the end of a sequence
    "loss_valid_tokens": # Number of tokens on which loss is computed
    "n_examples": # the total number of examples (sequences) that were processed
    "non_pad_tokens": # Non pad tokens
    "normalized_bytes_count": # the total number of bytes after normalization (e.g., UTF-8 encoding)
    "normalized_chars_count": # the total number of characters after normalization (e.g., lowercasing, removing special characters)
    "num_masked_tokens": # the total number of tokens that were masked (used in tasks like masked language modeling)
    "num_pad_tokens": # the total number of padding tokens used to equalize the length of the sequences
    "num_tokens": # Total number of tokens
    "pad_id": # the token ID used as padding
    "processed_files": # the number of files successfully processed
    "raw_bytes_count": # the total number of bytes before any processing
    "raw_chars_count": # the total number of characters before any processing
    "successful_files": # the number of files that were successfully processed without any issues
    "vocab_size": # the size of the vocabulary used in the tokenizer
}
```
