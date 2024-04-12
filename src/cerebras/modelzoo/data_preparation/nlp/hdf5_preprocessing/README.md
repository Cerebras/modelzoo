# Creating HDF5 dataset for GPT Models

For efficient implementation of data loader for GPT style models, we provide two methods to generate `.h5` files which can then be used in the input pipeline for GPT style models. If you already have a PyTorch dataset and need to convert it to HDF5 format, follow section [Converting a PyTorch dataset to HDF5 format](#converting-a-pytorch-dataset-to-hdf5-format) and if you want to generate HDF5 dataset from raw data, you can skip to section [Generating HDF5 data from raw data](#generating-hdf5-data-from-raw-data) of this document.


## Converting a PyTorch dataset to HDF5 format
If you have a PyTorch dataset for GPT models (from any source such HuggingFace, Map-Style or Iterable), you can easily write the samples of that dataset in HDF5 format to use with Cerebras optimized HDF5 DataProcessor. This can be done by calling the function `convert_dataset_to_HDF5()` which is defined in [`convert_dataset_to_HDF5.py`](./convert_dataset_to_HDF5.py). The following example shows conversion of [HuggingFace Eli5 dataset](../../../data_preparation/huggingface/HuggingFace_Eli5.py) to HDF5:

```python
from cerebras.modelzoo.data_preparation.huggingface.HuggingFace_Eli5 import (
    HuggingFace_Eli5,
)
from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.convert_dataset_to_HDF5 import (
    convert_dataset_to_HDF5,
)

dataset, data_collator = HuggingFace_Eli5(
  split="train", num_workers=8, sequence_length=128
)

convert_dataset_to_HDF5(
    dataset=dataset,
    data_collator=data_collator,
    output_dir="./eli5_hdf5_dataset/",
    num_workers=8,
)
```

The function `convert_dataset_to_HDF5()` uses a PyTorch Dataloader to fetch the samples from the specified dataset and writes those samples in `h5` files. Table 1, explains the arguments to the `convert_dataset_to_HDF5()` function:

##### Table 1: convert_dataset_to_HDF5 Arguments
Argument | Default Value | Description
--- | --- | ---
`dataset` | N/A |  PyTorch dataset to fetch the data from (IterableDataset or Dataset).
`output_dir` | ./hdf5_dataset/ | Directory where HDF5 will be stored.
`name` | dataset-partition | Name of the dataset; i.e. prefix to use for HDF5 file names.
`samples_per_file` | 2000 | Number of samples written to each HDF5 file
`num_workers` | 8 | Number of Python processes to use for generating data.
`batch_size` | 64 | The batch size to use fetching the data.
`data_collator` | N/A | Merges a list of samples to form a mini-batch of Tensor(s).
`dtype` | i4 | Data type for the HDF5 dataset.
`compression` | gzip | HDF5 Compression strategy.

While the function `convert_dataset_to_HDF5()` is generic and can be used with all transformer models, note that PyTorch dataset features dictionary should have the the following key/values GPT models:

- `input_ids`: Input token IDs, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `attention_mask`: Loss mask for positions that don't participate in loss backpropagation. Has values `0` on the padded positions and `1` elsewhere.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `labels`: Labels for language modeling pre-training task, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`

There is also two extra key/values needed to train GPT models on variable sequence length (VSL) samples that are packed into fixed length sequence:
"attention_span", "position_ids"
- `attention_span`: Specifies the attention span for each attention key to prevent attending to out-of-sample queries, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `position_ids`: Token position index realtive to the data sample, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`

> **NOTE**: 

1. More information on using of HuggingFace datasets can be found in this document: [Using HuggingFace datasets for auto-regressive LM](../../huggingface/README.md)
2. `attention_mask` here actually represents loss mask and is used as loss mask by our gpt-style models. It can mask out components like padding tokens or prompt tokens that shouldn't be in the loss calculation. For example: \
- If we do autoregressive language modeling with the input in format [`input_ids`, `padding_tokens`] (LMData), `attention_mask` will look something like [1, 1, ..., 1, 0, 0, 0, ..., 0] where the 1's corresponds to `input_ids` and 0's to `padding_tokens`
- If we do prompted generation like instruction tuning with input in format [`prompt_ids`, `input_ids`, `padding_tokens`(optional)] (Summarization), `attention_mask` will look something like [0, 0, ..., 0, ... 1, 1, ..., 1, 0, 0, 0, ..., 0] where the first chunk of 0's correspond to `prompt_ids`, the 1's correspond to `input_ids` and the second chunk of 0's correspond to `padding_tokens`

## Generating HDF5 data from raw data

We currently offer four modes for generating HDF5 files:

1. `LMData`: for processing language modelling datasets that are in `.jsonl`, `.jsonl.zst`, `.parquet` or `.txt` format.
2. `Summarization`: for processing fine-tuning datasets that are in `.jsonl`, `.jsonl.zst`, `.parquet` or `.txt` format.
3. `FIM`: for processing language modeling data to support Fill-in-the-Middle objective, also requires datasets that are in `.jsonl`, `.jsonl.zst`, `.parquet` or `.txt` format.
4. `LMData_VSL`: for processing language modelling datasets that are in `.jsonl`, `.jsonl.zst`, `.parquet` or `.txt` format for variable squence length (VSL) training.
5. `Summarization_VSL`: for processing fine-tuning datasets that are in `.jsonl`, `.jsonl.zst`, `.parquet` or `.txt` format for variable squence length (VSL) training.
6. `Customize`: for any dataset format, but requires supplying a module that specifies how to read the raw dataset files.
7. `LlavaPhaseOne`: for preprocessing llava phase 1 datasets that have a image in `.jpg` or `.png` format and text in `.jsonl`, `.jsonl.zst`, `.parquet` or `.txt` format.
8. `LlavaPhaseTwo`: for preprocessing llava phase 2 datasets that have a image in `.jpg` or `.png` format and text in `.jsonl`, `.jsonl.zst`, `.parquet` or `.txt` format.

Each of the above data processing can be done by running the provided script [`create_hdf5_dataset.py`](./create_hdf5_dataset.py) with the appropriate sub command (modes) to generate the `.h5` files for GPT style models. Each sub-commands takes in a set of arguments which are described below in [Generating HDF5 files](#generating-hdf5-files) section.

Before doing that, you need to setup a python virtual environment through [conda](https://www.anaconda.com/) as described below.

## Environment Setup

**NOTE**: Modelzoo environment setup is required for a clean run of the data preprocessing script. 

Setup the model zoo environment as described in [PYTHON-SETUP.md](../../../../../../PYTHON-SETUP.md).

## Input files format

The input text documents need to be in a specific file format before utilizing the provided script, except for the `Customize` mode. The acceptable file formats are `'.jsonl', '.json.gz', '.jsonl.zst', '.jsonl.zst.tar', '.parquet', '.txt'`. These files should have the data in a specific structure as described in [data format](#input-data-format) section.

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
`module` | N/A | Python file name contains the custom dataset processor for `Customize` mode only.
`dataset_processor` | N/A | Name of the custom dataset processor for `Customize` mode only.

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

The `FIM` mode is very similar to the `LMData` mode, and uses all the same other arguments as listed in the `LMData` table. These additional parameters determine whether what percentage of samples have the FIM transformation applied, and what percent of these end up in PSM (prefix, suffix, middle) or SPM format.  

> **Note**: For CodeLlama, to follow the note [here](https://huggingface.co/docs/transformers/main/model_doc/code_llama#transformers.CodeLlamaTokenizer.eos_token) specify the EOT token as the EOS token in the config.

##### Table 7: Dataset Arguments (`LMData_VSL` mode)
Argument | Default Value | Description
--- | --- | ---
`fold_long_doc` | `True` | Fold documents larger than `max_seq_length` into multiple sequences, instead of dropping them.
`position_ids_dtype` | `int32` | dtype of token position ids.

The `LMData_VSL` mode inherits all the other arguments from the `LMData` mode as listed in Table 4.

##### Table 8: Dataset Arguments (`Summarization_VSL` mode)
Argument | Default Value | Description
--- | --- | ---
`position_ids_dtype` | `int32` | dtype of token position ids.
`prompt_prefix` | `None` | If specified, this will be added before the prompt in every sequence. Example usage is to add `<\|user\|>` before the user message in a multi-turn dialogue. 
`completion_prefix` | `None` | Similar to `prompt_prefix`, but for the completion. Example usage is to add `<\|assistant\|>` before the model's response in a multi-turn dialogue. 
`eos_after_prompt` | `False` | Some current chat templates will include an EOS token after the end of the user input in a multi-turn dialogue. If this flag is specified, there will be EOS tokens after all prompts.
`multi_turn_key` | `None` | If specified, this replaces the `prompt_key` and `completion_key` usage. The assumption is that a multi-turn dialogue stores a list of the entries, which can be referenced by this key.
`multi_turn_content_key` | `None` | If the data stored at `multi_turn_key` is a list of dictionaries rather than a list of strings (of user and assistant responses), this key accesses the message content within the dictionary. For example, some data stores the dialogue as dictionaries of `{"content": ..., "user": ...}`, in which case `multi_turn_content_key` would be `content`.

The `Summarization_VSL` mode inherits all the other arguments from the `Summarization` mode as listed in Table 5.

##### Table 9: Dataset Arguments (`LlavaPhaseOne` mode)
Argument | Default Value | Description
--- | --- | ---
`eos_after_prompt` | `False` | Some current chat templates will include an EOS token after the end of the user input in a multi-turn dialogue. If this flag is specified, there will be EOS tokens after all prompts.
`multi_turn_key` | `None` | If specified, this replaces the `prompt_key` and `completion_key` usage. The assumption is that a multi-turn dialogue stores a list of the entries, which can be referenced by this key.
`multi_turn_content_key` | `None` | If the data stored at `multi_turn_key` is a list of dictionaries rather than a list of strings (of user and assistant responses), this key accesses the message content within the dictionary. For example, some data stores the dialogue as dictionaries of `{"content": ..., "user": ...}`, in which case `multi_turn_content_key` would be `content`.
`image_key` | `None` | Image key of the LLaVA dataset. For example a jsonl file might have the image path contained at the key, "image". 
`multi_modal_non_image_ex_key` | `None` | Some examples in LLaVA training are text-only and have no images, so that the model does not forget how to answer text-only questions while it is learning multi-modality. These examples will not have the `image_key`, but will have another key to represent that it is a no-image example. 
`image_token` | `None` | String that represents where in the text the image patches will be inserted. For example, the original LLaVA dataset contained the string "<image>" in the prompt. 
`num_patches` | `None` | Number of patches to represent an image. This is determined by the patch-size (in pixels) of the image-encoder, and the pixel count of the input images. 
`image_dir` | `None` |  Absolute path of image directory. Used along with the relative path under the `image_key` field to check that images exist, and throw out examples with no image.  

The `LlavaPhaseOne` mode inherits all the other arguments from the `Summarization` mode as listed in Table 8. Note that the LLaVA phase-one training removes the prompt text, and trains on the image + completion pair.  

Also note that both preprocessors for LLaVA currently only support tokenizers based on the Llama and Mistral models. 

##### Table 10: Dataset Arguments (`LlavaPhaseTwo` mode)
Argument | Default Value | Description
--- | --- | ---
`prompt_prefix` | `None` | If specified, this will be added before the prompt in every sequence. Example usage is to add `<\|user\|>` before the user message in a multi-turn dialogue. 
`completion_prefix` | `None` | Similar to `prompt_prefix`, but for the completion. Example usage is to add `<\|assistant\|>` before the model's response in a multi-turn dialogue. 
`system_prompt_style` | `None` | Key to obtain the system prompt used for the LLM backbone within LLaVA. The currently supported keys are `vicuna_v0`, `vicuna_v1`. For example, if you are training a LLaVA model based on the Vicuna model, you could specify `vicuna_v1`. 

The `LlavaPhaseTwo` mode inherits all the other arguments from `LlavaPhaseOne` mode as listed in Table 9. LLaVA phase-two training does *not* remove the prompt text as phase-one does. 

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

### `Customize` mode steps
1. Create a python file or put under `./hdf5_dataset_preprocessors.py`
2. Import the module `HDF5Preprocessor` in the file you created as follows:
```python
from cerebras.modelzoo.transformers.data_processing.scripts.hdf5_preprocessing.hdf5_preprocessor import HDF5Preprocessor
```
3. Create a class that inherits from `HDF5Preprocessor`. (e.g `CustomDataset`)
4. Implements init takes as input a dictionary contains the dataset parameters that is needed for `HDF5Preprocessor`.
5. Implements the method `file_read_generator` and `preprocessing_generator` following [Write Customized Preprocessor](#write-customized-preprocessor)
6. Run `create_hdf5_dataset.py` script.

### Write Customized Preprocessor

You can create customized preprocessor for various datasets or objectives. We provided `2` references at `hdf5_dataset_preprocessors.py` where:

1. LMDataPreprocessor is the preprocessor for autoregressive language modeling tasks.
2. SummarizationPreprocessor is the preprocessor for summarization tasks.

They both inherit from the `HDF5BasePreprocessor` at `hdf5_base_preprocessor.py` with 2 functions that can be overridden to customize for various cases:

1. file_read_generator(), this function takes a file path, reads from the file and yield the corresponding text documents.
You can customize how you want the file to be read based on its format (ex. csv, zip, etc.). Our defaults preprocessors use `lm_dataformat` reader with certain json keys.

2. preprocessing_generator(), This function takes in the output of file_read_generator(), performs tokenization and other preprocessing techniques and yield the data samples in np.array format.

For example, in the autoregressive language modeling task, file_read_generator yields a str object and the preprocessing_generator produces an np array with shape `[3, max_sequence_length]` with the following 3 features concatenated on the first dimension:
1. `input_ids`: Input token ids, padded with 0's to max_sequence_length.
2. `input_mask`: Loss mask for the sequence. It has 0's on padded positions like prompts or padding tokens and 1's elsewhere.
3. `labels`: input_ids shifted to the right by 1 position as the target labels.

> **NOTE**: To avoid tedious setup of arguments specific to your customized preprocessor, we recommend running with a yaml file config.

### Generation Notes

- It is recommended to use the ftfy module to fix the datasets. This can be enabled by the `--use_ftfy` argument.
- The NeoXTokenizer uses the HuggingFace library's inbuilt tokenizer and handles NFC normalization on its own. When using this tokenizer_type, it is recommended to set the `--ftfy_normalizer` argument to `None`. For the `GPT2Tokenizer`, use the default `NFC` value for the normalizer.
- To process HDF5 for training, we recommend using multi-processing. Moreover, we suggest using several input files such that the totalnum,ber of input files are greater than or equal to the number of processes provided by `--processes`. Note that this requires a high-spec CPU server, which can handle not only the concurrent running processes in RAM but also the I/O for reads and writes. If the I/O of the server is slow, the processes can appear to be hung for a very long while.
- For very large dataset (with several files with each file in the order of GBs) the recommendation is to split the data into smaller subsets and write out each subset. You can then mix all HDF5 in a common folder for use by the data pipeline, or just provide the locations of each subset in a list. The overall time to write out HDF5 can depend on the CPU server used.
- It is better to split the input dataset into multiple files, with similar size to leverage the full potential of parallel processing.
- For [CodeGen](https://arxiv.org/pdf/2203.13474.pdf) models processing please use `GPT2Tokenizer` along with the updated vocab files such that vocabulary of GPT-2 is extended by special tokens representing repeating tokens of tabs and white spaces.

### Output files structure

The output directory will contain a bunch of `h5` files as shown below (with 2 processes):

```bash
<path/to/output_dir>
├── checkpoint_0.txt
├── checkpoint_1.txt
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

Here `data_params.json` is the file which stores the parameters used for generating this set of files. `checkpoint_*.txt` can be used for resuming the processing in case the run script gets killed for some reason. There is one `checkpoint_*.txt` file for each process. To use this file, simply resume the previous command that you ran along with additional command line argument `--resume_from_checkpoint`

The `h5_dataset_stats` section in the generated `data_params.json` file contains the statistics on the generated dataset. The statistics are as follows:

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
