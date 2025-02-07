# Introduction

[OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/) (OWT) is one of the main datasets used by [RoBERTa](https://arxiv.org/abs/1907.11692) and, thus, can be used to train [BERT](https://arxiv.org/abs/1810.04805) as well. We offer four scripts to generate CSV files for OWT dataset that can be used to train BERT models:
1. [create_csv.py](./create_csv.py): generates features for both the Masked Language Model (MLM) and Next Sentence Prediction (NSP) pre-training tasks, and intended to be used with dynamic masking performed on the fly in [BertCSVDynamicMaskDataProcessor.py](../../../data/nlp/bert/BertCSVDynamicMaskDataProcessor.py).
2. [create_csv_mlm_only.py](./create_csv_mlm_only.py): generates features for only the Masked Language Model (MLM) pre-training task, and intended to be used with dynamic masking performed on the fly in [BertCSVDynamicMaskDataProcessor.py](../../../data/nlp/bert/BertCSVDynamicMaskDataProcessor.py).
3. [create_csv_static_masking.py](./create_csv_static_masking.py): generates features for both the Masked Language Model (MLM) and Next Sentence Prediction (NSP) pre-training tasks. It performs masking during preprocessing and intended to be used with [BertCSVDataProcessor.py](../../../data/nlp/bert/BertCSVDataProcessor.py).
4. [create_csv_mlm_only_static_masking.py](./create_csv_mlm_only_static_masking.py): generates features for only the Masked Language Model (MLM) pre-training task. It performs masking during preprocessing and intended to be used with [BertCSVDataProcessor.py](../../../data/nlp/bert/BertCSVDataProcessor.py).

## Prerequisites

The preprocessing scripts for BERT relies on the [spaCy](https://spacy.io/) package, if not installed:

```bash
pip install spacy
python -m spacy download en
```

## Input data

### Data download and extraction

To download the OWT dataset and extract them, run:

```bash
bash ../../../../data_processing/scripts/owt/download_and_extract.sh
```

Note that [download_and_extract.sh](../../../data_preparation/nlp/owt/download_and_extract.sh) may take a while to complete, as it unpacks 40GB of data (8,013,770 documents). Upon completion, the script will produce `openwebtext` directory in the same location. The directory has multiple subdirectories, each containing a collection of `*.txt` files of raw text data (one document per `.txt` file).

### Input files format

The following plain text files need to be prepared before generating the CSV files:
- Metadata file
  - Contains paths to raw data files (one data file per line). The paths are relative to `input_files_prefix` which should be the path to the `openwebtext` directory that was produced in the step [data download and extraction](#data-download-and-extraction).
  - Examples of Metadata files for OWT can be found in the directory [metadata](../../../data_preparation/nlp/owt/metadata/).
- Data files
  - If `multiple_docs_in_single_file` is set to `True`, then each data file can contain plain text that comes from multiple documents separated by `multiple_docs_separator` (e.g. `[SEP]`) as follows:
    ``` 
    <doc-1>
    [SEP]
    <doc-2>
    ```
    If `multiple_docs_in_single_file` is set to `False`, then each data file should contain plain text that comes from a single document.
  - If `single_sentence_per_line` is set to `True`, then each data file should contain a single sentence per line. Otherwise, the raw text will be segmented into sentences using the specified `spacy_model`.
    
    > **Note**: Ensure that the `spacy_model` is compatible with the language of the raw text.
- Vocab file
  - Contains a single token per line.
  - Includes the special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, and `[MASK]`.
  - Needs to be compatible with the raw text data (e.g. same language).
  - Is case sensitive. If the vocab file has only lowercased letters, then you need to set `do_lower_case` to `True` in order to avoid mapping many input tokens to `[UNK]`.
  - Examples of vocab files can be found in the folder [vocab](../../../models/vocab/).

## Running the scripts

### MLM+NSP scripts

Once the input files are prepared, you can start generating CSV files with MLM and NSP features using [create_csv_static_masking.py](./create_csv_static_masking.py) for static masking. The arguments for this script are detailed below:

```bash
usage: create_csv_static_masking.py [-h] --metadata_files METADATA_FILES
                                    [METADATA_FILES ...]
                                    [--multiple_docs_in_single_file]
                                    [--multiple_docs_separator MULTIPLE_DOCS_SEPARATOR]
                                    [--single_sentence_per_line]
                                    [--input_files_prefix INPUT_FILES_PREFIX]
                                    --vocab_file VOCAB_FILE
                                    [--split_num SPLIT_NUM] [--do_lower_case]
                                    [--max_seq_length MAX_SEQ_LENGTH]
                                    [--dupe_factor DUPE_FACTOR]
                                    [--short_seq_prob SHORT_SEQ_PROB]
                                    [--min_short_seq_length MIN_SHORT_SEQ_LENGTH]
                                    [--masked_lm_prob MASKED_LM_PROB]
                                    [--max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ]
                                    [--spacy_model SPACY_MODEL]
                                    [--mask_whole_word]
                                    [--output_dir OUTPUT_DIR]
                                    [--num_output_files NUM_OUTPUT_FILES]
                                    [--name NAME] [--init_findex INIT_FINDEX]
                                    [--seed SEED]

Required arguments:
  --metadata_files METADATA_FILES [METADATA_FILES ...]
                        path to text file containing a list of file names
                        corresponding to the raw input documents to be
                        processed and stored; can handle multiple metadata
                        files separated by space.
  --vocab_file VOCAB_FILE
                        path to vocabulary file.

optional arguments:
  -h, --help            show this help message and exit
  --multiple_docs_in_single_file
                        Pass this flag when a single text file contains
                        multiple documents separated by
                        <multiple_docs_separator> (default: False)
  --multiple_docs_separator MULTIPLE_DOCS_SEPARATOR
                        String which separates multiple documents in a single
                        text file. If newline character, pass \nThere can only
                        be one separator string for all the documents.
                        (default: `\n`)
  --single_sentence_per_line
                        Pass this flag when the document is already split into
                        sentences withone sentence in each line and there is
                        no requirement for further sentence segmentation of a
                        document (default: False)
  --input_files_prefix INPUT_FILES_PREFIX
                        prefix to be added to paths of the input files. For
                        example, can be a directory where raw data is stored
                        if the paths are relative. Defaults to current directory.
  --split_num SPLIT_NUM
                        number of input files to read at a given time for
                        processing. Defaults to 1000. (default: 1000)
  --do_lower_case       pass this flag to lower case the input text; should be
                        True for uncased models and False for cased models
                        (default: False)
  --max_seq_length MAX_SEQ_LENGTH
                        maximum sequence length (default: 128)
  --dupe_factor DUPE_FACTOR
                        number of times to duplicate the input data (with
                        different masks) (default: 10)
  --short_seq_prob SHORT_SEQ_PROB
                        probability of creating sequences which are shorter
                        than the maximum sequence length (default: 0.1)
  --min_short_seq_length MIN_SHORT_SEQ_LENGTH
                        The minimum number of tokens to be present in an
                        exampleif short sequence probability > 0.If None,
                        defaults to 2 Allowed values are [2, max_seq_length -
                        3) (default: None)
  --masked_lm_prob MASKED_LM_PROB
                        masked LM probability (default: 0.15)
  --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
                        maximum number of masked LM predictions per sequence
                        (default: 20)
  --spacy_model SPACY_MODEL
                        spaCy model to load, i.e. shortcut link, package name
                        or path. (default: en_core_web_sm)
  --mask_whole_word     whether to use whole word masking rather than per-
                        WordPiece masking. (default: False)
  --output_dir OUTPUT_DIR
                        directory where CSV files will be stored. Defaults to
                        ./csvfiles/. (default: ./csvfiles/)
  --num_output_files NUM_OUTPUT_FILES
                        number of files on disk to separate csv files into.
                        Defaults to 10. (default: 10)
  --name NAME           name of the dataset; i.e. prefix to use for csv file
                        names. Defaults to 'preprocessed_data'. (default:
                        preprocessed_data)
  --init_findex INIT_FINDEX
                        Index used in first output file. (default: 1)
  --seed SEED           random seed. Defaults to 0. (default: 0)
```

> **Note**: The script [create_csv.py](./create_csv.py) shares the same arguments as [create_csv_static_masking.py](./create_csv_static_masking.py), except for `dupe_factor` and `init_findex`.

### MLM-only scripts

Once the input files are prepared, you can start generating CSV files with only MLM features using [create_csv_mlm_only_static_masking.py](./create_csv_mlm_only_static_masking.py) for static masking. The arguments for this script are detailed below:

```bash
usage: create_csv_mlm_only_static_masking.py [-h] --metadata_files
                                             METADATA_FILES
                                             [METADATA_FILES ...]
                                             [--multiple_docs_in_single_file]
                                             [--overlap_size OVERLAP_SIZE]
                                             [--buffer_size BUFFER_SIZE]
                                             [--multiple_docs_separator MULTIPLE_DOCS_SEPARATOR]
                                             [--single_sentence_per_line]
                                             [--allow_cross_document_examples]
                                             [--document_separator_token DOCUMENT_SEPARATOR_TOKEN]
                                             [--input_files_prefix INPUT_FILES_PREFIX]
                                             --vocab_file VOCAB_FILE
                                             [--split_num SPLIT_NUM]
                                             [--do_lower_case]
                                             [--max_seq_length MAX_SEQ_LENGTH]
                                             [--dupe_factor DUPE_FACTOR]
                                             [--short_seq_prob SHORT_SEQ_PROB]
                                             [--min_short_seq_length MIN_SHORT_SEQ_LENGTH]
                                             [--masked_lm_prob MASKED_LM_PROB]
                                             [--max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ]
                                             [--spacy_model SPACY_MODEL]
                                             [--mask_whole_word]
                                             [--output_dir OUTPUT_DIR]
                                             [--num_output_files NUM_OUTPUT_FILES]
                                             [--name NAME]
                                             [--init_findex INIT_FINDEX]
                                             [--seed SEED]
Required arguments:
  --metadata_files METADATA_FILES [METADATA_FILES ...]
                        path to text file containing a list of file names
                        corresponding to the raw input documents to be
                        processed and stored; can handle multiple metadata
                        files separated by space.
  --vocab_file VOCAB_FILE
                        path to vocabulary file.

optional arguments:
  -h, --help            show this help message and exit
  --multiple_docs_in_single_file
                        Pass this flag when a single text file contains
                        multiple documents separated by
                        <multiple_docs_separator> (default: False)
  --overlap_size OVERLAP_SIZE
                        overlap size for generating sequences from buffered
                        data for mlm only sequencesDefaults to None, which
                        sets the overlap to max_seq_len/4. (default: None)
  --buffer_size BUFFER_SIZE
                        buffer_size number of elements to be processed at a
                        time (default: 1000000.0)
  --multiple_docs_separator MULTIPLE_DOCS_SEPARATOR
                        String which separates multiple documents in a single
                        text file. If newline character, pass \nThere can only
                        be one separator string for all the documents.
                        (default: `\n`)
  --single_sentence_per_line
                        Pass this flag when the document is already split into
                        sentences withone sentence in each line and there is
                        no requirement for further sentence segmentation of a
                        document (default: False)
  --allow_cross_document_examples
                        Pass this flag when examples can cross document
                        boundaries (default: False)
  --document_separator_token DOCUMENT_SEPARATOR_TOKEN
                        If examples can span documents, use this separator to
                        indicate separate tokens of current and next document
                        (default: [SEP])
  --input_files_prefix INPUT_FILES_PREFIX
                        prefix to be added to paths of the input files. For
                        example, can be a directory where raw data is stored
                        if the paths are relative. Defaults to current directory.
  --split_num SPLIT_NUM
                        number of input files to read at a given time for
                        processing. Defaults to 1000. (default: 1000)
  --do_lower_case       pass this flag to lower case the input text; should be
                        True for uncased models and False for cased models
                        (default: False)
  --max_seq_length MAX_SEQ_LENGTH
                        maximum sequence length (default: 128)
  --dupe_factor DUPE_FACTOR
                        number of times to duplicate the input data (with
                        different masks) if disable_masking is False (default:
                        10)
  --short_seq_prob SHORT_SEQ_PROB
                        probability of creating sequences which are shorter
                        than the maximum sequence length (default: 0.1)
  --min_short_seq_length MIN_SHORT_SEQ_LENGTH
                        The minimum number of tokens to be present in an
                        exampleif short sequence probability > 0.If None,
                        defaults to 2 Allowed values are [2, max_seq_length -
                        3) (default: None)
  --masked_lm_prob MASKED_LM_PROB
                        masked LM probability (default: 0.15)
  --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
                        maximum number of masked LM predictions per sequence
                        (default: 20)
  --spacy_model SPACY_MODEL
                        spaCy model to load, i.e. shortcut link, package name
                        or path. (default: en_core_web_sm)
  --mask_whole_word     whether to use whole word masking rather than per-
                        WordPiece masking. (default: False)
  --output_dir OUTPUT_DIR
                        directory where CSV files will be stored. Defaults to
                        ./csvfiles/. (default: ./csvfiles/)
  --num_output_files NUM_OUTPUT_FILES
                        number of files on disk to separate csv files into.
                        Defaults to 10. (default: 10)
  --name NAME           name of the dataset; i.e. prefix to use for csv file
                        names. Defaults to 'preprocessed_data'. (default:
                        preprocessed_data)
  --init_findex INIT_FINDEX
                        Index used in first output file. (default: 1)
  --seed SEED           random seed. Defaults to 0. (default: 0)
```

> **Note**: The script [create_csv_mlm_only.py](./create_csv_mlm_only.py) shares the same arguments as [create_csv_mlm_only_static_masking.py](./create_csv_mlm_only_static_masking.py), except for `dupe_factor` and `init_findex`.

## Output data

### Output directory structure

The output directory will contain CSV files as shown below:
```bash
<path/to/$OUTPUT_DIR>
├── data_params.json
├── meta.dat
├── $NAME-1.csv
├── $NAME-2.csv
├── $NAME-3.csv
└── $NAME-4.csv
```

### Output file data structure

#### Static masking scripts

The CSV files generated from the script [create_csv_static_masking.py](./create_csv_static_masking.py) can be used for pretraining BERT models and they contain the following features:

##### Table 1: Data features in the generated CVS files
Feature name | Data type | Sequence length | Description
--- | --- | --- | ---
`input_ids` | `int32` | `max_seq_length` | Input token IDs, padded with `0` to `max_seq_length`.
`attention_mask` | `int32` | `max_seq_length` | Mask for padded positions. Has values `0` on the padded positions and `1` elsewhere.
`token_type_ids` | `int32` | `max_seq_length` | Segment IDs. Has values `0` on the positions corresponding to the first segment, and `1` on positions corresponding to the second segment.
`masked_lm_positions` | `int32` | `max_predictions_per_seq` | Positions of masked tokens in the `input_ids` tensor, padded with `0` to `max_predictions_per_seq`.
`masked_lm_weights` | `float32` | `max_predictions_per_seq` | Mask for `masked_lm_positions`. Has values `1.0` on the positions corresponding to actually masked tokens in the given sample, and `0.0` elsewhere.
`labels` | `int32` | `max_predictions_per_seq` | IDs of masked tokens.
`next_sentence_label` | `int32` | `1` | Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair where 0 indicates sequence B is a continuation of sequence A, 1 indicates sequence B is a random sequence.

> **Note**: The script [create_csv_mlm_only_static_masking.py](./create_csv_mlm_only_static_masking.py) generates the same features except for `token_type_ids` and `next_sentence_label` are being omitted here.

We provide the following high level overview of the steps taken to generate the data features:

1. Given a list of raw text documents, generate raw examples by concatenating the two parts `tokens-a` and `tokens-b` as follows:

        `[CLS] <tokens-a> [SEP] <tokens-b> [SEP]`

    where:

    - `tokens-a` is a list of tokens taken from the current document. The list is of random length (less than the maximum sequence length or `MSL`).
    - `tokens-b` is a list of tokens chosen randomly to be either from the next sentence that comes after `tokens-a` or from another document. The list is of length `MSL-len(<tokens-a>)- 3` (to account for [CLS] and [SEP] tokens).

    If `next_sentence_labels` is 1:
    
            `tokens-b` is a tokenized sentence chosen randomly from different documents.
            
    else:
    
            `tokens-b` is the sentence that comes after `tokens-a` from the same document. The number of raw tokens depends on `short_sequence_prob` as well.

2. Mask the raw examples based on `max_predictions_per_seq` and `mask_whole_word` parameters.

3. Pad the masked example to `max_sequence_length` if less than MSL.

#### Dynamic masking scripts

The CSV files generated from the script [create_csv.py](./create_csv.py) can be used for pretraining BERT models and they contain the following features:

- `tokens`: Sequence of input tokens arranged in string format.

- `segment_ids`: Segment IDs arranged in string format. Has values `0` on the positions corresponding to the first segment, and `1` on positions corresponding to the second segment.

- `is_random_next`: Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair where 0 indicates sequence B is a continuation of sequence A, 1 indicates sequence B is a random sequence.

> **Note**: The script [create_csv_mlm_only.py](./create_csv_mlm_only.py) generates only the feature `tokens`, and omits the rest.
