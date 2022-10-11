# Scripts to Generate CSV files for BERT and RoBERTa pre-training

## Generating data for BERT

The `create_csv_static_masking.py` script generates CSV files for both the Masked Language Model (MLM) and Next Sentence Prediction (NSP) pre-training tasks for the BERT model.

NOTE: This script requires `spacy` and `tensorflow` to be installed in your Python environment in order to parse and tokenize the raw data. However, neither package is required for using the data loaders that come with this model.

### Description

A high level overview of the implementation is as follows:

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

The CSV generated from this script can be used for pretraining using the dataloader script [BertCSVDataProcessor.py](../BertCSVDataProcessor.py).


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

optional arguments:
  -h, --help            show this help message and exit
  --metadata_files METADATA_FILES [METADATA_FILES ...]
                        path to text file containing a list of file names
                        corresponding to the raw input documents to be
                        processed and stored; can handle multiple metadata
                        files separated by space (default: None)
  --multiple_docs_in_single_file
                        Pass this flag when a single text file contains
                        multiple documents separated by
                        <multiple_docs_separator> (default: False)
  --multiple_docs_separator MULTIPLE_DOCS_SEPARATOR
                        String which separates multiple documents in a single
                        text file. If newline character, pass \nThere can only
                        be one separator string for all the documents.
                        (default: )
  --single_sentence_per_line
                        Pass this flag when the document is already split into
                        sentences withone sentence in each line and there is
                        no requirement for further sentence segmentation of a
                        document (default: False)
  --input_files_prefix INPUT_FILES_PREFIX
                        prefix to be added to paths of the input files. For
                        example, can be a directory where raw data is stored
                        if the paths are relative (default: )
  --vocab_file VOCAB_FILE
                        path to vocabulary (default: None)
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

## Generating data for RobERTa

The `create_csv_mlm_only_static_masking.py` script generates CSV files for the Masked Language Model (MLM) pre-training task for the RoBERTa model.

NOTE: This script requires `spacy` and `tensorflow` to be installed in your Python environment in order to parse and tokenize the raw data. However, neither package is required for using the data loaders that come with this model.

### Description

A high level overview of the implementation is as follows:

1. Given a list of raw text documents, generate raw examples by adding two special tokens to every sequence of `<tokens>` as follows:

        `[CLS] <tokens> [SEP]`

2. Mask the raw examples based on `max_predictions_per_seq` and `mask_whole_word` parameters.

3. Pad the masked example to `max_sequence_length` if less than MSL.

The CSV generated from this script can be used for pretraining using the dataloader script [BertCSVDataProcessor.py](../BertCSVDataProcessor.py). To train RoBERTa, make sure that `disable_nsp` is set to `True` in the `model` section of the YAML.


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

optional arguments:
  -h, --help            show this help message and exit
  --metadata_files METADATA_FILES [METADATA_FILES ...]
                        path to text file containing a list of file names
                        corresponding to the raw input documents to be
                        processed and stored; can handle multiple metadata
                        files separated by space (default: None)
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
                        (default: )
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
                        if the paths are relative (default: )
  --vocab_file VOCAB_FILE
                        path to vocabulary
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
