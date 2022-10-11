# Scripts to Generate TFRecords

## Dataset preparation

The following plain text files need to be prepared before generating the TFRecords:
- Metadata file
  - Contains paths to raw data files (one data file per line). The paths are relative to `input_files_prefix`.
  - Examples of Metadata files can be found in the folder [metadata](../../../../data_processing/scripts/owt/metadata/).
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
  - Examples of vocab files can be found in the folder [vocab](../../../../vocab/).

## create_tf_records.py

This script generates TFRecords for both the Masked Language Model (MLM) and Next Sentence Prediction (NSP) pre-training tasks using the BERT model.

### Description

A high level overview of the implementation is as follows:

1. Given a list of raw text documents, generate raw examples by concatenating the two parts `tokens-a` and `tokens-b` as follows:

        [CLS] <tokens-a> [SEP] <tokens-b> [SEP]

    where:

    - `tokens-a` is a list of tokens taken from the current document. The list is of random length (less than `max_seq_length`).
    - `tokens-b` is a list of tokens chosen based on the randomly set `next_sentence_labels`. The list is of length `max_seq_length - len(<tokens-a>) - 3` (to account for `[CLS]` and `[SEP]` tokens).

    If `next_sentence_labels` is equal to `1`, (that is, if set to `1` with `0.5` probability):
    
            "tokens-b" is a list of tokens from sentences chosen randomly from different documents.
            
    else:
    
            "tokens-b" is a list of tokens taken from the same document and is a continuation of "tokens-a" in the document.

    > **Note**: Based on the probability `short_seq_prob`, the token sequence can have a length of `min_short_seq_length` instead of `max_seq_length`.

2. Map sample tokens to its indices in the vocabulary file.

3. Mask the raw examples based on `max_predictions_per_seq` and `mask_whole_word` parameters.

4. Pad the masked example to `max_seq_length` (if less than `max_seq_length`).

5. Create the feature dictionary that will be serialized into TFRecords with the features described in [Table 1](#table-1-data-features-in-the-generated-tfrecords).

#### Table 1: Data features in the generated TFRecords
Feature name | Data type | Sequence length | Description
--- | --- | --- | ---
`input_ids` | `tf.int64` | `max_seq_length` | Input token IDs.
`input_mask` | `tf.int64` | `max_seq_length` | Mask for padded positions (has values `1` on the padded positions, and `0` elsewhere).
`masked_lm_positions` | `tf.int64` | `max_predictions_per_seq` | Positions of masked tokens in the sequence.
`masked_lm_ids` | `tf.int64` | `max_predictions_per_seq` | IDs of masked tokens.
`masked_lm_weights` | `tf.float32` | `max_predictions_per_seq` | Mask for `masked_lm_ids` and `masked_lm_positions` (has values `1.0` on the positions corresponding to masked tokens, and `0.0` elsewhere).
`segment_ids` | `tf.int64` | `max_seq_length` | Segment IDs (has values `0` on the positions corresponding to the first segment, and `1` on the positions corresponding to the second segment).
`next_sentence_labels` | `tf.int64` | `1` | NSP label (has value `1` if second segment is next sentence, and `0` otherwise).

The TFRecords generated from this script can be used for pretraining using the dataloader script [BertTfRecordsProcessor.py](../BertTfRecordsProcessor.py). For more details, refer to [sentence_pair_processor.py](../../../../data_processing/sentence_pair_processor.py) and [create_tfrecords.py](./create_tfrecords.py).

```bash
Usage: create_tfrecords.py [-h] --metadata_files METADATA_FILES
                           [METADATA_FILES ...]
                           [--multiple_docs_in_single_file]
                           [--multiple_docs_separator MULTIPLE_DOCS_SEPARATOR]
                           [--single_sentence_per_line]
                           [--input_files_prefix INPUT_FILES_PREFIX]
                           --vocab_file VOCAB_FILE [--split_num SPLIT_NUM]
                           [--do_lower_case] [--max_seq_length MAX_SEQ_LENGTH]
                           [--dupe_factor DUPE_FACTOR]
                           [--short_seq_prob SHORT_SEQ_PROB]
                           [--min_short_seq_length MIN_SHORT_SEQ_LENGTH]
                           [--masked_lm_prob MASKED_LM_PROB]
                           [--max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ]
                           [--spacy_model SPACY_MODEL] [--mask_whole_word]
                           [--output_dir OUTPUT_DIR]
                           [--num_output_files NUM_OUTPUT_FILES] [--name NAME]
                           [--seed SEED]

Required arguments:
  --metadata_files METADATA_FILES [METADATA_FILES ...]
                      Path to the text file containing a list of file names
                      corresponding to the raw input documents to be
                      processed and stored; Multiple metadata
                      files must be separated by a space.
  --vocab_file VOCAB_FILE
                      Path to the vocabulary file.

Optional arguments:
  -h, --help            Show this help message and exit.
  --multiple_docs_in_single_file
                        Pass this flag when a single text file contains
                        multiple documents separated by
                        <multiple_docs_separator> (default: False).
  --multiple_docs_separator MULTIPLE_DOCS_SEPARATOR
                        String which separates multiple documents in a
                        single text file. If newline character,
                        pass `\n`. There can only
                        be one separator string for all the documents.
                        (default: `\n`)
  --single_sentence_per_line
                        Pass this flag when the document is already
                        split into sentences, with one sentence in
                        each line. There is no requirement for further
                        sentence segmentation of a document
                        (default: False).
  --input_files_prefix INPUT_FILES_PREFIX
                        Prefix to be added to paths of the input
                        files. For example, can be a directory where
                        raw data is stored if the paths are relative.
  --split_num SPLIT_NUM
                        Number of input files to read at a  given
                        time. It can speed up reading the data files through parallel 
                        processing (default: 1000).
  --do_lower_case       Pass this flag to lower case the input text.
                        Must be True for uncased models and False
                        for cased models. Note that if your vocab file has only 
                        lowercased letters, and you did not provide this flag, a lot of 
                        tokens will be mapped to `[UNK]` and vice versa (default: False).
  --max_seq_length MAX_SEQ_LENGTH
                        Maximum sequence length (default: 128).
  --dupe_factor DUPE_FACTOR
                        Number of times to duplicate the input data (with
                        different masks). For static masking, it is a common practice to 
                        duplicate the data, and provide different masking for the same 
                        input to learn more generalizable features (default: 10).
  --short_seq_prob SHORT_SEQ_PROB
                        Probability of creating sequences that are
                        shorter than the maximum sequence length
                        (default: 0.1).
  --min_short_seq_length MIN_SHORT_SEQ_LENGTH
                        The minimum number of tokens to be present in an
                        example if short sequence probability > 0. If None,
                        defaults to 2. Allowed values are [2, max_seq_length -
                        3) (default: None)
  --masked_lm_prob MASKED_LM_PROB
                        Probability of replacing input tokens with a mask token `[MASK]` 
                        for a language modeling task (default: 0.15).
  --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
                        Maximum number of masked tokens per
                        sequence (default: 20).
  --spacy_model SPACY_MODEL
                        The spaCy model to load (either a shortcut
                        link, a package name or a path). It is used to process the data 
                        files and segment them into sentences if the flag 
                        `single_sentence_per_line` is not set. Default model is set to 
                        the small English pipeline trained on written web text.
                        (default: en_core_web_sm).
  --mask_whole_word     Set to True to use whole word masking and
                        False to use per-WordPiece masking
                        (default: False).
  --output_dir OUTPUT_DIR
                        Directory where TFRecords will be stored
                        (default: "./tfrecords/").
  --num_output_files NUM_OUTPUT_FILES
                        TFRecords will be separated into the
                        specified number of files on disk. The larger the number of 
                        files, the easier it becomes to parallelize writing/reading of 
                        the TFRecords (default: 10).
  --name NAME           Name of the dataset, i.e., prefix to use
                        for TFRecord names (default: "examples").
  --seed SEED           Seed for the random number generators (default: 0).    
```

## create_tf_records_mlm_only.py

This script generates TFRecords for MLM-only pre-training task using the BERT model. The BERT model during pre-training will contain only the MLM head, and not the NSP head.

The script can generate two types of TFRecords based on `disable_masking` setting:

- If `disable_masking=True`, then raw text examples are written to TFRecords and these raw tokens will be masked dynamically during the pre-training process. Specifically, each raw text token is encoded with `UTF-8` into bytes list, and added to the variable length sequence of tokens that is written to the TFRecords.
- If `disable_masking=False`, the TFRecords will contain static masked examples as described in [Table 1](#table-1-data-features-in-the-generated-tfrecords), except for the last two features `segment_ids` and `next_sentence_labels` are being omitted here. 


The dataloaders that can be used for reading the TFRecords generated by this script are:
1. [BertMlmOnlyTfRecordsDynamicMaskProcessor.py](../BertMlmOnlyTfRecordsDynamicMaskProcessor.py) if `disable_masking=True` was used
2. [BertMlmOnlyTfRecordsStaticMaskProcessor.py](../BertMlmOnlyTfRecordsStaticMaskProcessor.py) if `disable_masking=False` was used

### Description

A high level overview of the implementation is as follows:

1. Generate raw examples with a list of `tokens` as follows:

        [CLS] <tokens-list> [SEP]

    where:

    - `tokens-list` is a list of tokens taken from the current document. The list is generated using a sliding window approach that includes tokens from the previous example based on the `overlap_size`. The list is of random length (less than `max_seq_length`) or can have the length `min_short_seq_length` based on `short_seq_prob`.
    
  > **Note**: If the flag `allow_cross_document_examples` was set `True`, the `tokens-list` can come from multiple documents and the token `document_separator_token` gets added to the list between the tokens of different documents.

 If `disable_masking=False`, then proceed to steps `2`-`5` as described [above](#description).

For more details, refer to [mlm_only_processor.py](../../../../data_processing/mlm_only_processor.py) and [create_tfrecords_mlm_only.py](./create_tfrecords_mlm_only.py).


```bash
Usage: create_tfrecords_mlm_only.py [-h] --metadata_files METADATA_FILES
                                    [METADATA_FILES ...]
                                    [--multiple_docs_in_single_file]
                                    [--multiple_docs_separator MULTIPLE_DOCS_SEPARATOR]
                                    [--single_sentence_per_line]
                                    [--allow_cross_document_examples]
                                    [--document_separator_token DOCUMENT_SEPARATOR_TOKEN]
                                    [--overlap_size OVERLAP_SIZE]
                                    [--buffer_size BUFFER_SIZE]
                                    [--input_files_prefix INPUT_FILES_PREFIX]
                                    --vocab_file VOCAB_FILE [--do_lower_case]
                                    [--max_seq_length MAX_SEQ_LENGTH]
                                    [--dupe_factor DUPE_FACTOR]
                                    [--short_seq_prob SHORT_SEQ_PROB]
                                    [--min_short_seq_length MIN_SHORT_SEQ_LENGTH]
                                    [--disable_masking]
                                    [--masked_lm_prob MASKED_LM_PROB]
                                    [--max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ]
                                    [--spacy_model SPACY_MODEL]
                                    [--mask_whole_word]
                                    [--output_dir OUTPUT_DIR]
                                    [--num_output_files NUM_OUTPUT_FILES]
                                    [--name NAME] [--seed SEED]

Required arguments:
  --metadata_files METADATA_FILES [METADATA_FILES ...]
                      Path to the text file containing a list of file
                      names corresponding to the raw input documents
                      to be processed and stored; Multiple metadata
                      files must be separated by a space.
  --vocab_file VOCAB_FILE
                        Path to the vocabulary file.


Optional arguments:
  -h, --help            Show this help message and exit.
  --multiple_docs_in_single_file
                        Pass this flag when a single text file contains
                        multiple documents separated by
                        <multiple_docs_separator> (default: False).
  --multiple_docs_separator MULTIPLE_DOCS_SEPARATOR
                        String which separates multiple documents in a single
                        text file. If newline character, pass `\n`.
                        There can only be one separator string for
                        all the documents.
                        (default: `\n`)
  --single_sentence_per_line
                        Pass this flag when the document is already
                        split into sentences, with one sentence in
                        each line. There is no requirement for further
                        sentence segmentation of a document
                        (default: False).
  --allow_cross_document_examples
                        Pass this flag when tokens for the same example can come from 
                        multiple documents (default: False).
  --document_separator_token DOCUMENT_SEPARATOR_TOKEN
                        If an example can span multiple documents, use this separator to 
                        indicate separate tokens of different documents 
                        (default: `[SEP]`).
  --overlap_size OVERLAP_SIZE
                        The overlap size between tokens of the current and previous 
                        example. Defaults to None, which sets the overlap to 
                        max_seq_len/4 (default: None).
  --buffer_size BUFFER_SIZE
                        Number of tokens to be processed at a time (default: 1000000).
  --input_files_prefix INPUT_FILES_PREFIX
                        Prefix to be added to paths of the input
                        files. For example, can be a directory where
                        raw data is stored if the paths are relative.
  --do_lower_case       Pass this flag to lower case the input text.
                        Must be True for uncased models and False for cased models. Note 
                        that if your vocab file has only lowercased letters, and you did 
                        not provide this flag, a lot of tokens will be mapped to `[UNK]` 
                        and vice versa (default: False).
  --max_seq_length MAX_SEQ_LENGTH
                        Maximum sequence length (default: 128).
  --dupe_factor DUPE_FACTOR
                        Number of times to duplicate the input data (with
                        different masks). For static masking, it is a common practice to 
                        duplicate the data, and provide different masking for the same 
                        input to learn more generalizable features (default: 10).
  --short_seq_prob SHORT_SEQ_PROB
                        Probability of creating sequences that are
                        shorter than the maximum sequence length
                        (default: 0.1).
  --min_short_seq_length MIN_SHORT_SEQ_LENGTH
                        The minimum number of tokens to be present in an
                        example if short sequence probability > 0. If None,
                        defaults to 2 + overlap_sizeAllowed values are between
                        [2 + overlap_size, max_seq_length-2) (default: None)
  --disable_masking     If False, TFRecords will be stored with
                        static masks. If True, masking will happen
                        dynamically during training (default: False).
  --masked_lm_prob MASKED_LM_PROB
                        Probability of replacing input tokens with a mask token `[MASK]` 
                        for a language modeling task (default: 0.15).
  --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
                        Maximum number of masked LM predictions per
                        sequence (default: 20).
  --spacy_model SPACY_MODEL
                        The spaCy model to load (either a shortcut
                        link, a package name or a path). It is used to process the data 
                        files and segment them into sentences if the flag 
                        `single_sentence_per_line` is not set. Default model is set to 
                        the small English pipeline trained on written web text.
                        (default: en_core_web_sm).
  --mask_whole_word     Set to True to use whole word masking and
                        False to use per-WordPiece masking
                        (default: False).
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

# Create your own TFRecords

## Description

The generated TFRecords will be used for pre-training BERT models using the dataloader script [BertTfRecordsProcessor.py](../BertTfRecordsProcessor.py) for MLM+NSP or [BertMlmOnlyTfRecordsStaticMaskProcessor.py](../BertMlmOnlyTfRecordsStaticMaskProcessor.py) for only MLM. Thus, to ensure compatibility when parsing the TFRecords, the five steps described [above](#description) need to be followed when generating your own TFRecords. Also, you need to ensure that all the features described in [Table 1](#table-1-data-features-in-the-generated-tfrecords) are present in the TFRecords you have generated, except when you pre-train for MLM only then you can omit the last two features `segment_ids` and `next_sentence_labels`.

> **Note**: If you need to disable static masking for MLM only pre-training and generate TFRecords that works with the dataloader in [BertMlmOnlyTfRecordsDynamicMaskProcessor.py](../BertMlmOnlyTfRecordsDynamicMaskProcessor.py), then you only need to perform step 1 described [above](#description) and encode the raw text tokens with `UTF-8` into bytes list, then add them to a variable length sequence of tokens and write the sequence into TFRecords.

## create_dummy_tfrecords.py

The script [create_dummy_tfrecords.py](./create_dummy_tfrecords.py) shows a simplified version of the steps taken to write a single data sample into TFRecords. It is meant as a guide to show you the necessary steps need to be taken when generating your own TFRecords to be compatible with either one of the three dataloader scripts based on the chosen settings:
- If `mlm_only=False`, then the generated TFRecords can be parsed by the dataloader at [BertTfRecordsProcessor.py](../BertTfRecordsProcessor.py).
- If `mlm_only=True` and `disable_masking=False`, then the generated TFRecords can be parsed by the dataloader at [BertMlmOnlyTfRecordsStaticMaskProcessor.py](../BertMlmOnlyTfRecordsStaticMaskProcessor.py).
- If `mlm_only=True` and `disable_masking=True`, then the generated TFRecords can be parsed by the dataloader at [BertMlmOnlyTfRecordsDynamicMaskProcessor.py](../BertMlmOnlyTfRecordsDynamicMaskProcessor.py).
 
Feel free to try this script or modify it as you need.

```bash
usage: create_dummy_tfrecords.py [-h] [--mlm_only] [--disable_masking]

optional arguments:
  -h, --help         show this help message and exit
  --mlm_only         If False, TFRecords will contain two extra features for
                     NSP which are `segment_ids` and `next_sentence_labels`.
                     (default: False)
  --disable_masking  If False, TFRecords will be stored with static masks. If
                     True, masking will happen dynamically during training.
                     (default: False)

``` 
