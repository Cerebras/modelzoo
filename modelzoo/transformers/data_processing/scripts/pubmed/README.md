# PubMed Dataset

PubMed dataset comprises of data from PubMed Baseline, Update and Full text commercial files. The hyperlinks to access the raw data for each of these are as below:

1. PubMed Baseline Abstracts : ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
2. PubMed UpdateFiles Abstracts: ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/
3. PubMed FullText Commercial: ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/

The following vocab files for PubMed dataset are located in [transformers/vocab](../../../../vocab)

* `uncased_pubmed_abstracts_and_fulltext_vocab.txt` - Size: 30522
* `uncased_pubmed_abstracts_only_vocab.txt` - Size: 28895

The following configuration files related to PubMed dataset are available at [transformers/bert/tf/configs](../../../../bert/tf/configs)

* `params_pubmedbert_base_msl128.yaml` 
* `params_pubmedbert_base_msl512.yaml`
* `params_pubmedbert_large_msl128.yaml`
* `params_pubmedbert_large_msl512.yaml`

The PubMed dataset creation for BERT pre-training can be broken down into 4 parts, namely

1. Downloading 
2. Formatting
3. Sharding  
4. Writing TFRecords 

### **Download**
In this stage, the dataset files are downloaded from the hyperlinks above and extracted to a subfolder.

Please refer to code in [Downloader.py](./preprocess/Downloader.py) for more details.

Please clone the `pubmed_parser` code from [this repo](https://github.com/titipata/pubmed_parser) in to the [preprocess](./preprocess/) directory.

### **Format**
In this stage, the raw data from extracted files is written to text files with one document/abstract in a single line and a newline separating documents. The filesize limit is set to 5GB inorder to ensure that files can be loaded into memory during subsequent stages. 

Please refer to [TextFormatting.py](./preprocess/TextFormatting.py) for more details

### **Shard**
In this stage,

1. Formatted files from stage 2 are loaded into memory and the total number of documents are counted
2. A fraction of these documents are distributed into `test` subset based on `fraction_test_set` argument. The remaining documents go into `training` dataset
3. The articles are segmented into individual sentences using `nltk` package and written to text files with one sentence in a line and couments separated by newline
4. The whole process is speeded up by using `multiprocessing`

Please refer to [TextSharding.py](./preprocess/TextSharding.py) for more details


### **Writing TFRecords**
The preprocessed dataset from stage 3 is now used to write to TFRecords. The examples written to TFrecords are targeted towards Masked Language Model(MLM) and Next Sentence Prediction(NSP) tasks of pretraining. The dataset written into TFrecords depends on

- `dupe_factor`:  Number of times to duplicate the dataset and write to TFrecords. Each of the duplicate example has a different random mask
- `max_sequence_length`: Maximum number of tokens to be present in a single example
-`max_predictions_per_seq`: Maximum number of tokens that can be masked per example
- `masked_lm_prob`: Masked LM Probability
- `do_lower_case`: Whether the tokens are to be converted to lower case or not
- `mask_whole_word`: Whether to mask all the tokens in a word or not


The examples in the TFRecords have the following key/values in itsfeatures dictionary:

`input_ids`:
Input token ids, padded with `0`'s to `max_sequence_length`.
Type: `int64`

`input_mask`:
Mask for padded positions. Has `0`'s on padded positions and `1`'s elsewhere in TFRecords.
BERT Model expects the `input_mask` to have 1's in padded positions and 0's elsewhere during pretraining. 
Therefore, when reading data from TFRecords, be sure to invert the `input_mask`. 
This is done in [BertTfRecordsProcessor](../../../../bert/tf/input/BertTfRecordsProcessor.py#L71)
Type: `int32`

`segment_ids`:
Segment ids. Has `0`'s on the positions corresponding to the first segment and `1`'s on positions corresponding to the second segment. The padded positions correspond to `0`.
Type: `int32`

`masked_lm_ids`:
Ids of masked tokens, padded with `0`'s to `max_predictions_per_seq` to accommodate a variable number of masked tokens per sample.
Type: `int32`

`masked_lm_positions`:
Positions of masked tokens in the `input_ids` tensor, padded with `0`'s to `max_predictions_per_seq`.
Type: `int32`

`masked_lm_weights`:
Mask for `masked_lm_ids` and `masked_lm_positions`. Has values `1.0` on the positions corresponding to actually masked tokens in the given sample and 0.0 elsewhere
Type: `float32`

`next_sentence_labels`: Carries the next sentence labels.
Type: `int32`


Please refer to `transformers/bert/tf/input/create_tfrecords.py` for more details.

### _**pubmedbert_prep.py**_

This scipt is a wrapper which performs the above four stages of PubMedBert dataset creation based on the action argument passed. Uses `multiprocessing` package to process commands in parallel wherever possible.

```
usage: pubmedbert_prep.py [-h]
                          [--action {sharding,text_formatting,create_tfrecord_files,download}]
                          --dataset {pubmed_baseline,pubmed_fulltext,pubmed_open_access,all,pubmed_daily_update}
                          --output_dir OUTPUT_DIR 
                          [--input_files INPUT_FILES]
                          [--n_training_shards N_TRAINING_SHARDS]
                          [--n_test_shards N_TEST_SHARDS]
                          [--fraction_test_set FRACTION_TEST_SET]
                          [--segmentation_method {nltk}]
                          [--n_processes N_PROCESSES]
                          [--random_seed RANDOM_SEED]
                          [--dupe_factor DUPE_FACTOR]
                          [--masked_lm_prob MASKED_LM_PROB]
                          [--max_seq_length MAX_SEQ_LENGTH]
                          [--max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ]
                          [--do_lower_case] 
                          [--mask_whole_word]
                          [--vocab_file VOCAB_FILE]
                          [--short_seq_prob SHORT_SEQ_PROB]

Preprocessing Application for PubMedBert Dataset

optional arguments:
  -h, --help            show this help message and exit
  --action {sharding,text_formatting,create_tfrecord_files,download}
                        Specify the action you want the app to take. e.g.,
                        generate vocab, segment, create tfrecords
  --dataset {pubmed_baseline,pubmed_fulltext,pubmed_open_access,all,pubmed_daily_update}
                        Specify the dataset to perform --action on
  --output_dir OUTPUT_DIR
                        Output parent folder where files/subfolders are
                        written to
  --input_files INPUT_FILES
                        Specify the input path for files
  --n_training_shards N_TRAINING_SHARDS
                        Specify the number of training shards to generate
  --n_test_shards N_TEST_SHARDS
                        Specify the number of test shards to generate
  --fraction_test_set FRACTION_TEST_SET
                        Specify the fraction (0..1) of the data to withhold
                        for the test data split (based on number of sequences)
  --segmentation_method {nltk}
                        Specify your choice of sentence segmentation
  --n_processes N_PROCESSES
                        Specify the max number of processes to allow at one
                        time
  --random_seed RANDOM_SEED
                        Specify the base seed to use for any random number
                        generation
  --dupe_factor DUPE_FACTOR
                        Specify the duplication factor
  --masked_lm_prob MASKED_LM_PROB
                        Specify the probability for masked lm
  --max_seq_length MAX_SEQ_LENGTH
                        Specify the maximum sequence length
  --max_predictions_per_seq MAX_PREDICTIONS_PER_SEQ
                        Specify the maximum number of masked words per
                        sequence
  --do_lower_case       pass this flag to lower case the input text; should be
                        True for uncased models and False for cased models
  --mask_whole_word     whether to use whole word masking rather than per-
                        WordPiece masking.
  --vocab_file VOCAB_FILE
                        Specify absolute path to vocab file to use)
  --short_seq_prob SHORT_SEQ_PROB
                        probability of creating sequences which are shorter
                        than the maximum sequence length

```

**Note**: One important input parameter to be aware of is the `input_files`. If sometimes, one would like to skip a few preprocessing steps, then this parameter could be used to pass the output folder from a previous step and trigger the step interested using `action` parameter.

```
python pubmedbert_prep.py --action=<action interested> --dataset=<dataset name> --output_dir=<path to new output directory> --input_files=<path to previously generated formatted folder>
```
For example: If we would like to start with `sharding`, then
```
# Generates sharded files at /tmp/new_output_dir/pubmed_fulltext/sharded

python pubmedbert_prep.py --action=sharding --dataset=pubmed_fulltext --output_dir=/tmp/new_output_dir --input_files=/tmp/output_dir/pubmed_fulltext/formatting

```

### _**create_pubmed_datasets.sh**_

Inorder to generate TFrecords end-to-end, please use this shell script. The script invokes `pubmedbert_prep.py` with `action` argument - `download`, `text_formatting`, `sharding`, `create_tfrecord_files`
in that order to generate TFrecord files.

```
source create_pubmed_datasets.sh ${OUTPUTDIR} ${DATASET_NAME} ${VOCAB_FILE}

    where OUTPUT_DIR:   Output folder where downloaded, formatted, sharded files and TFrecords are stored
          DATASET_NAME: One of the following choices:
                        pubmed_daily_update 
                        pubmed_baseline
                        pubmed_fulltext
          VOCAB_FILE: Location to vocab file which contains WordPiece to id mapping.
        

Example Usage: source create_pubmed_datasets.sh /tmp/pubmed_tfrecords pubmed_baseline ../vocab/uncased_pubmed_abstracts_and_fulltext_vocab.txt

```
Refer to [vocab directory](../../vocab) for PubMed and Google vocab files.

```
Directory Structure: 

Output directory = <args.output_dir>/<args.dataset>. Subfolder structure for this output directory is as shown below

<args.output_dir>/<args.dataset> (Example: /tmp/pubmed_tfrecords/pubmed_baseline)
├── download
│   ├── extracted
│   │   ├── <>.xml
│   │   ├── <>.xml
│   ├── <>.xml.gz
│   ├── <>.xml.gz
├── formatted
│   ├── <>.txt
│   ├── <>.txt
├── sharded
│   ├── test
│   │   ├── <>.txt
│   │   └── <>.txt
│   └── training
│       ├── <>.txt
│       └── <>.txt
└── tfrecord
    ├── uncased_msl128_mp20_wwmTrue_dupe5
    │   ├── params.json
    │   ├── test
    │   │   ├── <>.tfrecords
    │   │   ├── <>.tfrecords
    │   └── training
    │   │   ├── <>.tfrecords
    │   │   ├── <>.tfrecords    
    └── uncased_msl512_mp80_wwmTrue_dupe5
    │   ├── params.json
    │   ├── test
    │   │   ├── <>.tfrecords
    │   │   ├── <>.tfrecords
    │   └── training
    │   │   ├── <>.tfrecords
    │   │   ├── <>.tfrecords 

```
