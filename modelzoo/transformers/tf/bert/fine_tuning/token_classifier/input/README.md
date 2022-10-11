# BERT Token Classifier Input 

### NerDataProcessory.py

The features dictionary has the following key/values:

`input_ids`:
Input token ids, padded with 0's to max_sequence_length.
Shape: [batch_size, max_sequence_length].
Type: int64

`input_mask`:
Mask for padded positions. Has 0's on the padded positions and 1's elsewhere.
Shape: [batch_size, max_sequence_length]
Type: int32

`segment_ids`:
Segment ids. Has 0's on the positions
corresponding to the first segment and
1's on positions corresponding to the second segment.
The padded positions correspond to 0.
Shape: [batch_size, max_sequence_length]
Type: int32

`label_ids`
The label tensor of shape [batch_size, max_sequence_length].
Carries the labels for each token. Type: int32

___

## Generating TF records for Token Classifier

1. Download and preprocess the dataset from [BLURB dashboard](https://microsoft.github.io/BLURB/submit.html)

    ```bash
    wget https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz

    tar -xf data_generation.tar.gz
    ```
2. Refer to `README.md` from the `data_generation` (downloaded and extracted in Step 1) folder to download and preprocess the datasets for Named Entity Recognition (NER) - BC5CDR-Chem, BC5CDR-Disease, BC2GM, NCBI-Disease, JNLPBA

3. Generate TF Records from the raw data using the script `write_tfrecords_ner.py`

    ```python
    Example Usage: 

    python write_tfrecords_ner.py --data_dir=/cb/ml/language/datasets/blurb/data_generation/data/NCBI-disease --vocab_file=/cb/ml/language/datasets/pubmed_abstracts_baseline_fulltext_vocab/uncased_pubmed_abstracts_and_fulltext_vocab.txt --output_dir=/cb/ml/language/datasets/blurb/ner/ncbi-disease-tfrecords --do_lower_case
    ```

    Run `python write_tfrecords_ner.py --help` for more information about the arguments


## Directory structure

```
python write_tfrecords_ner.py --data_dir=/path/to/NCBI-disease/folder/containing/train.tsv/test.tsv/dev.tsv --vocab_file=../../../../../common/input/vocab/uncased_pubmed_abstracts_and_fulltext_vocab.txt --output_dir=./ncbi-disease-tfrecords --do_lower_case

ncbi-disease-tfrecords (<args.output_dir>)
├── ncbi-disease (<dataset_name>)
│   ├── dev
│   │   ├── label2id.json
│   │   ├── label2id.pkl
│   │   └── ncbi-disease_dev_0.tfrecord
│   ├── test
│   │   ├── label2id.json
│   │   ├── label2id.pkl
│   │   └── ncbi-disease_test_0.tfrecord
│   └── train
│       ├── label2id.json
│       ├── label2id.pkl
│       ├── ncbi-disease_train_0.tfrecord
│       ├── ncbi-disease_train_1.tfrecord
│       └── ncbi-disease_train_2.tfrecord
└── params.yaml

```



