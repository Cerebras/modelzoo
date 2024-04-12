# BERT Token Classifier Input

## BertTokenClassifierDataProcessor.py

The features dictionary has the following key/values:

* `input_ids`: Numpy array with input token indices.
    shape: (`max_sequence_length`), dtype: `int32`.
* `attention_mask`: Numpy array with attention mask.
    shape: (`max_sequence_length`), dtype: `int32`.
* `token_type_ids`: Numpy array with segment ids.
    shape: (`max_sequence_length`), dtype: `int32`.
* `labels`: Numpy array with labels.
    shape: (`max_sequence_length`), dtype: `int32`.

___

## Generating TSV files for Token Classifier Dataset

1. Download and preprocess the dataset from [BLURB dashboard](https://microsoft.github.io/BLURB/submit.html)

    ```bash
    wget https://microsoft.github.io/BLURB/sample_code/data_generation.tar.gz

    tar -xf data_generation.tar.gz
    ```

2. Refer to `README.md` from the `data_generation` (downloaded and extracted in Step 1) folder to download and preprocess the datasets for Named Entity Recognition (NER) - BC5CDR-Chem, BC5CDR-Disease, BC2GM, NCBI-Disease, JNLPBA

3. Download the vocab file from the Microsoft's model files on the [HuggingFace model card](microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext).

    ```bash
    wget https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/raw/main/vocab.txt -O pubmed_uncased_abstract_fulltext_vocab.txt
    ```

4. Generate TSV files from the raw data using the script `write_csv_ner.py`

    ```python
    Example Usage: 

    python write_csv_ner.py \
    --data_dir /cb/ml/language/datasets/blurb/data_generation/data/BC5CDR-chem/ \
    --vocab_file /cb/ml/language/datasets/ner-pt/pubmed_bert_base_uncased_abstract_fulltext_vocab.txt \
    --output_dir /cb/ml/language/datasets/ner-pt/bc5cdr-chem-tsv \
    --do_lower_case
    ```

    Run `python write_csv_ner.py --help` for more information about the arguments

## Directory structure

```bash
python write_csv_ner.py \
--data_dir /cb/ml/language/datasets/blurb/data_generation/data/BC5CDR-chem/ \
--vocab_file /cb/ml/language/datasets/ner-pt/pubmed_bert_base_uncased_abstract_fulltext_vocab.txt \
--output_dir /cb/ml/language/datasets/ner-pt/bc5cdr-chem-tsv \
--do_lower_case

bc5cdr-chem-csv/ <args.output_dir>
├── dev
│   ├── dev-1.csv
│   ├── dev-2.csv
│   ├── dev-3.csv
│   ├── dev-4.csv
│   ├── label2id.json
│   ├── label2id.pkl
│   └── meta.dat
├── test
│   ├── label2id.json
│   ├── label2id.pkl
│   ├── meta.dat
│   ├── test-1.csv
│   ├── test-2.csv
│   ├── test-3.csv
│   └── test-4.csv
├── train
│   ├── label2id.json
│   ├── label2id.pkl
│   ├── meta.dat
│   ├── train-1.csv
│   ├── train-2.csv
│   ├── train-3.csv
│   └── train-4.csv
└── params.yaml
```
