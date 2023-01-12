# OpenWebText (OWT) data preparation scripts

This directory contains scripts that can be used to download the [OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/) and create TFRecords containing masked sequences and labels. The TFRecords are then used by the `BertTfRecordsProcessor` and `GptTfRecordsProcessor` to produce inputs to `BertModel` and `Gpt2Model`, respectively.

## Data download and extraction

To download the OWT dataset, access the following link from a browser:

```url
https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
```

and manually download the `tar.xz` file there to the location you want.

To extract the manually downloaded files, run:

```bash
bash extract.sh
```

Note that `extract.sh` may take a while to complete, as it unpacks 40GB of data (8,013,770 documents). Upon completion, the script will produce `openwebtext` folder in the same location. The folder has multiple subfolders, each containing a collection of `*.txt` files of raw text data (one document per `.txt` file).

## Define train and evaluation datasets

Define metadata files that contain paths to subsets of documents in `openwebtext` folder to be used for training or evaluation. For example, for training, we use a subset of 512,000 documents. The associated metadata file can be found in `metadata/train_512k.txt`.

For evaluation, we choose 5,000 documents that are outside of the training set. The metadata file for evaluation can be found in `metadata/val_files.txt`. Users are free to create their own metadata files to define train and evaluation (as well as test) data subsets of different content and sizes.

## Create TFRecords

Given a metadata file that defines a data subset, create tf records containing masked sequences and labels coming from this data subset.

Install pre-requisites for BERT, if not installed:

```bash
pip install spacy
python -m spacy download en
```

For creating records for BERT, go to [scripts](../../../bert/input/scripts/):

```bash
python create_tfrecords.py --metadata_files /path/to/metadata_file.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file /path/to/vocab.txt --do_lower_case
```

For creating records for GPT-2, go to [input](../../../gpt2/input/):

```bash
python create_tfrecords.py --metadata_files /path/to/metadata_file.txt --vocab_file /path/to/vocab.txt --encoder_file /path/to/BPEencoder/file
```

where `metadata_file.txt` is a metadata file containing a list of paths to documents and `/path/to/vocab.txt` contains a vocabulary file to map WordPieces to word ids. For example, one can use the supplied `metadata/train_512k.txt` as an input to generate a train set based on 512,000 documents. We used the duplication factor of 10. The resulting dataset then ensures that after 900K steps with batch size 256 each sequence is seen approximately 40 times, with exactly the same mask for approximately 4 times. Sample vocabularies can be found in the [vocab](../../vocab) folder. `do_lower_case` allows to create records which work with uncased models, and not passing it creates records for the cased version of the model.

For more details, see `python create_tfrecords.py --help`.
