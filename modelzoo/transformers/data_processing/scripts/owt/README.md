# OpenWebText (OWT) data preparation scripts

This directory contains scripts that can be used to download the [OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/). Scripts for turning the raw OWT data into TFRecords can be found in `transformers/bert/tf/input` and `transformers/gpt2/tf/input`.

## Data download and extraction

To download the OWT dataset, access the following link from a browser:

```url
https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
```

and manually download the tar.xz file there to the location you want.

To extract the manually downloaded files, run:

```bash
bash extract.sh
```

Note that `extract.sh` may take a while to complete, as it unpacks 40GB of data (8,013,770 documents). Upon completion, the script will produce `openwebtext` folder in the same location. The folder has multiple subfolders, each containing a collection of `*.txt` files of raw text data (one document per `.txt` file).

## Define train and evaluation datasets

Define metadata files that contain paths to subsets of documents in `openwebtext` folder to be used for training or evaluation. For example, for training, we use a subset of 512,000 documents. The associated metadata file can be found in `metadata train_512k.txt`.

For evaluation, we choose 5,000 documents that are outside of the training set. The metadata file for evaluation can be found in `metadata/val_files.txt`. Users are free to create their own metadata files to define train and evaluation (as well as test) data subsets of different content and sizes.

## Create tf records

Given a metadata file that defines a data subset, create tf records containing masked sequences and labels coming from this data subset.

Install pre-requisites, if not installed:

```bash
pip install spacy
python -m spacy download en
```

For creating records for BERT, go to `transformers/bert/tf/input` and run:

```bash
python create_tfrecords.py --input_files /path/to/metadata_file.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file /path/to/vocab.txt --do_lower_case
```

where `metadata_file.txt` is a metadata file containing a list of paths to documents and `/path/to/vocab.txt` contains a vocabulary file to map WordPieces to word ids. For example, one can use the supplied `metadata/train_512k.txt` as an input to generate a train set based on 512,000 documents. In case of TFrecords generated for BERT, we used the duplication factor of 10. The resulting dataset then ensures that after 900K steps with batch size 256 each sequence is seen ~40 times, with exactly the same mask for ~4 times. Sample vocabularies can be found in the `transformers/vocab` folder. `do_lower_case` allows to create records which work with uncased models, and not passing creates records for the cased version of the model.

Similarly, records for GPT-2 can be created using `transformers/gpt2/tf/input/create_tfrecords.py.


### 2-Phase BERT Training

To generate datasets for 2-phase pre-training, with the first phase using sequences with maximum length of 128 and the second phase using sequences with maximum sequence length of 512, you will need to generate two separate training datastes, and if you want to run evaluation, two separate datasets for validation/evaluation. So, you will need to run the following commands.

__Phase 1: Maximum Sequence Length (MSL) 128__

- Generate training tfrecords:

```bash
python create_tfrecords.py --metadata_files metadata/train_512k.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file ../../../../common/input/tf/vocab/google_research_uncased_L-12_H-768_A-12.txt --output_dir train_512k_uncased_msl128 --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20
```

- Generate validation tfrecords:

```bash
python create_tfrecords.py --metadata_files metadata/val_files.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file ../../../../vocab/google_research_uncased_L-12_H-768_A-12.txt --output_dir val_uncased_msl128 --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20
```

__Phase 2: Maximum Sequence Length (MSL) 512__

- Generate training tfrecords:

```bash
python create_tfrecords.py --metadata_files metadata/train_512k.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file ../../../../vocab/google_research_uncased_L-12_H-768_A-12.txt --output_dir train_512k_uncased_msl512 --do_lower_case --max_seq_length 512 --max_predictions_per_seq 80
```

- Generate validation tfrecords:

```bash
python create_tfrecords.py --metadata_files metadata/val_files.txt --input_files_prefix /path/to/raw/data/openwebtext --vocab_file ../../../../vocab/google_research_uncased_L-12_H-768_A-12.txt --output_dir val_uncased_msl512 --do_lower_case --max_seq_length 512 --max_predictions_per_seq 80
```
