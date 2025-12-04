# OpenWebText (OWT) data preparation scripts

This directory contains scripts that can be used to download the [OpenWebText dataset](https://skylion007.github.io/OpenWebTextCorpus/) and create TFRecords containing masked sequences and labels. The TFRecords are then used by the `BertTfRecordsProcessor` and `GptTfRecordsProcessor` to produce inputs to `BertModel` and `Gpt2Model`, respectively.

## Data download and extraction

To download the OWT dataset, access the following link from a browser:

```url
https://drive.google.com/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
```

and manually download the `tar.xz` file there to the location you want. This is a google drive link, so you may need to manually download the file.

As an alternative, to automatically download and extract the files using [gdown](https://pypi.org/project/gdown/), run:

```bash
bash download_and_extract.sh
```

Note that `download_and_extract.sh` may take a while to complete, as it unpacks 40GB of data (8,013,770 documents). Upon completion, the script will produce `openwebtext` folder in the same location. The folder has multiple subfolders, each containing a collection of `*.txt` files of raw text data (one document per `.txt` file).

Note that this script installs `gdown` in your python environment so be sure you can install this without admin privileges.

## Define train and evaluation datasets

Define metadata files that contain paths to subsets of documents in `openwebtext` folder to be used for training or evaluation. For example, for training, we use a subset of 512,000 documents. The associated metadata file can be found in `metadata/train_512k.txt`.

For evaluation, we choose 5,000 documents that are outside of the training set. The metadata file for evaluation can be found in `metadata/val_files.txt`. Users are free to create their own metadata files to define train and evaluation (as well as test) data subsets of different content and sizes.

## Convert dataset to hdf5 files

After downloading the raw data, you can convert it to HDF5 files using `preprocess_data.py`. The instructions for this can be found [here](https://docs.cerebras.net/en/latest/wsc/Model-zoo/Components/Data-preprocessing/data_preprocessing.html).
