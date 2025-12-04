# Introduction

[The Pile](https://arxiv.org/abs/2101.00027) [1] is a dataset of diverse text for language modeling. It is constructed from 22 diverse high-quality subsets, both existing and newly constructed, many of which derive from academic or professional sources.

This directory contains scripts that provide a workflow from downloading the raw data to processing them as TFRecords or HDF5, to be used by the data pipelines of GPT style (auto-regressive) models for training and validation.

The output dataset can be used by models with an autoregressive language modeling task like GPT-Style models (GPT2, GPT3, GPTJ, GPT-Neox).

## Data download

The raw data for Pile is available at the [eye.ai website](https://mystic.the-eye.eu/public/AI/pile/). To download the data locally, you can use `download.py`, the arguments for which are detailed below:

```bash
usage: download.py [-h] --data_dir DATA_DIR [--name NAME] [--debug]

Download the raw Pile data and associated vocabulary for pre-processing.

optional arguments:
  -h, --help           show this help message and exit
  --data_dir DATA_DIR  Base directory where raw data is to be downloaded.
  --name NAME          Sub-directory where raw data is to be downloaded.
                       Defaults to `pile`.
  --debug              Checks if a given split exists in remote location.
```

This file automatically loops over the train, validation and test splits and downloads them to the given output folder. It also downloads the vocabulary files for two different tokenization mechanisms (one based on GPT2[4] and another based on GPT-NeoX[2, 3]).

### Download Notes

- The `train` split in particular is very large, and it takes approximately 15 hours @ 10MB/s to download. It also needs at least 500GB of space for storage.
- There is an additional `debug` flag, which lets you test if the remote files to download exist or not. To use this, pass an additional argument `--debug`.

### Some Metrics of Downloaded Files

All files are of `jsonl.zst` format.

- train: 30 compressed files, each with ~15GB.
- val: 1 compressed file with ~450MB.
- test: 1 compressed file with ~450MB.

## Convert dataset to hdf5 files

After downloading the raw data, you can convert it to hdf5 files using `preprocess_data.py`. The instructions for this can be found [here](https://docs.cerebras.net/en/latest/wsc/Model-zoo/Components/Data-preprocessing/data_preprocessing.html).

## References

1. The Pile: An 800GB Dataset of Diverse Text for Language Modeling, [arXiv 2021](https://arxiv.org/abs/2101.00027)
