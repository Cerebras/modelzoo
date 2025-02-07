# Introduction

[T5](https://arxiv.org/abs/1910.10683) model is trained on [Common Crawl (C4)](https://commoncrawl.org/) dataset which is available for download from the HuggingFace repo [link](https://huggingface.co/datasets/allenai/c4).

## Prerequisites

The preprocessing script uses [sentencepiece](https://github.com/google/sentencepiece) tokenizer package which needs to be installed in the Python virtual environment before launching the preprocessing job. 

## Input data

### Data download

The Common Crawl dataset is split into train and validation splits. It consists of `1024` total shards that are downloaded by the [preprocess_external.sh](./preprocess_external.sh) script in a compressed `gzip` format from this [HuggingFace repo](https://huggingface.co/datasets/allenai/c4/).

### Input files format

There are `1024` files that are downloaded from the HuggingFace Common Crawl dataset repo in `gzip` format and they are simultaneously downloaded and processed by `16` worker nodes. Compressed raw data records are uncompressed to `json` format and stored under  `$dataset_root/raw` directory.

Note that if you want to create your own dataloader, you will need to ensure that the input is prepared to match the following format. The model expects data of the specified format, and will not operate on other data.

If you want to use your custom dataset first it needs to be extracted or formatted to be in `json` format and place under `$dataset_root/raw` directory. You can remove these `2` lines from the preprocessing script:

```bash
wget https://huggingface.co/datasets/allenai/c4/resolve/main/en/${name}.json.gz\
         --no-verbose
gzip -d ${name}.json.gz
```

## Running the script

The script can be run using the following command:

```bash
cd ..
source preprocess_external.sh
```

This script downloads and preprocesses the C4 dataset for use in the T5 model.

Once finished execution, it will generate tokenized text of input features and labels in the respective `$dataset_root/train` and `$dataset_root/validation` directories.

## Output data

### Output directory structure

This process will generate tokenized `1024` training data files in the following naming format:
`$dataset_root/train/c4-train.0****-of-01024.txt`

And `8` validation data files:
`$dataset_root/validation/c4-validation.0****-of-00008.txt`

### Output file structure

Each text file in the output will contain tokenized text corresponding to sentences from the input dataset.

One such example of the tokenized text line is: 

```
▁F ight ▁G one ▁Bad ▁to ▁end ▁2009 ! ▁Monday ▁we ▁did ▁one ▁of ▁the ▁2009 ▁Affiliate ▁Cup ▁W OD s . ▁Tuesday ▁we ▁did ▁ a ▁her o ▁W OD . ▁So ▁as ▁I ▁already ▁know ▁I ▁keep ▁the ▁worst ▁log book ▁in ▁history . ▁It ▁has ▁been ▁ spor a dic ▁at ▁best ▁this ▁month , ▁but ▁then ▁my ▁workout s ▁have ▁been ▁ spor a dic ▁too . ▁Last ▁weekend ▁went ▁to ▁Cross Fi t ▁101 ▁at ▁ CF ▁N LP ▁in ▁Lake ▁Forest . ▁Coach ▁was ▁very ▁helpful ▁and ▁ reci e ve d ▁my ▁idea ▁about ▁ a ▁L 1 ▁in ▁Afghanistan ▁pretty ▁well . ▁As ▁long ▁as ▁I ▁can ▁lay ▁the ▁ground work ▁I ▁think ▁I ▁can ▁get ▁it ▁accomplished . ▁A ▁few ▁W OD s ▁crushed ▁me , ▁hit ▁it ▁Mon - T u e s , ▁Thur s - F r i . ▁But ▁so ▁little ▁to ▁do . ▁Another ▁bit ▁of ▁ s lac king , ▁then ▁work . ▁between ▁rounds ▁did ▁ 1-2 -3 ▁band ▁resistance ▁runs .
```

Each text file contains about 35k samples.
