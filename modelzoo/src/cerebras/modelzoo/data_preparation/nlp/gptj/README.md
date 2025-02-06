# Introduction

This document will walkthrough for preparing HDF5 dataset TRC2 dataset is used for continuous pre-training of the GPTJ model. 

This dataset comprises of Thomson Reuters Text Research Collection covering a variety of news stories. It has 1,800,370 news stories covering the period from 2008-01-01 00:00:03 to 2009-02-28 23:54:14 or 2,871,075,221 bytes, and was initially made available to participants of the 2009 blog track at the Text Retrieval Conference (TREC), to supplement the BLOGS08 corpus (that contains results of a large blog crawl carried out at the University of Glasgow). TRC2 is distributed via web [download](https://trec.nist.gov/data/reuters/reuters.html).

## Prerequisites

There's no special setup needed for running the pre-processing of this dataset.

## Input data

### Data download

To obtain the raw dataset, please request it from the NIST org by following the `Getting the corpus` section on the [TRC2 website](https://trec.nist.gov/data/reuters/reuters.html)

### Prepare raw dataset

The downloaded dataset is a text file of collection of the headlines and the news story along with extra formatting that needs to be stripped off before sending the dataset for the HDF5 file conversion and using it for the finetuning. The downloaded dataset file will be named as `TRC2-headlines-docs-TRECBLOG.v2`. Store this dataset in `<dataset_dir>`

Use the below set of commands to clean the dataset, i.e., to remove tags, date/time stamps, bad characters, and other extraneous information:

```bash
cd <dataset_dir>
sed 's|</[A-Z0-9]*>||' TRC2-headlines-docs-TRECBLOG.v2 > TRC2-headlines-docs-TRECBLOG.v2.clean.idm1
sed 's|<[A-Z0-9]*>||' TRC2-headlines-docs-TRECBLOG.v2.clean.idm1 > TRC2-headlines-docs-TRECBLOG.v2.clean.idm2
sed 's|[-0-9: ]*||' TRC2-headlines-docs-TRECBLOG.v2.clean.idm2 > TRC2-headlines-docs-TRECBLOG.v2.clean.idm3
sed 's|TRC2-[-0-9]*||' TRC2-headlines-docs-TRECBLOG.v2.clean.idm3 > TRC2-headlines-docs-TRECBLOG.v2.clean.idm4
sed 's|\[.*\]||' TRC2-headlines-docs-TRECBLOG.v2.clean.idm4 > TRC2-headlines-docs-TRECBLOG.v2.clean.idm5
sed 's|(Click on the Z.*||' TRC2-headlines-docs-TRECBLOG.v2.clean.idm5 > TRC2-headlines-docs-TRECBLOG.v2.clean.idm6
iconv -f utf-8 -t utf-8 -c TRC2-headlines-docs-TRECBLOG.v2.clean.idm6 > TRC2-headlines-docs-TRECBLOG.v2.clean
```

After running the above set of commands, the cleaned up dataset will be stored as: `<dataset_dir>/TRC2-headlines-docs-TRECBLOG.v2.clean`

In the next step, we'll split this dataset into a set of small `.txt` files and create meta data file which we'll use in creating the hdf5 files.

Use the script [split_trc_dataset](./split_trc_dataset.py) to split the dataset into smaller chunks which can then be used to get the train and validation sets.

```bash
$ python split_trc_dataset.py --help
usage: split_trc_dataset.py [-h] --input_file INPUT_FILE --out_dir OUT_DIR
                            [--buffer_len BUFFER_LEN]
                            [--val_split_ratio VAL_SPLIT_RATIO]

optional arguments:
  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        Path to the original source language dataset stored as
                        one file. (default: None)
  --out_dir OUT_DIR     Path to output directory with source language dataset
                        files. (default: None)
  --buffer_len BUFFER_LEN
                        Number of examples to store in one file. (default:
                        10000)
  --val_split_ratio VAL_SPLIT_RATIO
                        Ratio of the output number of files to be considered
                        as validation dataset. (default: 0.1)
```

The input to this script, `<INPUT_FILE>`, is the cleaned file from the last step above and the output dir `<OUT_DIR>` is required to store the split dataset. The split smaller files are stored in `<OUT_DIR>/split_files`. In addition to the split dataset, this script will also generate `train_meta.txt` and `val_meta.txt` in the `<OUT_DIR>` which can then be used in preparing HDF5 dataset to use in the input dataloader.

### Create HDF5 files

After splitting the dataset into smaller chunks from the previous step you can then convert the dataset into tokenized input ids in HDF5 file and use it with the `GptHDF5DataProcessor` class in [`GptHDF5DataProcessor.py`](../../../data/nlp/gpt/GptHDF5DataProcessor.py).

For the details on the processing, please refer to the section on [HDF5 preprocessing](https://docs.cerebras.net/en/latest/wsc/Model-zoo/Components/Data-preprocessing/data_preprocessing.html).

For processing the split files into HDF5, below command can be used as a reference:

```bash
python create_hdf5_dataset.py preprocessed_text \
    --input_dir <OUT_DIR>/split_files \
    --metadata_files <OUT_DIR>/<train/val_meta.txt>\
    --tokenizer_type GPT2Tokenizer \
    --vocab_file /path/to/gpt2-vocab.bpe \
    --encoder_file /path/to/gpt2-encoder.json \
    --max_seq_length 2048 \
    --output_dir <HDF5_OUT_DIR> \
    --seed 42 \
    --processes 8 \
    --ftfy \
    --write_remainder \
    --display_pbar \
```

In the above command, use the appropriate values for `tokenizer_type` and `vocab_file` for GPT NeoX, GPTJ based model.

- `<OUT_DIR>` is the same as the one used in previous step for splitting raw files.
- `<HDF5_OUT_DIR>` is the output dir where the HDF5 dataset is stored.
