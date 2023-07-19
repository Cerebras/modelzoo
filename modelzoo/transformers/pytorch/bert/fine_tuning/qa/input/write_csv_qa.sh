#!/usr/bin/bash

# Calls write_csv_qa.py from transformers/data_processing/ code with the correct command-line arguments for the current model (BERT)

script_path=$GITTOP/src/models/transformers/data_processing/qa/write_csv_qa.py
python $script_path \
    --do_lower_case \
    --data_dir /cb/ml/language/datasets/squad \
    --vocab_file /cb/ml/language/models/bert/pretrained/google-research/uncased_L-12_H-768_A-12/vocab.txt \
    --data_split_type all \
    --max_seq_length 384 \
    --output_dir /cb/ml/bert/squad/ \
    --tokenizer_scheme bert \
