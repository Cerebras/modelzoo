#!/bin/bash

# Copyright 2020 Cerebras Systems.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


pip install --user git+git://github.com/titipata/pubmed_parser.git
# https://bitbucket.org/mrabarnett/mrab-regex/issues/349/no-module-named-regex_regex-regex-is-not-a
pip install --user regex==2019.12.9
pip install --user -U nltk


OUTPUT_DIR=$1
DATASET_NAME=$2
VOCAB_FILE=$3

declare -A HASHMAP_TRAIN
HASHMAP_TRAIN=( ['pubmed_baseline']=2200 ['pubmed_daily_update']=1500 ['pubmed_fulltext']=6200 )

declare -A HASHMAP_TEST
HASHMAP_TEST=( ['pubmed_baseline']=500 ['pubmed_daily_update']=500 ['pubmed_fulltext']=1000 )

#### Download ####
python pubmedbert_prep.py --action download --dataset ${DATASET_NAME} --output_dir ${OUTPUT_DIR}

#### Properly format the text files ####
python pubmedbert_prep.py --action text_formatting --dataset ${DATASET_NAME} --output_dir ${OUTPUT_DIR}

#### Shard the text files ####
python pubmedbert_prep.py --action sharding --dataset ${DATASET_NAME} --n_training_shards ${HASHMAP_TRAIN[${DATASET_NAME}]} --n_test_shards ${HASHMAP_TEST[${DATASET_NAME}]} --output_dir ${OUTPUT_DIR}


#### Write TF records ####

## UNCASED MSL128 ##
python pubmedbert_prep.py --action create_tfrecord_files --dataset ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --max_seq_length 128 --max_predictions_per_seq 20 --mask_whole_word --do_lower_case --vocab_file ${VOCAB_FILE} --dupe_factor 5 --n_processes 8


## UNCASED MSL512 ##
python pubmedbert_prep.py --action create_tfrecord_files --dataset ${DATASET_NAME} --output_dir ${OUTPUT_DIR} --max_seq_length 512 --max_predictions_per_seq 80 --mask_whole_word --do_lower_case --vocab_file ${VOCAB_FILE} --dupe_factor 5 --n_processes 8
