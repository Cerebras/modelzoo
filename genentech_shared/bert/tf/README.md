# TAPE BERT Model

Implentation of the TAPE BERT model described in: [Evaluating Protein Transfer Learning with TAPE](https://arxiv.org/pdf/1906.08230.pdf) by Rao et al.

See the [modelzoo BERT implementation](https://github.com/Cerebras/modelzoo/tree/master/transformers/bert/tf) for model details.

## Data Preparation

Pfam Pretraining TFRecord data can be downloaded from the [TF TAPE repo](https://github.com/songlab-cal/tape-neurips2019#tfrecord-data).

The path to the directory containing the TFRecords should be provided through the `data_dir` field in the `train_input` section of the [config file](configs/params_tape_bert_base.yaml#L21).
