# Summarization task

- [Model overview](#model-overview)
  - [Rouge metric](#rouge-metric)
- [Sequence of steps to perform](#sequence-of-steps-to-perform)
- [Structure of the code](#structure-of-the-code)
- [Download and prepare the dataset](#download-and-prepare-the-dataset)
  - [BERT Summarization input](#bert-summarization-input)
    - [BERT Summarization features dictionary](#bert-summarization-features-dictionary)
    - [Prepare input format](#prepare-input-format)
- [Input function pipeline](#input-function-pipeline)
- [Run fine-tuning](#run-fine-tuning)
- [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
- [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
- [Configs included for this model](#configs-included-for-this-model)
- [References](#references)

## Model overview

A summarizaion task is the task of automatically generating a shorter version of the
document while retaining its most important information.

Usually the task is divided into two paradigms: abstractive summarization and extractive summarization.

In abstractive summarization, target summaries contain words
or phrases that were not in the original text, and usually require various text rewriting.

This code implements the extractive summarization approach,
where the summary is formed by copying and concatenating the most important
spans (usually sentences) in the document.

### Rouge metric
Rouge metric [wiki](https://en.wikipedia.org/wiki/ROUGE_\(metric\)) is a special
metric for asessing summarization tasks. It takes `1-grams`, `2-grams`, etc. and evaluates
the percentage of intersection between the referenced `n-grams` and the `n-grams` predicted by the system.
This percentage can be later on evaluated in terms of `f1-score`, `precition` and `recall`.

<img src="images/bertsum-rouge.png" width="1000">


## Sequence of steps to perform

The following block diagram shows a high-level view of the sequence of steps you will perform in this example.

<p align = "center">
<img src = ./images/steps-bert-extractive_summarization.png>
</p>
<p align = "center">
Fig.1 - Flow Chart of steps to fine-tune BERT summarization model
</p>

## Structure of the code

* `configs/`: YAML configuration files.
* [data/nlp/bert](../../../../data/nlp/bert/): Input pipeline implementation based on the [DeepMind Q&A Dataset](https://cs.nyu.edu/~kcho/DMQA/).
* `model.py`: Model implementation leveraging [BertSummarization](../bert_model.py) class.
* `run.py`: Training script. Performs training and validation.
* `utils.py`: Miscellaneous helper functions.


## Download and prepare the dataset

For information about how to download and prepare the dataset
refer to [README.md](../../../../data/nlp/bert/README.md).

### BERT Summarization input

#### BERT Summarization features dictionary

The features dictionary has the following key-value pairs (shown in [BertSumCSVDataProcessor.py](../../../../data/nlp/bert/BertSumCSVDataProcessor.py)):

`input_ids`: Input token IDs, padded with `0`s to `max_sequence_length`.

- Shape: `[batch_size, max_sequence_length]`.
- Type: `torch.int32`.

`attention_mask`: Mask for padded positions. Has `0`s on the padded positions
and `1`s elsewhere.

- Shape: `[batch_size, max_sequence_length]`.
- Type: `torch.int32`.

`token_type_ids`: Segment IDs. A segment is equal to `0` or `1`
 conditioned on index of the sentence is odd or even. For example, for `[sent1, sent2, sent3, sent4, sent5]` we will assign `[0, 1, 0, 1, 0]`.

`label_ids`: The label tensor. Carries the labels for each sentence. It's binary labels that indicate if each sub-sentences should be included in the summary. Please note that `max_cls_tokens` indicates the number of sub-sentences in our input text.

- Shape: `[batch_size, max_cls_tokens]`.
- Type: `torch.int32`.

`cls_indices`: CLS indices. Specifies indices in the `[CLS]` tokens in the `input_ids`.

- Shape: `[batch_size, max_cls_tokens]`.
- Type: `torch.int32`.

`cls_weights`: CLS tokens weights. Equal to `1` for all real `[CLS]` tokens, and `0` for all padded.

- Shape: `[batch_size, max_cls_tokens]`.
- Type: `torch.float32`.



A demo example of the input string and segment ID structure for single extractive_summarization:

``` bash
Tokens:   [CLS] sent one [SEP][CLS] sent two [SEP][CLS] sent three [SEP][CLS] sent four [SEP][PAD] [PAD] ...
Segments:   0    0    0    0    1    1    1    1   0     0     0     0    1    1    1    1     0    0    ...
label       0                   1                  0                      0
```
NOTE: The input tokens will be converted to IDs using the vocab file.

___

#### Prepare input format
To use BERT for extractive summarization, we follow the approach provided in the
paper [Fine-tune BERT for Extractive Summarization by Yang Liu](https://arxiv.org/pdf/1903.10318.pdf).

In order to apply BERT, authors suggest to fine-tune pre-trained BERT weights
by formatting the input into the special format:

![input](images/bertsum-input.png)

Each sentence starts with a special `[CLS]` token. The context representation of this token will be used to make
a decision if this sentence will be present in the predicted summary or not.

Each sentence is split with `[SEP]` tokens. Interval segments are also formatted in a special
way to help distinguish multiple sentences within one document: we alternate segment embeddings
between `E_A` and `E_B` for each subsequent sentence.
For example, for `[sent_1, sent_2, sent_3, sent_4, sent_5]` we will assign
`[E_A, E_B, E_A, E_B, E_A]`.



## Input function pipeline
For more details about the input function pipeline used for the models located in this folder, please refer to a separate documentation [README.md](../../../../data/nlp/bert/README.md).



## Run fine-tuning

**IMPORTANT**: See the following notes before proceeding further.

After obtaining the sentence vectors from BERT, following `Yang Liu`, we build several
summarization-specific layers stacked on top of the BERT outputs, to capture
document-level features for extracting summaries (See image below). For each sentence sent<sub>i</sub>,
we will calculate the final predicted score P<sub>i</sub> . The loss of the whole
model is the Binary Classification Entropy of P<sub>i</sub> against gold label Y<sub>i</sub>.
These summarization layers are jointly fine-tuned with BERT.

<img src="images/bertsum-model.png" width="300">

**Parameter settings in YAML config file**: The config YAML files are located in the [configs](configs/) directory. Before starting a fine-tuning run, make sure that in the YAML config file you are using:

- The `train_input.data_dir` parameter points to the correct dataset, and
- The `train_input.max_sequence_length` parameter corresponds to the sequence length of the dataset.
- The `train_input.batch_size` parameter will set the batch size for the training.

Same applies for the `eval_input`.

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our [quickstart in the Developer Docs](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html).

> **Note**: To specify a BERT pretrained checkpoint use: `--checkpoint_path` is the path to the saved checkpoint from BERT pre-training,`--load_checkpoint_states="model"` setting is needed for loading the pre-trained BERT model for fine-tuning and `--disable_strict_checkpoint_loading` is needed to be able to only partially load a model.

## To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../../../../PYTHON-SETUP.md), and simply run:

```
python run.py CPU --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```
or
```
python run.py GPU --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```

> **Note**: Change the command to `--mode eval` for evaluation.

## Configs included for this model

In order to train the model, you need to provide a yaml config file. Below is the list of yaml config files included for this model implementation at [configs](./configs/) folder. Also, feel free to create your own following these examples:

- `bert_base_summarization.yaml` have the standard bert-base config with `hidden_size=768, num_hidden_layers=12, num_heads=12` as a backbone.
- `bert_large_summarization.yaml` have the standard bert-large config with `hidden_size=1024, num_hidden_layers=24, num_heads=16` as a backbone.

## References

[1] [BERT paper](https://arxiv.org/abs/1810.04805)

[2] [Fine-tune BERT for Extractive Summarization](https://arxiv.org/abs/1903.10318)
