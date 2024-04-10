# Bloom language models

This directory contains the Pytorch ML reference for Bloom model.

## List of topics

- [Bloom language models](#bloom-language-models)
  - [Overview of the model](#overview-of-the-model)
    - [Bloom](#bloom)
  - [Steps for running model training](#steps-for-running-model-training)
  - [Structure of the code](#structure-of-the-code)
- [Download and prepare the dataset](#download-and-prepare-the-dataset)
- [How to run](#how-to-run)
- [Notes on configuration files](#notes-on-configuration-files)
  - [config files](#config-files)
  - [important yaml fields](#important-yaml-fields)
- [References](#references)

## Overview of the model

### Bloom

Bloom is a decoder-only transformer-based multilingual language model with up to 176B parameters from [BigScience](https://bigscience.huggingface.co/).
Its architecture is very similar to the [GPT2 model](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) with the following changes:

-   **Tokenizer**: Both the model and its tokenizer have a vocabulary size of around 250K which covers 46 natural languages and 13 programming languages.
-   **Position Embedding** Instead of using `learned` position embeddings like GPT2, bloom adopts the [ALiBi](https://arxiv.org/pdf/2108.12409.pdf) position embedding.
ALiBi position embedding biases the quer-key attention scores with a penalty that is proportional to their distance. 
This type of inductive bias on recency enables ALiBi to extrapolate to longer input sequences during inference and some performance boost. 
Please refer to the paper for more details.

**Reference**: 

Scao et al. (2023). [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100).

Press, et al. (2021). [TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/pdf/2108.12409.pdf).

## Steps for running model training
In order to run any of the models in this directory, you must go through the following steps:
- Download and preprocess the data (see [Prepare the data](#prepare-the-data) for more details)
- Run training for your desired model (see [Run pre-training](#run-pre-training))

## Structure of the code

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.
-   `utils.py`: Defined under [gpt2 directory](../gpt2/utils.py).
-   `data.py`: Defined under [gpt2 directory](../gpt2/data.py).
-   `model.py`: Defined under [gpt2 directory](../gpt2/model.py).

# Download and prepare the dataset

Please refer to the section `Download and prepare the dataset` in the gpt2 [README file](../gpt2/README.md) 

# How to run

Please refer to the section `How to run` in the gpt2 [README file](../gpt2/README.md) 

# Notes on configuration files

## config files

In order to train the model, you need to provide a yaml config file. Some reference yaml [configs](configs/) files are listed below for reference. 
Also, feel free to create your own following these examples:

- [params_bloom_7b.yaml](./configs/params_bloom_7b.yaml) have the model metrics with `hidden_size=4096`, `num_hidden_layers=30`, `num_heads=32`.

## important yaml fields

To use the alibi embedding in the bloom model, you need to pay attention to the following fields under the `model` tab:

-   `position_embedding_type (str)`: set the value of this field to `alibi`

-   `alibi_trainable_slopes (bool)`: whether the slopes of the alibi embedding is trainable (default to False). 
Note that based on the analysis of the original alibi paper, trainable slopes did not yield strong results (on-par with fixed slopes).

# References

**Reference**: 

Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

Scao et al. (2023). [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100).

Press, et al. (2021). [TRAIN SHORT, TEST LONG: ATTENTION WITH LINEAR BIASES ENABLES INPUT LENGTH EXTRAPOLATION](https://arxiv.org/pdf/2108.12409.pdf).
