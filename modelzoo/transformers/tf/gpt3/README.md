# GPT-3 Language Models

This directory contains the TensorFlow ML reference for GPT-2 and GPT-3 models.

- [GPT-3 Language Models](#gpt-3-language-models)
  - [Overview of the model](#overview-of-the-model)
  - [Steps for running model training](#steps-for-running-model-training)
  - [Key CSoft features](#key-csoft-features)
  - [Structure of the code](#structure-of-the-code)
  - [Prepare the data](#prepare-the-data)
  - [Input function](#input-function)
    - [Features dictionary](#features-dictionary)
    - [Label tensor](#label-tensor)
    - [Input pipeline with sharding](#input-pipeline-with-sharding)
  - [How to run](#how-to-run)
  - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
  - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
  - [Configs included for this model](#configs-included-for-this-model)
  - [Appendix](#appendix)

## Overview of the model

[GPT-3](https://arxiv.org/abs/2005.14165) is a very similar architecture to [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pd) except that every other self-attention layer in GPT-3 uses locally banded sparse attention in which tokens only attend to each other if they are nearby in the sequence
(see section 2.1 of the [GPT-3 paper](https://arxiv.org/abs/2005.14165) for more details). Figure below describes a high level model architectiure of GPT3 model.

![GPT3 Architecture Diagram](./images/architecture_diagram.png)

The larger versions of GPT-3 range from 1.3B to 175B parameters.

**NOTE:** In out current implementation, we use the code from [GPT2 implementation](../gpt2/) which does not have banded sparse attention implemented. We plan to add this in the future releases.

## Steps for running model training

In order to run any of the models in this directory, you must go through the following steps:

- Download and preprocess the data (see [gpt2/input/README.md](../gpt2/input/README.md) for more details)
  - Download the OpenWebText compressed tar file
  - Extract the text files
  - Allocate mutually exclusive subsets of the data for training and evaluation; create the corresponding metadata files
  - Use the [create_tfrecords.py](../gpt2/input/create_tfrecords.py) script to convert text files into TFRecords files
- Run training for your desired model (see [Run pre-training](#run-pre-training))


## Key CSoft features

The Cerebras Wafer Scale Engine supports two different execution modes:

- Layer pipelined: all layers of a model are loaded onto the Cerebras WSE at once and samples are streamed through one at a time in a pipeline parallel manner.
- Weight streaming: layers of the model are loaded onto the Cerebras WSE one at a time and executed in a data parallel manner.

This GPT3 implementation supports only weight streaming execution mode.

For more details on Cerebras execution modes, see [this explanation](https://docs.cerebras.net/en/latest/cerebras-basics/cerebras-execution-modes.html).

## Structure of the code

- `configs/`: YAML configuration files.
- `run-appliance.py`: Training script. Performs training and validation.

**NOTE:** In out current implementation, we use the code from [GPT2 implementation](../gpt2/), so the bare minimal files needed to run the model and configs are only provided in this directory.

## Prepare the data

First you need to download your raw data and create preprocessed TFRecords;
see details how to in [gpt2/input/README.md](../gpt2/input/README.md).

## Input function

This section describes the input data format expected by `Gpt2Model`. If you want to define your own dataloader for this model,
the easiest way to do it is to conform to this format in order to avoid model changes. See [GptTfRecordsProcessor](../gpt2/input/GptTfRecordsProcessor) for an example dataloader.

When you create your own custom input function, you must ensure that your input function produces a tuple of
`(features, labels)`, where the features dictionary and a label tensor as described in this section.

### Features dictionary

The features dictionary has the following key/values:

- `input_ids`: Input token IDs, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `tf.int32`
- `input_mask`: Mask for padded positions. Has values `1` on the padded positions and `0` elsewhere.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `tf.int32`

See the [input README](../gpt2/input/README.md#table-1-data-features-in-the-generated-tfrecords) for more details.

### Label tensor

The label tensor of shape `(batch_size,)`. Carries the next token labels.

### Input pipeline with sharding

In addition, the above-created TFRecords are used by the `GptTfRecordsProcessor` class to create a sharded dataset, using the `shard_dataset.py` utility. This allows multiple workers to stream data at once without repeating samples. For a detailed explanation of sharding, see <a href="https://docs.cerebras.net/en/latest/tensorflow-docs/preparing-tf-input/sharding-for-cs.html" class="external-link">Sharding For the Cerebras System</a>.

## How to run

**IMPORTANT**: See the following notes before proceeding further.

**Parameter settings in YAML config file**: The config YAML files are located in the [configs](configs/) directory. Before starting a pre-training run, make sure that in the YAML config file you are using:

- The `train_input.data_dir` parameter points to the correct dataset, and
- The `train_input.max_sequence_length` parameter corresponds to the sequence length of the dataset.
- The `model.max_position_embeddings` parameter corresponds to the maximum dimension of position embeddings.

**YAML config files**: Details on the configs for this model can be found in [Configs included for this model](#configs-included-for-this-model)

In the following example run commands, we use `/path/to/yaml`, `/path/to/model_dir`, and `train` as placeholders for user supplied inputs.

- `/path/to/yaml` is a path to the YAML config file with model parameters such one of the configurations described in [Configs included for this model](#configs-included-for-this-model).
- `/path/to/model_dir` is a path to the directory where you would like to store the logs and other artifacts of the run.
- `--mode` specifies the desired mode to run the model in. Change to `--mode eval` to run in eval mode.

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our Developer Docs at:
https://docs.cerebras.net/en/latest/getting-started/tensorflow/index.html

## To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../PYTHON-SETUP.md), and simply run:

```
python run.py --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```

Note that our model implementation and run scripts are compatible to run on GPU, however handling any GPU cluster related programming is up-to the user.

## Configs included for this model

For convenience, we provide different configurations of common model setups designed to give examples of models of different sizes intended for execution in either [weight streaming mode](https://docs.cerebras.net/en/latest/cerebras-basics/cerebras-execution-modes.html). 

Following are the convergent configs:

- [params_gpt3_xl_ws.yaml](./configs/params_gpt3_xl.yaml): A 1.3B parameter GPT-3 model designed to converge to the state-of-the-art. It uses hyperparameters as suggested in  [Chinchilla](https://arxiv.org/abs/2112.11446) and [Gopher](https://arxiv.org/abs/2112.11446). And, it uses 20 tokens per parameter as per the recommendation in Chinchilla. 

- [params_gpt3_2p7b_ws.yaml](./configs/params_gpt3_2p7b.yaml): A 6.7B parameter GPT-3 model designed to converge to the state-of-the-art. It uses hyperparameters as suggested in  Chinchilla and Gopher. And, it uses 20 tokens per parameter as per the recommendation in Chinchilla.
- [params_gpt3_6p7b_ws.yaml](./configs/params_gpt3_6p7b.yaml): A 6.7B parameter GPT-3 model designed to converge to the state-of-the-art. It uses hyperparameters as suggested in  Chinchilla and Gopher. And, it uses 20 tokens per parameter as per the recommendation in Chinchilla.
- [params_gpt3_13b_ws.yaml](./configs/params_gpt3_13b.yaml): A 13B parameter GPT-3 model designed to converge to the state-of-the-art. IIt uses hyperparameters as suggested in  Chinchilla and Gopher. And, it uses 20 tokens per parameter as per the recommendation in Chinchilla.
- [params_gpt3_20B_ws.yaml](./configs/params_gpt3_20B.yaml): A 20B parameter GPT-3 model designed to converge to the state-of-the-art. It uses hyperparameters as suggested in  Chinchilla and Gopher. And, it uses 20 tokens per parameter as per the recommendation in Chinchilla.

- [params_gpt3_xl.yaml](./configs/params_gpt3_xl.yaml): A 1.3B parameter GPT-2 model designed to match the configuration of the GPT-3 XL.
- [params_gpt3_xl_grad_accum.yaml](./configs/params_gpt3_xl_grad_accum.yaml): A 1.3B parameter GPT-2 model designed to match the configuration of the GPT-3 XL, with gradient accumulation enabled on CS2 to support larger batch sizes.
- [params_gpt3_6p7b.yaml](./configs/params_gpt3_6p7b.yaml): A 6.7B parameter GPT-2 model designed to match the configuration of the GPT-3 6.7B model.
- [params_gpt3_13b.yaml](./configs/params_gpt3_13b.yaml): A 13B parameter GPT-2 model designed to match the configuration of the GPT-3 13B model. Available as an early limited access.
- [params_gpt3_20b.yaml](./configs/params_gpt3_20b.yaml): A 20B parameter GPT-2 model designed to match the configuration of the GPT-NeoX. Available as an early limited access.

All configs are meant for running in Weight Streaming mode with Appliance mode and Kubernetes.

**NOTE**: In absence of banded sparse attention feature, the GPT3 small, medium and large models are equivalent to the corresponding GPT2 variants available in [gpt2 configs](../gpt2/configs/) directory.

## Appendix

**Reference**: Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

**Reference**: Brown, T.B. et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).

**Reference**: Hoffmann, J. et al. (2022). [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

**Reference**: Rae, J. W. et al. (2021). [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446).

