# GPT-3 Language Models

This directory contains the PyTorch ML reference for GPT-2 and GPT-3 models.

- [GPT-3 Language Models](#gpt-3-language-models)
  - [Overview of the model](#overview-of-the-model)
  - [Structure of the code](#structure-of-the-code)
  - [Prepare the data](#prepare-the-data)
    - [GPT-3 DataProcessor output](#gpt-3-dataprocessor-output)
  - [GPT-3 input function](#gpt-3-input-function)
    - [GPT-3 features dictionary](#gpt-3-features-dictionary)
- [How to run](#how-to-run)
  - [Steps to compile and validate](#steps-to-compile-and-validate)
  - [Steps to train on CS System](#steps-to-train-on-cs-system)
  - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
  - [Configs included for this model](#configs-included-for-this-model)
  - [Appendix](#appendix)

## Overview of the model

[GPT-3](https://arxiv.org/abs/2005.14165) is a very similar architecture to [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pd) except that every other self-attention layer in GPT-3 uses locally banded sparse attention in which tokens only attend to each other if they are nearby in the sequence
(see section 2.1 of the [GPT-3 paper](https://arxiv.org/abs/2005.14165) for more details). Figure below describes a high level model architecture of GPT3 model.

![GPT3 Architecture Diagram](./images/architecture_diagram.png)

The larger versions of GPT-3 range from 1.3B to 175B parameters.

**NOTE:** In out current implementation, we use the code from [GPT2 implementation](../gpt2/) which does not have banded sparse attention implemented. We plan to add this support in the future releases.

## Structure of the code

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.

## Prepare the data

You need to download your raw data and create preprocessed dataloader using [`create_hdf5_dataset.py`](../gpt2/input/scripts/create_hdf5_dataset.py). 

#### GPT-3 DataProcessor output
  The `GptHDF5DataProcessor` class in [`GptHDF5DataProcessor.py`](input/GptHDF5DataProcessor.py) creates `example_dict` iterative from the `self.features_list` which is returned on the call iteratively. 
 
## GPT-3 input function

If you want to use your own data loader with this example code, then this section describes the input data format expected by `Gpt2Model` class defined in [model.py](../gpt2/model.py). The `Gpt2Model` supports GPT-2 and GPT3 model architecture.

When you create your own custom GPT input function, you must ensure that your GPT input function produces a features dictionary as described in this section.

#### GPT-3 features dictionary

The features dictionary has the following key/values:

- `input_ids`: Input token IDs, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `attention_mask`: Mask for padded positions. Has values `0` on the padded positions and `1` elsewhere.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `labels`: Labels for language modeling pre-training task, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`

# How to run

**IMPORTANT**: See the following notes before proceeding further.

**Parameter settings in YAML config file**: The config YAML files are located in the [configs](configs/) directory. Before starting a pre-training run, make sure that in the YAML config file you are using:

-   The `train_input.data_dir` parameter points to the correct dataset, and
-   The `train_input.max_sequence_length` parameter corresponds to the sequence length of the dataset.
-   The `model.max_position_embeddings` parameter corresponds to the maximum dimension of position embeddings.

**YAML config files**: Details on the configs for this model can be found in [Configs included for this model](#configs-included-for-this-model)

In the following example run commands, we use `/path/to/yaml`, `/path/to/model_dir`, and `train` as placeholders for user supplied inputs.

-   `/path/to/yaml` is a path to the YAML config file with model parameters such one of the configurations described in [Configs included for this model](#configs-included-for-this-model).
-   `/path/to/model_dir` is a path to the directory where you would like to store the logs and other artifacts of the run.
-   `--mode` specifies the desired mode to run the model in. Change to `--mode eval` to run in eval mode.

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our Developer Docs at:
https://docs.cerebras.net/en/latest/getting-started/pytorch/index.html

## To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../PYTHON-SETUP.md), and simply run:

```
python run.py --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```
## Configs included for this model

For convenience, we provide different configurations of common model setups designed to give examples of models of different sizes intended for execution in  [weight streaming mode](https://docs.cerebras.net/en/latest/cerebras-basics/cerebras-execution-modes.html).

- [params_gpt3_xl.yaml](./configs/params_gpt3_xl.yaml): A 1.3B parameter model designed to match the configuration of the GPT-3 XL model.
- [params_gpt3_6p7b.yaml](./configs/params_gpt3_6p7b.yaml): A 6.7B parameter GPT-2 model designed to match the configuration of the GPT-3 6.7B model.
- [params_gpt3_13b.yaml](./configs/params_gpt3_13b.yaml): A 13B parameter GPT-2 model designed to match the configuration of the GPT-3 13B model. Available as an early limited access.
- [params_gpt3_20b.yaml](./configs/params_gpt3_20b.yaml): A 20B parameter GPT-2 model designed to match the configuration of the GPT-NeoX. Available as an early limited access.


All configs are meant to be run on Weight Streaming mode using Appliance mode and Kubernetes flow.

**NOTE**: In absence of banded sparse attention feature, the GPT3 small, medium and large models are equivalent to the corresponding GPT2 variants available in [gpt2 configs](../gpt2/configs/) directory.

## Appendix

**Reference**: Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

**Reference**: Brown, T.B. et al. (2020). [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165).
