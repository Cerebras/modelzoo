# BTLM

This directory contains the PyTorch ML reference for BTLM model.

- [BTLM](#btlm)
  - [Overview of the model](#overview-of-the-model)
  - [Structure of the code](#structure-of-the-code)
  - [Prepare the data](#prepare-the-data)
      - [BTLM DataProcessor output](#btlm-dataprocessor-output)
  - [BTLM input function](#btlm-input-function)
      - [BTLM features dictionary](#btlm-features-dictionary)
- [How to run](#how-to-run)
  - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
  - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
  - [Configs included for this model](#configs-included-for-this-model)

## Overview of the model

Released in July 2023, BTLM quickly became the most downloaded model of its size on [Hugging Face](https://huggingface.co/cerebras/btlm-3b-8k-base), amassing over 1 million downloads in three weeks. [BTLM](https://www.cerebras.net/machine-learning/btlm-3b-8k-7b-performance-in-a-3-billion-parameter-model/) is a very similar architecture to [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) with the exception of using [Maximal Update Parameterization (&mu;P)](https://arxiv.org/abs/2203.03466) and adding [SwiGLU activation](https://arxiv.org/abs/2002.05202) and [ALiBi](https://arxiv.org/abs/2108.12409) for [performance improvements](https://arxiv.org/abs/2210.15424). More details can be found in in the [BTLM paper](https://arxiv.org/abs//2309.11568).


## Structure of the code

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.

## Prepare the data

You may to download raw SlimPajama data from [here](https://huggingface.co/datasets/cerebras/SlimPajama-627B):

```
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```

and create preprocessed dataset files using [`create_hdf5_dataset.py`](../../../data_preparation/nlp/hdf5_preprocessing/).

#### BTLM DataProcessor output

The `GptHDF5MapDataProcessor` class in [`GptHDF5MapDataProcessor.py`](../../../data/nlp/gpt/GptHDF5MapDataProcessor.py) creates `example_dict` iterative from the `self.features_list` which is returned on the call iteratively.
 
## BTLM input function

If you want to use your own data loader with this example code, then this section describes the input data format expected by `Gpt2Model` class defined in [model.py](../gpt2/model.py). The `Gpt2Model` supports BTLM model architecture.

When you create your own custom GPT input function, you must ensure that your GPT input function produces a features dictionary as described in this section.

#### BTLM features dictionary

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
-   The `model.max_position_embeddings` parameter corresponds to the maximum dimension of position embeddings.

**YAML config files**: Details on the configs for this model can be found in the [Configs included for this model](#configs-included-for-this-model) section.

In the following example run commands, we use `/path/to/yaml`, `/path/to/model_dir`, and `train` as placeholders for user supplied inputs.

-   `/path/to/yaml` is a path to the YAML config file with model parameters such one of the configurations described in the [Configs included for this model](#configs-included-for-this-model) section.
-   `/path/to/model_dir` is a path to the directory where you would like to store the logs and other artifacts of the run.
-   `--mode` specifies the desired mode to run the model in. Change to `--mode eval` to run in eval mode.

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our [quickstart in the Developer Docs](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html).

## To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../../../PYTHON-SETUP.md), and simply run:

```
python run.py {CPU,GPU} --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```
## Configs included for this model

For convenience, we provide the configurations used to train BTLM.

- [params_btlm_2p7b_2k.yaml](./configs/params_btlm_2p7b_2k.yaml): A 2.7B parameter model designed to match the first stage of training BTLM at 2k maximum sequence length.
- [params_btlm_2p7b_8k.yaml](./configs/params_btlm_2p7b_8k.yaml): A 2.7B parameter model designed to match the second stage of training BTLM at 8k maximum sequence length.
