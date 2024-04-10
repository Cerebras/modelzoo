# Jais

This directory contains the PyTorch ML reference for Jais model.

- [Jais](#jais)
  - [Overview of the model](#overview-of-the-model)
  - [Structure of the code](#structure-of-the-code)
  - [Prepare the data](#prepare-the-data)
      - [Jais DataProcessor output](#jais-dataprocessor-output)
  - [Jais input function](#jais-input-function)
      - [Jais features dictionary](#jais-features-dictionary)
- [How to run](#how-to-run)
  - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
  - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
  - [Configs included for this model](#configs-included-for-this-model)

## Overview of the model

Jais is based on a transformer-based decoder-only (GPT-3) architecture and uses SwiGLU non-linearity. It implements ALiBi position embeddings, enabling the model to extrapolate to long sequence lengths, providing improved context handling and model precision. For more information, see the [paper](https://arxiv.org/abs/2308.16149).


## Structure of the code

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.

## Prepare the data

You may to download raw SlimPajama data from [here](https://huggingface.co/datasets/cerebras/SlimPajama-627B):

```
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```

and create preprocessed dataset files using [`create_hdf5_dataset.py`](../../data_processing/scripts/hdf5_preprocessing/).

#### Jais DataProcessor output

The `GptHDF5MapDataProcessor` class in [`GptHDF5MapDataProcessor.py`](input/GptHDF5MapDataProcessor.py) creates `example_dict` iterative from the `self.features_list` which is returned on the call iteratively.
 
## Jais input function

If you want to use your own data loader with this example code, then this section describes the input data format expected by `Gpt2Model` class defined in [model.py](../gpt2/model.py). The `Gpt2Model` supports Jais model architecture.

When you create your own custom GPT input function, you must ensure that your GPT input function produces a features dictionary as described in this section.

#### Jais features dictionary

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

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../PYTHON-SETUP.md), and simply run:

```
python run.py {CPU,GPU} --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```
## Configs included for this model

For convenience, we provide the configurations used to train Jais.

- [params_jais_30b.yaml](./configs/params_jais_30b.yaml): A 30B parameter pre-trained bilingual large language model for both Arabic and English.
- [params_jais_30b_chat.yaml](./configs/params_jais_30b_chat.yaml): A 30B parameter model fine-tuned over a curated Arabic and English prompt-response pairs dataset.
