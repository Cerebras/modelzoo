# Continuous Pre-training with GPT-J

- [Continuous Pre-training with GPT-J](#continuous-pre-training-with-gpt-j)
  - [Introduction](#introduction)
  - [Sequence of the steps to perform](#sequence-of-the-steps-to-perform)
  - [Structure of the code](#structure-of-the-code)
  - [Dataset](#dataset)
    - [Data preparation](#data-preparation)
  - [Input function](#input-function)
    - [Features Dictionary](#features-dictionary)
  - [Using pretrained checkpoints and configs from HuggingFace](#using-pretrained-checkpoints-and-configs-from-huggingface)
  - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
    - [Passing the pretrained checkpoint](#passing-the-pretrained-checkpoint)
    - [Run continuous-pretraining on GPU and CPU](#run-continuous-pretraining-on-gpu-and-cpu)
  - [Configuration files included for this model](#configuration-files-included-for-this-model)
  - [Citations](#citations)


## Introduction

In Natural Language Processing (NLP), in the presence of widely available pre-trained checkpoints on large datasets like [PILE](https://arxiv.org/abs/2101.00027), it may be desirable to fine-tune such checkpoints on domain-specific tasks.

Transformer [[1]](https://arxiv.org/pdf/1706.03762.pdf) based models such as GPT-J and GPT NeoX [[2]](https://github.com/kingoflolz/mesh-transformer-jax) are very effective in the continuous pre-training task on the domain-specific dataset. Here we demonstrate how to use GPTJ 6B model for continuous pre-training on [TRC2 dataset](https://trec.nist.gov/data/reuters/reuters.html).

## Sequence of the steps to perform

1. Download and clean the TRC Data
2. Convert dataset to hdf5 
3. Download pre-trained model checkpoint from HuggingFace
4. Convert checkpoint to CS format 
5. Run training

## Structure of the code

- `configs/`: YAML configuration files for continuous pre-training using GPT-J.
-  [data_preparation/nlp/gptj](../../../../data_preparation/nlp/gptj): contains scripts to clean and split the TRC2 dataset.

## Dataset

For the continuous pretraining example, we use the TRC2 dataset. The dataset consists of News articles from Reuters Corpora.

### Data preparation

Please see the instructions at [TRC2 dataset preparation](../../../../data_preparation/nlp/gptj/README.md).

## Input function

The `GptHDF5DataProcessor` class in [`GptHDF5DataProcessor.py`](../../../../data/nlp/gpt/GptHDF5DataProcessor.py) creates `example_dict` from the `self.features_list` which is returned on the call iteratively.

If you want to use your own data loader with this example code, then this section describes the input data format expected by `GptjModel` class defined in [model.py](model.py). The `GptjModel` supports GPT-J and GPT-Neox model architecture.

When you create your own custom GPT input function, you must ensure that your GPT input function produces a features dictionary as described in the next section.

### Features Dictionary

The features dictionary of a GPT-J (Neox) model has the following key/values:

- `input_ids`: Input token IDs, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `attention_mask`: Mask for padded positions. Has values `0` on the padded positions and `1` elsewhere.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`
- `labels`: Labels for language modeling pre-training task, padded with `0` to `max_sequence_length`.
  - Shape: `(batch_size, max_sequence_length)`
  - Type: `torch.int32`

## Using pretrained checkpoints and configs from HuggingFace

For continuous pre-training, we can use the checkpoints and configs from Hugging Face [repo](https://huggingface.co/EleutherAI/gpt-j-6B). These checkpoints and configs need to be converted into Cerebras Checkpoint format before being able to use them.

In order to convert the HF checkpoint and config, use the [checkpoint converter tool](https://docs.cerebras.net/en/latest/wsc/port/checkpoint-formats.html) as below:

```bash
python convert_checkpoint.py convert \
<path to checkpoint file to convert> \
--config <path to EleutherAI_gpt-j-6B_config.json >
--model gptj \
--src-fmt hf \
--tgt-fmt cs-2.3 \
--output-dir <location to save converted checkpoint and config> \
```

NOTE: use the above from the [checkpoint converter tool directory](../../../../tools/checkpoint_converters/).

It is also important to note that the config from Hugging Face may not be usable directly and the converted config from using the tool should be used for compatibility with Cerebras format. The converted config will only contain the `model` params and you will have to add `train_input`, `eval_input`, `optimizer` and `runconfig` as necessary. GPTJ [configs](../configs/) can be used as a reference for populating these.

## To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our [quickstart in the Developer Docs](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html).

Note that we do not have a separate `run.py` file for this, we continue using the gptj `run.py` [file](../run.py).

### Passing the pretrained checkpoint

Please provide the path of the converted pretrained checkpoint in the `--checkpoint_path` for fine-tuning to start from the pre-trained checkpoint. You can do this using the [checkpoint converter tool](../../../../tools/convert_checkpoint.py).

### Run continuous-pretraining on GPU and CPU

To run pre-training on GPU/CPU, use the following command:

```bash
python ../run.py {CPU,GPU} --mode train --params configs/params_gptj_6B_TRC2.yaml --model_dir </path/to/model_dir> --max_steps <num_train_steps>
```

Note that our model implementation and run scripts are compatible to run on a single GPU, however handling any GPU cluster related programming is up-to the user.

## Configuration files included for this model

This repository facilitates fine-tuning GPT-J [[2]](https://github.com/kingoflolz/mesh-transformer-jax) for continuous pre-training task. The config files are located under [./configs](./configs) directory.

- [configs/params_gptj_6B_TRC2.yaml](configs/params_gptj_6B_TRC2.yaml) does fine-tuning for GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax) model using a pre-trained checkpoint. The GPT-J model size is: `hidden_size=4096`, `num_hidden_layers=28`, `num_heads=16`.

## Citations

[1] [Attention Is All You Need by Vaswani, et al.](https://arxiv.org/pdf/1706.03762.pdf), 2017.

[2][Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX](https://github.com/kingoflolz/mesh-transformer-jax), May 2021.

[3] [TRC2 Dataset](https://trec.nist.gov/data/reuters/reuters.html)
