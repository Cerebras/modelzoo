# GPT-J, GPT-Neox

- [GPT-J, GPT-Neox](#running-gpt-j)
  - [Introduction](#introduction)
  - [Structure of the code](#structure-of-the-code)
  - [Data Preparation](#data-preparation)
    - [Download and Process the Pile Dataset](#download-and-process-the-pile-dataset)
    - [Run with Data Processors](#run-with-data-processors)
  - [Input Function](#input-function)
    - [Features Dictionary](#features-dictionary)
  - [Configuration files included for this model](#configuration-files-included-for-this-model)
  - [How to run](#how-to-run)
    - [To compile/validate, run train and eval on Cerebras System](#to-compilevalidate-run-train-and-eval-on-cerebras-system)
    - [To run train and eval on GPU/CPU](#to-run-train-and-eval-on-gpucpu)
  - [Citations](#citations)
 
## Introduction

#### GPT-J:
GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax) is an auto-regressive language model created by [EleutherAI](https://www.eleuther.ai/). A canonical configuration of the model, 
GPT-J-6B [[1]](https://github.com/kingoflolz/mesh-transformer-jax) ([param_file](configs/params_gptj.yaml)), has `6` billion parameters and it is has been trained by EleutherAI on a dataset called [Pile](https://arxiv.org/abs/2101.00027). Pile is
carefully assembled and curated from a large number of text datasets from different domains. 
GPT-J-6B has been demonstrated to perform reasonably well on a number of natural language tasks "as-is", 
without any further training, in a zero-shot setting. With our implementation of GPT-J it is now easy to pre-train a GPT-J model and fine-tune this model on a single CS system
with a custom domain-specific or task-specific dataset.

The design of the GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax) model is similar to GPT-3 [[4]](https://arxiv.org/abs/2005.14165) with a few notable differences:
* GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax) introduces a parallel decoder architecture, where attention and feed-forward layers in decoder are 
computed in parallel and then the results are added, as opposed to computing them  sequentially 
by feeding the attention output into the feed-forward layer, as in the standard transformer models (see the diagram with GPT and GPT-J architecture comparison below). This architectural 
change has been introduced by EleutherAI to achieve higher throughput with distributed training and it allows us to run the model on a single CS system without any model parallelism.
With the traditional design, residual attention with op-sharding requires one all-reduce operation in the forward pass and one in the backward pass [[2]](https://arxiv.org/abs/1909.08053). Op-sharding is a concept used in model parallelism paradigm especially [Tensor Parallelism](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-extended-features-pytorch-tensor-parallelism-how-it-works.html) where layers or modules are partitioned across tensor parallel ranks. 
By computing attention and the feed-forward layers in parallel, the results can be reduced locally before performing a single all-reduce on the coordinator host.
This leads to, on average, a `15%` increase in throughput on traditional hardware without noticeable impact on the convergence results.
* GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax) model uses Rotary Position Embeddings (RoPE) (see [[3]](https://arxiv.org/abs/2104.09864)), which has been shown to result in a better model quality 
in tasks with long textual inputs. We use 25% rotary embeddings, as it has been shown to get a good balance between 
computational efficiency and model quality [[1]](https://github.com/kingoflolz/mesh-transformer-jax). This means, given a sample,  we apply RoPE [[3]](https://arxiv.org/abs/2104.09864) to `25%` of the features and apply conventional [sinusoidal positional embedding](#https://openreview.net/pdf?id=onxoVA9FxMw) to the remaining `75%` of the features. Using this mixture of positional embeddings, it is shown to get a good balance between throughput and convergence speed [[5]](https://blog.eleuther.ai/rotary-embeddings/).
* GPT-J [[1]](https://github.com/kingoflolz/mesh-transformer-jax) uses dense attention instead of the efficient sparse attention used in [[4]](https://arxiv.org/abs/2005.14165). EleutherAI stated that dense attention has been used 
for simplicity, as sparse attention would not have significantly improved throughput at this scale. 

![](./images/GPT-vs-GPT-J.png)

#### GPT-Neox:
GPT-Neox [[6]](https://github.com/EleutherAI/gpt-neox) follows the same architecture as GPT-J with minor changes including:
* Untied layernorm: in the transformer block, 2 independent Layer Norms instead of a tied layer norm is used. Based on Eleuther's ablation study, this change doesn't make a difference in performance.
* Tokenizers: EleutherAI retrained the tokenizers on the [Pile](https://arxiv.org/abs/2101.00027) dataset which has more diverse text sources compared to gpt2. They also made the tokenizer more performant on dealing with whitespaces 
by adding token embeddings on repeated space tokens and applied consistent non-space-delimited token at the start of string. Changes on its tokenizer make GPT-Neox more capable at handling dataset with programming code. Details can be found in section 3.2 of the [paper](https://arxiv.org/pdf/2204.06745.pdf).

## Structure of the code

-   `configs/`: YAML configuration files.
-   `data.py`: The entry point to the data input pipeline code. Defines `train_input_dataloader`.
-   `gptj_model.py`: Defines the core model `GPTJModel` for GPT-J and GPT-Neox.
-   `model.py`: The entry point to the model. Defines `GptjModel` which supports GPT-J and GPT-Neox. 
-   `run.py`: Training script. Performs training and validation.
-   `utils.py`: Miscellaneous scripts to parse the `params` dictionary from the YAML files.

## Data Preparation

#### Download and Process the [Pile](https://arxiv.org/abs/2101.00027) Dataset:
Pile is a dataset of diverse text for language modeling. It is constructed from `22` diverse high-quality subsets, both existing and newly constructed, many of which derive from academic or professional sources.

In order to launch pre-training, you need to preprocess Pile to generate HDF5 files. Follow [these instructions](../../data_processing/scripts/pile)  to download the raw data, extract it and generate HDF5 files to be used by the dataloader.

#### Run with Data Processors:
The `GptHDF5DataProcessor` class in [`GptHDF5DataProcessor.py`](../gpt2/input/GptHDF5DataProcessor.py) creates `example_dict` iterative from the `self.features_list` which is returned on the call iteratively. 

## Input Function
If you want to use your own data loader with this example code, then this section describes the input data format expected by `GptjModel` class defined in [model.py](model.py). The `GptjModel` supports GPT-J and GPT-Neox model architecture.

When you create your own custom GPT input function, you must ensure that your GPT input function produces a features dictionary as described in this section.

#### Features Dictionary

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

## Configuration files included for this model

We provide the following config file for pre-training the GPT-J and GPT-Neox models, located under the [configs](configs) directory. 

* [params_gptj_6B.yaml](configs/params_gptj_6B.yaml): GPT-J model with `hidden_size=4096`, `num_hidden_layers=28`, `num_heads=16` and the gpt2 tokenizer.
* [params_gpt_neox_small.yaml](configs/params_gpt_neox_small.yaml): GPT-Neox model with `hidden_size=768`, `num_hidden_layers=12`, `num_heads=12` and the neox tokenizer.
* [params_gpt_neox_1p3b.yaml](configs/params_gpt_neox_1p3b.yaml): GPT-Neox model with `hidden_size=2048`, `num_hidden_layers=24`, `num_heads=16` and the neox tokenizer.
* [params_gpt_neox_2p7b.yaml](configs/params_gpt_neox_2p7b.yaml): GPT-Neox model with `hidden_size=2560`, `num_hidden_layers=32`, `num_heads=32` and the neox tokenizer.
* [params_gpt_neox_20B.yaml](configs/params_gpt_neox_20B.yaml): GPT-Neox model with `hidden_size=6144`, `num_hidden_layers=44`, `num_heads=64` and the neox tokenizer.

Dimensions of the rotary position embeddings is determined by setting `position_embedding_type` to `"rotary"` and `rotary_dim` to an even number.
By default, `rotary_dim` of the `small` and `20B` versions are calculated by `hidden_size` // `num_heads` // `4`, `rotary_dim` of the `1.3B` and `2.7B` versions are calculated by `hidden_size` // `num_heads`.
`use_untied_layer_norm` controls whether to use untied tied layer norm in the model. This parameter is set to `True` in GPT-J configs and `False` in GPT-Neox configs.
Other parameters in the config file performs the same as in our GPT-2 [implementation](../gpt2/).

## How to run

**IMPORTANT**: See the following notes before proceeding further.

**Parameter settings in YAML config file**: The config YAML files are located in the [configs](configs/) directory. Before starting a pre-training run, make sure that in the YAML config file you are using:

-   The `train_input.data_dir` parameter points to the correct dataset, and
-   The `train_input.max_sequence_length` parameter corresponds to the sequence length of the dataset.
-   The `model.max_position_embeddings` parameter corresponds to the maximum dimension of position embeddings.

**YAML config files**: Details on the configs for this model can be found in [Configuration files included for this model](#configuration-files-included-for-this-model)

In the following example run commands, we use `/path/to/yaml`, `/path/to/model_dir`, and `train` as placeholders for user supplied inputs.

-   `/path/to/yaml` is a path to the YAML config file with model parameters such one of the configurations described in [Configs included for this model](#configs-included-for-this-model).
-   `/path/to/model_dir` is a path to the directory where you would like to store the logs and other artifacts of the run.
-   `--mode` specifies the desired mode to run the model in. Change to `--mode eval` to run in eval mode.

### To compile/validate, run train and eval on Cerebras System

Please follow the instructions on our Developer Docs at:
https://docs.cerebras.net/en/latest/getting-started/pytorch/index.html

### To run train and eval on GPU/CPU

If running on a cpu or gpu, activate the environment from [Python GPU Environment setup](../../../../PYTHON-SETUP.md), and simply run:

```
python run.py --mode train --params /path/to/yaml --model_dir /path/to/model_dir
```

## Citations
[1] [Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX](https://github.com/kingoflolz/mesh-transformer-jax), May 2021.

[2] [Megatron-lm: Training multi-billion parameter language models using model parallelism](https://arxiv.org/abs/1909.08053), September 2019.

[3] [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), April 2021.

[4] [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), July 2020.

[5] [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/)

[6] [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/pdf/2204.06745)