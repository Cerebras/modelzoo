# Cerebras Model Zoo

## Introduction

This repository contains examples of common deep learning models that can be trained on Cerebras hardware. These models demonstrate the best practices for coding a model targeted at the Cerebras hardware so that you can take full advantage of this new powerful compute engine.

In order to get started with running your models on a Cerebras system, please refer to the [Developer Documentation](https://docs.cerebras.net/en/latest/index.html) along with this readme.

**NOTE**: If you are interested in trying out Cerebras Model Zoo on Cerebras Hardware (CS-2 Systems), we offer the following options:

- Academics - Please fill out our Partner Hardware Access Request form [here](https://www.cerebras.net/developers/partner-hardware-access-request/) and we will contact you about gaining access to a system from one of our partners.
- Commercial - Please fill out our Get Demo form [here]( https://www.cerebras.net/get-demo/) so that our team can provide you with a demo and discuss access to our system.
- For all others - Please contact us at developer@cerebras.net.

For a list of all supported models, please check [models in this repository](#models-in-this-repository).

## Installation

To install the Cerebras Model Zoo on the CSX system, please follow the instructions in [PYTHON-SETUP.md](./PYTHON-SETUP.md).

## Supported frameworks

We support the models developed in [PyTorch](https://pytorch.org/).

## Basic workflow

When you are targeting the Cerebras Wafer-Scale Cluster for your neural network jobs, please follow the [quick start guide](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html) from the developer docs to compile, validate and train the models in this Model Zoo for the framework of your choice.

For advanced use cases and porting your existing code please refer to the [developer docs](https://docs.cerebras.net/en/latest/wsc/port/index.html).

## Models in this repository

| Model   | Code pointer   |
|:-------|:-----------------------:|
| BERT | [Code](./modelzoo/transformers/pytorch/bert/) |
| BERT (fine-tuning) Classifier | [Code](./modelzoo/transformers/pytorch/bert/fine_tuning/classifier/) |
| BERT (fine-tuning) Named Entity Recognition | [Code](./modelzoo/transformers/pytorch/bert/fine_tuning/token_classifier/) |
| BERT (fine-tuning) Summarization | [Code](./modelzoo/transformers/pytorch/bert/fine_tuning/extractive_summarization/) |
| BERT (fine-tuning) Question Answering | [Code](./modelzoo/transformers/pytorch/bert/fine_tuning/qa/) |
| BLOOM | [Code](./modelzoo/transformers/pytorch/bloom/) |
| BTLM  | [Code](./modelzoo/transformers/pytorch/btlm/) |
| GPT-2 | [Code](./modelzoo/transformers/pytorch/gpt2/) |
| GPT-3 | [Code](./modelzoo/transformers/pytorch/gpt3/) |
| GPT-J | [Code](./modelzoo/transformers/pytorch/gptj/) |
| GPT-NeoX | [Code](./modelzoo/transformers/pytorch/gptj/) |
| GPT-J (fine-tuning) Summarization |[Code](./modelzoo/transformers/pytorch/gptj/fine_tuning/continuous_pretraining/) |
| RoBERTa | [Code](./modelzoo/transformers/pytorch/bert/) |
| T5 | [Code](./modelzoo/transformers/pytorch/t5/) |
| Transformer | [Code](./modelzoo/transformers/pytorch/transformer/) |
| MNIST (fully connected) | [Code](./modelzoo/fc_mnist/pytorch/) |
| UNet | [Code](./modelzoo/vision/pytorch/unet/) |
| Falcon | [Code](./modelzoo/transformers/pytorch/falcon) |
| StarCoder | [Code](./modelzoo/transformers/pytorch/starcoder) |
| LLaMA | [Code](./modelzoo/transformers/pytorch/llama) |

## License

[Apache License 2.0](./LICENSE)
