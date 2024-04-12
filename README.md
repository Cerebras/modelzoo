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
| BERT | [Code](./src/cerebras/modelzoo/models/nlp/bert/) |
| BERT (fine-tuning) Classifier | [Code](./src/cerebras/modelzoo/models/nlp/bert/classifier/) |
| BERT (fine-tuning) Named Entity Recognition | [Code](./src/cerebras/modelzoo/models/nlp/bert/token_classifier/) |
| BERT (fine-tuning) Summarization | [Code](./src/cerebras/modelzoo/models/nlp/bert/extractive_summarization/) |
| BLOOM | [Code](./src/cerebras/modelzoo/models/nlp/bloom/) |
| BTLM  | [Code](./src/cerebras/modelzoo/models/nlp/btlm/) |
| GPT-2 | [Code](./src/cerebras/modelzoo/models/nlp/gpt2/) |
| GPT-3 | [Code](./src/cerebras/modelzoo/models/nlp/gpt3/) |
| GPT-J | [Code](./src/cerebras/modelzoo/models/nlp/gptj/) |
| GPT-NeoX | [Code](./src/cerebras/modelzoo/models/nlp/gptj/) |
| GPT-J (fine-tuning) Summarization |[Code](./src/cerebras/modelzoo/models/nlp/gptj/continuous_pretraining/) |
| RoBERTa | [Code](./src/cerebras/modelzoo/models/nlp/bert/) |
| T5 | [Code](./src/cerebras/modelzoo/models/nlp/t5/) |
| Transformer | [Code](./src/cerebras/modelzoo/models/nlp/transformer/) |
| MNIST (fully connected) | [Code](./src/cerebras/modelzoo/fc_mnist/pytorch/) |
| Falcon | [Code](./src/cerebras/modelzoo/models/nlp/falcon) |
| StarCoder | [Code](./src/cerebras/modelzoo/models/nlp/starcoder) |
| LLaMA and LLaMA-2 | [Code](./src/cerebras/modelzoo/models/nlp/llama) |
| DiT | [Code](./src/cerebras/modelzoo/models/vision/dit) |
| MPT | [Code](./src/cerebras/modelzoo/models/nlp/mpt) |
| Mistral | [Code](./src/cerebras/modelzoo/models/nlp/mistral) |
| DPO | [Code](./src/cerebras/modelzoo/models/nlp/dpo) |
| DPR | [Code](./src/cerebras/modelzoo/models/nlp/dpr) |
| JAIS | [Code](./src/cerebras/modelzoo/models/nlp/jais) |
| LLaVA | [Code](./src/cerebras/modelzoo/models/multimodal/llava) |

## License

[Apache License 2.0](./LICENSE)
