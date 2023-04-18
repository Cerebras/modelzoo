# Cerebras Model Zoo

## Introduction

This repository contains examples of common deep learning models that can be trained on Cerebras hardware. These models demonstrate the best practices for coding a model targeted at the Cerebras hardware so that you can take full advantage of this new powerful compute engine.

In order to get started with running your models on a Cerebras system, please refer to the [Developer Documentation](https://docs.cerebras.net/en/latest/index.html) along with this readme.

**NOTE**: If you are interested in trying out Cerebras Model Zoo on Cerebras Hardware (CS-2 Systems), we offer the following options:

- Academics - Please fill out our Partner Hardware Access Request form [here](https://www.cerebras.net/developers/partner-hardware-access-request/) and we will contact you about gaining access to a system from one of our partners.
- Commercial - Please fill out our Get Demo form [here]( https://www.cerebras.net/get-demo/) so that our team can provide you with a demo and discuss access to our system.
- For all others - Please contact us at developer@cerebras.net.

For a list of all supported models, please check [models in this repository](#models-in-this-repository).

## Supported frameworks

We support the models developed in [PyTorch](https://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/).

## Basic workflow

When you are targeting the Cerebras Wafer-Scale Cluster for your neural network jobs, please follow the [quick start guide](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html) from the developer docs to compile, validate and train the models in this Model Zoo for the framework of your choice.

For advanced use cases and porting your existing code please refer to the [developer docs](https://docs.cerebras.net/en/latest/wsc/port/index.html).

## Execution modes

On the Cerebras Wafer Scale Cluster you can run neural networks of different model sizes. Cerebras Software supports different execution modes to efficiently run such variety of models.

The execution mode refers to how the Cerebras runtime loads your neural network model onto the Cerebras Wafer Scale Engine (WSE). Two execution modes are supported:

- **Weight streaming**: In this mode one layer of the neural network model is loaded at a time. This layer-by-layer mode is used to run extremely large models (with billions to trillions of parameters).
- **Layer pipelined**: In this mode all the layers of the network are loaded altogether onto the Cerebras WSE. This mode is selected for neural network models of small to medium sized models (with less than a billion parameters).

You can get more information about this on the developer page section on [Cerebras Execution Modes](https://docs.cerebras.net/en/latest/wsc/cerebras-basics/cerebras-execution-modes.html)

## Models in this repository

| Model                                       | Layer Pipeline mode                                                                                                                                                               | Weight Streaming mode                                                                                        |
|---------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| BERT                                        | [TensorFlow code](./modelzoo/transformers/tf/bert/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/)                                                                      | [PyTorch code](./modelzoo/transformers/pytorch/bert/)                                                                                                            |
| BERT (fine-tuning) Classifier               | [TensorFlow code](./modelzoo/transformers/tf/bert/fine_tuning/classifier/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/fine_tuning/classifier/)                        | -                                                                                                            |
| BERT (fine-tuning) Named Entity Recognition | [TensorFlow code](./modelzoo/transformers/tf/bert/fine_tuning/token_classifier/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/fine_tuning/token_classifier/)            | -                                                                                                            |
| BERT (fine-tuning) Summarization            | [TensorFlow code](./modelzoo/transformers/tf/bert/fine_tuning/extractive_summarization/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/fine_tuning/extractive_summarization/) | -                                                                                                            |
| BERT (fine-tuning) Question Answering       | [TensorFlow code](./modelzoo/transformers/tf/bert/fine_tuning/qa/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/fine_tuning/qa/)                                        | -                                                                                                            |
| GPT-2                                       | [TensorFlow code](./modelzoo/transformers/tf/gpt2/)<br>[PyTorch code](./modelzoo/transformers/pytorch/gpt2/)                                                                      | [TensorFlow code](./modelzoo/transformers/tf/gpt2/)<br>[PyTorch code](./modelzoo/transformers/pytorch/gpt2/) |
| GPT-3                                       | -                                                                                                                                                                                 | [TensorFlow code](./modelzoo/transformers/tf/gpt3/)<br>[PyTorch code](./modelzoo/transformers/pytorch/gpt3/) |
| GPT-J                                       | -                                                                                                                                                                                 | [TensorFlow code](./modelzoo/transformers/tf/gptj/) <br>[PyTorch code](./modelzoo/transformers/pytorch/gptj/) |
| GPT-NeoX                                    | -                                                                                                                                                                                 | [TensorFlow code](./modelzoo/transformers/tf/gptj/) <br>[PyTorch code](./modelzoo/transformers/pytorch/gptj/) |
| GPT-J (fine-tuning) Summarization           | -                                                                                                                                                                                 | [TensorFlow code](./modelzoo/transformers/tf/gptj/fine_tuning/abstractive_summarization/)                    |
| Linformer                                   | [TensorFlow code](./modelzoo/transformers/tf/linformer/)                                                                                                                          | -                                                                                                            |
| RoBERTa                                     | [TensorFlow code](./modelzoo/transformers/tf/bert/)<br>[PyTorch code](./modelzoo/transformers/pytorch/bert/)                                                                      | -                                                                                                            |
| T5                                          | [TensorFlow code](./modelzoo/transformers/tf/t5/)<br>[PyTorch code](./modelzoo/transformers/pytorch/t5/)                                                                          | [PyTorch code](./modelzoo/transformers/pytorch/t5/)                                                                                                                     |
| Transformer                                 | [TensorFlow code](./modelzoo/transformers/tf/transformer/)<br>[PyTorch code](./modelzoo/transformers/pytorch/transformer/)                                                        | -                                                                                                            |
| MNIST (fully connected)                     | [TensorFlow code](./modelzoo/fc_mnist/tf/)<br>[PyTorch code](./modelzoo/fc_mnist/pytorch/)                                                                                        | -                                                                                                            |
| UNet        | -                                                                                           | [PyTorch code](./modelzoo/vision/pytorch/unet/)                                                                                                               |

## License

[Apache License 2.0](./LICENSE)


