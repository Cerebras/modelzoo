# Cerebras Model Zoo

<picture>
  <source srcset="images/dark-image-nb.png" media="(prefers-color-scheme: dark)">
  <source srcset="images/light-image.png" media="(prefers-color-scheme: light)">
  <img src="images/light-image.png" alt="Cerebras banner">
</picture>

## Introduction

The Cerebras Model Zoo is a collection of deep learning models and utilities optimized to run on Cerebras hardware. The repository provides reference implementations, configuration files, and utilities that demonstrate best practices for training and deploying models using Cerebras systems. 

### Key Features and Components

* [CLI](https://training-docs.cerebras.ai/rel-2.7.0/model-zoo/cli-overview): The ModelZoo CLI is a comprehensive command-line interface that serves as a single entry point for all ModelZoo-related tasks. It streamlines workflows such as data preprocessing, model training, and validation.
* [Models](./src/cerebras/modelzoo/models): Includes configuration files and reference implementations for a wide range of NLP, vision, and multimodal models, including Llama, Mixtral, DINOv2, and Llava. These are optimized for Cerebras hardware and follow best practices for performance and scalability. 
* [Data Preprocessing Tools](https://training-docs.cerebras.ai/rel-2.7.0/model-zoo/core-workflows/quickstart-guide-for-data-preprocessing): Scripts and utilities for preparing datasets for training, including tokenization, formatting, and batching for supported models.
* [Checkpoint Converters and Porting Tools](https://training-docs.cerebras.ai/rel-2.7.0/model-zoo/migration/convert-checkpoints-and-model-configs/convert-checkpoints-and-model-configs): Tools for converting between checkpoint formats (e.g., Cerebras ↔ HuggingFace) and porting PyTorch models to run on Cerebras systems.
* Advanced Features: Support for training optimizations such as custom training loops, custom model implementations, µParam (μP) scaling, rotary position embedding (RoPE) scaling for extended sequence lengths, and more.

### Ready to Get Started?

* Reach out to us [here](https://cerebras.ai/contact) to get access to Cerebras Hardware and ModelZoo!
* Install Cerebras ModelZoo by following the steps in our [setup guide](https://training-docs.cerebras.ai/rel-2.7.0/getting-started/setup-and-installation). 
* Once you have ModelZoo installed, get started by [pretraining](https://training-docs.cerebras.ai/rel-2.7.0/getting-started/setup-and-installation) or [finetuning](https://training-docs.cerebras.ai/rel-2.7.0/getting-started/fine-tune-your-first-model) your first model!
* Visit our [developer documentation](https://training-docs.cerebras.ai) for comprehensive guides on everything you can do with Cerebras ModelZoo. 

## Models in this repository

| Model                           | Code pointer                                                                 | Model                                    | Code pointer                                                                 |
|:--------------------------------|:-----------------------------------------------------------------------------:|:-----------------------------------------|:-----------------------------------------------------------------------------:|
| BERT                            | [Code](./src/cerebras/modelzoo/models/nlp/bert/)                             | BLOOM                                    | [Code](./src/cerebras/modelzoo/models/nlp/bloom/)                            |
| BTLM                            | [Code](./src/cerebras/modelzoo/models/nlp/btlm/)                             | DiT                                      | [Code](./src/cerebras/modelzoo/models/vision/dit)                            |
| DINOv2                          | [Code](./src/cerebras/modelzoo/models/vision/dino/)                          | DPO                                      | [Code](./src/cerebras/modelzoo/models/nlp/dpo)                               |
| DPR                             | [Code](./src/cerebras/modelzoo/models/nlp/dpr)                               | ESM-2                                    | [Code](./src/cerebras/modelzoo/models/nlp/esm2)                              |
| Falcon                          | [Code](./src/cerebras/modelzoo/models/nlp/falcon)                            | Gemma 2                                    | [Code](./src/cerebras/modelzoo/models/nlp/gemma2/)                            |
| GPT-2                           | [Code](./src/cerebras/modelzoo/models/nlp/gpt2/)                             | GPT-3                                    | [Code](./src/cerebras/modelzoo/models/nlp/gpt3/)                             |
| GPT-J & GPT-Neox                | [Code](./src/cerebras/modelzoo/models/nlp/gptj/)                             | Jais                                     | [Code](./src/cerebras/modelzoo/models/nlp/jais)                              |
| LLaMA                           | [Code](./src/cerebras/modelzoo/models/nlp/llama)                             | LLaVA                                    | [Code](./src/cerebras/modelzoo/models/multimodal/llava)                      |
| Mistral                         | [Code](./src/cerebras/modelzoo/models/nlp/mistral)                           | Mixtral of Experts                       | [Code](./src/cerebras/modelzoo/models/nlp/mixtral)                           |
| Multimodal Simple               | [Code](./src/cerebras/modelzoo/models/multimodal/multimodal_simple)          | SantaCoder                               | [Code](./src/cerebras/modelzoo/models/nlp/santacoder)                        |
| StarCoder                       | [Code](./src/cerebras/modelzoo/models/nlp/starcoder)                         | Transformer                              | [Code](./src/cerebras/modelzoo/models/nlp/transformer/)                      |
| T5                              | [Code](./src/cerebras/modelzoo/models/nlp/t5/)                               |                                          |                                                                             |

## License

[Apache License 2.0](./LICENSE)
