# LLaMa Language Models

## Overview of the model

[LLaMa](https://arxiv.org/pdf/2302.13971.pdf) is a very similar architecture to [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pd) except that:
1. RMSNorm (equivalent to T5-style normalization) instead of LayerNorm during pre-normalization.
2. It uses the SwiGLU (equivalent to silu) activation rather than ReLU, similar to the PaLM model.
3. It uses rotary positional embeddings instead of absolute positional embeddings, like in the GPT-NeoX models.


## Structure of the code

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.

## Configs included for this model

For convenience, we provide different configurations of common model setups designed to give examples of models of different sizes intended for execution in  [weight streaming mode](https://docs.cerebras.net/en/latest/cerebras-basics/cerebras-execution-modes.html).

- [params_llama_7b_reference.yaml](./configs/params_llama_7b_reference.yaml): A 7B parameter model configured as described in the original paper.
- [params_llama_13b_reference.yaml](./configs/params_llama_13b_reference.yaml): A 13B parameter model configured as described in the original paper.
- [params_llama_33b_reference.yaml](./configs/params_llama_33b_reference.yaml): A 32.5B parameter model configured as described in the original paper.
- [params_llama_65b_reference.yaml](./configs/params_llama_65b_reference.yaml): A 65.2B parameter model configured as described in the original paper.


All configs are meant to be run on Weight Streaming mode using Appliance mode and Kubernetes flow.

## Appendix

**Reference**: Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

**Reference**: Touvron, Hugo, et al. (2023). [Llama: Open and efficient foundation language models] (https://arxiv.org/pdf/2302.13971.pdf)