# Mistral Language Models

## Overview of the model

[Mistral](https://arxiv.org/abs/2310.06825) is a very similar architecture to [LLaMa](../llama/) except that:
1. Grouped-query attention (GQA), which reduces the number of attention-heads for keys and values
2. Sliding window attention (SWA) of 4k, which attends to a smaller local window of a sequence rather than the full sequence
3. Higher default maximum sequence length (MSL) of 32k, rather than 4k

For more details on each technique we refer to the original papers in the `References` section. 

## Structure of the code

The code for Mistral uses the same infrastructure as our implementation of [GPT-2](../gpt2/); we refer to the README under GPT-2 for most instructions. The code in this directory contains:

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.

## Configs included for this model

For convenience, we provide different configurations of common model setups for Mistral.

- [params_mistral_7B.yaml](./configs/params_mistral_7B.yaml): A 7B parameter model configured as described in the original paper.
- [params_mistral_7B_msl128k.yaml](./configs/params_mistral_7B_msl128k.yaml): A 7B parameter model configured as above but with support for much higher sequence lengths. The sliding window attention allows Mistral to have much higher efficiency at longer sequence lengths. 


## Appendix

**Reference**: Touvron, Hugo, et al. (2023). [Llama: Open and efficient foundation language models](https://arxiv.org/pdf/2302.13971.pdf)

**Reference**: Jiang, Albert, et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825)

**Reference**: Ainslie, Joshua, et al. (2023). [GQA: Training Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)

**Reference**: Child, Rewon, et al. (2019). [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)