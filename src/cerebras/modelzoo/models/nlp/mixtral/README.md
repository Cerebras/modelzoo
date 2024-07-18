# Mixtral Language Models

## Overview of the model

[Mixtral of Experts](https://arxiv.org/abs/2401.04088) models (8x7B and 8x22B) have similar architecture to [Mistral](../mistral) model, except that they use Mixture of Expert (MoE) instead of regular FFNs. Each MoE block has 8 experts and each token is routed to top-2 expert FFNs.

For more details on each technique we refer to the original papers in the `References` section.

## Structure of the code

The code for Mixtral uses the same infrastructure as our implementation of [GPT-2](../gpt2/); we refer to the README under GPT-2 for most instructions. The code in this directory contains:

- `configs/`: YAML configuration files.
- `run.py`: Training script. Performs training and validation.

## Configs included for this model

For convenience, we provide two different configurations of common model setups for Mixtral models:

- [params_mixtral_8x7B.yaml](./configs/params_mixtral_8x7b.yaml): A 46.7B parameter model configured as described in the original paper.
- [params_mixtral_8x22B.yaml](./configs/params_mixtral_8x22b.yaml): Larger Mixtral model (141B parameters), announced in [Cheaper, Better, Faster, Stronger](https://mistral.ai/news/mixtral-8x22b/), the arXiv paper for this configuration is not published yet.

## Appendix

**Reference**: Jiang, Albert, et al. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825)

**Reference**: Jiang, Albert, et al. (2024). [Mixtral of Experts](https://arxiv.org/abs/2401.04088)
