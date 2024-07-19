# Santacoder Language Model

## Overview of the model

[Santacoder](https://arxiv.org/abs/2301.03988) is a very similar architecture to [GPT-2](../gpt2/) except that it uses multi-query attention [(MQA)](https://arxiv.org/abs/1911.02150) for faster inference. It is trained on parts of [The Stack](https://arxiv.org/abs/2211.15533) dataset.  

For more details we refer to the original papers in the `References` section. 

## Structure of the code

The code for Santacoder uses the same infrastructure as our implementation of [GPT-2](../gpt2/); we refer to the README under GPT-2 for most instructions. The code in this directory contains:

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.

## Configs included for this model

For convenience, we provide a configuration for the standard setup of Santacoder.

- [params_santacoder_1b.yaml](./configs/params_santacoder_1b.yaml): A 1B parameter model configured as described in the original paper.


## Appendix

**Reference**: Radford, Alec et al. (2018): Language Models are Unsupervised Multitask Learners

**Reference**: Shazeer, Noam (2019): Fast Transformer Decoding: One Write-Head is All You Need

**Reference**: Kocetkov, Denis et al. (2022): The Stack: 3 TB of permissively licensed source code

