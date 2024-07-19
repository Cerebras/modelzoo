# LLaMa Language Models

## Overview of the model

[LLaMa](https://arxiv.org/pdf/2302.13971) is a very similar architecture to [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) except that:
1. RMSNorm (equivalent to T5-style normalization) instead of LayerNorm during pre-normalization.
2. It uses the SwiGLU (equivalent to silu) activation rather than ReLU, similar to the PaLM model.
3. It uses rotary positional embeddings instead of absolute positional embeddings, like in the GPT-NeoX models.

[LLaMa-2](https://arxiv.org/pdf/2307.09288) is an updated version of LLaMa, with following changes:
1. Trained on a new mix of publicly available data, and increased the size of the pretraining corpus by 40%.
2. Doubled the context length of the model from 2K to 4K.
3. Adopted [grouped-query attention](https://arxiv.org/pdf/2305.13245) in 70B.

[Code Llama](https://arxiv.org/pdf/2308.12950) is a code-specialized version of Llama 2 that was created by further training Llama 2 on its code-specific datasets, sampling more data from that same dataset for longer.

[LLaMa-3](https://llama.meta.com/llama3/) made a major leap over LLaMa-2 and includes the following changes:
1. New tokenizer with 128K vocabulary that encodes language much more efficiently, which leads to substantially improved model performance.
2. Adopted [grouped-query attention](https://arxiv.org/pdf/2305.13245) for both 8B and 70B.
3. Trained the models on sequences of 8,192 tokens, using a mask to ensure self-attention does not cross document boundaries.

## Structure of the code

-   `configs/`: YAML configuration files.
-   `run.py`: Training script. Performs training and validation.

## Configs included for this model

For convenience, we provide different configurations of common model setups designed to give examples of models of different sizes.

### LLaMa-3
- [params_llama3_8b.yaml](./configs/params_llama3_8b.yaml): A 8B parameter model configured as described in the llama-3 blog.
- [params_llama3_70b.yaml](./configs/params_llama3_70b.yaml): A 70B parameter model configured as described in the llama-3 blog.

### LLaMa-2
- [params_llama2_7b.yaml](./configs/params_llama2_7b.yaml): A 7B parameter model configured as described in the llama-2 paper.
- [params_llama2_13b.yaml](./configs/params_llama2_13b.yaml): A 13B parameter model configured as described in the llama-2 paper.
- [params_llama2_70b.yaml](./configs/params_llama2_70b.yaml): A 70B parameter model configured as described in the llama-2 paper.


### Code LLaMa
- [params_code_llama_7b.yaml](./configs/params_code_llama_7b.yaml): A 7B parameter model configured as described in the code llama paper.
- [params_code_llama_13b.yaml](./configs/params_code_llama_13b.yaml): A 13B parameter model configured as described in the code llama paper.
- [params_code_llama_34b.yaml](./configs/params_code_llama_34b.yaml): A 34B parameter model configured as described in the code llama paper.
- [params_code_llama_70b.yaml](./configs/params_code_llama_70b.yaml): A 70B parameter model configured as described in the code llama paper.

### LLaMa
- [params_llama_7b.yaml](./configs/params_llama_7b.yaml): A 7B parameter model configured as described in the original paper.
- [params_llama_33b.yaml](./configs/params_llama_33b.yaml): A 32.5B parameter model configured as described in the original paper.

All configs are meant to be run on Weight Streaming mode using Appliance mode and Kubernetes flow.

## Appendix

**Reference**: Radford, A. et al. (2019). [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).

**Reference**: Touvron, Hugo, et al. (2023). [Llama: Open and efficient foundation language models](https://arxiv.org/pdf/2302.13971)

**Reference**: Touvron, Hugo, et al. (2023). [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288)

**Reference**: Rozière, Baptiste, et al. (2023). [Code Llama: Open Foundation Models for Code](https://arxiv.org/pdf/2308.12950)

**Reference**: Meta AI (2024). [Build the future of AI with Meta Llama 3](https://llama.meta.com/llama3)
