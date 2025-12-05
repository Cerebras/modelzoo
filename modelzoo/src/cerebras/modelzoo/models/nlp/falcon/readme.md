# Falcon

The Falcon series consists of causal decoder-only transformer models with 7B, 40B, and 180B parameters, developed by the Technology Innovation Institute (TII). The models follow an optimized GPT-style architecture with key changes for efficient scaling and throughput:

* Parallel attention and MLP layers within transformer blocks.
* Rotary positional embeddings (RoPE) and multigroup attention (a generalization of multiquery attention) for faster inference and better tensor parallelism.
* GELU activations, no dropout, and z-loss regularization for stable training.
* Context length of 2,048 tokens and a 65K vocabulary.

For more information on using our Falcon implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/falcon) in our documentation.
