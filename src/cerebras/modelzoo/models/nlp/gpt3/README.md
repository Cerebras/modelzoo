# GPT-3 Language Models

GPT-3 is a decoder-only transformer language model architecture designed for large-scale autoregressive pretraining. It extends GPT-2 with significantly more parameters (ranging from 1.3B to 175B) and introduces architectural refinements such as sparse attention layers, used in alternating blocks to reduce compute costs during training. However, this implementation uses the GPT-2-style dense attention in all layers.

Training occurs on next-token prediction using large text corpora like The PILE, with inputs represented as token sequences padded and masked to a fixed maximum sequence length.

For more information on using our GPT-3 implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/gpt3) in our documentation.
