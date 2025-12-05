# BLOOM language models

BLOOM is a decoder-only Transformer-based language model developed by the BigScience project. It supports multilingual training across 46 natural languages and 13 programming languages, with models ranging in size up to 176B parameters.

Architecturally, BLOOM resembles GPT-2 but introduces two important differences:

* Tokenizer: BLOOM uses a tokenizer and vocabulary specifically designed for multilingual generalization, consisting of ~250K tokens.
* Position Embeddings: Instead of learnable absolute position embeddings (as in GPT-2), BLOOM uses ALiBi — Attention with Linear Biases — which allows extrapolation to longer sequence lengths and introduces a recency bias in attention computation.

For more information on using our BLOOM implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/bloom) in our documentation.
