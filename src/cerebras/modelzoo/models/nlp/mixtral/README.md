# Mixtral Language Models

Mixtral is a family of decoder-only transformer models that use sparse Mixture of Experts (MoE) to scale model capacity without increasing inference cost. Instead of activating all model parameters for every input, Mixtral routes each token to a subset of experts: specialized feedforward networks. This allows Mixtral models to retain training efficiency while significantly expanding total parameter count.

The architecture builds on the Mistral base model, inheriting sliding window attention (SWA), grouped-query attention (GQA), SwiGLU activations, and a 32K maximum sequence length. Each expert block consists of multiple experts, and only a configurable top_k subset is selected per token during forward pass.

Mixtral models are effective for tasks requiring high capacity — such as long-context reasoning, coding, and instruction following — while remaining efficient at inference time.

For more information on using our Mixtral implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/mixtral) in our documentation.
