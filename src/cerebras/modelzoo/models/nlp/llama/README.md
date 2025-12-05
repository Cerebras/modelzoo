# LLaMa Language Models

The LLaMA family is a series of decoder-only transformer models designed for efficient, high-performance language modeling. Architecturally similar to GPT-2, the original LLaMA model uses RMSNorm instead of LayerNorm, SwiGLU activations, and rotary positional embeddings. LLaMA-2 improves on this with a larger training corpus, doubled context length, and grouped-query attention in its largest model. Code LLaMA specializes in programming tasks through continued pretraining on code-heavy data. LLaMA-3 introduces a more efficient 128K-token tokenizer, expands context, and adopts grouped-query attention across all sizes. These models excel at text generation, summarization, reasoning, coding, and instruction following.

For more information on using our LLaMa implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/llama) in our documentation.
