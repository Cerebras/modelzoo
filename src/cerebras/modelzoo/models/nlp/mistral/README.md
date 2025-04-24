# Mistral Language Models

Mistral is a family of decoder-only transformer models optimized for efficiency and throughput while preserving strong general performance. Architecturally, Mistral builds on the transformer decoder backbone with several key enhancements: it adopts grouped-query attention (GQA) for faster inference, replaces absolute positional encodings with sliding window attention for improved scalability, and utilizes SwiGLU activation functions. These models are well-suited for instruction following, reasoning, summarization, and coding tasks.

For more information on using our Mistral implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/mistral) in our documentation.
