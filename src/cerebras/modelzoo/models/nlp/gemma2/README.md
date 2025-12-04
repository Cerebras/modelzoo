# Gemma 2

Gemma 2 is a family of decoder-only transformer models developed by Google DeepMind, ranging from 2B to 27B parameters. Architecturally, Gemma 2 builds upon the Transformer backbone with several enhancements: it interleaves local sliding window and global attention layers, adopts grouped-query attention (GQA), and uses GeGLU activations with RMSNorm. The models support a context length of 8K and utilize a 256K-token multilingual tokenizer inherited from Gemini.

Gemma 2 models are well-suited for tasks involving instruction following, long-context understanding, multilingual reasoning, and coding.

For more information on using our Gemma 2 implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/gemma) in our documentation.
