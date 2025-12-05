# Transformer Language Models

This implementation reproduces the original Transformer model architecture introduced in Attention Is All You Need. It was first applied to English–German translation on the WMT16 dataset and introduced the now-standard building blocks of modern NLP models: multi-head self-attention, layer normalization, feed-forward networks, residual connections, and positional embeddings.

While this implementation shares much of its foundation with the T5 model, it includes important differences in architecture, datasets, model sizes, and training objectives. In particular, this model uses learned absolute positional embeddings rather than relative encodings, and the training task is translation rather than general sequence-to-sequence learning.

For more information on using our Transformer implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/nlp/transformer) in our documentation.
