# Multimodal Simple Model

This directory contains our multimodal library, which can be used to instantiate many of the current state-of-the-art models such as LLaVA, CogVLM, and MM1 among others. Our implementation supports multiple images interleaved with text as input, and can generate text as output. The building blocks for this implementation are as follows:
- **Vision Encoder**: Process images through one or more image encoders to produce embeddings.
- **Image Embedding Projector**: Projects the embeddings from vision encoder into a shared latent space with LLM using MLPs. 
- **Language Model**: Accepts the vision and language embeddings as input and produces text as output.

For more information on using Multimodal Simple, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/multimodal/multimodal-simple) in our documentation.
