# Multimodal LLaVA

This directory contains an implementation for the multimodal LLaVA model. Vision Language models in general accept image and text as inputs, and generate text outputs. The LLaVA model consists of a vision encoder backbone, a language model and a projector module that acts as a bridge between vision model and language model.

In the case of LLaVA, the vision encoder is initialized from pretrained OpenAI CLIP-ViT-L-336/14 weights, and the language model is initialized using Vicuna weights. The projector module which consists of MLP, is initialized using random weights.

LLaVA model is in general trained in two phases as below:
- **Phase 1: Pre-training for Feature alignment**: In this stage, only the projector weights are updated to align the image features with that of LLM word embedding.

- **Phase 2: Instruction Finetuning end-to-end**: In this stage, the model is trained on instruction finetuning data and enables the model with chatbot capabilities. In this stage, usually the LLM and projector weights are trained and the vision encoder remains frozen.

For more information on using our LLaVA implementation, visit its [model page](https://training-docs.cerebras.ai/rel-2.5.0/model-zoo/models/multimodal/llava) in our documentation.
