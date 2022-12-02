# Release Notes

The following are the release notes for the Model Zoo repository.

## Version 1.6.1

- First Weight Streaming model support for PyTorch GPT2 XL model as an early access.
- Improvements on Pipeline Legacy flow using Kubernetes and appliance mode.

## Version 1.6.0

### New features and enhancements

#### TensorFlow

- Increased support for GPT3 style models:
  - GPT3 style model with 6.7B parameters.
  - Early limited access for GPT3 style 13B parameter model.
  - Early limited access for GPT3 style 20B parameter model, inspired from GPT-NeoX architecture.

- Support for Appliance Mode run scripts to run models on Cerebras Wafer-Scale Cluster in Weight Streaming.

#### PyTorch

- PyTorch Layer API support for following layers:
  - AttentionLayer
  - EmbeddingLayer
  - FeedForwardNetwork
  - RelativePositionEmbeddingLayer
  - TransformerDecoderLayer
  - TransformerDecoder
  - TransformerEncoderLayer
  - TransformerEncoder

- Transformer style demo model using the Layer API.
- Migrated GPT2 model implementation to use PyTorch Layer API from the HuggingFace based implementation.
  - HuggingFace based implementation for these models is deprecated.

- Support for PyTorch Optimizers:
  - Adafactor, Adam (including AdamW), Lamb, RMSprop, SGD.
  - Experimental: RAdam, Rprop, ASGD, NAdam, Adadelta, Adagrad, Adamax.

#### Usability

- Support for Pipeline models in Kubernetes(k8s) workflow for running on Cerebras Wafer-Scale Clusters.
