# Release Notes

The following are the release notes for the Model Zoo repository.

## Version 1.7.1

### New features and enhancements

#### Unified workflow on the Wafer-Scale Cluster

- User workflow on the Wafer-Scale Cluster is now the same for Pipelined and Weight Streaming execution. The same launching scripts and the same environment can now be used to run larger models with Weight Streaming execution and smaller models with Pipelined execution. Use additional command line argument with `run.py` for PyTorch and with `run_appliance.py` for Tensorflow: set ``--execution_strategy`` argument to ``pipeline`` or ``weight_streaming`` to specify execution mode.
  - Note that Pipelined execution only supports using one CS-2 for each run. It is only valid to specify ``--num_csx=1`` in the run command for Pipelined execution. Weight Streaming does not have such requirement.
  
### Known issues

#### Specifying number of workers

- ``--num_workers_per_csx`` denotes the maximum number of Kubernetes worker pods per CS-X. Note that these are distributed across physical worker nodes attached to each CS-X on the Wafer-Scale Cluster.
- For this release, please specify ``--num_workers_per_csx=8`` as a command line argument in the run command for Pipelined execution. Weight Streaming execution does not need to specify this argument.

#### Specifying path in the configuration file

- Please specify absolute paths for any configuration parameters that are used by dataloaders in PyTorch or the input_fn in TensorFlow. More specifically, this applies to all variables in the ``train_input`` and/or ``eval_input`` sections of the configuration yamls.

#### Padding token in loss calculation

- TensorFlow BERT fine-tuning token classification model does not support padding tokens in loss on the Wafer-Scale Cluster for both Pipelined and Weight Streaming execution. Please set ``include_padding_in_loss: False`` in the configuration yaml. We believe it makes the most sense to exclude padding tokens in the loss calculation. Such setting differs from the original public implementation where token padding is included in the loss, which is most likely used for performance optimization on GPUs, leading to our eval accuracy being potentially different from published numbers. This does not apply to the PyTorch version or the Original Cerebras Installation.

#### TensorFlow BERT SQuAD Unsupported

- TensorFlow BERT fine-tuning model for SQuAD is not supported in appliance (neither Pipelined nor Weight Streaming). If you would like to fine-tune BERT with SQuAD, please use the PyTorch version.

## Version 1.7.0

### New features and enhancements

#### PyTorch

- Increased support for GPT2 style models:
  - First Weight Streaming model support for PyTorch GPT2 small model as an early access.
  - GPT2 XL style model.
  
- Increased support for GPT3 style models:
  - GPT3 XL style model.
  - GPT3 style model with 6.7B parameters.
  - GPT3 style model with 13B parameters.
  - GPT3 style model with 20B parameters.
  
- Added support for GPTj and NeoX style models:
  - GPT NeoX small style model.
  - GPT NeoX style model with 1.3B parameters.
  - GPT NeoX style model with 2.7B parameters.
  - GPT NeoX style model with 20B parameters.
  - GPTj style model with 6B parameters.
  
- Added support for UNet model
  - First Weight Streaming model support.

#### TensorFlow

- Increased support for GPT2 style models:
  - GPT2 XL style model.

- Increased support for GPT3 style models:
  - GPT3 XL style model.
  - GPT3 style model with 2.7B parameters.
  - GPT3 style model with 6.7B parameters.
  - GPT3 style model with 13B parameters.
  - GPT3 style model with 20B parameters.
  
- Added support for GPTj and NeoX style models:
  - GPT NeoX model with 20B parameters.

#### Other features

- Added support for bfloat16 data type and is enabled by default
- Training with static sparsity masks is now available for PyTorch models
- New utility to convert a dense PyTorch checkpoint into sparse version
- Added support for gradient accumulation on CS2 system in all Weight Streaming NLP models.
- Expanded support for PyTorch learning rate schedules and loss functions

#### Known Issues

##### Running eval with UNet in PyTorch, Weight Streaming execution

- UNet model eval with metrics mIOU, DSC on images of size 4096 x 4096 pixels causes failures and is a known issue. All default configurations and image sizes published in the ModelZoo have been tested and expected to work without issues. The issue may manifest itself with other non-tested image sizes. Please contact Cerebras support if you run into failures.

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
