# Release Notes

The following are the release notes for the Model Zoo repository.

## Version 1.8.0

### New features and enhancements

#### Large language models

* Added support for **T5 models up to 11B parameters** with Weight Streaming execution mode. T5 is supported **with source and target inputs up to 2K tokens**.
* Added support for **BERT pre-training** with Weight Streaming execution mode. BERT is supported **with input sequences up to 30K tokens**.
* Added support for gradient accumulation for GPT-style and BERT-style language models, allowing for larger effective batch sizes. See [our documentation](https://docs.cerebras.net/en/latest/wsc/general/grad_accumulation.html) for more details.
* Added support for deterministic checkpointing of dataloaders for language models to enable pausing and restarting of training runs without using duplicate samples or batches. See [our documentation](https://docs.cerebras.net/en/latest/wsc/general/deterministic_checkpoints.html) for more details.
* Loss scaling by ``num_tokens`` is enabled, allowing users to divide the loss value by the actual number of tokens in a batch, since input lengths are not constant. See [our documentation](https://docs.cerebras.net/en/latest/wsc/general/num-tokens-loss-scaling.html) for more details.
* In past releases pre-layer normalization in our T5 & Transformer models required setting ``use_pre_encoder_decoder_layer_norm: False``. This was confusing, and we have changed the behavior in 1.8. To enable pre-layer normalization you should instead set ``use_pre_encoder_decoder_layer_norm: True``. This update better aligns the naming of the parameter to its usage. To use release 1.7 checkpoints in release 1.8, you'll need to update the config to reflect this change. Directions for converting configuration files can be found in [our documentation](https://docs.cerebras.net/en/latest/wsc/port/porting-checkpoints.html).
* You may now control the activation function used by the BERT pooler (``pooler_nonlinearity``) and masked language model head (``mlm_nonlinearity``) independently of the activation used for the rest of the model (``encoder_nonlinearity``). Both will default to ``encoder_nonlinearity`` if not explicitly set. Use [our documentation](https://docs.cerebras.net/en/latest/wsc/port/porting-checkpoints.html#upgrading-checkpoints-configs-to-the-current-release) to convert 1.7 configuration files to 1.8 to have access to this feature.  

#### Computer vision models

* Added support for 3D UNet model in Weight Streaming execution to enable segmentation of large volumes of up to 512 x 512 x 160 pixels. Single CS-2 system support only. Check our reference implementation and learn more in the [Model Zoo](https://github.com/Cerebras/modelzoo/tree/main/modelzoo/vision/pytorch/unet#u-net-3d-model).
* Multi-channel multi-class segmentation is now supported for 2D UNet.
* 2D UNet now supports image sizes up to 7k x 7k
* 2D UNet now supports multi-channel inputs
* Added support for 2D ResNet and variants (see ModelZoo for more details)
* 2D ResNet supports input sizes of up to 1k x 1k

#### Other features

* We are releasing scripts and instrucions for checkpoint and configuration conversion to and from corresponding Hugging Face models and between Cerebras software versions. More details in [our documentation](https://docs.cerebras.net/en/latest/wsc/port/porting-checkpoints.html).
* Added support for ``eval_all`` and ``train_and_eval`` to enable users to evaluate models throughout long training runs or evaluate all checkpoints after training has completed. Mode details in [our documentation](https://docs.cerebras.net/en/latest/wsc/port/run-model/eval.html).
* The Cerebras Model Zoo run scripts have been updated with a more informative and explicit command line interface. For more details please read [our documentation](https://docs.cerebras.net/en/latest/wsc/getting-started/cs-appliance.html).
* The script ``run_appliance.py`` has now been deprecated for TensorFlow in favour of one single script called ``run.py`` that employs the aforementioned run script changes.
* More custom losses and non-linearities are now supported with AutoGen feature. It allows users to swap operations (losses, nonlinearities, positional encodings) for language models and improves performance of loss functions with fused kernels. Learn more about autogen in [our documentation](https://docs.cerebras.net/en/latest/wsc/general/autogen.html).
* You can now use scalar and tensor summaries in PyTorch to track various tensors of interest during training. Internally we heavily rely on these features for example to track parameters like gradient norms during training of large language models. Learn mode in [our documentation](https://docs.cerebras.net/en/latest/wsc/general/summaries.html).

#### Known Issues

* T5 with input or output sequences longer than 1024 tokens (``src_max_sequence_length`` and ``tgt_max_sequence_length`` parameters in model yaml config file) may have compile times of over 3 hours. T5 is only supported with input and output sequences up to 2048 tokens.
* T5 has limitations with respect to gradient accumulation and batch sizes (BS).
    * Gradient accumulation is not supported for T5.
    * At [precision optimization level](https://docs.cerebras.net/en/latest/wsc/general/cs-1-data-formats.html#precision-optimization-level) 0 (POL0), the largest supported batch size for T5 model with 11B parameters is 220.
    * At precision optimization levels 1 and 2 (POL1 and POL2) batch sizes over 770 for T5 3B and over 260 for T5 11B will result in a long compile time.
    * Models will not compile if ``(vocabulary V / (heads * Greatest_Common_Divisor(Sin, Sout)) > 2^11``.
* Maximum supported vocabulary size for language models is 1 million.
* Downloading data sets from the internet within data loaders is not supported. As a workaround, please download data sets and prepare them outside the dataloader function. See [PyTorch FC MNIST implementation](https://github.com/Cerebras/modelzoo/blob/main/modelzoo/fc_mnist/pytorch/prepare_data.py) for an example and additional details.
* Users creating their own models and training scripts must separate the dataloader or input_fn into a separate Python file from the rest of the training script, in order to avoid the error described [in the documentation](https://docs.cerebras.net/en/latest/wsc/troubleshooting/error_receiving_activations.html).
* The [experimental PyTorch API](https://docs.cerebras.net/en/latest/wsc/port/porting-pytorch-to-cs/cstorch-api.html) does not save checkpoints at steps 0 and 1 in the correct format. No issues with checkpoints at other steps or outside the experimental API.


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
