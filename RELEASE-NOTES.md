# Release Notes

The following are the release notes for the Model Zoo repository.

## Version 2.1.0

### New features and enhancements

#### Large language models

* [MPT](./modelzoo/transformers/pytorch/mpt) and [Mistral](./modelzoo/transformers/pytorch/mistral) are now part of Model Zoo models.
* Release 2.1.0 introduces dynamic loss scaling for [cbfloat16](https://docs.cerebras.net/en/latest/wsc/how_to_guides/cs-1-data-formats.html#ref-cbfloat16) training. When initializing from ``bfloat`` checkpoints from past releases without loss scaling, explicitly specify ``--load_checkpoint_states`` or its ``runconfig`` equivalent to ensure parameter loading from ``params.yaml``. Subsequent checkpoints will inherit dynamic loss scaling and not require this.
* Release 2.1.0 includes support for running non-generative (non-autoregressive) evaluation tasks in [Eleuther AI's Evaluation Harness (EEH)](https://docs.cerebras.net/en/latest/wsc/general/eval_harness.html) on the Cerebras Wafer-Scale cluster. The supported EEH version is v0.3.0. Supported Model Zoo models are GPT2, GPT3, BTLM, BLOOM, LLaMA, Mistral, MPT, StarCoder, and SantaCoder on CS-2.
* Release 2.1.0 introduces [Map and Iterable dataloaders](https://docs.cerebras.net/en/latest/wsc/tutorials/dataloader-checkpointing.html) for large language models in Model Zoo, enhancing training workflow efficiency.
* Cerebras now enables direct HDF5 generation from raw sources, streamlining workflow efficiency and enabling unparalleled control over data format and granularity. Check out our [detailed guide](https://docs.cerebras.net/en/latest/wsc/port/prepare-data/chunk_preprocessing.html) to learn about the process.

#### Sparsity

* In release 2.1.0, we introduce [Sparse Iso-FLOP Transformations for Maximizing Training Efficiency](https://arxiv.org/abs/2303.11525), a technique designed to improve model quality over dense without increasing training FLOPs. To get started with Sparse-IFT, we have provided a comprehensive [Sparsity how-to-guide](https://docs.cerebras.net/en/latest/wsc/how_to_guides/sparsity.html). Additionally, you can explore reference configurations in the Model Zoo to leverage it effectively in your projects. The Model Zoo reference configuration is accessible [SPDF Model Zoo configuration](./modelzoo/transformers/pytorch/gpt3/configs/sparsity/pretraining). For more information, you can read our [blog](https://www.cerebras.net/blog/can-sparsity-make-ai-models-more-accurate) or contact the support team.

## Release 2.0.2

### New features and enhancements

- All Model Zoo models now use the new [Cerebras PyTorch 2.0 API](https://docs.cerebras.net/en/latest/wsc/api/cerebras_pytorch/index.html).
- We improved deterministic restart of custom dataloaders with the new Cerebras PyTorch API. Refer to our [documentation](https://docs.cerebras.net/en/latest/wsc/tutorials/dataloader-checkpointing.html) to see how to save and load the dataloader state along with existing mechanisms for saving model checkpoints during a training run.

#### Sparsity

- With release 2.0.2, we introduce [Sparse Pretraining and Dense Finetuning (SPDF)](https://arxiv.org/abs/2303.10464), a technique designed to accelerate pretraining by incorporating high levels of sparsity while maintaining downstream task accuracy through dense finetuning. To get started with SPDF, we have provided a comprehensive [Sparsity how-to-guide](https://docs.cerebras.net/en/latest/wsc/how_to_guides/sparsity.html). Additionally, you can explore reference configurations in the Model Zoo to leverage SPDF effectively in your projects. The Model Zoo reference configuration is accessible in the Cerebras Model Zoo [here](./transformers/pytorch/gpt3/configs/sparsity). For more information, contact our support team.

#### Large Language Models

- Cerebras released **BTLM**, the best performing and the most downloaded 3B model in the world, in July. G42, a Cerebras strategic partner, released the #1 Arabic language model in the world, **Jais**, in September. Both models used high-performing architectures (maximal update parameterization, SwiGLU activations, ALiBi position encodings). Examples of this style of configuration are available [here](./transformers/pytorch/btlm/configs/).
- Both static and dynamic weight sparsity are supported in release 2.0.2 for [faster training](https://www.cerebras.net/blog/harnessing-the-power-of-sparsity-for-large-gpt-ai-models) and [higher accuracy](https://www.cerebras.net/blog/can-sparsity-make-ai-models-more-accurate). We provide example sparse model configurations in the Cerebras Model Zoo. For more information, contact our support team. Information on using how to use sparsity can be found [here](https://docs.cerebras.net/en/latest/wsc/how_to_guides/sparsity.html) in the Cerebras Developer Documentation.
- GPT style models train with ~30% improved performance in release 2.0.2.
- **LLaMA v2** 7B, 13B, 70B is supported for training from scratch, continuous pretraining, or fine-tuning from a pretrained checkpoint. Reference ModelZoo configs are available [here](./transformers/pytorch/llama/configs/).
- **Falcon 40B** is supported for training from scratch, continuous pretraining, or fine-tuning from a pretrained checkpoint. Reference ModelZoo configs are available [here](./transformers/pytorch/falcon/configs/).
- **StarCoder 15B** is supported for training from scratch, continuous pretraining, or fine-tuning from a pretrained checkpoint. Reference ModelZoo configs are available [here](./transformers/pytorch/starcoder/configs/).
- The default dataloader for GPT-style models is now GptHDF5MapDataProcessor.

#### Computer Vision Models

- Added support for the [Diffusion Transformer](https://arxiv.org/abs/2212.09748). DiT supports AdaLN conditioning and the following model sizes: Small, Base, Large, XL, 2B. Diffusion Transformer also supports multiple patch-sizes like /2, /4, and /8 and image sizes up to 512 x 512.

#### Other features

- We have deprecated old PyTorch BaseModel and BaseRunner classes as part of our update to PyTorch 2.0. Check out our [PyTorch documentation](https://docs.cerebras.net/en/latest/wsc/api/cerebras_pytorch/index.html).
- Enabling gradient accumulation now makes the stack search for a micro-batch size that provides good training throughput performance. This makes compile times longer. Users may avoid this compile time by supplying a micro-batch size with the ``micro_batch_size`` parameter within the ``train_input`` and ``eval_input`` sections of the model configuration YAML. Note that ``batch_size/num_csx`` must be a multiple of ``micro_batch_size``. Micro-batch sizes with good performance are recommended within the gradient accumulation [Micro-batch size setting in YAML params](https:/docs.cerebras.net/en/latest/wsc/general/grad_accumulation.rst#micro-batch-size-setting-in-yaml-params) within the Cerebras Developer Documentation.
- Distributed data parallel model evaluation is now supported on multiple CS-2 systems in a Wafer-Scale Cluster.
- Previous limitations in T5 compile times have been addressed. T5 XXL compile time is now less than 90 minutes with a specified micro-batch size.
- Jobs submitted from the user nodes to the Wafer-Scale cluster now include a token that identifies the user submitting the job. This token can be validated on the Wafer-Scale cluster for user authentication. This change is made to improve security. Machine learning users will not notice any difference in their workflows.
- We improved messages related to job scheduling errors to provide clear guidance for users to take corrective action.
- Loss scaling by number of tokens is supported on single box and multi-box, with and without gradient accumulation. See our [documentation](https://docs.cerebras.net/en/latest/wsc/general/num-tokens-loss-scaling.html) for more information.
- The ``is_pretrained_checkpoint`` flag has been deprecated for clarity. Users should instead use the ``load_checkpoint_states`` in conjunction with ``checkpoint_path`` to specify which components are loaded from the checkpoint. Allowed values are ``model``, ``optimizer``, ``dataloader``, ``grad_scaler``, ``lr_scheduler``. For more information, see the [PyTorch params documentation](https://docs.cerebras.net/en/latest/wsc/port/yaml-params/pytorch_params.html).

#### Known Issues

- Diffusion Transformer (DiT) supports up to 1k by 1k image sizes, but compile time for this input size is extremely long.
- We encourage users to save models and artifacts (with model_dir) on fast storage (SSD backed, local or NFS) to achieve significant improvement in weight initialization, checkpoint loading, and sending weights from host to wafer when using cached compilation.
- Using larger batch sizes provides better training performance but increases compile times. We encourage using batch sizes that have multiple factors as it enables more options for micro-batch for the stack to choose from. This is especially important for distributed runs.
- Dynamic sparsity cannot be used with gradient accumulation (``use_cs_grad_accum`` in ``runconfig`` of YAML) in release 2.0.2.
- Computer vision workloads (UNet and ResNet) will cause out of memory errors if scheduled in parallel with other jobs on the appliance.
- Hugging Face's Transformers library does not support Maximal Update Parameterization (muP) or models with SwiGLU and ALiBi. If you have a Cerebras GPT2/3 checkpoint that uses muP, it is possible to :doc:`convert it to the GPT2 Hugging Face model](https://docs.cerebras.net/en/latest/wsc/how_to_guides/mup_docs>` to perform inference. Custom models can still be used with Hugging Face via the Hugging Face Hub.  
- Gradient accumulation for computer vision models is supported by the software stack but has not been fully tested across all model variants. We plan to perform comprehensive qualification testing for CV models with gradient accumulation as part of the upcoming 2.1 release. This will ensure that larger batch sizes can be confidently utilized for your computer vision tasks.
- The number of heads ``num_heads`` within a transformer block should not be a prime number.

*Note: Version 2.0.0 and 2.0.1 were special, small-distribution releases. 2.0.2 is our general release.*

## Version 1.9.1

### New features and enhancements

#### Large Language Models

- Maximal Update Parameterization (muP), used for improving training stability and transferring hyperparameters from smaller language models to Larger Language Models (including CerebrasGPT), is now available for GPT-2 and GPT-3 style models. See the [How-to guide](https://docs.cerebras.net/en/latest/wsc/how_to_guides/mup_docs.html) for usage.
- New checkpoint converters between Hugging Face and Cerebras formats have been added. See more at [Convert checkpoints and configurations](https://docs.cerebras.net/en/latest/wsc/port/porting-checkpoints.html).
- Gradient accumulation is enabled for all transformer language models in 1.9.1 through YAML config.
- Pre-trained [Falcon 7B](./modelzoo/transformers/pytorch/falcon/) is supported in Model Zoo.
- Pre-trained [LLaMA 7B, 13B, and 33B](./modelzoo/transformers/pytorch/llama/) are supported in Model Zoo.
- [BLOOM 7B](./modelzoo/transformers/pytorch/bloom/) is available in Model Zoo.
- ALiBi positional encodings can be enabled in all GPT-style models through the model section in the configuration yaml as shown below:

```yaml
position_embedding_type: 'alibi'

alibi_trainable_slopes: False # whether the slopes of the alibi embedding is trainable (default to False).

alibi_implementation: 'expand' # We support `embedding` and `expand` with default set to `expand`.
```

#### Computer vision models

- Fixed bugs and improved performance for computer vision models.

#### Other features

- Pipeline mode and TensorFlow support is deprecated. All models must use PyTorch and weight streaming functionality. There is no longer a need to specify a `{pipelined,weight_streaming}` argument in `run.py` because all models will run in `weight_streaming` mode by default. All models previously supported in Pipeline are now supported for Weight Streaming.
- The `batch_size` parameter in Model Zoo yaml configuration files now represents the total effective batch size of the model and is divided evenly across the specified `num_csx` CSX systems. This differs from pre-1.9.0 behavior, where the `batch_size` parameter defined the batch size per CSX, not globally. Note that `batch_size` must now be divisible by `num_csx`.

#### Known Issues

- Some dataloader implementations from Model Zoo require evaluation to be done on a single CS-2 rather than multiple CS-2s. Multi-box evaluation has no explicit limitation, but these dataloaders require the dataset to be sharded in such a way that each worker gets at least one file. Evaluation datasets are often small and not split into many files.
- All T5 limitations from Release 1.8 remain.
- Loss scaling by number of tokens is not yet fully supported and requires coordination with the Cerebras team.
- GPT NeoX suffers NaNs when trained with extremely long sequence lengths (30k, 50k).
- The base, pre-trained Falcon and LLaMA variants are supported. Other variants, such as those with long sequence lengths or different numbers of heads, may not be supported.

*Note: Version 1.9.0 was a special, small-distribution release. 1.9.1 is our general release.*

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
