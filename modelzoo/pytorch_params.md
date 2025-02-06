# YAML params for PyTorch models

## Model params

### Common

| Parameter Name | Description |
| --- | --- |
| mixed_precision | Whether to use mixed precision training or not. (`bool`, optional) Default: `None` |
| fp16_type | The 16-bit floating type to use. Accepted values:<br>`"float16"`<br>`"bfloat16"`<br>`"cfloat16"`. See [more](https://docs.cerebras.net/en/latest/general/cs-1-data-formats.html?#bfloat16-floating-type) (`bool`, optional) Default: `False` |

### Transformer based models

| Parameter Name | Description | Supported Models |
| --- | --- | --- |
| attention_dropout_rate | Dropout rate for attention layer. (`float`, optional) Default: same as `dropout` | All |
| attention_softmax_fp32 | Whether to use fp32 precision for attention softmax. (`bool`, optional)  Default: `True`) | All |
| attention_type | Type of attention. Accepted values:<br>`"dot_product"`<br>`"scaled_dot_product"`.<br>(`str`,  optional) Default: `"scaled_dot_product"` | All |
| d_ff | Size of the intermediate feed forward layer in each `T5Block`. (`int`,  optional) Default: `2048` | T5, Transformer |
| d_kv | Size of the query/key/value projections per attention head. `d_kv` does *not* have to be equal to `d_model//num_heads`. (`int`,  optional) Default: `64` | T5, Transformer |
| d_model | The number of expected features in the encoder/decoder inputs. (`int`,  optional) Default `512` | All |
| decoder_nonlinearity | Type of nonlinearity to be used in decoder. (`str`,  optional) Default: `"relu"` | T5, Transformer |
| decoder_num_hidden_layers | Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set. (`int`, optional) | T5, Transformer |
| disable_nsp | Whether to disable the next sentence prediction task. (`bool`, optional) Default: False | BERT (pre-training, fine-tuning) |
| dropout_rate | The dropout probability for all fully connected layers. (`float`, optional), Default: `0.1` | All |
| embedding_dropout_rate | Dropout rate for embeddings. (`float`, optional) Default: `0.1` | All |
| embedding_initializer | Initializer to use for embeddings. See [supported initializers](./layers/create_initializer.py). (`str`, optional) Default: "normal" | GPT2, GPT3, GPTJ |
| encoder_nonlinearity | Type of nonlinearity to be used in encoder. (`str`, optional) Default: varies per model | BERT (pre-training, fine-tuning), T5, Transformer |
| encoder_num_hidden_layers  | Number of hidden layers in the encoder. (`int`, optional) Default: `6` | T5, Transformer |
| extra_ids | The number of extra ids used for additional vocabulary items  (`int`, optional) Default: `0` | T5, Transformer
| filter_size |  Dimensionality of the feed-forward layer in the Transformer block. (`int`, optional) Default: `3072` |  BERT (pre-training, fine-tuning), GPT2, GPT3, GPTJ |
| hidden_size | The size of the transformer hidden layers (`int`, optional) Default: `768` |  BERT (pre-training, fine-tuning), GPT2, GPT3, GPTJ |
| initializer | The initializer to be used for all the initializers used in the model. See [supported initializers](./layers/create_initializer.py). (`str`, optional) Default: varies based on model | BERT (pre-training, fine-tuning), GPT2, GPT3, GPTJ |
| initializer_range | The standard deviation of the truncated_normal_initializer as the default initializer. (`float`, optional) Default: `0.02` | BERT (pre-training), GPT2, GPT3, GPTJ |
| layer_norm_epsilon | The epsilon value used in layer normalization layers. (`float`, optional) Default: `1e-5`)| All |
| lm_loss_weight | Value that scales loss by the mean number of predictions per sequence in the dataset. This number varies per dataset and can be calculated by getting the reciprocal of average number of tokens per sequence in the training dataset. This is only needed when setting loss scaling to `"batch_size"`.  (`float`, optional) Default: `1.0` | T5, Transformer |
| loss_scaling | The scaling type used to calculate the loss. Accepts: <br> `batch_size`, `num_tokens`. See [more](https://docs.cerebras.net/en/latest/wsc/general/num-tokens-loss-scaling.html). **Note:** It is recommended to set this to `batch_size` when gradient accumulation is enabled for training stability. (`str`, optional) Default: `num_tokens` | GPT2, GPT3, GPTJ |
| loss_weight | The weight for the loss scaling when `loss_scaling: "batch_size"`, generally set to `1/max_sequence_length`. (`float`, optional) Default: `1.0` | GPT2, GPT3, GPTJ |
| max_position_embeddings | The maximum sequence length that the model can handle. (`int`, optional) Default: `1024` | All |
| mlm_loss_scaling | A string specifying the scaling factor type used for the language modeling loss. Accepts one of: `"num_masked"` - uses the off-the shelf loss scaling by number of valid (non-padding) tokens the cross entropy loss function, `"precomputed_num_masked"` - uses loss scaling from the computed num valid masks in the data loader, when enabling `dynamic_loss_weight` in the data loader params, `"batch_size"` - uses loss scaling by `"batch_size"` and `lm_loss_weight` should be provided when using `"batch_size"`. (`str`, optional) Default: `"batch_size"` | T5, Transformer |
| mlm_loss_weight | The weight for the masked language modeling loss used when scaling the loss with `"batch_size"`. This number varies per dataset and can be calculated by getting the reciprocal of average number of masked tokens per sequence in the training dataset. (`float`, optional) Default: `1.0` | BERT (pre-training) |
| nonlinearity | The non-linear activation function used in the feed forward network in each transformer block. See list of non-linearity functions [here](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity). (`str`, optional) Default: varies per model | BERT (pre-training, fine-tuning), GPT2, GPT3, GPTJ |
| num_heads | The number of attention heads in the multi-head attention layer. (`int`, optional) Default: varies per model| All |
| num_hidden_layers | Number of hidden layers in the Transformer encoder/decoder. (`int`, optional) Default: `12` | All |
| output_layer_initializer | The name of the initializer for the weights of the output layer. See [supported initializers](./common/pytorch/model_utils/create_initializer.py). (str, optional) Default: varies based on model | GPT2, GPT3, GPTJ |
| position_embedding_type | The type of position embedding to use in the model. Can be one of: `"fixed"` - Sinusoidal from original [Transformer](https://arxiv.org/abs/1706.03762), `"relative"` - Relative position embedding, [to exploit pairwise, relative positional information](https://arxiv.org/abs/1803.02155)., `"rotary"` - a.k.a [RoPE](https://arxiv.org/pdf/2104.09864v4.pdf) , `"learned"` - Learned embedding matrix, `None` (`str`, optional) Default: varies per model | All |
| relu_dropout_rate | The dropout rate for ReLU activation function. (`float`, optional) Default: varies per model | T5, Transformer |
| residual_dropout_rate | The dropout rate for residual connections. (`float`, optional) Default: `0.1` | GPTJ |
| rotary_dim | The number of dimensions used for the rotary position encoding. Must be an even number. (`int`, optional) Default: `None` | GPTJ |
| share_embedding_weights | Whether to share the embedding weights between the input and out put embedding. (`bool`, optional) Default: `True` | All |
| share_encoder_decoder_embedding | Whether to share the embedding weights between the encoder and decoder (`bool`, optional) Default: `True`| T5, Transformer |
| src_vocab_size | The size of the source vocabulary. Max supported value: `512000`. (`int`, optional) Default: `32128` | T5, Transformer |
| tgt_vocab_size | The size of the target vocabulary. Max supported value: `512000`. (`int`, optional) Default: `32128` | T5, Transformer |
| use_bias_in_output | Whether to use bias in the final output layer. (`bool`, optional) Default: `False` | GPT2, GPT3, GPTJ |
| use_dropout_outside_residual_path | Whether to set dropout calculations outside of the residual path. (`bool`, optional) Default: `True` for T5, `False` for Transformer | T5, Transformer |
| use_ffn_bias | Whether to use bias in the feedforward network (FFN). (`bool`, optional) Default: varies per model | All |
| use_ffn_bias_in_attention | Whether to include bias in the attention layer for feed-forward network (FFN). (`bool`, optional) Default: varies per model | All |
| use_pre_encoder_decoder_dropout | Whether to use dropout layer after positional embedding layer and encoder/decoder. (`bool`, optional) Default: `False` | T5, Transformer |
| use_pre_encoder_decoder_layer_norm | Whether to use layer norm before passing input tensors into encoder/decoder. (`bool`, optional) Default: `True` | T5, Transformer |
| use_projection_bias_in_attention | Whether to include bias in the attention layer for projection.  (`bool`, optional) Default: varies per model | All |
| norm_type | Whether to use T5 layer norm (a.k.a `rmsnorm`, with no mean subtraction and bias correction) or use the regular `nn.LayerNorm` module. (`str`, optional) Default: `layernorm` | T5, Transformer |
| use_transformer_initialization | The Transformer model tends to converge best with a scaled variant on Xavier uniform initialization used for linear layers. This contrasts the initialization used for the original T5 paper, which uses He normal initialization for linear layers. Setting this flag to `True` switches the initialization to the Transformer specific scaled Xavier initialization. (`bool`, optional) Default: `False` | T5, Transformer |
| use_untied_layer_norm | Whether to use untied layer normalization. (`bool`, optional) Default: `False` | GPTJ |
| vocab_size | The size of the vocabulary used in the model. Max supported value: `512000`. (`int`, optional) Default: varies per model | All |

### Computer Vision models

| Parameter | Description | Supported Models |
| --- | --- | --- |
| bias_initializer | Initializer for the bias. (`str`, optional) Default: `"zeros"` | UNet |
| convs_per_block | List of conv specifications for each conv in the block. (`List[str]`, required) | UNet |
| decoder_filters | List of filter sizes for each block in the decoder. (`List[str]`, required) | UNet |
| downscale_bottleneck | Whether to downsample the spatial dimensions in the UNet bottleneck block. (`bool`, optional) Default: `False`| UNet |
| downscale_encoder_blocks | Determine whether each block in the Encoder includes downsampling. Length of the list must correspond to the number of UNetBlocks in the Encoder. If a single bool is provided, all blocks will use this value. (`bool`/`List[bool]`, optional) Default: `True` | UNet |
| downscale_first_conv | If True, the first convolution operation in each UNetBlock will be downscaled. If False, the last convolution in each UNetBlock will be downscaled. (`bool`, optional) Default: `False` | UNet |
| downscale_method | Downscaling method at the end of each block. One of  `"max_pool"` or `"strided_conv"`. (`str`, optional) Default: `"max_pool"` | UNet |
| enable_bias | Whether to include a bias operation following convolution layers. By default, bias will only be included when no normalization is used after the convolution layers. | UNet |
| encoder_filters | List of filter sizes for each block in the encoder. (`List[str]`, required) | UNet |
| eval_ignore_classes | List of classes to ignore during evaluation of model. (`List[int]`, optional) | UNet |
| eval_metrics | List of evaluation metrics to use during training and validation. Available options are accuracy (`Acc`), mean IOU (`mIOU`) or Dice (`DSC`).  (`List[str]`, optional). | UNet |
| initializer | Initializer for the convolution weights. See [supported initializers](./layers/create_initializer.py) (`str`, required) | UNet |
| input_channels | Number of channels in the input images to the model. (`int`, required) | UNet |
| loss |  Loss type, supported: values: `"bce"`, `"multilabel_bce"`, `"ssce"` (`str`, required) | UNet |
| nonlinearity | Activation function used in the model following convolutions in the encoder and decoder. (`str`, required) | UNet |
| norm_kwargs | args to be passed to norm layers during initialization. For <br>`norm_type` = `group`, `norm_kwargs` must include `num_groups` key value pair. <br>`norm_type` = `layer`, `norm_kwargs` must include `normalized_shape` key value pair. <br>(`dict`, optional) Default: `None` | UNet |
| norm_layer | Type of normalization to be used. See [supported norm layers]](./layers/norms.py). (`str`, optional) Default: `"batchnorm2d"` | UNet |
| residual_blocks | Flag for using residual connections at the end of each block. (`bool`, optional) Default: `False` | UNet |
| skip_connect | Flag for if the model concatenates encoder outputs to decoder inputs. (`bool`, optional) Default: `True` | UNet |
| use_conv3d | Whether to use 3D convolutions in the model. (`bool`, optional) Default: `False` | UNet |
| frequency_embedding_size | Size of Sinusoidal Timestep embeddings. (`int`, required) Default: `256` | DiT |
| label_dropout_rate | probability of dropout applied to label tensor. (`float`, required) Default: `0.1` | DiT |
| patch_size | Size of patch used to convert image to tokens. (`[int, int]`, required)| DiT |
| use_conv_patchified_embedding |  If True, use conv2D to convert image to patches (`bool`, option) Default: `True` | DiT |
| block_type | DiT Block variant. Accepted values: `adaln_zero`. (`str`, optional) Default: `adaln_zero` | DiT |
| | | |
| vae |Params related to Pretrained Variational Auto Encoder(VAE). | DiT |
| vae.down_block_types | List of downsample block types used in Encoder of VAE. (`List[DownEncoderBlock2D]`, optional) Default: `[DownEncoderBlock2D]`  | DiT |
| vae.up_block_types | List of upsample block types used in Decoder of VAE. (`List[UpDecoderBlock2D]`, optional) Default: `[UpDecoderBlock2D]` | DiT |
| vae.block_out_channels | Number of output channels after each of downsample(upsample) blocks in Encoder(Decoder). (`List[int]`, optional) Default: `[64, ]` | DiT |
| vae.layers_per_block | Number of ResNet2D blocks(`norm->conv2D->norm->conv2D->activation`) per downsample(upsample) blocks. (`int`, optional) Default: `1` | DiT |
| vae.act_fn | Activation function to use in VAEModel. (`str`, optional) Default: `silu` | DiT |
| vae.latent_size | Latent Tensor(output of VAEEncoder) [height, width]. (`List[int]`, required) Default: `[32, 32]` | DiT |
| vae.latent_channels | Number of channels in Latent Tensor. (`int`, optional) Default: `4` | DiT |
| vae.norm_num_groups | Number of groups in GroupNorm of ResNet2D block. (`int`, optional) Default: `32` | DiT |
| vae.scaling_factor | The component-wise standard deviation of the trained latent space computed using the first batch of the training set. This is used to scale the latent space to have unit variance when training the diffusion model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1/ scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper. (`float`, optional) Default: `0.18215` | DiT |
| vae.sample_size | size of tiles when VAE is used with tiling where input tensor is split into tiles for forward pass(`int`, optional) Default: `256` | DiT |
| | | |
| reverse_process | Params related to Reverse Diffusion Process | DiT |
| | | |
| reverse_process.sampler | Params related to the sampler being used (required) | DiT |
| reverse_process.sampler.name | Name of sampler to use (`str`, required). Accepted values: `ddpm`, `ddim`| DiT |
| reverse_process.sampler.beta_start | The starting `beta` value of inference. (`float`, optional) Default: `0.0001` | DiT |
| reverse_process.sampler.beta_end | The final `beta` value. (`float`, optional) Default: `0.02` | DiT |
| reverse_process.sampler.num_inference_steps | Number of intermediate diffusion timesteps. (`List[int] (or) int`, optional) Default: `250` | DiT |
| reverse_process.sampler.custom_timesteps | List of timesteps to be used during sampling. Should be in decreasing order.Can either pass `custom_timesteps` (or) `num_inference_steps`, but not both. (`List[int]`, optional) Default: `None` | DiT |
| | | |
| reverse_process.pipeline | Diffusion Pipeline params that ties samplers and generation of multiple samples. | DiT |
| reverse_process.pipeline.guidance_scale | Controls classifier free guidance scale. guidance_scale = `1.0` disables guidance (`float`, required) | DiT |
| reverse_process.pipeline.num_cfg_channels | Number of latent channels to use for classifier free guidance. (`int`, optional) Default: `3` | DiT |
| reverse_process.pipeline.custom_labels | Generate samples from this label subset if provided.  (`List[int]`, optional) Default: `None` | DiT |


## Data loader params

### Common

| Parameter Name | Description |
| --- | --- |
| batch_size | Effective batch size of the input data. (`int`, required) |
| data_dir | Path/s to the data files to use. (`str`/`List[str]`, required) |
| data_processor | Name of the data processor to be used. (`str`, required)  |
| mixed_precision | Flag to cast input to fp16. (`bool`, optional) Default: `None` |
| micro_batch_size | Micro batch size settings for gradient accumulation. Only applies to 'CSX' runs. Please set `num_csx` and `batch_size` such that `batch_size//num_csx` is a multiple of `micro_batch_size`.  (`Union[None, int, Literal["auto", "explore"]]`, optional) Default: `auto` |
| num_workers | Number of workers to use in the dataloader. See [more](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). (`int`, optional) Default: `0` |
| persistent_workers | For multi-worker dataloader controls if the workers are recreated at the end of each epoch (see [PyTorch docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)). (`bool`, optional) Default: `True` |
| prefetch_factor |  Number of samples loaded in advance by each worker. (`int`, optional) Default: `10` |
| shuffle | Flag to enable data shuffling. (`bool`, optional) Default: `True` |
| shuffle_buffer | Size of shuffle buffer in samples. (`int`, optional) Default: `10 * batch_size` |
| shuffle_seed | Shuffle seed. (`int`, optional) Default: `None` |

### Transformers

| Parameter Name | Description | Supported Models |
| --- | --- | --- |
| do_lower | Flag to lower case the texts. (`bool`, optional) Default: `False` | BERT (pre-training, fine-tuning), T5, Transformer |
| dynamic_loss_weight | Flag to dynamically scale the loss. If set, will divide the loss for a token by the length of the sequence that the token comes from. Use with `"precomputed_num_tokens"` loss scaling. (`bool`, optional) Default: `False` | T5, Transformer |
| dynamic_mlm_scale | Flag to dynamically scale the loss. If set, MLM Loss is scaled by the number of masked tokens in the current batch using the `masked_lm_weights` from the input data features.  (`bool`, optional) Default: `False` | BERT (pre-training) |
| extra_ids | Number of sentinel tokens for T5 objective. (`int`, optional) Default: `0` | T5, Transformer |
| masked_lm_prob  | Ratio of the masked tokens over the sequence length. (`float`, optional) Default: `0.15`| BERT (pre-training) |
| max_predictions_per_seq  |  Maximum number of masked tokens per sequence. (`int`, required) | BERT (pre-training) |
| max_sequence_length  | Maximum sequence length of the input data. (`int`, optional) Default: varies per model | All |
| src_data_dir | Path to directory containing all the files of tokenized data for source sequence. (`str`, required) | T5, Transformer |
| src_max_sequence_length | Largest possible sequence length for the input source sequence. If longer it will be truncated. All other sequences padded to this length. (`int`, required) | T5, Transformer |
| src_vocab_file | Path to vocab file for source input. (`str`, required) | T5, Transformer |
| tgt_data_dir | Path to directory containing all the files of tokenized data for target sequence. (`str`, required) | T5, Transformer |
| tgt_max_sequence_length | Largest possible sequence length for the input target sequence. If longer it will be truncated. All other sequences padded to this length. (`int`, required) | T5, Transformer |
| tgt_vocab_file | Path to vocab file for target input. (`str`, required) | T5, Transformer |
| vocab_file | Path to vocab file. (`str`, required) | BERT (pre-training, fine-tuning) |
| vocab_size | The size of the vocabulary used in the model. (`int`, required) | BERT (pre-training, fine-tuning) |

### Computer Vision

| Parameter Name | Description | Supported Models |
| --- | --- | --- |
| aggregate_cartilage | For SKM-TEA dataset only. Combines medial and lateral classes into single class. (`bool`, optional) Default: `True` | UNet |
| augment_data | Apply data augmentation to the data. (`bool`, optional) Default: `True` | UNet |
| class_id | For the Severstal Dataset this sets which class id to be considered as the positive class. All other classes will be considered negative examples. (`int`, optional) | UNet |
| echo_type | For SKM-TEA dataset only. Specifies training data configuration. Allowed options are: `echo1`, `echo2`, or `root_sum_of_squares`. (`str`, required) Default: `echo1` | UNet |
| image_shape | Expected shape of output images in format (H, W, C), (`List[int]`, required) | UNet |
| normalize_data_method | Specify the strategy to normalize the input data. One of: `"zero_centered"`,`"zero_one"`,`"standard_score"`. (`str`, required) | UNet |
| num_classes | Number of classes in the training dataset. (`int`, required) | UNet |
| train_test_split | Percentage of data to be used in the training dataset. | UNet |
| use_fast_dataloader | If set to True, mapstyle datasets that use the UNetDataProcessor perform faster data processing. (`bool`, optional) Default: `False` | UNet |
| use_worker_cache | If set to True data will be read from local SSD memory on the individual worker nodes during training. If the data does not exist on the worker nodes it will be automatically copied from the host node. This will cause a slowdown the first time this copy takes place. (`bool`, optional) Default: `True` | UNet |
| num_diffusion_steps | Number of timesteps in the diffusion forward process(`int`, required) | DiT |
| split | Dataset split to use(`str`, required) | DiT |


## Optimizer params

| Parameter Name | Description |
| --- | --- |
| initial_loss_scale | Initial loss scale to be used in the grad scale. (`int`, optional) Default: `2 ** 15` |
| learning_rate | Learning rate scheduler to be used. See [supported LR schedulers](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-ops/supported-pt-learning-rate-schedulers.html). (`dict`, required) |
| log_summaries | Flag to log per layer gradient norm in Tensorboard (`bool`, optional) Default: `False` |
| loss_scaling_factor | Loss scaling factor for gradient calculation in learning step. (`float`/`str`, optional) Default: `1.0` |
| max_gradient_norm | Max norm of the gradients for learnable parameters. Used for gradient clipping.(`float`, optional) Default: `None` |
| min_loss_scale | The minimum loss scale value that can be chosen by dynamic loss scaling. (`float`, optional) Default: `None` |
| max_loss_scale | The maximum loss scale value that can be chosen by dynamic loss scaling. (`float`, optional) Default: `None` |
| optimizer_type | Optimizer to be used. See [supported optimizers](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-ops/supported-pytorch-optimizers.html). (`str`, required) |

## Runconfig params

| Key | Description | Supported mode |
| --- | --- | --- |
| autoload_last_checkpoint | Flag to automatically load the last checkpoint in the `model_dir`. (`bool`, optional) Default: `True` | All |
| check_loss_values | Flag to check the loss values to see if it is `Nan/inf`. (`bool`, optional) Default: `True` | All |
| checkpoint_path | The path to load checkpoints from during training. (`str`, optional) Default: `None` | All |
| checkpoint_steps | The number of steps between saving model checkpoints during training. `0` means no checkpoints saved. (`int`, optional) Default: `0` | All |
| compile_dir | Compile directory where compile artifacts will be written. (`str`, optional) Default: `None` | All |
| compile_only | Enables compile only workflow. (`bool`, optional) Default: `False` | All |
| credentials_path | Credentials for cluster access. If `None`, the value from a pre-configured location will be used if available. (`str`, optional) Default: `None`| CSX |
| debug_args_path | Path to debugs args file.  (`str`, optional) Default: `None` | CSX |
| disable_strict_checkpoint_loading | Flag used in conjunction with `checkpoint_path`, to avoid enforcing strict model state loading. (`bool`, optional) Default: `False`    | All            |
| dist_addr | To init master_addr and master_port of distributed. (`str`, optional) Default: `localhost:8888` | GPU |
| dist_backend | Distributed backend engine. (`str`, optional) Default: `"nccl"` | GPU |
| enable_distributed | Flag to enable distributed training on GPU. (`bool`, optional) Default: `False` | GPU |
| enable_summaries | Enable summaries when running on CS-X hardware. (`bool`, optional) Default: `False` | CSX |
| eval_frequency | Specifies the evaluation frequency during training. Only used for `train_and_eval` mode.  (`int`, optional) Default: `None` | All |
| eval_steps | Specifies the number of steps to run the model evaluation. (`int`, optional) Default: `None` | All |
| init_method | URL specifying how to initialize the process group. (`str`, optional) Default: `"env://"` | GPU |
| job_labels | A list of equal-sign-separated key value pairs served as job labels. (`str`, optional) Default: `None` | CSX |
| load_checkpoint_states | Comma-separated string of keys used in conjunction with `checkpoint_path` to explicitly specify what components' state should be loaded if present in a checkpoint. If this flag is used, any component whose key isn't specified will not load state from the checkpoint. For example, if `load_checkpoint_states` is `"model"`, we only load the model state and enforce resetting of optimizer states and training steps after loading a given checkpoint; i.e., matching weights are initialized from checkpoint provided by `checkpoint_path`, training starts from step 0, and optimizer states present in the checkpoint are ignored. This is useful for fine-tuning runs on different tasks (e.g., classification, Q&A, etc.) where weights from a pre-trained model trained on language modeling (LM) tasks are loaded or fine-tuning on a different dataset on the same LM task. If `dataloader` state exists in the checkpoint, that will also be ignored. In this case, the dataloaders will yield samples from the beginning. However, if `load_checkpoint_states` is `"model,dataloader"`then only the model and dataloader states will be loaded. By default, this config is `None` meaning that we load state for every compononent found in the checkpoint. (`str`, optional) Default: `None` | All |
| steps_per_epoch | The number of steps per epoch. (`int`, optional) Default: `None` | All |
| log_steps | Specifies the number of steps between logging during training. Same number controls the summary steps in Tensorboard. (`int`, optional) Default: `None` | All |
| logging | Specifies the logging level during training. (`str`, optional) Default: `"INFO"` | All |
| max_steps | Specifies the maximum number of steps for training. `max_steps` is optional unless neither `num_epochs` nor `num_steps` are provided, in which case `max_steps` must be provided. (`int`, required) | All |
| mgmt_address | The address of the management service used for coordinating the training job as `<host>:<port>`. (`str`, optional) | CSX |
| mode | The mode of the training job, either '`"train"`', '`"eval"`', `"eval_all"` or `"train_and_eval"`. (`str`, required) | All |
| model_dir | The directory where the model checkpoints and other metadata will be saved during training. (`str`, optional) Default: `./model_dir` | All |
| mount_dirs | A list of paths to be mounted to the appliance containers. It should generally contain path to the directory containing the Cerebras model zoo and data dir. (`List[str]`, optional) Default: `None` | CSX |
| num_act_servers |  Number of activation servers per CS-X dedicated to stream samples to the WSE. Input workers stream data to these activation servers, and the activation servers to hold and further stream the data to the WSE. For LLMs, we generally choose 1 because they're compute-bound. For CV models we choose a higher number, a crude rule of thumb is to have one activation server for every 4 workers (i.e. `num_workers_per_csx // 4 if num_workers_per_csx > 4, else 1`). It is suggested to keep the default values for this param when possible. (`int`, optional) Default: `1` | CSX |
| num_csx | The number of CSX systems to use in Cerebras WSE cluster. (`int`, optional) Default: `1` | CSX |
| num_epochs | The number of epochs to train for. (`int`, optional) Default: `None` | All |
| num_steps | The number of steps to train for. (`int`, optional) Default: `None` | All |
| num_wgt_servers | Upper bound on the number of MemoryX servers used for storing the model weights. Compilation may choose a smaller number depending on the model topology. A sensible upper bound (currently 24) is selected if a value is not provided. (`int`, optional) Default: `None` | CSX |
| num_workers_per_csx | Number of input workers, per CSX, to use for streaming samples. This setting depends on whether the model is compute-bound or input-bound and how efficient the dataloader implementation is. For compute-bound models (e.g., LLM), even 1 input worker per csx is enough to saturate the input buffers on CSX systems. But for smaller models a larger number may be used. We currently default to 1 worker per CSX. (`int`, optional) Default: `0` | CSX |
| precision_opt_level | Setting to control the level of numerical precision used for training runs for large NLP models. See [more](https://docs.cerebras.net/en/latest/general/performance-optimization.html?#precision-optimization-level). (`int`, optional) Default: `1` | CSX |
| python_paths | A list of paths to be exported into `PYTHONPATH` for worker containers. It should generally contain path to the directory containing the Cerebras model zoo. (`List[str]`, optional) Default: `None` | CSX |
| save_initial_checkpoint | Whether to save an initial checkpoint before training starts. (`bool`, optional) Default: `False` | All |
| seed | The seed to use for random number generation for reproducibility. (`int`, optional) Default: `None` | All |
| sync_batchnorm | Whether to use synchronized batch normalization on multi GPU setup. (`bool`, optional) Default: `False` | GPU |
| target_device | The target device to run the training on. One of: `CPU`, `GPU`, `CSX`. Required in command line. (`str`, optional) Default: command line value | All |
| validate_only | Enables validate only workflow, stops the compilation at kernel matching stage. (`bool`, optional) Default: `False` | CSX |
| wsc_log_level | Specifes the logging level for particular Wafer-Scale Cluster servers or tasks. Input can be either a single value setting a global log level (i.e. `--wsc_log_level DEBUG`) or a list of equal-sign-separated key value pairs in the format of `<task or server>=<log level>`. A task and server can be combined to specify a server only during a specific task (i.e. `<execute>.<crd>`). The log level can be either an int or a string (i.e. `INFO`, `DEBUG`, `VERBOSE`, `20`, `10`). See [more](https://docs.python.org/3/library/logging.html#logging-levels). (`str`, optional) Default: `None` | All |