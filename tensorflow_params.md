# YAML params for TensorFlow models

## Model params

### Transformer based models

| Parameter Name | Description | Supported Models |
| --- | --- | --- |
| attention_dropout_rate | Dropout rate for attention layer. (`float`, optional) Default: same as `dropout` | All |
| attention_type | Type of attention. Accepted values:<br>`"dot_product"`<br>`"scaled_dot_product"`.<br>(`str`,  optional) Default: `"scaled_dot_product"` | All |
| boundary_casting | Flag to cast outputs the values in half precision and casts the input values up to full precision. (`bool`, optional) Default: `False` | All |
| decoder_nonlinearity | Type of nonlinearity to be used in decoder. (`str`,  optional) Default: `"relu"` | T5, Transformer |
| decoder_num_hidden_layers | Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set. (`int`, optional) | T5, Transformer |
| disable_nsp | Whether to disable the next sentence prediction task. (`bool`, optional) Default: False | BERT |
| dropout_rate | The dropout probability for all fully connected layers. (`float`, optional), Default: `0.1` | All |
| dropout_seed | Seed with which to initialize the dropout layer. (`int`, optional) Default: `None` | All |
| embedding_initializer | Initializer to use for embeddings. See [supported initializers](#supported-initializers). (`str`, optional) Default: varies per model | All |
| encoder_nonlinearity | Type of nonlinearity to be used in encoder. (`str`, optional) Default: varies per model | BERT, Linformer, T5, Transformer |
| encoder_num_hidden_layers | Number of hidden layers in the encoder. (`int`, required) | T5, Transformer |
| filter_size | Dimensionality of the feed-forward layer in the Transformer block. (`int`, required) |  All |
| hidden_size | The size of the transformer hidden layers (`int`, required) | All |
| initializer | The initializer to be used for all the initializers used in the model. See [supported initializers](#supported-initializers). (`str`, optional) Default: varies per model | All |
| layer_norm_epsilon | The epsilon value used in layer normalization layers. (`float`, optional) Default: `1e-5`)| All |
| loss_scaling | The scaling type used to calculate the loss. Accepts: <br> `batch_size`, `num_tokens` (`str`, optional) Default: `num_tokens` | GPT2, GPT3, GPTJ |
| loss_weight | The weight for the loss scaling when `loss_scaling: "batch_size"`, generally set to `1/max_sequence_length`. (`float`, optional) Default: `1.0` | GPT2, GPT3, GPTJ |
| max_position_embeddings | The maximum sequence length that the model can handle. (`int`, required) | All |
| mixed_precision | Whether to use mixed precision training or not. (`bool`, optional) Default: `None` | All |
| nonlinearity | The non-linear activation function used in the feed forward network in each transformer block. (`str`, optional) Default: varies per model | All |
| num_heads | The number of attention heads in the multi-head attention layer. (`int`, required) | All |
| num_hidden_layers |  Number of hidden layers in the Transformer encoder/decoder. (`int`, required) | All |
| output_layer_initializer | The name of the initializer for the weights of the output layer. See [supported initializers](#supported-initializers). (str, optional) Default: varies based on model | GPT2, GPT3, GPTJ, T5 |
| position_embedding_type | The type of position embedding to use in the model. Can be one of: `"fixed"`, `"learned"` (`str`, required) | All |
| precision_opt_level | Setting to control the level of numerical precision used for training runs for large NLP models in weight-streaming. See [more](https://docs.cerebras.net/en/latest/general/performance-optimization.html?#precision-optimization-level). (`int`, optional) Default: `1` | GPT2, GPT3, GPTJ |
| rotary_dim | The number of dimensions used for the rotary position encoding. (`int`, optional) Default: `None` | GPTJ |
| share_embedding_weights | Whether to share the embedding weights between the input and out put embedding. (`bool`, optional) Default: `True` | BER, GPT2, GPT3, GPTJ, Linformer |
| share_encoder_decoder_embedding | Whether to share the embedding weights between the encoder and decoder (`bool`, optional) Default: `True`| T5, Transformer |
| tf_summary | Flag to save the activations with the `summary_layer`. (`bool`, optional) Default: `False` | All |
| use_bias_in_output | Whether to use bias in the final output layer. (`bool`, optional) Default: `False` | GPT2, GPT3, GPTJ |
| use_ffn_bias | Whether to use bias in the feedforward network (FFN). (`bool`, optional) Default: varies per model | All |
| use_ffn_bias_in_attention | Whether to include bias in the attention layer for feed-forward network (FFN). (`bool`, optional) Default: varies per model | All |
| use_position_embedding | Whether to use position embedding in the model. (`bool`, required) | BERT, GPT2, GPT3, Linformer |
| use_projection_bias_in_attention | Whether to include bias in the attention layer for projection.  (`bool`, optional) Default: varies per model | All |
| use_segment_embedding | Whether to use segment embedding in the model. (`bool`, required) | BERT, Linformer |
| use_untied_layer_norm | Flag to use untied layer norm in addition to the default layer norm. (`bool`, optional) Default: `False` | GPTJ |
| use_vsl | Whether to enable variable sequence length. See [more](https://docs.cerebras.net/en/latest/pytorch-docs/pytorch-vts.html). (`bool`, optional) Default: `False` | BERT (pre-training), T5, Transformer |
| vocab_size | The size of the vocabulary used in the model. (`int`, optional) Default: varies per model | All |
| weight_initialization_seed | Seed applied for weights initialization. (`int`, optional) Default: `None` | All |

#### Supported Initializers

Supported initializers include:

- `"constant"`
- `"uniform"`
- `"glorot_uniform"`
- `"normal"`
- `"glorot_normal"`
- `"truncated_normal"`
- `"variance_scaling"`

## Data loader params

### Transformers

| Parameter Name | Description | Supported Models |
| --- | --- | --- |
| add_special_tokens | Flag to add special tokens in the data loader. Special tokens are defined based on the data processor.  (`bool`, optional) Default: `False` | BERT, GPT2, GPT3 |
| batch_size | Batch size of the data. (`int`, required) | All |
| buckets | A list of boundaries for sequence lengths to bucket together in order to speed up VTS/VSL. (`list`, optional) Default: `None` | BERT (pre-training), T5, Transformer |
| data_dir |  Path/s to the data files to use. (`str`/`List[str]`, required) | All |
| data_processor | Name of the data processor to be used. (`str`, required) | All |
| do_lower | Flag to lower case the texts. (`bool`, optional) Default: `False` | BERT, Linformer, T5, Transformer |
| mask_whole_word | Flag to mask the whole words. (`bool`, optional) Default: `False` | BERT, Linformer |
| max_predictions_per_seq |  Maximum number of masked tokens per sequence. (`int`, required) | BERT, Linformer |
| max_sequence_length | Maximum sequence length of the input data. (`int`, optional) Default: varies per model | All |
| mixed_precision | Flag to cast input to fp16. (`bool`, optional) Default: `None` | All |
| n_parallel_reads | For call to `tf.data.Dataset.interleave` (`int`, optional) Default: `4` | All |
| repeat | Flag to specify if the dataset should be repeated. (`bool`, optional) Default: `True` | All |
| scale_mlm_weights | Scales `lm_weights` with the value of `batch_size/sum(mlm_weights)` for loss calculation. (`bool`, optional) Default: `True` | BERT, Linformer |
| shuffle | Flag to enable data shuffling. (`bool`, optional) Default: `True` | All |
| shuffle_buffer | Size of shuffle buffer in samples. (`int`, optional) Default: `10 * batch_size` | All |
| shuffle_seed | Shuffle seed. (`int`, optional) Default: `None` | All |
| src_data_dir | Path to directory containing all the files of tokenized data for source sequence. (`str`, required) | T5, Transformer |
| src_max_sequence_length | Largest possible sequence length for the input source sequence. If longer it will be truncated. All other sequences padded to this length. (`int`, required) | T5, Transformer |
| src_vocab_file | Path to vocab file for source input. (`str`, required) | T5, Transformer |
| tgt_data_dir | Path to directory containing all the files of tokenized data for target sequence. (`str`, required) | T5, Transformer |
| tgt_max_sequence_length | Largest possible sequence length for the input target sequence. If longer it will be truncated. All other sequences padded to this length. (`int`, required) | T5, Transformer |
| tgt_vocab_file | Path to vocab file for target input. (`str`, required) | T5, Transformer |
| use_multiple_workers | Flag to specify if the dataset will be sharded. (`bool`, optional) Default: `False` | All |
| vocab_file | Path to vocab file. (`str`, required) | BERT (pre-training, fine-tuning) | BERT, Linformer |
| vocab_size | The size of the vocabulary used in the model. (`int`, required) | BERT (pre-training, fine-tuning) | All |

## Optimizer params

| Parameter Name | Description |
| --- | --- |
| initial_loss_scale | Initial loss scale to be used in the grad scale. (`int`, optional) Default: `2 ** 15` |
| learning_rate | Learning rate scheduler to be used. See [supported LR schedulers](#supported-learning-rate-schedulers)  (`float/dict`, required)) |
| log_summaries | Flag to log per layer gradient norm in Tensorboard (`bool`, optional) Default: `False` |
| loss_scaling_factor | Loss scaling factor for gradient calculation in learning step. (`float`/`str`, optional) Default: `1.0` |
| max_gradient_norm | Max norm of the gradients for learnable parameters. Used for gradient clipping.(`float`, optional) Default: `None` |
| min_loss_scale | The minimum loss scale value that can be chosen by dynamic loss scaling. (`float`, optional) Default: `None` |
| max_loss_scale | The maximum loss scale value that can be chosen by dynamic loss scaling. (`float`, optional) Default: `None` |
| optimizer_type | Optimizer to be used. Supported optimizers: `"sgd"`, `"momentum"`, `"adam"`, `"adamw"`. (`str`, required) |
| ws_summary | Flag to add weights summary into the Tensorboard. (`bool`, optional) Default: `False` |

### Supported learning rate schedulers

Currently supports for following learning rates:

- constant
- cosine
- exponential
- linear
- polynomial
- piecewise constant

`learning_rate` can be specified in yaml as:

- a single float for a constant learning rate
- a dict representing a single decay schedule
- a list of dicts (for a series of decay schedules)

## Runconfig params

| Key | Description | Supported mode |
| --- | --- | --- |
| enable_distributed | Flag to enable distributed training on GPU. (`bool`, optional) Default: `False` | GPU |
| eval_steps | Specifies the number of steps to run the model evaluation. (`int`, optional) Default: `None` | All |
| keep_checkpoint_max | Total number of most recent checkpoints to keep in the `model_dir`.  (`int`, optional) Default: `5` | All |
| log_step_count_steps | Specifies the number of steps between logging during training. (`int`, optional) Default: `None` | All |
| max_steps | Specifies the maximum number of steps for training. `max_steps` is optional unless neither `num_epochs` nor `num_steps` are provided, in which case `max_steps` must be provided. (`int`, required) | All |
| mode | The mode of the training job, either '`"train"`', '`"eval"`', `"eval_all"`. (`str`, required) | All |
| model_dir | The directory where the model checkpoints and other metadata will be saved during training. (`str`, optional) Default: `./model_dir` | All |
| multireplica | Whether to allow multiple replicas for the same graph. See [more](https://docs.cerebras.net/en/latest/original/general/multi-replica-data-parallel-training.html). (`bool`, optional)  Default: `False` | CSX (pipeline mode) |
| num_wgt_servers | The number of weight servers to use in weight streaming execution. (`int`, optional) Default: `None` | CSX (weight streaming) |
| save_checkpoints_steps | The number of steps between saving model checkpoints during training. `0` means no checkpoints saved. (`int`, optional) Default: `0` | All |
| save_summary_steps | This number controls the summary steps in Tensorboard. (`int`, optional) Default: `None` | All |
| tf_random_seed | The seed to use for random number generation for reproducibility. (`int`, optional) Default: `None` | All |
| use_cs_grad_accum | Whether to use gradient accumulation to support larger batch sizes. (`bool`, optional) Default: `False` | CSX |
