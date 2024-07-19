# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from cerebras.modelzoo.common.registry import registry
from cerebras.modelzoo.config_manager.config_classes.base.base_config import (
    BaseConfig,
    required,
)
from cerebras.modelzoo.config_manager.config_classes.base.data_config import (
    DataConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.model_config import (
    InitializerConfig,
    ModelConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.optimizer_config import (
    OptimizerConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.run_config import (
    RunConfig,
)
from cerebras.modelzoo.config_manager.config_classes.base.sparsity_config import (
    SparsityConfig,
)
from cerebras.modelzoo.models.vision.dit.samplers.sampler_utils import (
    configure_sampler_params,
)


@dataclass
class ReverseProcessConfig(BaseConfig):
    pipeline: Optional[dict] = None
    "Pipeline configuration for reverse process."

    sampler: Optional[dict] = None
    "Sampler configuration for reverse process."

    batch_size: int = 32

    def __post_init__(self):
        configure_sampler_params(copy.deepcopy(self.sampler))


@dataclass
class VAEConfig(BaseConfig):
    act_fn: str = "silu"
    "Activation function to use in VAE autoencoder."

    in_channels: int = 3
    "Number of channels in the input image."

    out_channels: int = 3
    "Number of channels in the output."

    block_out_channels: Tuple[int] = (64,)
    "Tuple of block output channels"

    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    "Tuple of downsample block types"

    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    "Tuple of upsample block types."

    latent_channels: int = 4
    "Number of channels in the latent space."

    latent_size: List[int] = field(default_factory=lambda: [32, 32])
    "Size of latent space (height, width)."

    layers_per_block: Optional[int] = 1
    "Number of layers per block."

    norm_num_groups: int = 32
    "The number of groups to use for the first normalization layer of each block."

    sample_size: Optional[int] = None
    "Sample size to use when tiling is enabled"

    scaling_factor: float = 0.18215
    """The component-wise standard deviation of the trained latent space computed using the first batch of the
    training set. This is used to scale the latent space to have unit variance when training the diffusion
    model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
    diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z = 1
    / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution Image
    Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper."""


@dataclass
class DiTModelConfig(ModelConfig):
    # Embedding
    embedding_dropout_rate: float = 0.1
    "Dropout rate for embeddings."

    embedding_nonlinearity: str = "silu"
    "Type of nonlinearity to be used in embedding."

    position_embedding_type: str = "learned"
    """The type of position embedding to use in the model. Can be one of:
    `fixed` - Sinusoidal from original Transformer, (https://arxiv.org/abs/1706.03762).
    `relative` - Relative position embedding, to exploit pairwise,
      relative positional information, (https://arxiv.org/abs/1803.02155).
    `rotary` - a.k.a RoPE
    `learned` - Learned embedding matrix, (https://arxiv.org/pdf/2104.09864v4.pdf)"""

    # Transformer Attention
    attention_type: str = "scaled_dot_product"
    "Type of attention. Accepted values: `dot_product`, `scaled_dot_product`."

    attention_softmax_fp32: bool = True
    "Whether to use fp32 precision for attention softmax."

    attention_dropout_rate: float = 0.0
    "Dropout rate for attention layer."

    use_projection_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for projection."

    use_ffn_bias_in_attention: bool = True
    "Whether to include bias in the attention layer for feed-forward network."

    # Model
    num_classes: int = 1000
    "Number of classes in the training dataset."

    hidden_size: int = 768
    "The size of the transformer hidden layers."

    num_hidden_layers: Optional[int] = 12
    "Number of hidden layers in the Transformer encoder/decoder."

    layer_norm_epsilon: Optional[float] = 1.00e-5
    "The epsilon value used in layer normalization layers."

    num_heads: int = 12
    "The number of attention heads in the multi-head attention layer."

    dropout_rate: float = 0.0
    "The dropout probability for all fully connected layers."

    nonlinearity: str = "gelu"
    """The non-linear activation function used in the feed forward network
    in each transformer block. Some may have to use autogen_policy: `medium`."""

    filter_size: int = 3072
    "Dimensionality of the feed-forward layer in the Transformer block."

    use_ffn_bias: Optional[bool] = True
    "Whether to use bias in the feedforward network."

    norm_first: Optional[bool] = True
    """Enables normalization before the Attention & FFN blocks (i.e Pre-LN as
    described in https://arxiv.org/pdf/2002.04745.pdf. When disabled,
    normalization is applied *after* the residual (Post-LN)"""

    latent_size: Optional[List] = None
    "Latent Tensor(output of VAEEncoder) [height, width]."

    latent_channels: Optional[int] = None
    "Number of channels in Latent Tensor."

    patch_size: List[int] = field(default_factory=lambda: [16, 16])
    "Size of patch used to convert image to tokens."

    use_conv_patchified_embedding: bool = False
    "If True, use conv2D to convert image to patches."

    use_conv_transpose_unpatchify: Optional[bool] = True
    "If True, use transposed convolution for unpatchify step."

    frequency_embedding_size: int = 256
    "Size of sinusoidal timestep embeddings."

    label_dropout_rate: float = 0.1
    "Probability of dropout applied to label tensor."

    block_type: Literal["adaln_zero",] = "adaln_zero"
    "DiT Block variant. Accepted values: adaln_zero."

    encoder_nonlinearity: str = "gelu"
    "Type of nonlinearity to be used in encoder."

    vae: Optional[VAEConfig] = None
    """Only used for evaluation. This should be left as None in train mode since we don't run the
    VAE during training."""

    # Initialization
    initializer_range: float = 0.02
    """The standard deviation of the truncated_normal_initializer as the
    default initializer"""

    projection_initializer: InitializerConfig = InitializerConfig(
        name="xavier_uniform", gain=1.0
    )
    """Initializer for embedding linear layer. Either a string indicating the name of the
    initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""

    position_embedding_initializer: Optional[str] = None
    """Initializer for position embedding layer. Either a string indicating the name of
    the initializer or a dict that includes the name + other params if relevant. If left
    unspecified will apply truncated normal initialization."""

    init_conv_like_linear: Optional[bool] = None
    "If True, modify fan-in fan-out by reshaping before initializing."

    attention_initializer: InitializerConfig = InitializerConfig(
        name="xavier_uniform", gain=1.0
    )
    "Attention layer initializer. Defaults to `xavier_uniform`."

    ffn_initializer: InitializerConfig = InitializerConfig(
        name="xavier_uniform", gain=1.0
    )
    "FFN layer initializer. Defaults to `xavier_uniform`."

    timestep_embedding_initializer: Optional[InitializerConfig] = None
    "Initializer used in timestep embedding."

    label_embedding_initializer: Optional[InitializerConfig] = None
    "Initializer used in label embedding."

    head_initializer: InitializerConfig = InitializerConfig(name="zeros")
    "Initializer used in regression head."

    # Diffusion
    num_diffusion_steps: int = 1000
    "Number of timesteps in the diffusion forward process."

    schedule_name: str = "linear"
    "Scheduler for gaussian diffusion steps."

    beta_start: float = 0.0001
    "The starting beta value of inference."

    beta_end: float = 0.02
    "The final beta value."

    reverse_process: Optional[ReverseProcessConfig] = None
    """Configuration for reverse diffusion process. This is only used for inference
    and should be left as None in train mode."""

    var_loss: bool = False

    fp16_type: Literal["bfloat16", "float16", "cbfloat16"] = "bfloat16"
    "Type of 16bit precision used"

    def __post_init__(self):
        super().__post_init__()

        if self.latent_channels is None:
            self.latent_channels = self.vae.latent_channels
        if self.latent_size is None:
            self.latent_size = self.vae.latent_size
        if self.init_conv_like_linear is None:
            self.init_conv_like_linear = self.use_conv_patchified_embedding
        if self.timestep_embedding_initializer is None:
            self.timestep_embedding_initializer = InitializerConfig(
                name="normal", mean=0.0, std=self.initializer_range
            )
        if self.label_embedding_initializer is None:
            self.label_embedding_initializer = InitializerConfig(
                name="normal", mean=0.0, std=self.initializer_range
            )


@dataclass
class DiTTrainDataConfig(DataConfig):
    split: Optional[Literal["train", "val"]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        self.params.setdefault("shuffle", True)
        self.params.setdefault("shuffle_seed", 4321)
        self.params.setdefault("num_classes", 1000)
        self.params.setdefault("noaugment", False)
        self.params.setdefault("drop_last", True)
        self.params.setdefault("num_workers", 0)
        self.params.setdefault("prefetch_factor", 10)
        self.params.setdefault("persistent_workers", True)
        self.params.setdefault("use_worker_cache", False)
        if self.params["noaugment"]:
            self.params["transforms"] = None

        super().__post_init__()


@dataclass
class DiTEvalDataConfig(DataConfig):
    split: Optional[Literal["train", "val"]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __post_init__(self):
        self.params.setdefault("shuffle", False)
        self.params.setdefault("shuffle_seed", 4321)
        self.params.setdefault("noaugment", False)
        self.params.setdefault("drop_last", True)
        self.params.setdefault("num_workers", 0)
        self.params.setdefault("prefetch_factor", 10)
        self.params.setdefault("persistent_workers", True)
        self.params.setdefault("use_worker_cache", False)
        if self.params["noaugment"]:
            self.params["transforms"] = None

        super().__post_init__()


@registry.register_config("dit")
@dataclass
class DiTConfig(BaseConfig):
    eval_input: Optional[DiTEvalDataConfig] = None
    "Input params class for eval mode."

    model: DiTModelConfig = required
    "Model level params class. Supported params differ for each model."

    optimizer: OptimizerConfig = required
    "Optimizer specific parameters captured in this class."

    runconfig: RunConfig = None
    "Params class to define params for controlling runs."

    train_input: Optional[DiTTrainDataConfig] = None
    "Input params class for train mode."

    sparsity: Optional[SparsityConfig] = None
    "Params class for sparsity related configurations."

    def __post_init__(self):
        super().__post_init__()
        self.set_config_defaults(
            self.model,
            self.train_input.params if self.train_input else None,
            [self.eval_input.params if self.eval_input else None],
        )

    @staticmethod
    def set_config_defaults(mparams, tparams, eparams_list):
        from .utils import _model_to_input_map

        # Set Model related defaults
        if tparams is not None:
            mparams.num_diffusion_steps = tparams["num_diffusion_steps"]
            mparams.num_classes = tparams["num_classes"]
            mparams.vae.in_channels = tparams["image_channels"]
            mparams.vae.out_channels = tparams["image_channels"]

        if mparams.reverse_process:
            rparams = mparams.reverse_process
            rparams.sampler.setdefault(
                "num_diffusion_steps", mparams.num_diffusion_steps
            )
            rparams.pipeline.setdefault("num_classes", mparams.num_classes)
            rparams.pipeline.setdefault("custom_labels", None)
            # For DDPM Sampler only

        # Copy params across
        for section in [tparams, *eparams_list]:
            if section is None:
                continue

            section["vae_scaling_factor"] = mparams.vae.scaling_factor

            for _key_map in _model_to_input_map:
                if isinstance(_key_map, tuple):
                    assert (
                        len(_key_map) == 2
                    ), f"Tuple {_key_map} does not have len=2"
                    model_key, input_key = _key_map
                else:
                    model_key = input_key = _key_map

                section[input_key] = getattr(mparams, model_key)
