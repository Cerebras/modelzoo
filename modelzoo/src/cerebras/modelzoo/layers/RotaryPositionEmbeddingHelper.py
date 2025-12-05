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

import math
from enum import Enum, auto
from typing import Tuple

import torch

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.half_dtype import maybe_to_half_dtype
from cerebras.modelzoo.common.utils.model.transformer_utils import (
    create_sliding_window_mask_with_complement,
)
from cerebras.modelzoo.common.utils.model.wit_utils import (
    wafer_instruction_tuning,
)


class RopeRelDistanceMode(Enum):
    default = auto()
    capped = auto()
    grouped = auto()


class NoRopeModification:
    """Sentinel object signifying that unmodified
    queries/keys have to be returned by `rotate_tensor`
    """


no_rope_modification_pos_id = NoRopeModification


def rotate_every_two(x):
    head_dim = x.shape[-1]
    odd_dim = head_dim % 2
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_odd_or_null = x[..., :0]
    if odd_dim:
        x_odd_or_null = x1[..., -1:]
        x1 = x1[..., :-1]
    x = torch.stack((-x2, x1), dim=-1)
    return torch.cat((x.flatten(-2), x_odd_or_null), dim=-1)


class RotaryPosEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, sin, cos):
        ctx.save_for_backward(x, sin, cos)
        x_dtype = x.dtype
        if cstorch.current_pol() >= 1:
            x, sin, cos = maybe_to_half_dtype((x, sin, cos))
        return (x * cos + rotate_every_two(x) * sin).to(x_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, sin, cos = ctx.saved_tensors
        x_dtype = x.dtype
        if cstorch.current_pol() >= 1:
            grad_output, sin, cos = maybe_to_half_dtype((grad_output, sin, cos))
        grad_x = grad_output * cos - rotate_every_two(grad_output * sin)
        return grad_x.to(x_dtype), None, None


rotary_pos_emb = RotaryPosEmb.apply


def _compute_default_inv_freq(
    x_shape: torch.Size,
    rotary_dim: int,
    theta: float,
    device: torch.device,
    fold_const: bool,
) -> torch.tensor:
    """Returns repeated default inverse frequency tensor for RoPE.

    inv_freq = theta ^ -(2i / d) for i in [0, d/2)
    return repeat_interleave(inv_freq, 2)
    """
    theta = torch.tensor(theta, device=device, dtype=torch.float32)
    head_dim = x_shape[-1]
    # NOTE: Values past rotary_dim are garbage and will be overwritten later.
    dim_range = torch.arange(
        head_dim, device=device, dtype=torch.float32
    ).broadcast_to(x_shape)
    dim_pairs = dim_range / 2
    dim_pairs = dim_pairs.to(torch.int16).to(torch.float32)

    if fold_const:
        return 1 / theta ** (2 * dim_pairs / rotary_dim)

    # NOTE: This is numerically different from the above implementation.
    neg_ln_theta_by_dim = -torch.log(theta) * 2 / rotary_dim
    inv_freq = torch.exp(dim_pairs * neg_ln_theta_by_dim)
    return inv_freq


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (
        dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
    ) / (2 * math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(x_shape, min, max, device):
    if min == max:
        max += 0.001  # Prevent singularity

    # repeat interleave and broadcast. values past rotary dim are garbage and
    # will be overwritten.
    dim_range = torch.arange(
        x_shape[-1], device=device, dtype=torch.float32
    ).broadcast_to(x_shape)
    dim_pairs = dim_range / 2
    dim_pairs = dim_pairs.to(torch.int16).to(torch.float32)

    linear_func = (dim_pairs - min) / (max - min)
    ramp_func = torch.where(linear_func < 0, 0, linear_func)
    ramp_func = torch.where(linear_func > 1, 1, linear_func)
    return ramp_func


def _yarn_get_mscale(scale=1, a=0.1, b=1.0):
    if scale <= 1:
        return 1.0
    return a * math.log(scale) + b


def _llama_3_update_inv_freq(
    inv_freq,
    scaling_factor=8,
    low_freq_factor=1,
    high_freq_factor=4,
    original_max_position_embeddings=8192,
):
    # Ported from https://github.com/huggingface/transformers/blob/44f6fdd74f84744b159fa919474fd3108311a906/src/transformers/modeling_rope_utils.py#L298-L339
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # wavelen < high_freq_wavelen: do nothing
    lhs = wavelen - high_freq_wavelen + 1
    lt_mask = torch.where(lhs < 1.0, 1.0, 0.0)
    inv_freq_llama = inv_freq * lt_mask

    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_scaled = inv_freq / scaling_factor
    lhs = wavelen - low_freq_wavelen + 1
    gt_mask = torch.where(lhs > 1.0, 1.0, 0.0)
    inv_freq_llama += inv_freq_scaled * gt_mask

    # otherwise: interpolate between the two, using a smooth factor
    smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    inv_freq_rest = (1 - smooth) * inv_freq_scaled + smooth * inv_freq
    rest_mask = torch.ones_like(inv_freq) - lt_mask - gt_mask
    inv_freq_llama += inv_freq_rest * rest_mask

    return inv_freq_llama


class RotaryPositionEmbeddingHelper:
    def __init__(
        self,
        max_position_embeddings,
        rotary_dim,
        base=10000,
        scaling_factor=1.0,
        scaling_type="linear",
        scaling_extra_args=None,
        rel_distance_mode=RopeRelDistanceMode.default,
        rel_distance_extra_args=None,
        fold_rope_consts=False,
        constant_pos_embedding=None,
    ):
        super(RotaryPositionEmbeddingHelper, self).__init__()
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.base = base
        self.scaling_factor = scaling_factor

        self.scaling_type = scaling_type
        self.scaling_extra_args = scaling_extra_args

        self.fold_rope_consts = fold_rope_consts
        self.constant_pos_embedding = constant_pos_embedding
        if constant_pos_embedding is not None and fold_rope_consts:
            raise ValueError(
                "fold_rope_consts should be False when constant_pos_embedding is enabled."
            )

        self._rope_cache = {}
        self.swa_mask, self.swa_complement_mask = None, None

        self.scaling_type = self.scaling_type.lower()
        if scaling_type not in ["linear", "yarn", "llama3", "longrope"]:
            raise ValueError(
                f"Position scaling type {self.scaling_type} not supported! Use 'linear', 'yarn', or 'llama3' or 'longrope"
            )

        self.rel_distance_mode = rel_distance_mode
        if isinstance(rel_distance_mode, str):
            self.rel_distance_mode = RopeRelDistanceMode.__members__.get(
                rel_distance_mode, None
            )

        self.rel_distance_extra_args = rel_distance_extra_args
        if self.rel_distance_extra_args is None:
            self.rel_distance_extra_args = dict()

        if self.rel_distance_mode is None:
            raise ValueError(
                f"Relative distance mode should be one of "
                f"{tuple(RopeRelDistanceMode.__members__.keys())}, but got {self.rel_distance_mode}"
            )

        # YaRN specific params
        if scaling_extra_args is None:
            scaling_extra_args = dict()
        if scaling_type == "yarn":
            self.extrapolation_factor = scaling_extra_args.get(
                "extrapolation_factor", 1.0
            )
            self.original_max_position_embeddings = scaling_extra_args.get(
                "original_max_position_embeddings"
            )
            self.beta_fast = scaling_extra_args.get("beta_fast", 32)
            self.beta_slow = scaling_extra_args.get("beta_slow", 1)
            self.mscale_a = scaling_extra_args.get("mscale_a", 0.1)
            self.mscale_b = scaling_extra_args.get("mscale_b", 1.0)

    @property
    def is_rel_distance_default(self):
        return self.rel_distance_mode == RopeRelDistanceMode.default

    def _constant_image_pos(self, x, t, constant_pos_mask):
        constant_pos_mask = constant_pos_mask[:, :, None, None].broadcast_to(
            x.shape
        )
        # TODO: only support two modalities, the values in `constant_pos_mask`
        # are either 0 or 1.
        image_mask = 1 - constant_pos_mask
        image_pos_id = constant_pos_mask * self.constant_pos_embedding
        t = t * image_mask  # Mask image portion to 0
        t = t + image_pos_id  # set pos id of image portion to the constant
        return t

    def _create_fixed_pos_emb(
        self,
        x,
        offset,
        position_ids,
        rope_cache_tag,
        constant_pos_mask,
        fold_const,
    ):
        assert (
            self.max_position_embeddings >= x.shape[1] + offset
        ), "RoPE requires max position embeddings ({}) >= sequence length ({}) + offset ({})".format(
            self.max_position_embeddings,
            x.shape[1],
            offset,
        )
        assert (self.constant_pos_embedding is None) or (
            self.constant_pos_embedding is not None
            and constant_pos_mask is not None
        ), "constant_pos_embedding is enabled, but 'constant_pos_mask' is not provided."

        if rope_cache_tag is None:
            rope_cache_tag = "default"

        # Cache values based on the tag, offset, input shape and dtype
        cache_key = (rope_cache_tag, x.shape, x.dtype, offset)

        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        import cerebras.pytorch as cstorch

        device = "cpu" if fold_const and cstorch.use_cs() else x.device

        x_shape = x.shape
        if fold_const:
            # Broadcast later to keep the serialized constants small.
            x_shape = x[0:1, :, 0:1, :].shape  # [1, seq_len, 1, head_dim]

        inv_freq = _compute_default_inv_freq(
            x_shape, self.rotary_dim, self.base, device, fold_const=fold_const
        )

        if position_ids is None:
            t = torch.arange(
                offset,
                x.shape[1] + offset,
                device=device,
                dtype=torch.float32,
            )
        else:
            if (
                position_ids.dim() not in (1, 2)
                or position_ids.shape[-1] != self.max_position_embeddings
                or position_ids.dim() == 2
                and position_ids.shape[0] != x.shape[0]
            ):
                raise ValueError(
                    f"position_ids tensor passed to 'rotate_tensor' method of RoPE helper "
                    f"should be 1- or 2-dimensional with last dim size {self.max_position_embeddings} "
                    f"but got shape {position_ids.shape}"
                )
            if fold_const:
                if (
                    torch.max(position_ids) >= self.max_position_embeddings
                    or torch.min(position_ids) < 0
                ):
                    raise ValueError(
                        f"Cannot use `position_ids` with values out of [0, {self.max_position_embeddings}) range with `fold_rope_consts: true` in RoPE"
                    )
            t = position_ids
        t = t[..., None, None].broadcast_to(x_shape)

        if (
            constant_pos_mask is not None
            and self.constant_pos_embedding is not None
        ):
            t = self._constant_image_pos(x, t, constant_pos_mask)

        self.mscale = None
        if self.scaling_type == "linear":
            self.mscale = 1.0
            t = t / self.scaling_factor
        elif self.scaling_type == "yarn":
            inv_freq_extrapolation = inv_freq
            inv_freq_interpolation = inv_freq / self.scaling_factor
            low, high = _yarn_find_correction_range(
                self.beta_fast,
                self.beta_slow,
                self.rotary_dim,
                self.base,
                self.original_max_position_embeddings,
            )
            inv_freq_mask = (
                1 - _yarn_linear_ramp_mask(x_shape, low, high, device)
            ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )
            self.mscale = _yarn_get_mscale(
                self.scaling_factor, self.mscale_a, self.mscale_b
            )
        elif self.scaling_type == "llama3":
            self.mscale = 1.0
            inv_freq = _llama_3_update_inv_freq(
                inv_freq,
                scaling_factor=self.scaling_factor,
                **self.scaling_extra_args,
            )
        elif self.scaling_type == "longrope":
            inv_freq, self.mscale = self._longrope_computations(
                x_shape, self.scaling_extra_args, device
            )

        sinusoid_inp = t * inv_freq

        sin, cos = (
            torch.sin(sinusoid_inp) * self.mscale,
            torch.cos(sinusoid_inp) * self.mscale,
        )

        if x.shape[-1] != self.rotary_dim:
            # TODO: Delete padding rewrites.
            rotary_mask = torch.arange(
                x.shape[3], device=device, dtype=torch.float32
            ).broadcast_to(x_shape) - (self.rotary_dim - 1)
            sin = torch.where(rotary_mask <= 0, sin, 0)
            cos = torch.where(rotary_mask < 1, cos, 1)

        # constant folding is performed.
        if fold_const:
            sin, cos = cstorch.make_constant(sin), cstorch.make_constant(cos)
            self._rope_cache[cache_key] = sin, cos

        return sin, cos

    def rotate_tensor(
        self,
        x,
        offset=0,
        constant_pos_mask=None,
        position_ids=None,
        rope_cache_tag=None,
    ):
        assert (
            len(x.shape) == 4
        ), "Tensor should be of shape [batch_size, seq_length, num_heads, head_dim] !"
        if position_ids is no_rope_modification_pos_id:
            return x

        @wafer_instruction_tuning(force_recompute_opt=True)
        def apply_otf_rope(
            x, offset, position_ids, constant_pos_mask, rope_cache_tag
        ):
            sin, cos = self._create_fixed_pos_emb(
                x,
                offset,
                position_ids,
                rope_cache_tag,
                constant_pos_mask=constant_pos_mask,
                fold_const=False,
            )
            return rotary_pos_emb(x, sin, cos)

        def apply_folded_rope(
            x, offset, position_ids, constant_pos_mask, rope_cache_tag
        ):
            sin, cos = self._create_fixed_pos_emb(
                x,
                offset,
                position_ids,
                rope_cache_tag,
                constant_pos_mask=constant_pos_mask,
                fold_const=True,
            )
            return rotary_pos_emb(x, sin, cos)

        if self.fold_rope_consts:
            return apply_folded_rope(
                x,
                offset,
                position_ids,
                constant_pos_mask=constant_pos_mask,
                rope_cache_tag=rope_cache_tag,
            )
        else:
            return apply_otf_rope(
                x,
                offset,
                position_ids,
                constant_pos_mask=constant_pos_mask,
                rope_cache_tag=rope_cache_tag,
            )

    def _longrope_computations(
        self, x_shape, rope_kwargs, device
    ) -> Tuple["torch.Tensor", float]:
        """
        Compute longrope based parameters calculating inv_freq
        and attention scaling factors
        Input args:
            rope_kwargs : Extra longrope params picked from this dict
        Output:
            Tuple of tensors for frequencies and scale factors generated
        """

        base = self.base
        dim = self.rotary_dim
        factor = self.scaling_factor
        head_dim = x_shape[-1]

        assert (
            "original_max_position_embeddings" in rope_kwargs
        ), "LongRope needs original context size"
        self.original_max_position_embeddings = rope_kwargs.get(
            "original_max_position_embeddings"
        )
        max_position_embeddings = self.original_max_position_embeddings
        expanded_max_position_embeddings = self.max_position_embeddings
        factor = expanded_max_position_embeddings / max_position_embeddings

        # Compute the inverse frequencies -- scaled based on the target sequence length
        if expanded_max_position_embeddings > max_position_embeddings:
            assert (
                "long_factor" in rope_kwargs
            ), "LongRope needs long_factor scaling params in config"
            rescale_factors = rope_kwargs["long_factor"]
            if "long_mscale" in rope_kwargs:
                self.mscale = rope_kwargs["long_mscale"]

        else:
            assert (
                "short_factor" in rope_kwargs
            ), "LongRope needs short_factor scaling params in config"
            rescale_factors = rope_kwargs["short_factor"]
            if "short_mscale" in rope_kwargs:
                self.mscale = rope_kwargs["short_mscale"]

        rescale_factors = [v for v in rescale_factors for _ in (0, 1)]  # repeat
        rescale_factors += [0] * (head_dim - dim)  # pad zeros, masked out later
        if cstorch.use_cs() and self.fold_rope_consts is False:
            ext_factors = torch.tensor(
                rescale_factors, dtype=torch.float32, device="cpu"
            )
            ext_factors = cstorch.make_constant(ext_factors)
        else:
            ext_factors = torch.tensor(
                rescale_factors, dtype=torch.float32, device=device
            )

        # Sets the attention factor as suggested in the paper
        if self.mscale is None:
            # Check if the config is using attention_factor param instead
            if "attention_factor" in rope_kwargs:
                self.mscale = rope_kwargs["attention_factor"]
            elif factor <= 1.0:
                self.mscale = 1.0
            else:
                self.mscale = math.sqrt(
                    1 + math.log(factor) / math.log(max_position_embeddings)
                )

        dim_range = torch.arange(
            head_dim, device=device, dtype=torch.float32
        ).broadcast_to(x_shape)
        dim_pairs = dim_range / 2
        dim_pairs = dim_pairs.to(torch.int16).to(torch.float32)
        inv_freq = 1.0 / (ext_factors * base ** (2 * dim_pairs / dim))

        return inv_freq, self.mscale

    def get_attn_region_masks(self, shape, device):
        """Returns a sliding window causal mask with a complement mask
        to combine logit values inside and outside the sliding window region
        """
        if "rope_local_window_size" not in self.rel_distance_extra_args:
            raise ValueError(
                f"Attempt to create RoPE relative distance region masks but "
                f"'rope_local_window_size' is not specified"
            )

        if self.swa_mask is not None:
            return self.swa_mask, self.swa_complement_mask

        batch_size, num_heads, seq_len, _ = shape
        sliding_window_length = self.rel_distance_extra_args[
            "rope_local_window_size"
        ]

        self.swa_mask, self.swa_complement_mask = (
            create_sliding_window_mask_with_complement(
                batch_size=batch_size,
                num_heads=num_heads,
                tgt_seq_length=seq_len,
                sliding_window_length=sliding_window_length,
                device=device,
            )
        )
        return self.swa_mask, self.swa_complement_mask

    def get_distant_pos_id_vectors(self, device):
        """Depending on whether capped relative distances (LM-Infinite style)
        or grouped relative distances (Self-Extend style) are used returns 1D tensors
        of position IDs for queries/keys to compute relative distances
        outside the sliding window region

        Returns: a tuple of (q_pos_id, k_pos_id) 1D tensors of shape [max_position_embeddings]
        """
        if self.rel_distance_mode == RopeRelDistanceMode.default:
            raise ValueError(
                "Invalid attempt to create position ID vectors for "
                "out-of-sliding-window logits when rel_distance_mode='default'. "
                "rel_distance_mode value should be 'capped' or 'grouped'."
            )

        import cerebras.pytorch as cstorch

        device = "cpu" if cstorch.use_cs() and self.fold_rope_consts else device

        if self.rel_distance_mode == RopeRelDistanceMode.capped:
            if "rope_local_window_size" not in self.rel_distance_extra_args:
                raise ValueError(
                    "'rope_local_window_size' has to be specified for rel_distance_mode='capped'"
                )

            q_pos_id = (
                torch.ones(
                    self.max_position_embeddings,
                    device=device,
                    dtype=torch.float32,
                )
                * self.rel_distance_extra_args["rope_local_window_size"]
            )
            k_pos_id = no_rope_modification_pos_id
        elif self.rel_distance_mode == RopeRelDistanceMode.grouped:
            if (
                "rope_local_window_size" not in self.rel_distance_extra_args
                or "rope_group_size" not in self.rel_distance_extra_args
            ):
                raise ValueError(
                    "Both 'rope_local_window_size' and 'rope_group_size' have to be "
                    "specified for rel_distance_mode='grouped'"
                )

            max_rel_distance = self.rel_distance_extra_args[
                "rope_local_window_size"
            ]
            group_size = self.rel_distance_extra_args["rope_group_size"]
            constant_dist = max_rel_distance - max_rel_distance // group_size
            k_pos_id = (
                torch.arange(
                    self.max_position_embeddings,
                    device=device,
                    dtype=torch.float32,
                )
                // group_size
            )
            q_pos_id = k_pos_id + constant_dist

        return q_pos_id, k_pos_id
