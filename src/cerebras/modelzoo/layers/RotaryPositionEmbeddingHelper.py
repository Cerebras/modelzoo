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

import torch


def _duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dims = m.shape[:-1]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(*dims, -1)  # reshape into a matrix, interleaving the copy
    return m


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


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
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
    # Ported from https://github.com/huggingface/transformers/blob/bc2adb0112b6677b0dfb4105c74570a0f92183eb/src/transformers/modeling_rope_utils.py#L298-L339
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    new_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scaling_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (
                original_max_position_embeddings / wavelen - low_freq_factor
            ) / (high_freq_factor - low_freq_factor)
            new_freqs.append(
                (1 - smooth) * freq / scaling_factor + smooth * freq
            )
    inv_freq = torch.tensor(
        new_freqs, dtype=inv_freq.dtype, device=inv_freq.device
    )
    return inv_freq


class RotaryPositionEmbeddingHelper:
    def __init__(
        self,
        max_position_embeddings,
        rotary_dim,
        base=10000,
        scaling_factor=1.0,
        scaling_type="linear",
        scaling_extra_args=None,
        pad_fixed_pos_emb=False,
        constant_pos_embedding=None,
    ):
        super(RotaryPositionEmbeddingHelper, self).__init__()
        self.max_position_embeddings = max_position_embeddings
        self.rotary_dim = rotary_dim
        self.base = base
        self.sin_cached = None
        self.cos_cached = None
        self.offset = 0
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type
        self.scaling_extra_args = scaling_extra_args
        self.pad_fixed_pos_emb = pad_fixed_pos_emb
        self.constant_pos_embedding = constant_pos_embedding
        # force pad_fixed_pos_emb to True when constant_pos_embedding is enabled.
        if constant_pos_embedding is not None:
            self.pad_fixed_pos_emb = True

        self.scaling_type = self.scaling_type.lower()
        if scaling_type not in ["linear", "yarn", "llama3"]:
            raise ValueError(
                f"Position scaling type {self.scaling_type} not supported! Use 'linear', 'yarn', or 'llama3'."
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

        if self.scaling_type != "linear" and pad_fixed_pos_emb:
            raise ValueError(
                f"Position scaling type {self.scaling_type} does not work with pad_fixed_pos_emb=True"
            )

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

    def _create_padded_fixed_pos_emb(self, x, offset, constant_pos_mask=None):
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

        inv_freq_arange = (
            torch.arange(
                0, x.shape[3] / 2, 0.5, device=x.device, dtype=torch.float32
            )
            .broadcast_to(x.shape)
            .floor()
            * 2
        )
        inv_freq = 1.0 / (self.base ** (inv_freq_arange / self.rotary_dim))

        t = torch.arange(
            offset, x.shape[1] + offset, device=x.device, dtype=torch.float32
        )[:, None, None].broadcast_to(x.shape)

        if (
            constant_pos_mask is not None
            and self.constant_pos_embedding is not None
        ):
            t = self._constant_image_pos(x, t, constant_pos_mask)

        if self.scaling_type == "linear":
            t = t / self.scaling_factor

        sinusoid_inp = t * inv_freq

        sin, cos = (
            torch.sin(sinusoid_inp).to(x.dtype),
            torch.cos(sinusoid_inp).to(x.dtype),
        )

        if self.rotary_dim == x.shape[3]:
            return sin, cos

        rotary_arange = (
            torch.arange(
                x.shape[3], device=x.device, dtype=torch.float32
            ).broadcast_to(x.shape)
            - self.rotary_dim
        )

        rotary_mask = torch.clamp(-rotary_arange, min=0, max=1).to(x.dtype)

        sin_masked = sin * rotary_mask
        cos_masked = cos * rotary_mask - (rotary_mask - 1)
        return sin_masked, cos_masked

    def _create_fixed_pos_emb(self, x, offset):
        if self.sin_cached is not None and self.cos_cached is not None:
            if self.offset == offset:
                return self.sin_cached, self.cos_cached
        self.offset = offset

        import cerebras.pytorch as cstorch

        device = "cpu" if cstorch.use_cs() else x.device

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.rotary_dim, 2, device=device)
                / self.rotary_dim
            )
        )
        t = torch.arange(self.max_position_embeddings, device=device)

        self.mscale = 1.0

        if self.scaling_type == "linear":
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
                1
                - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2)
                .float()
                .to(device)
            ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
            inv_freq = (
                inv_freq_interpolation * (1 - inv_freq_mask)
                + inv_freq_extrapolation * inv_freq_mask
            )
            self.mscale = _yarn_get_mscale(
                self.scaling_factor, self.mscale_a, self.mscale_b
            )
        elif self.scaling_type == "llama3":
            inv_freq = _llama_3_update_inv_freq(
                inv_freq,
                scaling_factor=self.scaling_factor,
                **self.scaling_extra_args,
            )

        sinusoid_inp = torch.einsum(
            "i , j -> i j",
            t,
            inv_freq,
        )

        sin, cos = (
            (torch.sin(sinusoid_inp) * self.mscale).to(x.dtype),
            (torch.cos(sinusoid_inp) * self.mscale).to(x.dtype),
        )

        sin, cos = map(_duplicate_interleave, (sin, cos))

        def slice_at_offset(t):
            return t[None, offset : x.shape[1] + offset, None, :]

        assert (
            self.max_position_embeddings >= x.shape[1] + offset
        ), "RoPE requires max position embeddings ({}) >= sequence length ({}) + offset ({})".format(
            self.max_position_embeddings,
            x.shape[1],
            offset,
        )
        sin, cos = map(slice_at_offset, (sin, cos))

        # For cs runs, wrap the sin and cos matrices in xla_literal so that
        # constant folding is performed.
        self.sin_cached = cstorch.make_constant(sin)
        self.cos_cached = cstorch.make_constant(cos)
        return self.sin_cached, self.cos_cached

    def _apply_rotary_pos_emb(
        self, x, real_seq_length, offset=0, pad=False, constant_pos_mask=None
    ):
        def rotate_every_two(x):
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            # in einsum notation: rearrange(x, '... d j -> ... (d j)')
            return x.flatten(-2)

        if pad:
            sin, cos = self._create_padded_fixed_pos_emb(
                x, offset, constant_pos_mask=constant_pos_mask
            )
        else:
            sin, cos = self._create_fixed_pos_emb(x, offset)

        # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
        return (x * cos) + (rotate_every_two(x) * sin)

    def _rotate_sliced_tensor(self, x, real_seq_length, offset=0):
        x_rotary = x[:, :, :, : self.rotary_dim]
        x_pass = x[:, :, :, self.rotary_dim :]
        x_rotated = self._apply_rotary_pos_emb(
            x_rotary, real_seq_length, offset=offset, pad=False
        )
        x = torch.cat([x_rotated, x_pass], dim=-1)
        return x

    def rotate_tensor(
        self, x, real_seq_length, offset=0, constant_pos_mask=None
    ):
        assert (
            len(x.shape) == 4
        ), "Tensor should be of shape [batch_size, seq_length, num_heads, head_dim] !"
        if self.pad_fixed_pos_emb:
            return self._apply_rotary_pos_emb(
                x,
                real_seq_length,
                offset=offset,
                pad=True,
                constant_pos_mask=constant_pos_mask,
            )
        return self._rotate_sliced_tensor(x, real_seq_length, offset=offset)
