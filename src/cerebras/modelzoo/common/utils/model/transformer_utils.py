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

import logging
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional, Tuple

# Doesn't work with `multiprocessing`, because Pickle cannot handle closures.
# `multiprocess` uses Dill instead, which works for closures.
import multiprocess as mp
import torch
from typing_extensions import Self

import cerebras.pytorch as cstorch
from cerebras.modelzoo.common.half_dtype import maybe_to_half_dtype


def _extend_mask_to_shape_of_4(mask: torch.Tensor):
    assert len(mask.shape) in [
        2,
        3,
        4,
    ], "Masks with shape 2, 3, 4 are supported"
    if len(mask.shape) == 2:
        # [batch_size, target_seq_len]
        mask = mask[:, None, None, :]
    elif len(mask.shape) == 3:
        # [batch_size, src_seq_len, target_seq_len]
        mask = mask[:, None, :, :]
    else:
        # len(key_padding_mask.shape) == 4
        # [batch_size, num_heads, src_seq_len, target_seq_len]
        mask = mask
    return mask


def replace_with_zero_and_neg_inf(
    mask: torch.Tensor, neg_inf: bool = True
) -> torch.Tensor:
    """Replace the values in mask tensor with 0 and -inf.

    Nonpositive values are replaced with 0. Positive values are replaced with -inf.

    Args:
        mask: Mask tensor with nonpositive values indicating tokens to attend to and
            positive values for tokens to ignore.
        neg_inf: Use negative infinity instead of one in the resulting mask.
            defaults to True.
    Returns:
        The mask tensor with values replaced.
    """
    mask_val = torch.tensor(float("-inf") if neg_inf else 1, dtype=mask.dtype)
    return torch.where(mask > 0, mask_val, 0)


def make_key_padding_mask_broadcastable(
    key_padding_mask: torch.Tensor,
    dtype=None,
    revert_mask: bool = True,
    use_neg_inf: bool = True,
):
    """Makes broadcastable key_padding masks so that padding tokens are ignored.

    Args:
        key_padding_mask (torch.Tensor): Key padding mask with shape in [2,3,4], with entry values either 1 or 0.
        dtype (torch.dtype): Dtype of the resulting mask.
        revert_mask (bool): Whether to flip the 1's and 0's of the attention mask, default to True.
        use_neg_inf (bool): Use negative infinity instead of one in the resulting mask, default to True.

    Returns:
        The key padding mask of shape [batch_size, num_heads, src_seq_len, target_seq_len],
        with broadcast dimensions set to 1.
    """

    if dtype is None:
        dtype = torch.float16
    key_padding_mask = key_padding_mask.to(dtype=dtype)

    # Since `key_padding_mask` is passed as `1.0` for positions we want to
    # attend and `0.0` for masked positions, this operation will "invert"
    # the mask due to the "negative infinity" scaling at the end.
    if revert_mask:
        key_padding_mask = 1.0 - key_padding_mask

    if use_neg_inf:
        key_padding_mask = replace_with_zero_and_neg_inf(key_padding_mask)

    extended_key_padding_mask = _extend_mask_to_shape_of_4(key_padding_mask)

    return extended_key_padding_mask


# Experimental FlexAttention-like API.
from cerebras.modelzoo.common.utils.model.attn_mask_ranges import (
    AttentionMaskRangesInfo,
)


@dataclass(frozen=True)
class SparseAttentionMask:
    '''
    A mask value of `True` means the token should be atteneded to, `False` should be masked.
    In the end, the mask will be converted to a float representation which, exponentiated,
    will yield the correct probability score. For instance, the float value for `True`
    will be 0 - generating probability 1 - and the value for `False` will be -inf -
    genenerating probability 0.
    '''

    mask_mod: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    batch_size: int = 0
    num_heads: int = 0
    tgt_seq_length: int = 0
    max_num_ranges: int = 256
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32
    use_neg_inf: bool = True

    def gen_qk(self) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_shape = (
            self.batch_size,
            self.num_heads,
            self.tgt_seq_length,
            self.tgt_seq_length,
        )
        seq_range = torch.arange(
            self.tgt_seq_length, device=self.device, dtype=torch.float32
        )
        q_range = seq_range[:, None].broadcast_to(mask_shape)
        k_range = seq_range[None, :].broadcast_to(mask_shape)
        return q_range, k_range

    def get_mask(self, use_probabilities=True) -> torch.Tensor:
        mask = self.mask_mod(*self.gen_qk())
        zero = torch.tensor(0, dtype=torch.float32)
        if not use_probabilities:
            return mask.to(self.dtype)
        if 'lazy' in str(self.device):
            return torch.where(
                mask != zero,
                zero,
                torch.tensor(
                    float('-inf') if self.use_neg_inf else -1,
                    dtype=self.dtype,
                    device=self.device,
                ),
            )
        else:
            return torch.where(
                mask,
                0.0,
                torch.tensor(
                    float('-inf') if self.use_neg_inf else -1,
                    dtype=self.dtype,
                    device=self.device,
                ),
            )

    @cached_property
    def mask_tensor(self):
        return self.get_mask()

    @cached_property
    def sparsity_annotation(self):
        return self.get_ranges()

    def and_mask(self, other: Self) -> Self:
        def and_mask_mod(q, k):
            if 'lazy' in str(q.device):
                # We first flip the representation from (0->False, 1->True) to (0->True, 1->False),
                # then AND using float addition (which can happen on wafer), then flip back.
                one = torch.tensor(1.0, dtype=self.dtype)
                float_mask_a = -(
                    self.mask_mod(q, k).to(self.dtype) - one
                )  # invert
                float_mask_b = -(
                    other.mask_mod(q, k).to(self.dtype) - one
                )  # invert
                float_and_mask = -torch.maximum(
                    -(float_mask_a + float_mask_b), -one
                )  # take the minimum (i.e., clamp to [0,1]), still inverted
                return -(
                    float_and_mask - one
                )  # flip back to (0->False, 1->True)
            else:
                return torch.bitwise_and(
                    self.mask_mod(q, k), other.mask_mod(q, k)
                )

        return SparseAttentionMask(
            and_mask_mod,
            **{i: d for i, d in self.__dict__.items() if i != 'mask_mod'},
        )

    def or_mask(self, other: Self) -> Self:
        def or_mask_mod(q, k):
            if 'lazy' in str(q.device):
                # here we OR in the regular (non-inverted) representation using + and clamp to [0,1]
                one = torch.tensor(1.0, dtype=self.dtype)
                float_mask_a = self.mask_mod(q, k).to(self.dtype)
                float_mask_b = other.mask_mod(q, k).to(self.dtype)
                return -torch.maximum(-(float_mask_a + float_mask_b), -one)
            else:
                return torch.bitwise_or(
                    self.mask_mod(q, k), other.mask_mod(q, k)
                )

        return SparseAttentionMask(
            or_mask_mod,
            **{i: d for i, d in self.__dict__.items() if i != 'mask_mod'},
        )

    def not_mask(self) -> Self:
        def not_mask_mod(q, k):
            if 'lazy' in str(q.device):
                one = torch.tensor(1.0, dtype=self.dtype)
                return -(self.mask_mod(q, k).to(self.dtype) - one)
            else:
                return torch.bitwise_not(self.mask_mod(q, k))

        return SparseAttentionMask(
            not_mask_mod,
            **{i: d for i, d in self.__dict__.items() if i != 'mask_mod'},
        )

    # This generates the mask range annotation iteratively by materializing
    # `range_size` tiles of it at a time on CPU.
    def get_ranges(
        self, override_max_num_ranges_with=None, do_not_parallelize=False
    ):
        max_num_ranges = self.max_num_ranges
        if override_max_num_ranges_with is not None:
            max_num_ranges = override_max_num_ranges_with
        start_ranges = []
        end_ranges = []
        sin_begin = 0
        sout_begin = 0
        sin_end = self.tgt_seq_length
        sout_end = self.tgt_seq_length
        range_size = max(-(self.tgt_seq_length // -max_num_ranges), 1)
        q_range = torch.arange(range_size)[:, None].broadcast_to(
            [range_size, range_size]
        )
        k_range = q_range.T
        k_values = torch.arange(0, self.tgt_seq_length + 1, range_size)
        _k_values = k_values[:, None]
        use_parallelization = (
            False if do_not_parallelize else self.tgt_seq_length >= 2**15
        )  # 32k and up

        def analyze_tile(_q):
            tmp = _k_values.broadcast_to([k_values.shape[0], range_size])[
                :, None
            ].broadcast_to([k_values.shape[0], range_size, range_size])
            active = self.mask_mod(_q + q_range, tmp + k_range)
            begins = torch.nn.functional.pad(
                _k_values, (0, 1), mode='constant', value=_q
            )
            ends = torch.nn.functional.pad(
                _k_values, (0, 1), mode='constant', value=_q + range_size
            )
            mask = active.any(dim=(1, 2))
            padded = torch.nn.functional.pad(
                mask, (1, 0), mode='constant', value=False
            )
            i_begins = (mask & ~padded[:-1]).nonzero(as_tuple=False).squeeze()
            i_ends = ((~mask) & padded[:-1]).nonzero(as_tuple=False).squeeze()
            return begins[i_begins], ends[i_ends]

        def is_in_range(start, end):
            return (start_range < self.tgt_seq_length).all() and (
                end_range <= self.tgt_seq_length
            ).all()

        wrap = lambda x: x.tolist() if len(x.shape) > 1 else [x.tolist()]
        logging.info(
            f"Attention mask auto-sparsifier will run analysis "
            + (f"in parallel" if use_parallelization else "sequentially")
            + f" with tile size {range_size} for MSL {self.tgt_seq_length}."
        )
        if use_parallelization:
            with mp.Pool() as pool:
                for start_range, end_range in pool.map(
                    analyze_tile, range(0, self.tgt_seq_length + 1, range_size)
                ):
                    if is_in_range(start_range, end_range):
                        start_ranges += [wrap(start_range)]
                        end_ranges += [wrap(end_range)]
        else:
            for q in range(0, self.tgt_seq_length + 1, range_size):
                start_range, end_range = analyze_tile(q)
                if is_in_range(start_range, end_range):
                    start_ranges += [wrap(start_range)]
                    end_ranges += [wrap(end_range)]
        assert len(end_ranges) == len(start_ranges)
        logging.info("Attention mask auto-sparsifier finished analysis.")
        return AttentionMaskRangesInfo(
            start_ranges, end_ranges, self.tgt_seq_length
        )


def create_mask_using_flex_api(
    batch_size: int,
    num_heads: int,
    tgt_seq_length: int,
    attention_span: Optional[torch.Tensor] = None,
    sliding_window_length: Optional[int] = None,
    num_sink_tokens: Optional[int] = None,
    attention_vertical_column_spacing: Optional[int] = None,
    attention_vertical_column_width: Optional[int] = None,
    attention_chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    use_neg_inf: bool = True,
):
    causal_mask = SparseAttentionMask(
        lambda q, k: q >= k,
        batch_size=batch_size,
        num_heads=num_heads,
        tgt_seq_length=tgt_seq_length,
        device=device,
        dtype=dtype,
        use_neg_inf=use_neg_inf,
    )
    attn_mask = causal_mask

    if attention_chunk_size is not None:

        def attn_chunk(q, k):
            return k < float(attention_chunk_size) * maybe_to_half_dtype(
                (q / attention_chunk_size).short()
            )

        attn_mask = attn_mask.and_mask(
            SparseAttentionMask(attn_chunk).not_mask()
        )

    if sliding_window_length is not None:

        def sliding_window(q, k):
            return q - k < sliding_window_length

        attn_mask = attn_mask.and_mask(SparseAttentionMask(sliding_window))
        if num_sink_tokens:
            attn_mask = attn_mask.or_mask(
                SparseAttentionMask(lambda q, k: k < num_sink_tokens)
            ).and_mask(causal_mask)

    if attention_vertical_column_spacing is not None:
        for i in range(
            attention_vertical_column_spacing - 1,
            tgt_seq_length,
            attention_vertical_column_spacing,
        ):
            left_span = SparseAttentionMask(lambda q, k, i=i: i - 1 < k)
            right_span = SparseAttentionMask(
                lambda q, k, i=i: k < i + attention_vertical_column_width
            )
            attn_mask = attn_mask.or_mask(
                causal_mask.and_mask(left_span.and_mask(right_span))
            )
    return attn_mask


def create_broadcasted_autoregressive_mask(
    batch_size: int,
    num_heads: int,
    tgt_seq_length: int,
    attention_span: Optional[torch.Tensor] = None,
    attention_sliding_window_length: Optional[int] = None,
    attention_sink_tokens: Optional[int] = None,
    attention_vertical_column_spacing: Optional[int] = None,
    attention_vertical_column_width: Optional[int] = None,
    attention_chunk_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    use_neg_inf: bool = True,
    use_experimental_flex_api: bool = False,
):
    """Create broadcasted causal attention mask optionally with VSL masking.

    For VSL, `attention_span` is required and past tokens out of the current sequence
    are additionally masked.

    For sliding window, `attention_sliding_window_length` is required and cannot be used along with VSL.

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of heads.
        tgt_seq_length (int): Target sequence length.
        attention_span (torch.Tensor): Attention span of keys for VSL, has shape [batch_size, target_seq_len].
        attention_sliding_window_length (int): If specified, the current token would only attend the current
        token and attention_sliding_window_length previous tokens.
        attention_sink_tokens (int): Number of attention sink tokens to be used for StreamingLLM-style inference.
        attention_chunk_size (int): If specified, the attention mask will have a chunked pattern of `attention_chunk_size` length windows.
        device (torch.device): The device of the input to the model, used for causal mask creation.
        dtype (torch.dtype): Dtype of the resulting mask, default to torch.float16.
        use_neg_inf (bool): Use negative infinity instead of one in the resulting mask, default to True.
        use_experimental_flex_api (bool): Use a FlexAttention-like API for mask construction (experimental).


    Returns:
        The attention mask of shape [batch_size, num_heads, tgt_seq_len, tgt_seq_len].
    """
    if (
        attention_span is not None
        and attention_sliding_window_length is not None
    ):
        raise ValueError(f"Sliding window used with VSL.")

    if (
        attention_chunk_size is not None
        and attention_sliding_window_length is not None
    ):
        raise ValueError(
            f"Chunked attention mask incompatible with sliding window."
        )

    if (
        attention_sink_tokens is not None
        and attention_sliding_window_length is None
    ):
        raise ValueError(
            f"got {attention_sink_tokens=} but {attention_sliding_window_length=}"
        )

    if (attention_vertical_column_spacing is None) ^ (
        attention_vertical_column_width is None
    ):
        raise ValueError("column spacing and width must both be specified")

    if attention_span is not None:
        return create_vsl_mask(
            attention_span=attention_span,
            num_heads=num_heads,
            is_causal=True,
            device=device,
            dtype=dtype,
            use_neg_inf=use_neg_inf,
        )

    if use_experimental_flex_api:
        return create_mask_using_flex_api(
            batch_size,
            num_heads,
            tgt_seq_length,
            attention_span=attention_span,
            sliding_window_length=attention_sliding_window_length,
            num_sink_tokens=attention_sink_tokens,
            attention_vertical_column_spacing=attention_vertical_column_spacing,
            attention_vertical_column_width=attention_vertical_column_width,
            attention_chunk_size=attention_chunk_size,
            device=device,
            dtype=dtype,
            use_neg_inf=use_neg_inf,
        )

    mask_shape = (
        batch_size,
        num_heads,
        tgt_seq_length,
        tgt_seq_length,
    )
    seq_range = torch.arange(tgt_seq_length, device=device, dtype=torch.float32)
    q_range = seq_range[:, None].broadcast_to(mask_shape)
    k_range = seq_range[None, :].broadcast_to(mask_shape)
    diff = q_range - k_range

    # We want mask construction written as supported float ops.
    #
    # For example,
    #   causal_mask = (diff < 0) | (diff > attention_span)
    # can be written as
    #   max(diff - attention_span, 0) - min(diff, 0)
    # for integer tensors diff, attention_span.

    # Use -diff directly for pure triangular mask.
    attention_mask = -diff

    if attention_chunk_size is not None:
        attention_span = create_chunked_attention_span(
            batch_size,
            tgt_seq_length,
            attention_chunk_size,
            device=attention_mask.device,
        )
        zero = torch.tensor(0, dtype=torch.float32)
        attention_mask = torch.maximum(
            diff - attention_span[:, None, None, :], zero
        )
        attention_mask += torch.maximum(-diff, zero)

    if attention_sliding_window_length is not None:
        # Set upper triangular part to positive integers.
        zero = torch.tensor(0, dtype=torch.float32)
        attention_mask = torch.maximum(attention_mask, zero)

        window_span = torch.tensor(
            attention_sliding_window_length - 1,
            device=device,
            dtype=torch.float32,
        ).broadcast_to(mask_shape)
        if attention_sink_tokens:
            offset = tgt_seq_length - attention_sink_tokens
            window_span = torch.where(
                k_range + offset >= tgt_seq_length, window_span, tgt_seq_length
            )
        attention_mask += torch.maximum(diff - window_span, zero)

    if attention_vertical_column_spacing is not None:
        zero = torch.tensor(0, dtype=torch.float32)
        attention_mask = torch.maximum(attention_mask, zero)
        for i in range(
            attention_vertical_column_spacing - 1,
            tgt_seq_length,
            attention_vertical_column_spacing,
        ):
            window_span = torch.tensor(
                i,
                device=device,
                dtype=torch.float32,
            ).broadcast_to(mask_shape)
            # first construct the column span
            right_span = torch.where(
                k_range <= window_span, tgt_seq_length, window_span
            )
            left_span = torch.where(
                k_range + attention_vertical_column_width <= window_span,
                tgt_seq_length,
                window_span,
            )
            col_part_one = torch.where(
                0.0 < right_span - left_span, float(tgt_seq_length), 0.0
            )
            col_part_two = torch.where(
                0.0 < col_part_one,
                float(attention_vertical_column_width),
                0.0,
            )
            col_span = col_part_one + col_part_two
            # then select the lower triangular part of the columns
            final_span = torch.maximum(diff - col_span, zero)
            upper_mask = torch.where(diff < 0.0, attention_mask, 0.0)
            attention_mask = (
                -torch.maximum(-final_span, -attention_mask) + upper_mask
            )
    attention_mask = attention_mask.to(dtype)
    attention_mask = replace_with_zero_and_neg_inf(attention_mask, use_neg_inf)

    return attention_mask


def create_vsl_mask(
    attention_span: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    is_causal: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    use_neg_inf: bool = True,
):
    """Creates a VSL attention mask.

    E.g. for a VSL sequence that consists of a sequence of length 3 and a sequence of length 2, then the causal mask is:
    ```
    [
        [0,     -inf,   -inf,   -inf,   -inf],
        [0,     0,      -inf,   -inf,   -inf],
        [0,     0,      0,      -inf,   -inf],
        [-inf,  -inf,   -inf,   0,      -inf,],
        [-inf,  -inf,   -inf,   0,      0],
    ]
    ```
    whereas the non-causal mask is:
    ```
    [
        [0,     0,      0,      -inf,   -inf],
        [0,     0,      0,      -inf,   -inf],
        [0,     0,      0,      -inf,   -inf],
        [-inf,  -inf,   -inf,   0,      0],
        [-inf,  -inf,   -inf,   0,      0],
    ]
    ```

    Args:
        attention_span (torch.Tensor): Attention span of keys for VSL, has shape [batch_size, seq_len].
        position_ids (torch.Tensor): Optional position id of keys for VSL, has shape [batch_size, seq_len].
        num_heads (int): Number of heads.
        is_causal (bool): The mask is causal or not (bidirectional), default to True.
        device (torch.device): The device of the input to the model, used for causal mask creation.
        dtype (torch.dtype): Dtype of the resulting mask, default to torch.float16.
        use_neg_inf (bool): Use negative infinity instead of one in the resulting mask, default to True.

    Returns:
        The attention mask of shape [batch_size, num_heads, seq_len, seq_len].
    """
    if not is_causal and position_ids is None:
        raise ValueError(f"Creating bidirectional mask requires position_ids.")

    batch_size, seq_len = attention_span.shape
    mask_shape = (batch_size, num_heads, seq_len, seq_len)
    seq_range = torch.arange(seq_len, device=device, dtype=torch.float32)
    q_range = seq_range[:, None].broadcast_to(mask_shape)
    k_range = seq_range[None, :].broadcast_to(mask_shape)
    diff = q_range - k_range

    # We want mask construction written as supported float ops.
    #
    # For example,
    #   causal_mask = (diff < 0) | (diff > attention_span)
    # can be written as
    #   max(diff - attention_span, 0) - min(diff, 0)
    # for integer tensors diff, attention_span.

    # Set out of sequence VSL regions to positive integers.
    zero = torch.tensor(0, dtype=torch.float32)
    attention_mask = torch.maximum(
        diff - attention_span[:, None, None, :], zero
    )

    if is_causal:
        # Set upper triangular part to positive integers.
        attention_mask += torch.maximum(-diff, zero)
    else:
        # Set out of sequence VSL regions in the upper triangular part to positive integers.
        attention_mask += torch.maximum(
            -diff - position_ids[:, None, None, :], zero
        )

    attention_mask = attention_mask.to(dtype)
    attention_mask = replace_with_zero_and_neg_inf(attention_mask, use_neg_inf)

    return attention_mask


def create_2D_autoregressive_mask(
    src_sequence_length: int,
    target_sequence_length: int,
    dtype=None,
    device=None,
):
    """Creates a reverted autoregressive (upper triangular) mask where the 0s refers to the tokens
        to attend to and 1s refer to the tokens that are skipped.

    Args:
        batch_size (int): Batch size.
        src_sequence_length (int): Sequence length of the source (num query vectors).
        target_sequence_length (int): Sequence length of the target (num key vectors).
        dtype (torch.dtype): Dtype of the resulting mask.
        device: (torch.device): The device of the input to the model, used for causal mask creation.

    Returns:
        The causal mask of shape [src_seq_len, target_seq_len].
    """
    if dtype is None:
        dtype = torch.float16
    causal_mask = torch.triu(
        torch.ones(
            (src_sequence_length, target_sequence_length),
            device=device,
            dtype=dtype,
        ),
        diagonal=1,
    )
    return causal_mask


def create_2D_full_mask(
    src_sequence_length: int,
    target_sequence_length: int,
    dtype=None,
    device=None,
):
    """Create autoregressive (triangular) mask.

    Args:
        batch_size (int): Batch size.
        src_sequence_length (int): Sequence length of the source (num query vectors).
        target_sequence_length (int): Sequence length of the target (num key vectors).
        dtype (torch.dtype): Dtype of the resulting mask.
        device: (torch.device): The device of the input to the model, used for causal mask creation.

    Returns:
        The causal mask of shape [src_seq_len, target_seq_len].
    """
    if dtype is None:
        dtype = torch.float16
    full_mask = torch.ones(
        (src_sequence_length, target_sequence_length),
        device=device,
        dtype=dtype,
    )
    return full_mask


def make_sparse_mask_broadcastable(
    sparse_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
    dtype=None,
    device=None,
    revert_mask: bool = True,
    use_neg_inf: bool = True,
):
    """Create broadcastable sparse mask so that masked positions are ignored.

    Args:
        sparse_mask (torch.Tensor): Sparse mask with shape [src_seq_len, target_seq_len].
        key_padding_mask (torch.Tensor): Key padding mask with shape in [2,3,4].
        dtype (torch.dtype): Dtype of the resulting mask.
        device: (torch.device): The device to move the sparse mask to.
        revert_mask (bool): Whether to flip the 1's and 0's of the attention mask, default to True.
        use_neg_inf (bool): Use negative infinity instead of one in the resulting mask, default to True.

    Returns:
        The attention mask of shape [batch_size, num_heads, src_seq_len, target_seq_len],
        with broadcast dimensions set to 1.
    """
    if dtype is None:
        dtype = torch.float16

    if revert_mask:
        sparse_mask = 1.0 - sparse_mask

    # When running on CS, move constant from CPU to device wrapped with
    # XLA literal
    if cstorch.use_cs():
        fixed_sparsity = cstorch.make_constant(sparse_mask.to(dtype=dtype))
    else:
        # When running on GPU, move constant from CPU to GPU
        fixed_sparsity = sparse_mask.to(device=device)

    extended_key_padding_mask = make_key_padding_mask_broadcastable(
        key_padding_mask,
        dtype=dtype,
        revert_mask=False,
        use_neg_inf=False,
    )

    sparse_attention_mask, _ = torch.broadcast_tensors(
        fixed_sparsity,
        extended_key_padding_mask,
    )

    if use_neg_inf:
        extended_key_padding_mask = replace_with_zero_and_neg_inf(
            extended_key_padding_mask
        )

    return sparse_attention_mask


def get_extended_attention_mask(
    attention_mask: torch.Tensor,
    input_shape: Optional[Tuple[int]] = None,
    causal: bool = False,
    device: Optional[torch.device] = None,
    dtype=None,
) -> torch.Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model (required for causal masks).
        causal: (`bool`): If enabled the returned mask will be causal.
        device: (:obj:`torch.device`):
            The device of the input to the model.
    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    if dtype is None:
        dtype = torch.float16
    attention_mask = attention_mask.to(dtype=dtype)

    # Since `attention_mask` is passed as `1.0` for positions we want to
    # attend and `0.0` for masked positions, this operation will "invert"
    # the mask due to the "negative infinity" scaling at the end.
    attention_mask = 1.0 - attention_mask

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is an encoder, make the mask broadcastable to
        # [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]

        # - if the model is a decoder, apply a causal mask instead of the
        # padding mask
        if causal:
            batch_size, seq_length = input_shape
            # build seq_length x seq_length lower triangular boolean
            # mask(i, j) = i > j
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, :] > seq_ids[:, None]
            causal_mask = causal_mask.to(attention_mask.dtype)
            # in case past_key_values are used we need to add a prefix
            # zeros mask to the causal mask
            if attention_mask.shape[1] > seq_length:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.zeros(
                            (seq_length, prefix_seq_len),
                            device=device,
                            dtype=causal_mask.dtype,
                        ),
                        causal_mask,
                    ],
                    axis=-1,
                )

            extended_attention_mask, _ = torch.broadcast_tensors(
                causal_mask, extended_attention_mask
            )
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    # Replace all the `1.0` masked-off values with -inf float value, since we
    # are adding it to the raw scores before the softmax; this is effectively
    # the same as removing these entirely.
    return replace_with_zero_and_neg_inf(extended_attention_mask)


def create_chunked_attention_span(
    batch_size: int,
    target_seq_len: int,
    chunk_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create an attention span tensor to create a
    chunked attention mask pattern, similar to VSL masking.
    For a batch size of 1, sequence length of 10 and chunk size of 3, the attention span tensor is:
    ```
    [
        [2, 1, 0, 2, 1, 0, 2, 1, 0, 2],
    ]
    ```

    Args:
        batch_size (int): Input batch size.
        target_seq_len (int): Input sequence length.
        chunk_size (int): Size of local attention chunk window.
        device (Optional[torch.device]): The device of the input to the model.

    Returns:
        Attention span tensor of shape [batch_size, target_seq_len].
    """
    seq_range = torch.arange(
        target_seq_len, device=device, dtype=torch.float32
    ).broadcast_to((batch_size, target_seq_len))

    zero = torch.tensor(0, dtype=torch.float32)
    one = torch.tensor(1, dtype=torch.float32)

    attention_span = torch.zeros_like(seq_range)

    for i in range(chunk_size, target_seq_len + chunk_size, chunk_size):
        shifted_range = torch.tensor(i, dtype=torch.float32) - seq_range
        right_bounded = torch.where(shifted_range > 0, shifted_range, zero)
        attention_span += torch.where(
            right_bounded - chunk_size <= 0, right_bounded, zero
        )
    return attention_span - one


def create_sliding_window_mask_with_complement(
    batch_size: int,
    num_heads: int,
    tgt_seq_length: int,
    sliding_window_length: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns two boolean masks, one is a sliding window causal mask,
    the second is a complement so that both form a lower-triangular causal mask.
    That is, the sliding window mask would look like:
    ```
    [
        [True,  False, False, False, False],
        [True,  True,  False, False, False],
        [False, True,  True,  False, False],
        [False, False, True,  True,  False],
        [False, False, False, True,  True],
    ]
    ```
    whereas the complement mask is:
    ```
    [
        [False, False,  False,  False,  False],
        [False, False,  False,  False,  False],
        [True,  False,  False,  False,  False],
        [True,  True,   False,  False,  False],
        [True,  True,   True,   False,  False],
    ]
    ```

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of heads.
        tgt_seq_length (int): Target sequence length.
        sliding_window_length (int): Mask sliding window length.
        device (torch.device): The device of logit tensors to be masked.

    Returns:
        Tuple of two attention masks of shape [batch_size, num_heads, tgt_seq_length, tgt_seq_length].
    """
    mask_shape = (
        batch_size,
        num_heads,
        tgt_seq_length,
        tgt_seq_length,
    )
    seq_range = torch.arange(tgt_seq_length, device=device, dtype=torch.float32)
    q_range = seq_range[:, None].broadcast_to(mask_shape)
    k_range = seq_range[None, :].broadcast_to(mask_shape)
    diff = q_range - k_range

    attention_mask = -diff

    if sliding_window_length is not None:
        # Set upper triangular part to positive integers.
        zero = torch.tensor(0, dtype=torch.float32)
        attention_mask = torch.maximum(attention_mask, zero)

        window_span = torch.tensor(
            sliding_window_length - 1, device=device, dtype=torch.float32
        ).broadcast_to(mask_shape)
        attention_mask += torch.maximum(diff - window_span, zero)

    swa_mask = attention_mask == 0
    swa_complement_mask = (-diff + float(tgt_seq_length) * swa_mask) <= 0
    return swa_mask, swa_complement_mask


def smooth_loss(prediction_scores, loss, label_smoothing, classes):
    """
    Add label smoothing to loss function,
    this is a workaround method of label smoothing in our system.
    """
    logits = prediction_scores.view(-1, classes)
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    smooth_loss = -1.0 * logprobs.mean(dim=-1)
    loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth_loss

    return loss


def get_embedding_dtype(mixed_precision=True, dtype=None):
    if mixed_precision and torch.cuda.is_available():
        if dtype in ["bfloat16", "cbfloat16"]:
            return torch.bfloat16
        if dtype == "float16":
            return torch.float16

    return None
