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

from typing import Optional, Tuple

import torch

import cerebras.pytorch as cstorch


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


def create_broadcasted_autoregressive_mask(
    batch_size: int,
    num_heads: int,
    tgt_seq_length: int,
    attention_span: Optional[torch.Tensor] = None,
    sliding_window_length: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
    use_neg_inf: bool = True,
):
    """Create broadcasted causal attention mask optionally with VSL masking.

    For VSL, `attention_span` is required and past tokens out of the current sequence
    are additionally masked.

    Args:
        batch_size (int): Batch size.
        num_heads (int): Number of heads.
        tgt_seq_length (int): Target sequence length.
        attention_span (torch.Tensor): Attention span of keys for VSL, has shape [batch_size, target_seq_len].
        sliding_window_length (int): If specified, the current token would only attend the current
        token and sliding_window_length previous tokens.
        device (torch.device): The device of the input to the model, used for causal mask creation.
        dtype (torch.dtype): Dtype of the resulting mask, default to torch.float16.
        use_neg_inf (bool): Use negative infinity instead of one in the resulting mask, default to True.

    Returns:
        The attention mask of shape [batch_size, num_heads, tgt_seq_len, tgt_seq_len].
    """
    if attention_span is not None and sliding_window_length is not None:
        raise ValueError(f"Sliding window used with VSL.")

    if attention_span is not None:
        return create_vsl_mask(
            attention_span=attention_span,
            num_heads=num_heads,
            is_causal=True,
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

    if sliding_window_length is not None:
        # Set upper triangular part to positive integers.
        zero = torch.tensor(0, dtype=torch.float32)
        attention_mask = torch.maximum(attention_mask, zero)

        window_span = torch.tensor(
            sliding_window_length - 1, device=device, dtype=torch.float32
        ).broadcast_to(mask_shape)
        attention_mask += torch.maximum(diff - window_span, zero)

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


def smooth_loss(prediction_scores, loss, label_smoothing, classes):
    """
    Add label smoothing to loss function,
    this is a workaround method of label smoothing in our system
    """
    logits = prediction_scores.view(-1, classes)
    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
    smooth_loss = -1.0 * logprobs.mean(dim=-1)
    loss = (1.0 - label_smoothing) * loss + label_smoothing * smooth_loss

    return loss


def get_embedding_dtype(mixed_precision, fp16_type):
    if mixed_precision and torch.cuda.is_available():
        if fp16_type in ["bfloat16", "cbfloat16"]:
            dtype = torch.bfloat16
        elif fp16_type == "float16":
            dtype = torch.float16
        else:
            dtype = None
    else:
        dtype = None
