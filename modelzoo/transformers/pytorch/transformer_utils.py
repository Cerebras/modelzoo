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

NEGATIVE_INFINITY = -1.0e4


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


def make_key_padding_mask_broadcastable(
    key_padding_mask: torch.Tensor,
    dtype=None,
    revert_mask: bool = True,
    multiply_neg_inf: bool = True,
):
    """Makes broadcastable key_padding masks so that padding tokens are ignored.

    Args:
        key_padding_mask (torch.Tensor): key padding mask with shape in [2,3,4], with entry values either 1 or 0.
        dtype (torch.dtype): Dtype of the resulting mask.
        revert_mask (bool): whether to flip the 1's and 0's of the attention mask, default to True.
        multiply_neg_inf (bool): whether to multiply the resulting mask by a negative infinity constant, default to True.

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

    extended_key_padding_mask = _extend_mask_to_shape_of_4(key_padding_mask)

    if multiply_neg_inf:
        extended_key_padding_mask = (
            extended_key_padding_mask * NEGATIVE_INFINITY
        )

    return extended_key_padding_mask


def build_broadcastable_attention_mask(
    attention_mask: torch.Tensor,
    build_causal: bool = False,
    device: Optional[torch.device] = None,
    dtype=None,
    revert_mask: bool = True,
    multiply_neg_inf: bool = True,
):
    """Create broadcastable attention mask (full or causal) so that masked positions are ignored.

    Args:
        attention_mask (torch.Tensor): attention mask with shape in [2,3,4], with entry values either 1 or 0.
        build_causal (bool): If enabled a causal mask will be created according to the dims of attention_mask.
        device: (torch.device): The device of the input to the model, used for causal mask creation.
        dtype (torch.dtype): Dtype of the resulting mask.
        revert_mask (bool): whether to flip the 1's and 0's of the attention mask, default to True.
        multiply_neg_inf (bool): whether to multiply the resulting mask by a negative infinity constant, default to True.

    Returns:
        The attention mask of shape [batch_size, num_heads, src_seq_len, target_seq_len],
        with broadcast dimensions set to 1.
    """

    assert len(attention_mask.shape) in [
        2,
        3,
        4,
    ], "Masks with shape 2, 3, 4 are supported"
    if dtype is None:
        dtype = torch.float16
    attention_mask = attention_mask.to(dtype=dtype)

    # Since `attention_mask` is passed as `1.0` for positions we want to
    # attend and `0.0` for masked positions, this operation will "invert"
    # the mask due to the "negative infinity" scaling at the end.
    if revert_mask:
        attention_mask = 1.0 - attention_mask

    extended_attention_mask = _extend_mask_to_shape_of_4(attention_mask)

    target_sequence_length = extended_attention_mask.shape[-1]
    src_sequence_length = (
        extended_attention_mask.shape[-2]
        if extended_attention_mask.shape[-2] != 1
        else target_sequence_length
    )

    if build_causal:
        causal_mask = create_2D_autoregressive_mask(
            src_sequence_length,
            target_sequence_length,
            dtype=dtype,
            device=device,
        )
        extended_attention_mask, _ = torch.broadcast_tensors(
            causal_mask, extended_attention_mask
        )

    if multiply_neg_inf:
        extended_attention_mask = extended_attention_mask * NEGATIVE_INFINITY

    return extended_attention_mask


def create_2D_autoregressive_mask(
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
    causal_mask = torch.triu(
        torch.ones(
            (src_sequence_length, target_sequence_length),
            device=device,
            dtype=dtype,
        ),
        diagonal=1,
    )
    return causal_mask


def make_sparse_mask_broadcastable(
    sparse_mask: torch.Tensor,
    key_padding_mask: torch.Tensor,
    dtype=None,
    device=None,
    revert_mask: bool = True,
    multiply_neg_inf: bool = True,
):
    """Create broadcastable sparse mask so that masked positions are ignored.

    Args:
        sparse_mask (torch.Tensor): sparse_mask mask with shape [src_seq_len, target_seq_len].
        key_padding_mask (torch.Tensor): key padding mask with shape in [2,3,4].
        dtype (torch.dtype): Dtype of the resulting mask.
        device: (torch.device): The device to move the sparse mask to.
        revert_mask (bool): whether to flip the 1's and 0's of the attention mask, default to True.
        multiply_neg_inf (bool): whether to multiply the resulting mask by a negative infinity constant, default to True.

    Returns:
        The attention mask of shape [batch_size, num_heads, src_seq_len, target_seq_len],
        with broadcast dimensions set to 1.
    """
    if dtype is None:
        dtype = torch.float16

    if revert_mask:
        sparse_mask = 1.0 - sparse_mask

    from modelzoo.common.pytorch import cb_model as cm

    # When running on CS, move constant from CPU to device wrapped with
    # XLA literal
    if cm.use_cs():
        fixed_sparsity = cm.make_constant(sparse_mask.to(dtype=dtype))
    else:
        # When running on GPU, move constant from CPU to GPU
        fixed_sparsity = sparse_mask.to(device=device)

    extended_key_padding_mask = make_key_padding_mask_broadcastable(
        key_padding_mask,
        dtype=dtype,
        revert_mask=False,
        multiply_neg_inf=False,
    )

    sparse_attention_mask, _ = torch.broadcast_tensors(
        fixed_sparsity, extended_key_padding_mask,
    )

    if multiply_neg_inf:
        sparse_attention_mask = sparse_attention_mask * NEGATIVE_INFINITY

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
                The shape of the input to the model (required for causal masks)
            causal: (`bool`): if enabled the returned mask will be causal
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
    # Scale all the `1.0` masked-off values with `-10000.0`, since we are
    # adding it to the raw scores before the softmax; this is effectively
    # the same as removing these entirely.
    return extended_attention_mask * -1e4


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
