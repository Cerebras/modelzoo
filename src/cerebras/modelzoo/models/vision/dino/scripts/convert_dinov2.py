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

#!/usr/bin/env python
# coding=utf-8
"""
DINOv2 two-way converter: original <-> Cerebras

- Handles patch embedding permute, positional embedding CLS token reordering,
- Splits/merges Q, K, V for attention layers,
- Renames keys (MLP, norms, heads, centers, etc.) to match Cerebras or original naming.

Usage:
  python convert_dinov2.py \
    --input_ckpt /path/to/checkpoint.mdl \
    --output_ckpt /path/to/converted.mdl \
    --convert_to cerebras \
"""

import argparse
import logging
import re
from typing import Dict

import torch

import cerebras.pytorch as cstorch

logger = logging.getLogger(__name__)


###############################################################################
#                  FINAL NORM & CLS/MASK TOKEN RENAMING
###############################################################################
def rename_final_norm_original(
    state_dict: Dict[str, torch.Tensor], trunk_out: str
) -> Dict[str, torch.Tensor]:
    """
    Rename final norm keys when converting FROM Cerebras BACK to the original format.

    Example:
      trunk_out + 'encoder.transformer_encoder.norm.weight'
      becomes
      trunk_out + 'norm.weight'
    """
    new_sd = {}
    pattern = rf"{trunk_out}encoder\.transformer_encoder\.norm\.(weight|bias)"
    for k, v in state_dict.items():
        if re.match(pattern, k):
            new_k = k.replace("encoder.transformer_encoder.norm", "norm")
            new_sd[new_k] = v
        else:
            new_sd[k] = v
    return new_sd


def rename_final_norm_cerebras(
    state_dict: Dict[str, torch.Tensor], trunk_out: str
) -> Dict[str, torch.Tensor]:
    """
    Rename final norm keys when converting FROM the original format TO Cerebras.

    Example:
      trunk_out + 'norm.weight'
      becomes
      trunk_out + 'encoder.transformer_encoder.norm.weight'
    """
    new_sd = {}
    pattern = rf"{trunk_out}norm\.(weight|bias)"
    for k, v in state_dict.items():
        if re.match(pattern, k):
            new_k = k.replace("norm.", "encoder.transformer_encoder.norm.")
            new_sd[new_k] = v
        else:
            new_sd[k] = v
    return new_sd


def rename_cls_mask_tokens_cerebras(
    state_dict: Dict[str, torch.Tensor], trunk_out: str
) -> Dict[str, torch.Tensor]:
    """
    Rename CLS and mask tokens when converting FROM the original format TO Cerebras.
    Example:
      '.cls_token' -> '.embedding_layer.cls_embedding'
      '.mask_token' -> '.embedding_layer.mask_token'
    """
    new_sd = {}
    for k, v in state_dict.items():
        if k.endswith(".cls_token"):
            new_k = k.replace(".cls_token", ".embedding_layer.cls_embedding")
            new_sd[new_k] = v.reshape(-1)
        elif k.endswith(".mask_token"):
            new_k = k.replace(".mask_token", ".embedding_layer.mask_token")
            new_sd[new_k] = v.reshape(-1)
        else:
            new_sd[k] = v
    return new_sd


def rename_cls_mask_tokens_original(
    state_dict: Dict[str, torch.Tensor], trunk_out: str
) -> Dict[str, torch.Tensor]:
    """
    Rename CLS and mask tokens when converting FROM Cerebras BACK to the original format.
    Example:
      '.embedding_layer.cls_embedding' -> '.cls_token'
      '.embedding_layer.mask_token'    -> '.mask_token'
    """
    new_sd = {}
    for k, v in state_dict.items():
        if ".embedding_layer.cls_embedding" in k:
            new_k = k.replace(".embedding_layer.cls_embedding", ".cls_token")
            # Original shape was [1, 1, ...]
            new_sd[new_k] = v.unsqueeze(0).unsqueeze(0)
        elif ".embedding_layer.mask_token" in k:
            new_k = k.replace(".embedding_layer.mask_token", ".mask_token")
            # Original shape was [1, ...]
            new_sd[new_k] = v.unsqueeze(0)
        else:
            new_sd[k] = v
    return new_sd


###############################################################################
#                POSITION EMBEDDING REORDERING (CLS TOKEN)
###############################################################################
def reorder_pos_embed_for_cerebras(tensor: torch.Tensor) -> torch.Tensor:
    """
    Move the CLS token from index 0 to the end.

    Shape assumptions:
      Original: [1, N, D]
      Returned: [N, D] (then reinsert batch dimension if needed).
    """
    assert (
        tensor.dim() == 3 and tensor.shape[0] == 1
    ), "Expected position embedding shape [1, N, D]."
    squeezed = tensor.squeeze(0)  # => [N, D]
    reordered = torch.cat((squeezed[1:], squeezed[0:1]), dim=0)
    return reordered


def reorder_pos_embed_for_original(tensor: torch.Tensor) -> torch.Tensor:
    """
    Inverse operation for the above reorder:
    If the model has CLS at the end, move it back to index 0.

    Shape assumptions:
      Input: [N, D]
      Returned: [1, N, D]
    """
    assert (
        tensor.dim() == 2 and tensor.size(0) > 1
    ), "Expected position embedding shape [N, D]."
    return torch.cat((tensor[-1:, :], tensor[:-1, :]), dim=0).unsqueeze(0)


###############################################################################
#           PATCH EMBEDDING RESHAPE/PERMUTE UTILS (for 2-way)
###############################################################################
def reshape_patch_embed_original_to_cerebras(
    weight: torch.Tensor, patch_size: int, num_channels: int
) -> torch.Tensor:
    """
    Convert patch embedding from original format to Cerebras format.

    Original shape:   [out_dim, in_dim, patch_size, patch_size]
    Cerebras shape:   [out_dim, in_dim * patch_size^2]
    """
    assert weight.dim() == 4, "Expected 4D tensor for patch embedding."
    out_dim, in_dim, ph, pw = weight.shape
    assert (
        ph == patch_size and pw == patch_size
    ), "Patch embedding weight shape does not match the specified patch_size."
    return weight.permute(0, 2, 3, 1).reshape(out_dim, in_dim * ph * pw)


def reshape_patch_embed_cerebras_to_original(
    weight: torch.Tensor, patch_size: int, num_channels: int
) -> torch.Tensor:
    """
    Inverse of `reshape_patch_embed_original_to_cerebras`.

    Cerebras shape:   [out_dim, in_dim * patch_size^2]
    Original shape:   [out_dim, in_dim, patch_size, patch_size]
    """
    assert weight.dim() == 2, "Expected 2D tensor for patch embedding."
    out_dim, in_dim_times_patch = weight.shape
    # Reshape into [out_dim, patch_size, patch_size, num_channels]
    reshape_4d = weight.reshape(out_dim, patch_size, patch_size, num_channels)
    return reshape_4d.permute(0, 3, 1, 2)


###############################################################################
#       ORIGINAL -> CEREBRAS: PATCH + POS REORDER
###############################################################################
def original_rename_patch_embed_and_pos(
    state_dict: Dict[str, torch.Tensor],
    trunk_prefix: str,
    patch_size: int,
    num_channels: int,
) -> Dict[str, torch.Tensor]:
    """
    Rename and reshape patch embedding and position embedding for
    conversion FROM original TO Cerebras format.

    patch_embed.proj.weight => embedding_layer.linear_proj.weight (with permute/reshape)
    patch_embed.proj.bias   => embedding_layer.linear_proj.bias
    pos_embed               => embedding_layer.position_embeddings.weight (reordered for Cerebras)
    """
    new_sd = {}
    for k, v in state_dict.items():
        if "patch_embed.proj.weight" in k:
            new_k = k.replace(
                "patch_embed.proj.weight", "embedding_layer.linear_proj.weight"
            )
            new_sd[new_k] = reshape_patch_embed_original_to_cerebras(
                v, patch_size, num_channels
            )
        elif "patch_embed.proj.bias" in k:
            new_k = k.replace(
                "patch_embed.proj.bias", "embedding_layer.linear_proj.bias"
            )
            new_sd[new_k] = v
        elif "pos_embed" in k:
            new_k = k.replace(
                "pos_embed", "embedding_layer.position_embeddings.weight"
            )
            new_sd[new_k] = reorder_pos_embed_for_cerebras(v)
        else:
            new_sd[k] = v
    return new_sd


###############################################################################
#       CEREBRAS -> ORIGINAL: PATCH + POS REORDER
###############################################################################
def cerebras_rename_patch_embed_and_pos(
    state_dict: Dict[str, torch.Tensor],
    trunk_prefix: str,
    patch_size: int,
    num_channels: int,
) -> Dict[str, torch.Tensor]:
    """
    Rename and reshape patch embedding and position embedding for
    conversion FROM Cerebras BACK TO the original format.

    embedding_layer.linear_proj.weight => patch_embed.proj.weight
    embedding_layer.linear_proj.bias   => patch_embed.proj.bias
    embedding_layer.position_embeddings.weight => pos_embed
    """
    new_sd = {}
    for k, v in state_dict.items():
        if "embedding_layer.linear_proj.weight" in k:
            new_k = k.replace(
                "embedding_layer.linear_proj.weight", "patch_embed.proj.weight"
            )
            new_sd[new_k] = reshape_patch_embed_cerebras_to_original(
                v, patch_size, num_channels
            )
        elif "embedding_layer.linear_proj.bias" in k:
            new_k = k.replace(
                "embedding_layer.linear_proj.bias", "patch_embed.proj.bias"
            )
            new_sd[new_k] = v
        elif "embedding_layer.position_embeddings.weight" in k:
            new_k = k.replace(
                "embedding_layer.position_embeddings.weight", "pos_embed"
            )
            new_sd[new_k] = reorder_pos_embed_for_original(v)
        else:
            new_sd[k] = v
    return new_sd


###############################################################################
#   ORIGINAL -> CEREBRAS: SPLIT QKV, ATTENTION, MLP, NORM, ETC.
###############################################################################
def original_split_qkv(
    state_dict: Dict[str, torch.Tensor],
    old_prefix: str,
    new_prefix: str,
    remove_old: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Split QKV from original format into separate Q, K, V in Cerebras format.

    Example:
      blocks.<L>.attn.qkv.(weight|bias) => self_attn.proj_{q,k,v}_dense_layer.(weight|bias)
    """
    new_sd = {}
    pattern = rf"{old_prefix}(?:\.\d+)?\.(\d+)\.attn\.qkv\.(weight|bias)"

    for k, v in list(state_dict.items()):
        match_obj = re.match(pattern, k)
        if match_obj:
            layer_idx = match_obj.group(1)
            w_or_b = match_obj.group(2)
            hidden_size = v.shape[0] // 3

            q_val = v[:hidden_size]
            k_val = v[hidden_size : 2 * hidden_size]
            v_val = v[2 * hidden_size :]

            base = f"{new_prefix}.{layer_idx}.self_attn"
            new_sd[f"{base}.proj_q_dense_layer.{w_or_b}"] = q_val
            new_sd[f"{base}.proj_k_dense_layer.{w_or_b}"] = k_val
            new_sd[f"{base}.proj_v_dense_layer.{w_or_b}"] = v_val

            # Keep the old key if remove_old=False
            if not remove_old:
                new_sd[k] = v
        else:
            new_sd[k] = v
    return new_sd


def original_rename_attn_proj(
    state_dict: Dict[str, torch.Tensor], old_prefix: str, new_prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Rename attention projection keys for original->Cerebras.

    Example:
      blocks.<L>.attn.proj.(weight|bias) => .self_attn.proj_output_dense_layer.(weight|bias)
    """
    new_sd = {}
    pattern = rf"{old_prefix}(?:\.\d+)?\.(\d+)\.attn\.proj\.(weight|bias)"
    for k, v in state_dict.items():
        match_obj = re.match(pattern, k)
        if match_obj:
            layer_idx = match_obj.group(1)
            w_or_b = match_obj.group(2)
            new_key = f"{new_prefix}.{layer_idx}.self_attn.proj_output_dense_layer.{w_or_b}"
            new_sd[new_key] = v
        else:
            new_sd[k] = v
    return new_sd


def original_rename_mlp(
    state_dict: Dict[str, torch.Tensor], old_prefix: str, new_prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Rename MLP blocks for original->Cerebras.

    Examples:
      blocks.<L>.mlp.fc1.(weight|bias) => .ffn.ffn.0.linear_layer.(weight|bias)
      blocks.<L>.mlp.fc2.(weight|bias) => .ffn.ffn.1.linear_layer.(weight|bias)
    """
    new_sd = {}
    fc1_pat = rf"{old_prefix}(?:\.\d+)?\.(\d+)\.mlp\.fc1\.(weight|bias)"
    fc2_pat = rf"{old_prefix}(?:\.\d+)?\.(\d+)\.mlp\.fc2\.(weight|bias)"

    for k, v in state_dict.items():
        m1 = re.match(fc1_pat, k)
        m2 = re.match(fc2_pat, k)
        if m1:
            layer_idx, w_or_b = m1.group(1), m1.group(2)
            new_k = f"{new_prefix}.{layer_idx}.ffn.ffn.0.linear_layer.{w_or_b}"
            new_sd[new_k] = v
        elif m2:
            layer_idx, w_or_b = m2.group(1), m2.group(2)
            new_k = f"{new_prefix}.{layer_idx}.ffn.ffn.1.linear_layer.{w_or_b}"
            new_sd[new_k] = v
        else:
            new_sd[k] = v
    return new_sd


def original_rename_norms_and_scales(
    state_dict: Dict[str, torch.Tensor], old_prefix: str, new_prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Rename norm and layer scale keys for original->Cerebras.

    blocks.<L>.norm1.(weight|bias) => .norm1.(weight|bias)
    blocks.<L>.ls1.gamma => .layer_scale1
    """
    new_sd = {}
    norm1_pat = rf"{old_prefix}(?:\.\d+)?\.(\d+)\.norm1\.(weight|bias)"
    norm2_pat = rf"{old_prefix}(?:\.\d+)?\.(\d+)\.norm2\.(weight|bias)"
    ls1_pat = rf"{old_prefix}(?:\.\d+)?\.(\d+)\.ls1\.gamma"
    ls2_pat = rf"{old_prefix}(?:\.\d+)?\.(\d+)\.ls2\.gamma"

    for k, v in state_dict.items():
        n1 = re.match(norm1_pat, k)
        n2 = re.match(norm2_pat, k)
        l1 = re.match(ls1_pat, k)
        l2 = re.match(ls2_pat, k)
        if n1:
            layer_idx, w_or_b = n1.group(1), n1.group(2)
            new_sd[f"{new_prefix}.{layer_idx}.norm1.{w_or_b}"] = v
        elif n2:
            layer_idx, w_or_b = n2.group(1), n2.group(2)
            new_sd[f"{new_prefix}.{layer_idx}.norm2.{w_or_b}"] = v
        elif l1:
            layer_idx = l1.group(1)
            new_sd[f"{new_prefix}.{layer_idx}.layer_scale1"] = v
        elif l2:
            layer_idx = l2.group(1)
            new_sd[f"{new_prefix}.{layer_idx}.layer_scale2"] = v
        else:
            new_sd[k] = v
    return new_sd


def convert_original_backbone_to_cerebras(
    src_sd: Dict[str, torch.Tensor],
    trunk_in: str,
    trunk_out: str,
    patch_size: int,
    num_channels: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert teacher or student trunk FROM the original format TO Cerebras format.

    Steps:
      1) Filter trunk keys
      2) Add trunk_out prefix
      3) Rename: split QKV, rename attn proj, rename MLP, rename norms/scales
      4) Rename patch & pos embed, final norms, cls/mask tokens
    """
    # 1) filter trunk
    filtered = {
        k.replace(trunk_in, ""): v
        for k, v in src_sd.items()
        if k.startswith(trunk_in)
    }

    if not filtered:
        raise ValueError(
            f"No keys found with prefix {trunk_in}, the input model is not suitable for DINOv2 pretraining."
        )

    # 2) prefix them
    with_prefix = {f"{trunk_out}{k}": v for k, v in filtered.items()}

    old_p = f"{trunk_out}blocks"
    new_p = f"{trunk_out}encoder.transformer_encoder.layers"

    # 3) rename
    step1 = original_split_qkv(with_prefix, old_prefix=old_p, new_prefix=new_p)
    step2 = original_rename_attn_proj(step1, old_prefix=old_p, new_prefix=new_p)
    step3 = original_rename_mlp(step2, old_prefix=old_p, new_prefix=new_p)
    step4 = original_rename_norms_and_scales(
        step3, old_prefix=old_p, new_prefix=new_p
    )

    # 4) rename patch & pos, final norm, cls/mask
    step5 = original_rename_patch_embed_and_pos(
        step4, trunk_out, patch_size, num_channels
    )
    step6 = rename_final_norm_cerebras(step5, trunk_out)
    step7 = rename_cls_mask_tokens_cerebras(step6, trunk_out)

    return step7


def convert_original_to_cerebras_checkpoint(
    state_dict_in: Dict[str, torch.Tensor],
    patch_size: int,
    num_channels: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert the entire checkpoint (teacher + student + heads + centers)
    FROM the original format TO Cerebras format.
    """
    # Convert teacher backbone
    teacher_sd = convert_original_backbone_to_cerebras(
        state_dict_in,
        "teacher.backbone.",
        "image_model_trunks.model.0.",
        patch_size,
        num_channels,
    )
    # Convert student backbone
    student_sd = convert_original_backbone_to_cerebras(
        state_dict_in,
        "student.backbone.",
        "image_model_trunks.model.1.",
        patch_size,
        num_channels,
    )

    # Convert heads
    heads_sd = {}
    for k, v in state_dict_in.items():
        if k.startswith("teacher.dino_head."):
            new_k = k.replace("teacher.dino_head.", "heads.model.0.")
            heads_sd[new_k] = v
        elif k.startswith("student.dino_head."):
            new_k = k.replace("student.dino_head.", "heads.model.1.")
            heads_sd[new_k] = v
        elif k.startswith("teacher.ibot_head."):
            new_k = k.replace("teacher.ibot_head.", "heads.model.2.")
            heads_sd[new_k] = v
        elif k.startswith("student.ibot_head."):
            new_k = k.replace("student.ibot_head.", "heads.model.3.")
            heads_sd[new_k] = v

    # Merge everything
    out_sd = {}
    out_sd.update(teacher_sd)
    out_sd.update(student_sd)
    out_sd.update(heads_sd)

    # Convert centers
    if "dino_loss.center" in state_dict_in:
        out_sd["losses.model.0.center"] = state_dict_in[
            "dino_loss.center"
        ].squeeze()
    if "ibot_patch_loss.center" in state_dict_in:
        out_sd["losses.model.1.center"] = state_dict_in[
            "ibot_patch_loss.center"
        ].squeeze()

    # Add EMA step and losses steps
    out_sd["ema_step"] = torch.zeros(1, dtype=torch.int32)
    for i in range(2):
        out_sd[f"losses.model.{i}.step"] = torch.zeros(1, dtype=torch.int32)

    return out_sd


###############################################################################
#   CEREBRAS -> ORIGINAL: MERGE QKV, ATTENTION, MLP, NORM, ETC.
###############################################################################
def cerebras_join_qkv(
    state_dict: Dict[str, torch.Tensor], prefix: str, new_prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Merge separate Q, K, V in Cerebras format back into a single qkv tensor
    in the original format.

    Cerebras: self_attn.proj_q_dense_layer.(weight|bias) -> Original: attn.qkv.(weight|bias)
    """
    new_sd = {}
    buffer = {}
    pattern = (
        rf"{prefix}\.(\d+)\.self_attn\.proj_([qkv])_dense_layer\.(weight|bias)"
    )

    for k, v in state_dict.items():
        match_obj = re.match(pattern, k)
        if match_obj:
            layer_idx, qkv_letter, w_or_b = match_obj.groups()
            merged_key = f"{new_prefix}.{layer_idx}.attn.qkv.{w_or_b}"
            if merged_key not in buffer:
                buffer[merged_key] = {"q": None, "k": None, "v": None}
            buffer[merged_key][qkv_letter] = v
        else:
            new_sd[k] = v

    # Combine Q, K, V
    for out_key, sub_dict in buffer.items():
        q = sub_dict["q"]
        k = sub_dict["k"]
        v = sub_dict["v"]
        if None in (q, k, v):
            logger.warning(f"Incomplete Q, K, V for {out_key}.")
            continue

        # Concatenate along dim=0 for both weight and bias
        merged = torch.cat([q, k, v], dim=0)
        new_sd[out_key] = merged
    return new_sd


def cerebras_rename_attn_proj(
    state_dict: Dict[str, torch.Tensor], prefix: str, new_prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Cerebras: .self_attn.proj_output_dense_layer.(weight|bias)
      -> Original: .attn.proj.(weight|bias)
    """
    new_sd = {}
    pattern = (
        rf"{prefix}\.(\d+)\.self_attn\.proj_output_dense_layer\.(weight|bias)"
    )
    for k, v in state_dict.items():
        match_obj = re.match(pattern, k)
        if match_obj:
            layer_idx, w_or_b = match_obj.group(1), match_obj.group(2)
            new_sd[f"{new_prefix}.{layer_idx}.attn.proj.{w_or_b}"] = v
        else:
            new_sd[k] = v
    return new_sd


def cerebras_rename_mlp(
    state_dict: Dict[str, torch.Tensor], prefix: str, new_prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Cerebras: .ffn.ffn.0.linear_layer.(weight|bias)
      -> Original: .mlp.fc1.(weight|bias)
    """
    new_sd = {}
    pat_fc1 = rf"{prefix}\.(\d+)\.ffn\.ffn\.0\.linear_layer\.(weight|bias)"
    pat_fc2 = rf"{prefix}\.(\d+)\.ffn\.ffn\.1\.linear_layer\.(weight|bias)"

    for k, v in state_dict.items():
        m1 = re.match(pat_fc1, k)
        m2 = re.match(pat_fc2, k)
        if m1:
            layer_idx, w_or_b = m1.group(1), m1.group(2)
            new_sd[f"{new_prefix}.{layer_idx}.mlp.fc1.{w_or_b}"] = v
        elif m2:
            layer_idx, w_or_b = m2.group(1), m2.group(2)
            new_sd[f"{new_prefix}.{layer_idx}.mlp.fc2.{w_or_b}"] = v
        else:
            new_sd[k] = v
    return new_sd


def cerebras_rename_norms_scales(
    state_dict: Dict[str, torch.Tensor], prefix: str, new_prefix: str
) -> Dict[str, torch.Tensor]:
    """
    Cerebras: .layer_scale1 -> Original: .ls1.gamma
    """
    new_sd = {}
    pat_ls1 = rf"{prefix}\.(\d+)\.layer_scale1"
    pat_ls2 = rf"{prefix}\.(\d+)\.layer_scale2"
    pat_n1 = rf"{prefix}\.(\d+)\.norm1\.(weight|bias)"
    pat_n2 = rf"{prefix}\.(\d+)\.norm2\.(weight|bias)"

    for k, v in state_dict.items():
        ls1_m = re.match(pat_ls1, k)
        ls2_m = re.match(pat_ls2, k)
        n1_m = re.match(pat_n1, k)
        n2_m = re.match(pat_n2, k)

        if ls1_m:
            layer_idx = ls1_m.group(1)
            new_sd[f"{new_prefix}.{layer_idx}.ls1.gamma"] = v
        elif ls2_m:
            layer_idx = ls2_m.group(1)
            new_sd[f"{new_prefix}.{layer_idx}.ls2.gamma"] = v
        elif n1_m:
            layer_idx, w_or_b = n1_m.group(1), n1_m.group(2)
            new_sd[f"{new_prefix}.{layer_idx}.norm1.{w_or_b}"] = v
        elif n2_m:
            layer_idx, w_or_b = n2_m.group(1), n2_m.group(2)
            new_sd[f"{new_prefix}.{layer_idx}.norm2.{w_or_b}"] = v
        else:
            new_sd[k] = v
    return new_sd


def convert_cerebras_backbone_to_original(
    state_dict: Dict[str, torch.Tensor],
    trunk_in: str,
    trunk_out: str,
    patch_size: int,
    num_channels: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert teacher or student trunk FROM Cerebras BACK TO the original format.

    Steps:
      1) Filter trunk keys
      2) Re-prefix with trunk_out
      3) Merge QKV, rename attn proj, MLP & norms
      4) Revert patch & pos reorder, final norm, cls/mask tokens
    """
    # 1) filter
    filtered = {
        k.replace(trunk_in, ""): v
        for k, v in state_dict.items()
        if k.startswith(trunk_in)
    }

    # 2) prefix
    with_prefix = {f"{trunk_out}{k}": v for k, v in filtered.items()}

    old_p = f"{trunk_out}encoder.transformer_encoder.layers"
    new_p = f"{trunk_out}blocks"

    # 3) merge QKV, rename
    s1 = cerebras_join_qkv(with_prefix, prefix=old_p, new_prefix=new_p)
    s2 = cerebras_rename_attn_proj(s1, prefix=old_p, new_prefix=new_p)
    s3 = cerebras_rename_mlp(s2, prefix=old_p, new_prefix=new_p)
    s4 = cerebras_rename_norms_scales(s3, prefix=old_p, new_prefix=new_p)

    # 4) revert patch/pos reordering, final norm, cls/mask tokens
    s5 = cerebras_rename_patch_embed_and_pos(
        s4, trunk_out, patch_size, num_channels
    )
    s6 = rename_final_norm_original(s5, trunk_out)
    s7 = rename_cls_mask_tokens_original(s6, trunk_out)
    return s7


def convert_cerebras_to_original_checkpoint(
    state_dict_in: Dict[str, torch.Tensor], patch_size: int, num_channels: int
) -> Dict[str, torch.Tensor]:
    """
    Convert the entire checkpoint (teacher + student + heads + centers)
    FROM Cerebras BACK TO the original format.
    """
    # Convert teacher backbone
    teacher_sd = convert_cerebras_backbone_to_original(
        state_dict_in,
        "image_model_trunks.model.0.",
        "teacher.backbone.",
        patch_size,
        num_channels,
    )
    # Convert student backbone
    student_sd = convert_cerebras_backbone_to_original(
        state_dict_in,
        "image_model_trunks.model.1.",
        "student.backbone.",
        patch_size,
        num_channels,
    )

    # Convert heads
    heads_sd = {}
    for k, v in state_dict_in.items():
        if k.startswith("heads.model.0."):
            new_k = k.replace("heads.model.0.", "teacher.dino_head.")
            heads_sd[new_k] = v
        elif k.startswith("heads.model.1."):
            new_k = k.replace("heads.model.1.", "student.dino_head.")
            heads_sd[new_k] = v
        elif k.startswith("heads.model.2."):
            new_k = k.replace("heads.model.2.", "teacher.ibot_head.")
            heads_sd[new_k] = v
        elif k.startswith("heads.model.3."):
            new_k = k.replace("heads.model.3.", "student.ibot_head.")
            heads_sd[new_k] = v

    out_sd = {}
    out_sd.update(teacher_sd)
    out_sd.update(student_sd)
    out_sd.update(heads_sd)

    # Convert centers
    if "losses.model.0.center" in state_dict_in:
        out_sd["dino_loss.center"] = state_dict_in[
            "losses.model.0.center"
        ].unsqueeze(0)
    if "losses.model.1.center" in state_dict_in:
        out_sd["ibot_patch_loss.center"] = (
            state_dict_in["losses.model.1.center"].unsqueeze(0).unsqueeze(0)
        )

    return out_sd


def convert_checkpoint(
    input_ckpt_path: str,
    output_ckpt_path: str,
    convert_to: str,
    patch_size: int = 14,
    num_channels: int = 3,
):
    """
    Converts a checkpoint between original and cerebras formats.

    :param input_ckpt_path: Path to the input checkpoint file (.mdl/.pth).
    :param output_ckpt_path: Path to save the output checkpoint file (.pth/.mdl).
    :param convert_to: Either "cerebras" or "original" indicating the target format.
    :param patch_size: Patch size used in the model (default=14).
    :param num_channels: Number of channels used in the model (default=3).
    """
    logger.info(f"Loading input checkpoint: {input_ckpt_path}")
    if convert_to == "cerebras":
        loaded_ckpt = torch.load(
            input_ckpt_path, weights_only=True, map_location="cpu"
        )
    else:
        loaded_ckpt = cstorch.load(input_ckpt_path)

    # Some checkpoints use "model" key; others are directly the state_dict
    state_dict_in = loaded_ckpt.get("model", loaded_ckpt)

    if convert_to == "cerebras":
        logger.info("Converting from original => cerebras...")
        state_dict_out = convert_original_to_cerebras_checkpoint(
            state_dict_in, patch_size, num_channels
        )
    else:
        logger.info("Converting from cerebras => original...")
        state_dict_out = convert_cerebras_to_original_checkpoint(
            state_dict_in, patch_size, num_channels
        )

    final_ckpt = {"model": state_dict_out}
    logger.info(f"Saving converted checkpoint to {output_ckpt_path}")
    if convert_to == "cerebras":
        cstorch.save(final_ckpt, output_ckpt_path)
    else:
        torch.save(final_ckpt, output_ckpt_path)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Two-way converter: original <-> cerebras."
    )
    parser.add_argument(
        "--input_ckpt",
        type=str,
        required=True,
        help="Path to the input checkpoint (.mdl/.pth).",
    )
    parser.add_argument(
        "--output_ckpt",
        type=str,
        required=True,
        help="Path to the output (converted) checkpoint (.pth/.mdl).",
    )
    parser.add_argument(
        "--convert_to",
        type=str,
        choices=["cerebras", "original"],
        required=True,
        help="Specify the target format: 'cerebras' or 'original'.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Specify the patch_size (default=14).",
    )
    parser.add_argument(
        "--num_channels",
        type=int,
        default=3,
        help="Specify the num_channels (default=3).",
    )
    args = parser.parse_args()

    convert_checkpoint(
        input_ckpt_path=args.input_ckpt,
        output_ckpt_path=args.output_ckpt,
        convert_to=args.convert_to,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
    )


if __name__ == "__main__":
    main()
