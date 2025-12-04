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

from functools import partial
from typing import List

import torch
import torch.nn as nn

import cerebras.pytorch.nn.functional as CF

# This is necessary because nn.* modules get redefined by the context manager
# defined in `per_example_grad_norm_utils.py` so we would end up in infinite
# recursion if we use the directly imported nn.* modules.
Linear = nn.Linear
Embedding = nn.Embedding
LayerNorm = nn.LayerNorm

# utility functions


class Noop(torch.autograd.Function):
    """Noop function that does nothing to the input or output. Used for testing
    the registration of custom autograd functions. Subclasses of this class may
    override the forward and backward methods but should not change the input or
    output tensors."""

    @staticmethod
    def forward(ctx, input, output, *args):
        ctx.save_for_backward(input, output, *args)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # note: pe grad norm here should probably be a norm
        # rather than a squared norm so it can be accumulated
        # over microbatches as a sum, then we can take the mean
        # in norm space rather than squared norm space
        input, output = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = None
        if not ctx.needs_input_grad[1]:
            grad_output = None
        return grad_input, grad_output


@torch.no_grad()
def stack_with_ones(x, dim):
    # stacks a tensor with ones along a given dim
    ones = torch.ones_like(x)
    return torch.stack([x, ones], dim=dim)


class ModuleWithBuffers(nn.Module):
    """Subclass nn.Module to add functionality for registering buffers and
    maintaining a dict to mark what the buffers are used for."""

    def _to_pegsqnorm(self, buffer_name):
        """Multiplies the buffer by the accumulated number of samples seen to
        compensate for the mean."""
        return torch.prod(getattr(self, buffer_name))

    def register_marked_buffer(self, name, init_func, marker):
        self.register_buffer(name, init_func())
        if not hasattr(self, 'marked_buffers'):
            self.marked_buffers = {}
        self.marked_buffers[name] = marker

    def register_upegsqnorm_buffer(self, name, marker="is_pegsqnorm"):
        """Registers a buffer to store the unnormalized per-example squared norm
        of the gradients for a given parameter. The buffer contains two scalars
        because it needs to track the number of examples seen, so we can use
        that later to normalize the value.
        After registering the buffer, a method is added to the module to
        retrieve the normalized per-example squared norm of the gradients for
        the parameter."""
        buffer_name = f"{name}_upegsqnorm"
        self.register_marked_buffer(buffer_name, lambda: torch.zeros(2), marker)
        # Add a method to the module for this buffer
        setattr(
            self, f"{name}_pegsqnorm", partial(self._to_pegsqnorm, buffer_name)
        )

    def named_buffers_with_marker(self, marker):
        """Gather all buffers marked with is_custom_buffer attribute."""
        return {
            name: buffer
            for name, buffer in self.named_buffers()
            if self.marked_buffers.get(name) == marker
        }

    # functions to make sure the marked buffers are saved and loaded in the
    # state dict
    def get_extra_state(self):
        if not hasattr(self, "marked_buffers"):
            return {}
        return {"marked_buffers": self.marked_buffers}

    def set_extra_state(self, state):
        if "marked_buffers" in state:
            self.marked_buffers = state["marked_buffers"]


def gather_marked_buffers(root_module: nn.Module, marker: str) -> List[str]:
    """
    Gather buffer names that share the same marker across all modules that have marked_buffers.

    Args:
        root_module: The root module to search through
        marker: The marker to search for

    Returns:
        List of fully qualified buffer names that share the marker
    """
    buffer_names = []

    # Go through all modules in the hierarchy
    for module_path, module in root_module.named_modules():
        # Check if this module has marked_buffers
        if hasattr(module, 'marked_buffers'):
            # For each buffer in this module's marked_buffers
            for buffer_name, buffer_marker in module.marked_buffers.items():
                if buffer_marker == marker:
                    # Construct full path to buffer
                    full_path = (
                        f"{module_path}.{buffer_name}"
                        if module_path
                        else buffer_name
                    )
                    buffer_names.append(full_path)

    return buffer_names


def get_buffers_by_marker(root_module: nn.Module, marker: str):
    """
    Get all buffer tensors that share the same marker across all modules.

    Args:
        root_module: The root module to search through
        marker: The marker to search for

    Returns:
        List of buffer tensors that share the given marker
    """
    buffers = []
    buffer_names = gather_marked_buffers(root_module, marker)

    # Get the actual buffer tensors using get_buffer
    for buffer_name in buffer_names:
        buffer = root_module.get_buffer(buffer_name)
        buffers.append((buffer_name, buffer))

    return buffers


############################################
# Begin: Linear Layer ######################
############################################


def linear_per_example_grad_norm(input, grad):
    """Core function to compute per-example squared norm of the gradients for
    a linear layer. This function is used in the backward pass of the linear
    layer defined in PEGLinearGradNormNoop."""
    batch_size, grad_dim, input_dim = (
        input.shape[0],
        grad.shape[-1],
        input.shape[-1],
    )

    bias_pe_grads = grad.sum(1)
    bias_pe_grads = stack_with_ones(
        (bias_pe_grads * bias_pe_grads).sum(-1), -1
    ).sum(0)
    # Ensure that input and grad are 3D for MatMul
    input = input.reshape(batch_size, -1, input_dim).float()
    grad = grad.reshape(batch_size, -1, grad_dim).float()
    # reduce to per-example gradients
    wt_grad = torch.matmul(input.transpose(1, 2), grad)
    wt_pe_grads = (wt_grad * wt_grad).sum(1).sum(1)
    wt_pe_grads = stack_with_ones(wt_pe_grads, -1).sum(0)
    return wt_pe_grads, bias_pe_grads


class PEGLinearGradNormNoop(Noop):
    """
    torch.autograd.Function that updates the unnormalized per-example squared
    in place. As a Noop, it doesn not change the forward and backward passes
    of the Linear layer.
    """

    @staticmethod
    def backward(ctx, grad_output):
        input, output, weight_upegsqnorm, bias_upegsqnorm = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = None
        if not ctx.needs_input_grad[1]:
            grad_output = None
        assert not ctx.needs_input_grad[
            2
        ]  # the buffers should not require grad
        assert not ctx.needs_input_grad[3]
        w_buf, b_buf = linear_per_example_grad_norm(
            input, grad_output
        )  # w_buf, b_buf are the upegsqnorms
        weight_upegsqnorm.add_(w_buf)  # update buffer in place
        if bias_upegsqnorm is not None:
            bias_upegsqnorm.add_(b_buf)
        return grad_input, grad_output, None, None


class ShimLinear(Linear):
    """Test layer for checking that registering a Noop torch.autograd.Function
    works."""

    def forward(self, input):
        output = super(ShimLinear, self).forward(input)
        return Noop.apply(input, output)


class PEGradNormShimLinear(ModuleWithBuffers, Linear):
    """Linear layer that accumulates the unnormalized per-example squared
    gradients of the weights and bias. Buffers must be manually cleared on each
    step."""

    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super(PEGradNormShimLinear, self).__init__(
            in_features, out_features, bias=bias, **kwargs
        )
        self.register_upegsqnorm_buffer("weight")
        if bias:
            self.register_upegsqnorm_buffer("bias")
        else:
            self.bias_upegsqnorm = None

    def forward(self, input):
        output = super(PEGradNormShimLinear, self).forward(input)
        return PEGLinearGradNormNoop.apply(
            input, output, self.weight_upegsqnorm, self.bias_upegsqnorm
        )


############################################
# End: Linear Layer ########################
# Begin: Embedding Layer ###################
############################################


def embedding_dense_backward(
    g, ids, vocab_size, padding_idx, scale_grad_by_freq
):
    """Function to replicate the backward pass of torch.Embedding to extract
    per-example gradients."""
    # first we make a binary tensor that indicates which embedding was selected
    # on each forward pass
    b = ids.shape[0]
    ids = ids.long()
    delta = CF.one_hot(ids, num_classes=vocab_size)  # [*ids.shape, vocab_size]
    # then we scale the delta by the gradient
    delta = delta.to(g.dtype)
    delta = delta.view(b, -1, vocab_size)
    # delta = delta.view(b, -1, vocab_size).transpose(2, 0) # [vocab_size, -1, b]
    # then use this to reduce
    _, d = g.shape[0], g.shape[-1]
    g = g.view(b, -1, d)
    peg = torch.einsum(
        'bnd,bnv->vbd', g, delta
    )  # this is not a sparse matmul but it should be
    return peg


def embedding_per_example_grad_norm(ids, g, vocab):
    """Uses `embedding_dense_backward` to compute the unnormalized per-example
    squared norm of the gradients."""
    # assumes 2d ids and 3d g
    peg = embedding_dense_backward(g, ids, vocab, -1, False)
    peg = peg.transpose(1, 0)
    s = (peg**2).sum(2).sum(1)
    return stack_with_ones(s, -1).sum(0)


class PEGEmbeddingGradNormNoop(Noop):
    """Noop for orchestrating the accumulation of the unnormalized per-example
    squared norm of the gradients for an embedding layer."""

    @staticmethod
    def forward(ctx, input, output, weight_upegsqnorm, vocab_size):
        ctx.save_for_backward(input, output, weight_upegsqnorm)
        ctx.vocab_size = vocab_size  # need to know vocab size for backward
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output, weight_upegsqnorm = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = None
        if not ctx.needs_input_grad[1]:
            grad_output = None
        assert not ctx.needs_input_grad[
            2
        ]  # the buffers should not require grad
        grad_buffer = embedding_per_example_grad_norm(
            input, grad_output, ctx.vocab_size
        )
        weight_upegsqnorm.add_(grad_buffer)
        return grad_input, grad_output, None, None


class ShimEmbedding(Embedding):
    """Test layer for checking that registering a Noop torch.autograd.Function works."""

    def forward(self, input):
        output = super(ShimEmbedding, self).forward(input)
        return Noop.apply(input, output)


class PEGradNormShimEmbedding(ModuleWithBuffers, Embedding):
    """Embedding layer that accumulates the unnormalized per-example squared
    norm of the gradients of the embedding weights. Buffers must be manually
    cleared on each step."""

    def __init__(self, *args, **kwargs):
        super(PEGradNormShimEmbedding, self).__init__(*args, **kwargs)
        # self.weight_upegsqnorm = nn.Parameter(torch.full((2,), torch.nan))
        self.register_upegsqnorm_buffer("weight")

    def forward(self, input):
        output = super(PEGradNormShimEmbedding, self).forward(input)
        return PEGEmbeddingGradNormNoop.apply(
            input, output, self.weight_upegsqnorm, self.weight.shape[0]
        )

    def weight_pegsqnorm(self):
        return torch.prod(self.weight_upegsqnorm.grad)


############################################
# End: Embedding Layer #####################
# Begin: LayerNorm Layer ###################
############################################


class ElementWiseAffine(nn.Module):
    """The Element-wise affine part of LayerNorm."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        else:
            self.register_buffer(
                'bias', torch.zeros(ndim)
            )  # extra flops of doing this are negligible

    def forward(self, input):
        castable = lambda x: x.view(*[1 for _ in range(x.dim())], -1)
        out = input * castable(self.weight)
        out += castable(self.bias)
        return out


class PEGradNormEANoop(torch.autograd.Function):
    """Noop for orchestrating the accumulation of the unnormalized per-example
    squared norm of the gradients for an element-wise affine layer."""

    @staticmethod
    def forward(
        ctx, input, output, weight, bias, weight_upegsqnorm, bias_upegsqnorm
    ):
        ctx.save_for_backward(input, weight_upegsqnorm, bias_upegsqnorm)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_upegsqnorm, bias_upegsqnorm = ctx.saved_tensors
        grad_input = grad_buffer = None
        if ctx.needs_input_grad[0]:
            grad_input = None  # don't pass any extra gradients through
        if not ctx.needs_input_grad[1]:
            grad_output = None
        z = input
        b, d = z.shape[0], z.shape[-1]
        # reshape to 3d to make the next steps easier
        z = z.view(b, -1, d)
        g = grad_output.view(b, -1, d)
        pe_dgamma = (g * z).sum(1)
        pe_dbeta = g.sum(1)
        w_upegsqnorm = stack_with_ones((pe_dgamma * pe_dgamma).sum(1), -1).sum(
            0
        )
        b_upegsqnorm = stack_with_ones((pe_dbeta * pe_dbeta).sum(1), -1).sum(0)
        if weight_upegsqnorm is not None:
            weight_upegsqnorm.add_(w_upegsqnorm)
        if bias_upegsqnorm is not None:
            bias_upegsqnorm.add_(b_upegsqnorm)
        return grad_input, grad_output, None, None, None, None


class PEGradNormShimElementWiseAffine(ModuleWithBuffers, ElementWiseAffine):
    """Element-wise affine layer that accumulates the unnormalized per-example
    squared norm of the gradients of the weights and bias. Buffers must be
    manually cleared on each step."""

    def __init__(self, ndim, bias):
        ElementWiseAffine.__init__(self, ndim, bias)
        self.register_upegsqnorm_buffer('weight')
        if bias:
            self.register_upegsqnorm_buffer('bias')
        else:
            self.bias_upegsqnorm = None

    def forward(self, input):
        out = ElementWiseAffine.forward(self, input)
        return PEGradNormEANoop.apply(
            input,
            out,
            self.weight,
            self.bias,
            self.weight_upegsqnorm,
            self.bias_upegsqnorm,
        )


class PEGradNormShimLayerNorm(LayerNorm):
    """LayerNorm layer constructed of a LayerNorm layer followed by an
    ElementWiseAffine layer. The PEGradNormShimElementWiseAffine layer is used
    to accumulate the unnormalized per-example squared norm of the gradients."""

    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        device=None,
        dtype=None,
    ):
        LayerNorm.__init__(
            self,
            normalized_shape,
            eps=eps,
            elementwise_affine=False,
            bias=False,
            device=device,
            dtype=dtype,
        )
        assert type(normalized_shape) == int, "Only 1D normalization supported"
        if elementwise_affine:
            self.affine = PEGradNormShimElementWiseAffine(
                normalized_shape, bias
            )
        else:
            self.affine = lambda x: x  # do nothing

    def forward(self, input):
        return self.affine(LayerNorm.forward(self, input))


class ShimLayerNorm(LayerNorm):
    """Test layer for checking that registering a Noop torch.autograd.Function works."""

    def forward(self, input):
        output = super(ShimLayerNorm, self).forward(input)
        return Noop.apply(input, output)


############################################
# End: LayerNorm Layer #####################
############################################
