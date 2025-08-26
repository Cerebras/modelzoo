import torch
import torch.nn as nn


class GRPOLoss(nn.Module):
    def __init__(self, eps_clip: float = 0.2, kl_loss_coeff: float = 0.1):
        """
        GRPO Loss with optional KL regularization.
        Args:
            eps_clip (float): Clipping epsilon.
            kl_loss_coeff (float): Coefficient for KL loss term.
        """
        super().__init__()
        self.eps_clip = eps_clip
        self.kl_loss_coeff = kl_loss_coeff
        self.epsilon = 1e-6

    def forward(
        self,
        old_log_probs,
        curr_log_probs,
        advantages,
        loss_mask,
        ref_log_probs=None,
    ):
        """
        Args:
            log_probs (Tensor): New log probabilities.
            sub_old_log_probs (Tensor): Log probs from old policy.
            advantages (Tensor): Advantage estimates.
            loss_mask (Tensor): Binary mask for valid entries.
            sub_ref_log_probs (Tensor, optional): Reference policy log probs for KL regularization.
        Returns:
            policy_loss (Tensor): Scalar GRPO loss.
        """
        sampling_ratio = torch.exp(curr_log_probs - old_log_probs)

        # Clipping for stability
        clipped_ratio = torch.clamp(
            sampling_ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip
        )

        unclipped_loss = sampling_ratio * advantages
        clipped_loss = clipped_ratio * advantages

        loss = -torch.min(unclipped_loss, clipped_loss)

        # Apply mask
        policy_loss = (loss_mask * loss).sum() / (
            loss_mask.sum() + self.epsilon
        )

        # Optional KL divergence loss
        if self.kl_loss_coeff > 0.0 and ref_log_probs is not None:
            kl = ref_log_probs - curr_log_probs
            ratio = torch.exp(kl)
            kld = ratio - kl - 1.0
            kl_loss = torch.clamp(kld, min=-10.0, max=10.0)
            masked_kl_loss = (loss_mask * kl_loss).mean()
            policy_loss += self.kl_loss_coeff * masked_kl_loss

        return policy_loss
