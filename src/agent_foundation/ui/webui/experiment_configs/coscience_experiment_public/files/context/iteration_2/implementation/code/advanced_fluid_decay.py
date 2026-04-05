"""
SYNAPSE v2: Advanced FLUID Decay Module

Multi-timescale exponential decay with learnable timescale parameters.
This module addresses the fixed τ=24h limitation from Iteration 1 by
learning content-appropriate decay rates.

Key Features:
- 3 learnable timescales (fast/medium/slow)
- Log-space parameterization for training stability
- Gradient clipping support for log_tau parameters
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdvancedFLUIDDecay(nn.Module):
    """Multi-timescale exponential decay with learnable parameters.

    Instead of using a fixed τ=24h for all items, this module learns
    3 timescales that can be mixed based on content type:
    - Fast (τ ≈ 3h): News, trending topics, viral content
    - Medium (τ ≈ 24h): Movies, TV shows, games
    - Slow (τ ≈ 168h): Music albums, books, evergreen content

    The timescales are parameterized in log-space for numerical stability
    during training.
    """

    def __init__(
        self,
        num_timescales: int = 3,
        init_timescales: Tuple[float, float, float] = (3.0, 24.0, 168.0),
        min_timescale: float = 0.5,
        max_timescale: float = 720.0,
    ):
        """Initialize the decay module.

        Args:
            num_timescales: Number of learnable timescales
            init_timescales: Initial values for timescales (in hours)
            min_timescale: Minimum allowed timescale (prevents collapse)
            max_timescale: Maximum allowed timescale (prevents explosion)
        """
        super().__init__()

        assert num_timescales == len(
            init_timescales
        ), f"Expected {num_timescales} initial timescales, got {len(init_timescales)}"

        # Store bounds for clamping
        self.log_min_tau = math.log(min_timescale)
        self.log_max_tau = math.log(max_timescale)

        # Learnable timescale parameters in log-space
        init_log_tau = torch.tensor([math.log(t) for t in init_timescales])
        self.log_tau = nn.Parameter(init_log_tau)

        self.num_timescales = num_timescales

    @property
    def timescales(self) -> Tensor:
        """Get current timescale values in hours (clamped to valid range)."""
        clamped_log_tau = torch.clamp(self.log_tau, self.log_min_tau, self.log_max_tau)
        return torch.exp(clamped_log_tau)

    def forward(self, delta_t: Tensor, weights: Tensor) -> Tensor:
        """Compute weighted multi-timescale decay.

        Args:
            delta_t: Time differences in hours, shape [batch, seq_len]
            weights: Mixture weights for each timescale, shape [batch, seq_len, num_timescales]

        Returns:
            Decay factors, shape [batch, seq_len]
        """
        # Get timescales (clamped for stability)
        tau = self.timescales  # [num_timescales]

        # Compute decay for each timescale
        # delta_t: [B, N] -> [B, N, 1]
        # tau: [K] -> [1, 1, K]
        decay_per_scale = torch.exp(-delta_t.unsqueeze(-1) / tau.view(1, 1, -1))
        # decay_per_scale: [B, N, K]

        # Weighted combination
        weighted_decay = (decay_per_scale * weights).sum(dim=-1)
        # weighted_decay: [B, N]

        return weighted_decay

    def get_timescale_names(self) -> list:
        """Get descriptive names for each timescale."""
        return ["fast", "medium", "slow"][: self.num_timescales]

    def get_timescale_stats(self) -> dict:
        """Get current timescale statistics for logging."""
        tau = self.timescales.detach()
        names = self.get_timescale_names()
        return {f"tau_{name}": tau[i].item() for i, name in enumerate(names)}


class TimescaleSeparationLoss(nn.Module):
    """Regularization loss to prevent timescale collapse.

    This loss encourages the learned timescales to remain well-separated,
    preventing them from collapsing to similar values during training.
    """

    def __init__(self, min_ratio: float = 4.0, weight: float = 0.01):
        """Initialize the separation loss.

        Args:
            min_ratio: Minimum desired ratio between adjacent timescales
            weight: Loss weight for balancing with main objective
        """
        super().__init__()
        self.min_ratio = min_ratio
        self.weight = weight

    def forward(self, log_tau: Tensor) -> Tensor:
        """Compute separation loss.

        Args:
            log_tau: Log-timescale parameters, shape [num_timescales]

        Returns:
            Scalar separation loss
        """
        tau = torch.exp(log_tau)
        n = len(tau)

        # Sort timescales to compute adjacent ratios
        sorted_tau, _ = torch.sort(tau)

        # Compute ratios between adjacent timescales
        ratios = sorted_tau[1:] / sorted_tau[:-1]

        # Penalize if any ratio is below minimum
        violations = F.relu(self.min_ratio - ratios)

        return self.weight * violations.mean()


class GradientClippedDecay(AdvancedFLUIDDecay):
    """AdvancedFLUIDDecay with built-in gradient clipping for log_tau.

    This variant automatically clips gradients on the log_tau parameters
    during the backward pass to prevent training instability.
    """

    def __init__(self, *args, max_grad_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_grad_norm = max_grad_norm

        # Register backward hook for gradient clipping
        self.log_tau.register_hook(self._clip_grad)

    def _clip_grad(self, grad: Tensor) -> Tensor:
        """Clip gradients to prevent instability."""
        return torch.clamp(grad, -self.max_grad_norm, self.max_grad_norm)


# Example usage and testing
if __name__ == "__main__":
    # Test basic functionality
    decay_module = AdvancedFLUIDDecay()

    batch_size, seq_len = 4, 10
    delta_t = torch.rand(batch_size, seq_len) * 48  # 0-48 hours
    weights = F.softmax(torch.randn(batch_size, seq_len, 3), dim=-1)

    decay = decay_module(delta_t, weights)
    print(f"Decay shape: {decay.shape}")  # [4, 10]
    print(f"Decay range: [{decay.min():.3f}, {decay.max():.3f}]")
    print(f"Timescales: {decay_module.get_timescale_stats()}")

    # Test separation loss
    sep_loss = TimescaleSeparationLoss()
    loss = sep_loss(decay_module.log_tau)
    print(f"Separation loss: {loss.item():.6f}")
