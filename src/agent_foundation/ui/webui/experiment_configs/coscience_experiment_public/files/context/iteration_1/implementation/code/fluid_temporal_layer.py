"""
FLUID Temporal Layer: Fluid Latent Update via Integrated Dynamics.

Key Innovation: Analytical closed-form solution to continuous-time temporal decay
without requiring numerical ODE solvers.

Core equation:
    h(t+Δt) = exp(-Δt/τ) · h(t) + (1 - exp(-Δt/τ)) · f(x)

Benefits:
- GPU-parallelizable (no sequential ODE solving)
- Handles arbitrary time gaps (seconds to months)
- Fixed timescale τ in Iteration 1 (learnable τ in future iterations)
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FLUIDConfig:
    """Configuration for FLUID Temporal Layer."""

    d_model: int = 512
    base_timescale: float = 86400.0  # τ = 24 hours in seconds (fixed in Iter 1)
    learnable_timescale: bool = False  # Will be True in Iteration 2
    min_timescale: float = 3600.0  # 1 hour minimum
    max_timescale: float = 2592000.0  # 30 days maximum
    decay_type: str = "exponential"  # "exponential", "power_law"
    dropout: float = 0.1


class AnalyticalDecay(nn.Module):
    """
    Analytical exponential decay with closed-form solution.

    Given the continuous-time ODE: dh/dt = -h/τ + f(x)/τ
    The closed-form solution is: h(t+Δt) = exp(-Δt/τ)·h(t) + (1-exp(-Δt/τ))·f(x)

    This is GPU-parallelizable because:
    1. exp(-Δt/τ) can be computed in parallel for all Δt
    2. No sequential dependency in the decay computation
    """

    def __init__(self, config: FLUIDConfig):
        super().__init__()
        self.config = config

        # Fixed timescale in Iteration 1
        self.register_buffer("base_tau", torch.tensor(config.base_timescale))

    def compute_decay_weight(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Compute decay weight exp(-Δt/τ).

        Args:
            delta_t: Time differences [batch, seq_len] in seconds

        Returns:
            Decay weights [batch, seq_len] in range (0, 1)
        """
        # Clamp delta_t to avoid numerical issues
        delta_t = torch.clamp(delta_t, min=0.0)

        if self.config.decay_type == "exponential":
            decay = torch.exp(-delta_t / self.base_tau)
        elif self.config.decay_type == "power_law":
            decay = 1.0 / (1.0 + delta_t / self.base_tau)
        else:
            raise ValueError(f"Unknown decay type: {self.config.decay_type}")

        return decay

    def forward(
        self, h_prev: torch.Tensor, f_x: torch.Tensor, delta_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply analytical decay update.

        Args:
            h_prev: Previous hidden state [batch, seq_len, d_model]
            f_x: Input contribution [batch, seq_len, d_model]
            delta_t: Time differences [batch, seq_len] in seconds

        Returns:
            Updated hidden state [batch, seq_len, d_model]
        """
        # Compute decay weight
        decay = self.compute_decay_weight(delta_t)  # [batch, seq_len]
        decay = decay.unsqueeze(-1)  # [batch, seq_len, 1]

        # Apply closed-form solution
        # h(t+Δt) = decay · h(t) + (1 - decay) · f(x)
        h_new = decay * h_prev + (1 - decay) * f_x

        return h_new


class TimescalePredictor(nn.Module):
    """
    Predicts item-specific timescales τ(x).

    NOTE: In Iteration 1, this returns fixed τ=24h for all items.
    In Iteration 2, we will enable learned timescales.

    The intuition:
    - News articles: short τ (2-4 hours) → fast decay, relevance drops quickly
    - Music albums: long τ (weeks/months) → slow decay, stays relevant longer
    """

    def __init__(self, config: FLUIDConfig):
        super().__init__()
        self.config = config

        if config.learnable_timescale:
            # Will be enabled in Iteration 2
            self.tau_predictor = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Linear(config.d_model // 2, 1),
                nn.Sigmoid(),  # Output in (0, 1), scaled to (min_tau, max_tau)
            )

        self.register_buffer("min_tau", torch.tensor(config.min_timescale))
        self.register_buffer("max_tau", torch.tensor(config.max_timescale))
        self.register_buffer("base_tau", torch.tensor(config.base_timescale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict timescale for each item.

        Args:
            x: Item features [batch, seq_len, d_model]

        Returns:
            Timescales [batch, seq_len] in seconds
        """
        if self.config.learnable_timescale:
            # Learned timescale (Iteration 2)
            tau_normalized = self.tau_predictor(x).squeeze(-1)  # [batch, seq_len]
            tau = self.min_tau + tau_normalized * (self.max_tau - self.min_tau)
        else:
            # Fixed timescale (Iteration 1)
            batch, seq_len, _ = x.shape
            tau = self.base_tau.expand(batch, seq_len)

        return tau


class FLUIDTemporalLayer(nn.Module):
    """
    FLUID: Complete continuous-time temporal layer for sequential recommendation.

    Handles arbitrary time gaps between interactions using analytical decay,
    without the computational overhead of numerical ODE solvers.

    Key advantages over positional encoding:
    1. Semantically meaningful: 1-week gap treated differently from 1-second gap
    2. Continuous: handles any time granularity
    3. Learnable: can adapt to item-specific temporal dynamics (Iteration 2)

    Example:
        >>> config = FLUIDConfig(d_model=512, base_timescale=86400)
        >>> fluid = FLUIDTemporalLayer(config)
        >>>
        >>> # Sequence of interactions with timestamps
        >>> x = torch.randn(32, 100, 512)
        >>> timestamps = torch.arange(100).float().unsqueeze(0).expand(32, -1)
        >>> timestamps = timestamps * 3600  # hourly interactions
        >>>
        >>> output = fluid(x, timestamps)
    """

    def __init__(self, config: FLUIDConfig):
        super().__init__()
        self.config = config

        # Input transformation f(x)
        self.input_transform = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # Analytical decay computation
        self.decay = AnalyticalDecay(config)

        # Timescale predictor (fixed in Iter 1, learnable in Iter 2)
        self.timescale_predictor = TimescalePredictor(config)

        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def compute_delta_t(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute time differences between consecutive interactions.

        Args:
            timestamps: Interaction timestamps [batch, seq_len] in seconds

        Returns:
            Time differences [batch, seq_len]
        """
        # Pad with 0 at the beginning (no previous interaction for first item)
        padded = F.pad(timestamps, (1, 0), value=timestamps[:, 0:1].squeeze(-1).mean())
        delta_t = timestamps - padded[:, :-1]

        # Ensure non-negative
        delta_t = torch.clamp(delta_t, min=0.0)

        return delta_t

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        return_decay_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply continuous-time temporal modeling.

        Args:
            x: Input sequence [batch, seq_len, d_model]
            timestamps: Interaction times [batch, seq_len] in seconds
            return_decay_weights: Whether to return decay weights for analysis

        Returns:
            output: Temporally-modulated sequence [batch, seq_len, d_model]
            decay_weights: (optional) Decay weights [batch, seq_len]
        """
        batch, seq_len, d_model = x.shape

        # Compute time differences
        delta_t = self.compute_delta_t(timestamps)  # [batch, seq_len]

        # Get item-specific timescales (fixed τ=24h in Iteration 1)
        tau = self.timescale_predictor(x)  # [batch, seq_len]

        # Transform input
        f_x = self.input_transform(x)

        # Initialize hidden state with first input
        h = torch.zeros_like(f_x)
        h[:, 0] = f_x[:, 0]

        # Apply analytical decay sequentially
        # (Can be parallelized with scan operations in production)
        decay_weights = []
        for t in range(1, seq_len):
            dt = delta_t[:, t : t + 1]  # [batch, 1]

            # Compute decay weight using current timescale
            decay_weight = torch.exp(-dt / tau[:, t : t + 1])
            decay_weights.append(decay_weight)

            # Apply decay update
            h[:, t] = decay_weight * h[:, t - 1] + (1 - decay_weight) * f_x[:, t]

        # Output projection
        output = self.dropout(self.output_proj(h))

        if return_decay_weights:
            decay_weights = torch.cat(decay_weights, dim=1)  # [batch, seq_len-1]
            return output, decay_weights

        return output, None

    def get_effective_timescale(self) -> float:
        """Get the current effective timescale in hours."""
        return self.config.base_timescale / 3600.0


class FLUIDTemporalLayerV1(FLUIDTemporalLayer):
    """
    FLUID V1: Iteration 1 version with fixed τ=24h timescale.

    Known Limitations (to be addressed in Iteration 2):
    - Fixed τ=24h for all items (news vs albums treated the same)
    - May over-value stale news articles
    - May under-value evergreen content like music albums

    Expected Performance:
    - Overall re-engagement: +6% (target: +8-12%)
    - Temporal-sensitive items: +4% (below target)
    - Non-temporal items: +7% (meets target)
    """

    def __init__(self, config: FLUIDConfig):
        # Ensure fixed timescale for Iteration 1
        config.learnable_timescale = False
        config.base_timescale = 86400.0  # 24 hours
        super().__init__(config)

    def get_iteration_info(self) -> Dict[str, str]:
        """Get information about this iteration's configuration."""
        return {
            "version": "v1",
            "iteration": "1",
            "timescale_type": "fixed",
            "base_timescale": "24 hours",
            "known_limitation": "Fixed τ treats all items the same, "
            "hurting temporal-sensitive items like news",
            "improvement_potential": "Learn per-category timescales in Iteration 2",
        }
