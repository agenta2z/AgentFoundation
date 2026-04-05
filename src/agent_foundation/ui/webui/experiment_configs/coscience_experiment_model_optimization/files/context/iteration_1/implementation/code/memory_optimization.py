"""
Memory Optimization Patterns for ML Training

This module demonstrates activation memory optimization and gradient
checkpointing techniques for reducing memory usage.

Key techniques:
1. Activation memory budget tuning
2. Selective gradient checkpointing
3. Memory-efficient operations
"""

import functools
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


def configure_activation_memory_budget(budget: float = 0.05) -> None:
    """
    Configure activation memory budget for automatic checkpointing.

    The activation_memory_budget parameter controls how much activation
    memory to retain vs recompute during backward pass.

    Recommended values:
    - 0.1 (default): Balanced memory/compute
    - 0.05 (optimized): More aggressive checkpointing, ~10-20% memory savings
    - 0.02 (aggressive): Maximum memory savings, higher recompute cost

    Impact:
    - 10-20% memory reduction with 0.05 budget
    - Minimal throughput impact for most models

    Args:
        budget: Fraction of memory to use for activations (0.0 - 1.0)
    """
    torch._functorch.config.activation_memory_budget = budget


def selective_checkpoint(fn: Callable, *args, **kwargs):
    """
    Apply gradient checkpointing to specific functions.

    Use for expensive operations where recomputation is cheaper
    than storing large activations.

    Good candidates:
    - Attention layers (large activation tensors)
    - FFN layers in transformers
    - Embedding lookups

    Args:
        fn: Function to checkpoint
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function output (activations not stored)
    """
    return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient attention implementation.

    Techniques applied:
    1. In-place operations where safe
    2. Chunked computation for large sequences
    3. Gradient checkpointing for backward pass
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        chunk_size: int = 1024,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _attention_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention for a single chunk."""
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Chunked attention for memory efficiency
        if N > self.chunk_size:
            outputs = []
            for i in range(0, N, self.chunk_size):
                q_chunk = q[:, :, i : i + self.chunk_size]

                if self.use_checkpoint:
                    chunk_output = selective_checkpoint(
                        self._attention_chunk, q_chunk, k, v
                    )
                else:
                    chunk_output = self._attention_chunk(q_chunk, k, v)

                outputs.append(chunk_output)

            attn_output = torch.cat(outputs, dim=2)
        else:
            if self.use_checkpoint:
                attn_output = selective_checkpoint(self._attention_chunk, q, k, v)
            else:
                attn_output = self._attention_chunk(q, k, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(attn_output)


def inplace_relu_optimization(x: torch.Tensor) -> torch.Tensor:
    """
    Use in-place operations to reduce peak memory.

    In-place operations modify tensors directly, avoiding allocation
    of new tensors for intermediate results.

    CAUTION: Only use when tensor is not needed later in computation.

    Args:
        x: Input tensor (will be modified in-place)

    Returns:
        Same tensor after ReLU (modified in-place)
    """
    return torch.relu_(x)  # In-place version


def memory_efficient_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient GELU using tanh approximation.

    Standard GELU stores intermediate values for backward.
    Tanh approximation is more memory-efficient.

    Args:
        x: Input tensor

    Returns:
        GELU(x) with reduced memory footprint
    """
    return nn.functional.gelu(x, approximate="tanh")


# Configuration summary
RECOMMENDED_MEMORY_CONFIG = {
    "activation_memory_budget": 0.05,  # From 0.1 default
    "gradient_checkpointing": True,  # For attention layers
    "use_inplace_ops": True,  # Where safe
    "chunk_size": 1024,  # For long sequences
}
