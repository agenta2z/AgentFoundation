"""
PyTorch 2 Compilation Optimization Patterns

This module demonstrates techniques for enabling effective torch.compile
optimization by eliminating graph breaks.

Key techniques:
1. @torch.fx.wrap for custom operations
2. Replace .item() with torch operations
3. Remove data-dependent control flow
4. Use torch.where instead of Python conditionals

Impact: 15-30% QPS improvement when graph breaks are eliminated
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# Use @torch.fx.wrap to make custom ops compatible with torch.compile
@torch.fx.wrap
def fx_compatible_embedding_lookup(
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    num_candidates: torch.Tensor,
) -> torch.Tensor:
    """
    FX-compatible embedding lookup that works with torch.compile.

    The @torch.fx.wrap decorator tells the compiler to treat this
    function as an opaque operation, avoiding graph breaks from
    unsupported operations inside.

    Args:
        embeddings: [B, N, D] embedding tensor
        indices: [B] start indices for each sample
        num_candidates: [B] number of candidates per sample

    Returns:
        Selected embeddings
    """
    B, N, D = embeddings.shape
    device = embeddings.device

    # Create mask using tensor operations (no .item())
    positions = torch.arange(N, device=device).expand(B, N)
    start_mask = positions >= indices.unsqueeze(1)
    end_mask = positions < (indices + num_candidates).unsqueeze(1)
    mask = start_mask & end_mask

    # Apply mask
    return embeddings * mask.unsqueeze(-1).float()


def replace_item_with_where(
    x: torch.Tensor,
    threshold: torch.Tensor,
    if_true: torch.Tensor,
    if_false: torch.Tensor,
) -> torch.Tensor:
    """
    Replace Python conditionals using .item() with torch.where.

    BAD (causes graph break):
        if x.max().item() > threshold.item():
            return if_true
        return if_false

    GOOD (compilable):
        return torch.where(x.max() > threshold, if_true, if_false)

    Args:
        x: Input tensor
        threshold: Threshold tensor (not scalar!)
        if_true: Result if condition is true
        if_false: Result if condition is false

    Returns:
        Selected result based on condition
    """
    condition = x.max() > threshold
    return torch.where(condition, if_true, if_false)


class CompileFriendlyModule(nn.Module):
    """
    Example module designed for torch.compile compatibility.

    Key principles:
    1. No .item() calls
    2. No Python control flow based on tensor values
    3. All custom ops wrapped with @torch.fx.wrap
    4. Use torch operations instead of Python operations
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass designed for torch.compile.

        Args:
            x: [B, N, D] input tensor
            mask: [B, N] optional mask tensor

        Returns:
            Processed output
        """
        # Use torch.where instead of Python if/else for mask
        if mask is not None:
            # Expand mask for broadcasting
            expanded_mask = mask.unsqueeze(-1).float()
            x = x * expanded_mask

        # Standard operations (all compilable)
        x = self.norm(x)
        x = self.linear(x)
        x = torch.relu(x)

        return x


def analyze_graph_breaks(model: nn.Module, sample_input: torch.Tensor) -> None:
    """
    Analyze model for torch.compile graph breaks.

    Use this to identify operations that prevent compilation.

    Args:
        model: Model to analyze
        sample_input: Sample input tensor
    """
    import torch._dynamo as dynamo

    explanation = dynamo.explain(model)(sample_input)
    print("Graph Breaks Analysis:")
    print(f"  Number of graphs: {explanation.graph_count}")
    print(f"  Graph breaks: {len(explanation.break_reasons)}")

    for i, reason in enumerate(explanation.break_reasons):
        print(f"  Break {i+1}: {reason}")


def compile_model_optimally(
    model: nn.Module,
    mode: str = "reduce-overhead",
) -> nn.Module:
    """
    Compile model with optimal settings for training throughput.

    Modes:
    - "default": Balanced compilation
    - "reduce-overhead": Aggressive optimization, best for training
    - "max-autotune": Maximum optimization, longer compile time

    Args:
        model: Model to compile
        mode: Compilation mode

    Returns:
        Compiled model
    """
    return torch.compile(
        model,
        mode=mode,
        fullgraph=False,  # Allow partial compilation if needed
        dynamic=True,  # Support dynamic shapes
    )


# Pattern: Replace dynamic shape operations
def compilable_dynamic_slice(
    x: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """
    Dynamic slicing that works with torch.compile.

    BAD (causes graph break due to dynamic indexing):
        results = []
        for i in range(B):
            results.append(x[i, :lengths[i].item()])

    GOOD (compilable with masking):
        mask = arange < lengths.unsqueeze(1)
        return x * mask.unsqueeze(-1)

    Args:
        x: [B, N, D] input tensor
        lengths: [B] length for each sample
        max_length: Maximum sequence length

    Returns:
        Masked output (zeros beyond each sample's length)
    """
    B, N, D = x.shape
    device = x.device

    # Create position indices
    positions = torch.arange(N, device=device).expand(B, N)

    # Create mask from lengths
    mask = positions < lengths.unsqueeze(1)

    # Apply mask (compilable operation)
    return x * mask.unsqueeze(-1).float()


# Configuration for optimal PT2 compilation
PT2_OPTIMIZATION_CONFIG = {
    "mode": "reduce-overhead",
    "fullgraph": False,  # Allow partial graphs
    "dynamic": True,  # Support dynamic batch sizes
    "disable": False,  # Enable compilation
}
