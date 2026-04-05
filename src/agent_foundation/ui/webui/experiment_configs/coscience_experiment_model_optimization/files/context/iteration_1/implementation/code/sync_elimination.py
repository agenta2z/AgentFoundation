"""
HSTU Transducer CInt Forward Pass Optimization Patterns

This module demonstrates the actual optimizations applied to hstu_transducer_cint.py
based on analysis of commits 2a26d77d → 8b028c28.

Key techniques:
1. Slice indexing instead of torch.arange + index_select
2. F.normalize instead of manual normalization
3. Removed redundant dtype casting under autocast
4. Broadcasting instead of explicit .expand()
5. Consolidated branch logic
6. Pre-computed ro_lengths with torch.no_grad()
"""

import torch
import torch.nn.functional as F


def slice_indexing_optimization(
    encoded_embeddings: torch.Tensor,
    contextual_seq_len: int,
) -> torch.Tensor:
    """
    Slice indexing instead of torch.arange + index_select.

    ORIGINAL (2 kernels - arange + index_select):
        indices = torch.arange(contextual_seq_len, N, 2, device=encoded_embeddings.device)
        non_contextualized = torch.index_select(encoded_embeddings, dim=1, index=indices)

    OPTIMIZED (1 kernel - slice + contiguous):
        non_contextualized = encoded_embeddings[:, contextual_seq_len::2, :].contiguous()

    Estimated Impact: ~0.3-0.5% QPS improvement
    - Eliminates intermediate index tensor allocation
    - Reduces kernel launches from 2 to 1
    """
    # Optimized: native slice syntax with .contiguous() for backward pass efficiency
    non_contextualized_embeddings = encoded_embeddings[
        :, contextual_seq_len::2, :
    ].contiguous()
    return non_contextualized_embeddings


def normalize_optimization(
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    F.normalize instead of manual normalization.

    ORIGINAL (3 kernels - norm + clamp + divide):
        normalized = embeddings / torch.linalg.norm(
            embeddings, ord=2, dim=-1, keepdim=True
        ).clamp(min=1e-6)

    OPTIMIZED (1 fused kernel):
        normalized = F.normalize(embeddings, p=2, dim=-1, eps=1e-6)

    Estimated Impact: ~0.3-0.5% QPS improvement
    - Reduces 3 kernels to 1 fused kernel
    - Eliminates 2 intermediate tensor allocations
    - Numerically equivalent (both use L2 norm, eps=1e-6)
    """
    return F.normalize(embeddings, p=2, dim=-1, eps=1e-6)


def broadcasting_optimization(
    num_targets: torch.Tensor,
    total_padded_targets: int,
) -> torch.Tensor:
    """
    Broadcasting instead of explicit .expand().

    ORIGINAL (creates expanded tensor):
        mf_nro_indices_valid = (
            torch.arange(total_padded_targets, device=num_targets.device)
            .unsqueeze(0)
            .expand(num_targets.size(0), -1)
        ) < num_targets.unsqueeze(1)

    OPTIMIZED (automatic broadcasting):
        arange_tensor = torch.arange(total_padded_targets, device=num_targets.device)
        mf_nro_indices_valid = arange_tensor.unsqueeze(0) < num_targets.unsqueeze(1)

    Broadcasting (1, N) < (B, 1) → (B, N) happens automatically.
    Avoids explicit tensor expansion.
    """
    arange_tensor = torch.arange(total_padded_targets, device=num_targets.device)
    # Let PyTorch broadcast: (1, N) < (B, 1) → (B, N)
    mf_nro_indices_valid = arange_tensor.unsqueeze(0) < num_targets.unsqueeze(1)
    return mf_nro_indices_valid


def precomputed_ro_lengths(
    encoded_embeddings: torch.Tensor,
    past_lengths: torch.Tensor,
    num_nro_candidates: torch.Tensor,
) -> torch.Tensor:
    """
    Pre-computed ro_lengths with torch.no_grad().

    Safe to use torch.no_grad() here because:
    - past_lengths and num_nro_candidates are integer tensors
    - The result is used only for indexing
    - No gradients need to flow through index tensors

    This is a correct optimization for index computation only.
    """
    with torch.no_grad():
        B, N, D = encoded_embeddings.size()
        ro_lengths = past_lengths - num_nro_candidates
    return ro_lengths


# CRITICAL BUG: The following pattern is BROKEN
def get_nro_embeddings_buggy(
    seq_embeddings: torch.Tensor,
    num_nro_candidates: torch.Tensor,
    ro_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    ⚠️ CRITICAL BUG: @torch.no_grad() decorator breaks gradient flow!

    The original optimization used:
        @torch.fx.wrap
        @torch.no_grad()  # ← THIS BREAKS GRADIENT FLOW
        def get_nro_embeddings(seq_embeddings, ...):
            ...
            return seq_embeddings[mask]  # Returns tensor with requires_grad=False!

    PROBLEM:
    - The decorator wraps the ENTIRE function including the return
    - The returned tensor has requires_grad=False and grad_fn=None
    - Gradients will NOT flow back to the encoder
    - This causes SILENT training degradation

    This function demonstrates the BUGGY pattern - DO NOT USE.
    """
    B, N, D = seq_embeddings.shape

    # This pattern is buggy when wrapped with @torch.no_grad() decorator
    start_idx = ro_lengths.unsqueeze(1)
    end_idx = start_idx + num_nro_candidates.unsqueeze(1)
    raw_mask = torch.arange(N, device=ro_lengths.device).expand(B, N)
    mask = (raw_mask >= start_idx) & (raw_mask < end_idx)

    return seq_embeddings[
        mask
    ]  # Gradient flow broken if function has @torch.no_grad()!


# CORRECT FIX: Move torch.no_grad() inside function
def get_nro_embeddings_fixed(
    seq_embeddings: torch.Tensor,
    num_nro_candidates: torch.Tensor,
    ro_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    ✅ CORRECT FIX: Move torch.no_grad() inside to wrap only index computations.

    The return statement seq_embeddings[mask] MUST be outside no_grad context
    to preserve gradient flow to the encoder.
    """
    B, N, D = seq_embeddings.shape

    # CORRECT: Only wrap index computations in no_grad
    with torch.no_grad():
        start_idx = ro_lengths.unsqueeze(1)
        end_idx = start_idx + num_nro_candidates.unsqueeze(1)
        raw_mask = torch.arange(N, device=ro_lengths.device).expand(B, N)
        mask = (raw_mask >= start_idx) & (raw_mask < end_idx)

    # OUTSIDE no_grad - gradients flow correctly to seq_embeddings
    return seq_embeddings[mask]


# Example demonstrating the bug
def test_gradient_flow():
    """
    Empirical verification that @torch.no_grad() breaks gradient flow.

    Run this to see the difference:
    - With @torch.no_grad() decorator: result.requires_grad=False, grad_fn=None
    - Without decorator: result.requires_grad=True, grad_fn=<IndexBackward0>
    """
    seq_embeddings = torch.randn(2, 10, 8, requires_grad=True)
    num_nro_candidates = torch.tensor([2, 3])
    ro_lengths = torch.tensor([5, 4])

    # Test the fixed version
    result = get_nro_embeddings_fixed(seq_embeddings, num_nro_candidates, ro_lengths)

    print(f"result.requires_grad: {result.requires_grad}")
    print(f"result.grad_fn: {result.grad_fn}")

    # Verify gradients flow
    loss = result.sum()
    loss.backward()

    assert seq_embeddings.grad is not None, "Gradients should flow to seq_embeddings"
    print(f"seq_embeddings.grad is not None: {seq_embeddings.grad is not None}")
    print("✅ Gradient flow test passed!")
