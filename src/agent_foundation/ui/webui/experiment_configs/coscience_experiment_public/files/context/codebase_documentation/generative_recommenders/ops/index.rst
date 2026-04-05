==================================
Generative Recommenders Operations
==================================

This section documents the optimized operations used in the Generative Recommenders framework.

.. toctree::
   :maxdepth: 2

   attention
   compute
   kernels

Operations Overview
===================

The ``generative_recommenders/ops/`` directory contains optimized GPU operations:

.. code-block:: text

    ops/
    ├── hstu_attention.py       # Multi-head attention dispatcher
    ├── hstu_compute.py         # Fused linear operations
    ├── jagged_tensors.py       # Jagged tensor utilities
    ├── layer_norm.py           # Group/Layer normalization
    ├── position.py             # Relative position encoding
    ├── triton/                 # Triton kernel implementations
    │   └── triton_hstu_attention.py
    ├── pytorch/                # PyTorch reference implementations
    │   └── pt_hstu_attention.py
    └── cpp/                    # CUDA/C++ kernels

Kernel Selection
================

The ``HammerKernel`` enum provides multiple backend implementations:

.. code-block:: python

    from enum import Enum

    class HammerKernel(Enum):
        PYTORCH = "pytorch"       # Reference implementation
        TRITON = "triton"         # GPU-optimized Triton kernels
        TRITON_CC = "triton_cc"   # Compiled Triton (H100+)
        CUDA = "cuda"             # Custom CUDA kernels

Selection is automatic based on mode:

- **Training**: Uses ``TRITON`` kernels
- **Inference**: Uses ``TRITON_CC`` (compiled) for H100+ GPUs

Attention Operations
====================

hstu_mha
--------

.. function:: hstu_mha(max_seq_len, alpha, q, k, v, seq_offsets, causal=True, dropout_pr=0.0, training=True, num_targets=None, max_attn_len=0, contextual_seq_len=0, sort_by_length=False, kernel=HammerKernel.PYTORCH)

   Multi-head attention for HSTU with pointwise aggregation (non-softmax).

   :param max_seq_len: Maximum sequence length in batch
   :param alpha: Attention scale factor (typically 1/sqrt(d_attn))
   :param q: Query tensor ``[L, H, d_qk]``
   :param k: Key tensor ``[L, H, d_qk]``
   :param v: Value tensor ``[L, H, d_v]``
   :param seq_offsets: Jagged sequence offsets ``[B+1]``
   :param causal: Use causal (autoregressive) masking
   :param dropout_pr: Dropout probability
   :param training: Whether in training mode
   :param num_targets: Number of target tokens per sample (for target-aware attention)
   :param max_attn_len: Maximum attention span (for sparsity)
   :param contextual_seq_len: Length of contextual prefix
   :param sort_by_length: Sort sequences by length for efficiency
   :param kernel: Which kernel backend to use
   :returns: Attention output ``[L, H, d_v]``

   **Key Innovation**: Uses pointwise aggregated attention (not softmax normalized),
   which preserves the intensity of user preferences.

delta_hstu_mha
--------------

.. function:: delta_hstu_mha(max_seq_len, alpha, delta_q, k, v, seq_offsets, num_targets=None, max_attn_len=0, contextual_seq_len=0, kernel=HammerKernel.PYTORCH)

   Incremental attention for cached inference (M-FALCON algorithm).

   :param max_seq_len: Maximum sequence length
   :param alpha: Attention scale factor
   :param delta_q: New query tokens ``[bm, H, d_qk]``
   :param k: Cached K values ``[full_seq, H, d_qk]``
   :param v: Cached V values ``[full_seq, H, d_v]``
   :param seq_offsets: Sequence offsets
   :param num_targets: Target counts
   :param max_attn_len: Attention span limit
   :param contextual_seq_len: Contextual prefix length
   :param kernel: Kernel backend
   :returns: Attention output for new queries

   **Key Feature**: Only attends to cached K/V, not to other delta_q tokens.
   Complexity: O(bm × n × d) instead of O(m × n × d)

Fused Compute Operations
========================

hstu_preprocess_and_attention
-----------------------------

.. function:: hstu_preprocess_and_attention(x, norm_weight, norm_bias, num_heads, attn_dim, hidden_dim, uvqk_weight, uvqk_bias, causal=True, num_targets=None, sort_by_length=True, kernel=HammerKernel.PYTORCH)

   Fused operation combining layer norm, linear projection, and attention.

   :param x: Input embeddings ``[L, d]``
   :param norm_weight: Layer norm weight
   :param norm_bias: Layer norm bias
   :param num_heads: Number of attention heads
   :param attn_dim: Attention dimension
   :param hidden_dim: Hidden dimension
   :param uvqk_weight: Unified projection weight ``[d, 2h(d_h + d_a)]``
   :param uvqk_bias: Unified projection bias
   :param causal: Use causal masking
   :param num_targets: Target counts for target-aware attention
   :param sort_by_length: Sort by length for efficiency
   :param kernel: Kernel backend
   :returns: Tuple of (U, attn_output, K, V)

   **Fused Operations**:

   1. Layer norm on input
   2. Single linear projection → U, V, Q, K
   3. Multi-head attention

hstu_compute_output
-------------------

.. function:: hstu_compute_output(attn, u, x, norm_weight, norm_bias, output_weight, dropout_ratio, num_heads, linear_dim, concat_ux=True, training=True, kernel=HammerKernel.PYTORCH)

   Fused output computation with gating and residual.

   :param attn: Attention output ``[L, h*d_h]``
   :param u: Gating tensor ``[L, h*d_h]``
   :param x: Residual input ``[L, d]``
   :param norm_weight: Layer norm weight
   :param norm_bias: Layer norm bias
   :param output_weight: Output projection ``[3*h*d_h, d]``
   :param dropout_ratio: Dropout ratio
   :param num_heads: Number of heads
   :param linear_dim: Linear dimension
   :param concat_ux: Whether to concatenate U and X
   :param training: Training mode
   :param kernel: Kernel backend
   :returns: Output embeddings ``[L, d]``

   **Fused Operations**:

   1. Normalize: attn·V
   2. Gate: norm(attn·V) ⊙ U(X)
   3. Project: [U ⊙ attn; U; X] → [d]
   4. Residual + Dropout

Jagged Tensor Operations
========================

.. module:: generative_recommenders.ops.jagged_tensors
   :synopsis: Utilities for variable-length sequence processing

The framework uses jagged tensors to efficiently handle variable-length sequences
without padding overhead.

Key Functions
-------------

- ``split_2D_jagged``: Split embeddings by sequence boundaries
- ``reorder_batched_ad_indices``: Reorder indices for batched processing
- ``cumsum``: Cumulative sum for offset computation

These operations use ``fbgemm_gpu`` for maximum performance.

Triton Kernels
==============

The ``ops/triton/`` directory contains Triton implementations of core operations:

triton_hstu_mha
---------------

.. code-block:: python

    @triton.jit
    def triton_hstu_mha_kernel(
        q_ptr, k_ptr, v_ptr, output_ptr,
        seq_offsets_ptr, num_targets_ptr,
        L, H, D_QK, D_V,
        alpha, max_attn_len,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Optimized attention kernel for GPU
        pass

**Performance Features**:

- Block-tiled computation for memory efficiency
- Shared memory utilization
- Warp-level parallelism
- Coalesced memory access patterns

PyTorch Reference
=================

The ``ops/pytorch/`` directory contains reference implementations:

.. code-block:: python

    def pytorch_hstu_mha(
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        causal: bool = True,
        num_targets: Optional[torch.Tensor] = None,
        max_attn_len: int = 0,
    ) -> torch.Tensor:
        """Reference PyTorch implementation for validation."""
        pass

Used for:

- CPU fallback
- Numerical validation
- Debugging

Performance Comparison
======================

+----------------+------------------+------------------+---------+
| Kernel         | Training Speed   | Inference Speed  | Memory  |
+================+==================+==================+=========+
| PYTORCH        | 1.0x (baseline)  | 1.0x             | 1.0x    |
+----------------+------------------+------------------+---------+
| TRITON         | 5.3x             | 4.1x             | 0.6x    |
+----------------+------------------+------------------+---------+
| TRITON_CC      | 5.3x             | 15.2x (H100)     | 0.6x    |
+----------------+------------------+------------------+---------+
| CUDA           | 5.1x             | 5.0x             | 0.5x    |
+----------------+------------------+------------------+---------+

Cross-References
================

- :doc:`/architecture/generative_recommenders` - Architecture overview
- :doc:`/generative_recommenders/modules/index` - Module documentation
- :doc:`/workflows/inference` - Inference workflows
