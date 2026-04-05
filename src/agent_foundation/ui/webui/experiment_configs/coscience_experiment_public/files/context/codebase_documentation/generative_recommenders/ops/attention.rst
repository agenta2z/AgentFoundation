=======================
Attention Operations
=======================

.. module:: generative_recommenders.ops.hstu_attention
   :synopsis: Multi-head attention for HSTU

This document describes the attention operations used in the HSTU architecture.

Overview
========

The attention module provides:

- Multi-head attention for sequential recommendation
- Target-aware attention for candidate scoring
- Cached incremental attention for M-FALCON inference
- Multiple kernel backends (PyTorch, Triton, CUDA)

Main Functions
==============

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

   **Example**:

   .. code-block:: python

      from generative_recommenders.ops import hstu_mha

      output = hstu_mha(
          max_seq_len=512,
          alpha=1.0 / math.sqrt(128),
          q=queries,
          k=keys,
          v=values,
          seq_offsets=offsets,
          causal=True,
          num_targets=targets,
      )

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
   Complexity: O(bm x n x d) instead of O(m x n x d)

Attention Masking
=================

Causal Masking
--------------

Standard causal mask for autoregressive modeling:

.. code-block:: text

   Position:  0  1  2  3  4
           0 [1  0  0  0  0]
           1 [1  1  0  0  0]
           2 [1  1  1  0  0]
           3 [1  1  1  1  0]
           4 [1  1  1  1  1]

Target-Aware Masking
--------------------

When ``num_targets`` is provided, candidates (targets) can attend to user history
but not to each other:

.. code-block:: text

   UIH positions: 0-4 (history)
   Target positions: 5-7 (candidates)

   Position:  0  1  2  3  4 | 5  6  7
           0 [1  0  0  0  0 | 0  0  0]  UIH
           1 [1  1  0  0  0 | 0  0  0]  UIH
           2 [1  1  1  0  0 | 0  0  0]  UIH
           3 [1  1  1  1  0 | 0  0  0]  UIH
           4 [1  1  1  1  1 | 0  0  0]  UIH
           ----------------------------
           5 [1  1  1  1  1 | 0  0  0]  Target (attends to UIH only)
           6 [1  1  1  1  1 | 0  0  0]  Target (no cross-attention)
           7 [1  1  1  1  1 | 0  0  0]  Target (independent)

This enables efficient batch inference over multiple candidates.

Kernel Selection
================

The ``HammerKernel`` enum provides multiple backends:

.. code-block:: python

   from generative_recommenders.ops import HammerKernel

   # PyTorch reference (CPU/GPU)
   output = hstu_mha(..., kernel=HammerKernel.PYTORCH)

   # Triton optimized (GPU)
   output = hstu_mha(..., kernel=HammerKernel.TRITON)

   # Compiled Triton for H100+ (GPU)
   output = hstu_mha(..., kernel=HammerKernel.TRITON_CC)

   # Custom CUDA kernels (GPU)
   output = hstu_mha(..., kernel=HammerKernel.CUDA)

Cross-References
================

- :doc:`compute` - Fused compute operations
- :doc:`kernels` - Kernel implementations
- :doc:`/architecture/generative_recommenders` - Architecture overview
