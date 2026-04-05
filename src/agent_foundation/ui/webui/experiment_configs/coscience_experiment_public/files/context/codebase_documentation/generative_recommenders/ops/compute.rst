==================
Compute Operations
==================

.. module:: generative_recommenders.ops.hstu_compute
   :synopsis: Fused compute operations for HSTU

This document describes the fused compute operations for HSTU efficiency.

Overview
========

Fused operations combine multiple sequential operations into single kernel launches:

- Reduces memory bandwidth requirements
- Eliminates intermediate tensor allocations
- Maximizes GPU utilization

Main Functions
==============

hstu_preprocess_and_attention
-----------------------------

.. function:: hstu_preprocess_and_attention(x, norm_weight, norm_bias, num_heads, attn_dim, hidden_dim, uvqk_weight, uvqk_bias, causal=True, num_targets=None, sort_by_length=True, kernel=HammerKernel.PYTORCH)

   Fused operation combining layer norm, linear projection, and attention.

   :param x: Input embeddings ``[L, d]``
   :param norm_weight: Layer norm weight ``[d]``
   :param norm_bias: Layer norm bias ``[d]``
   :param num_heads: Number of attention heads
   :param attn_dim: Attention dimension per head
   :param hidden_dim: Hidden dimension per head
   :param uvqk_weight: Unified projection weight ``[d, 2h(d_h + d_a)]``
   :param uvqk_bias: Unified projection bias ``[2h(d_h + d_a)]``
   :param causal: Use causal masking
   :param num_targets: Target counts for target-aware attention
   :param sort_by_length: Sort by length for efficiency
   :param kernel: Kernel backend
   :returns: Tuple of (U, attn_output, K, V)

   **Fused Operations**:

   1. Layer norm on input: ``x_norm = LN(x)``
   2. Single linear projection: ``[U, V, Q, K] = x_norm @ uvqk_weight + uvqk_bias``
   3. Multi-head attention: ``attn = MHA(Q, K, V)``

   **Example**:

   .. code-block:: python

      from generative_recommenders.ops import hstu_preprocess_and_attention

      u, attn, k, v = hstu_preprocess_and_attention(
          x=embeddings,
          norm_weight=layer.input_norm_weight,
          norm_bias=layer.input_norm_bias,
          num_heads=4,
          attn_dim=128,
          hidden_dim=128,
          uvqk_weight=layer.uvqk_weight,
          uvqk_bias=layer.uvqk_bias,
          causal=True,
          num_targets=targets,
      )

hstu_compute_output
-------------------

.. function:: hstu_compute_output(attn, u, x, norm_weight, norm_bias, output_weight, dropout_ratio, num_heads, linear_dim, concat_ux=True, training=True, kernel=HammerKernel.PYTORCH)

   Fused output computation with gating and residual.

   :param attn: Attention output ``[L, h*d_h]``
   :param u: Gating tensor ``[L, h*d_h]``
   :param x: Residual input ``[L, d]``
   :param norm_weight: Layer norm weight ``[d]``
   :param norm_bias: Layer norm bias ``[d]``
   :param output_weight: Output projection ``[3*h*d_h, d]``
   :param dropout_ratio: Dropout ratio
   :param num_heads: Number of heads
   :param linear_dim: Linear dimension per head
   :param concat_ux: Whether to concatenate U and X
   :param training: Training mode
   :param kernel: Kernel backend
   :returns: Output embeddings ``[L, d]``

   **Fused Operations**:

   1. Normalize attention: ``attn_norm = GroupNorm(attn)``
   2. Gate: ``gated = attn_norm * sigmoid(u)``
   3. Concatenate: ``concat = [gated; u; x]`` (if concat_ux=True)
   4. Project: ``projected = concat @ output_weight``
   5. Normalize: ``output_norm = LN(projected)``
   6. Residual: ``output = x + Dropout(output_norm)``

   **Example**:

   .. code-block:: python

      from generative_recommenders.ops import hstu_compute_output

      output = hstu_compute_output(
          attn=attention_output,
          u=gating_tensor,
          x=residual_input,
          norm_weight=layer.output_norm_weight,
          norm_bias=layer.output_norm_bias,
          output_weight=layer.output_weight,
          dropout_ratio=0.1,
          num_heads=4,
          linear_dim=128,
          training=True,
      )

Memory Efficiency
=================

Fusion Benefits
---------------

+--------------------------+------------------+------------------+
| Operation                | Unfused Memory   | Fused Memory     |
+==========================+==================+==================+
| Layer Norm               | 2 x [L, d]       | -                |
+--------------------------+------------------+------------------+
| UVQK Projection          | 4 x [L, h*d]     | -                |
+--------------------------+------------------+------------------+
| Attention                | 3 x [L, h*d]     | -                |
+--------------------------+------------------+------------------+
| **Total (Unfused)**      | 9 x [L, d]       | -                |
+--------------------------+------------------+------------------+
| **Total (Fused)**        | -                | 4 x [L, d]       |
+--------------------------+------------------+------------------+

**Memory Reduction**: ~2.25x

Cross-References
================

- :doc:`attention` - Attention operations
- :doc:`kernels` - Kernel implementations
- :doc:`/generative_recommenders/modules/stu` - STU layer usage
