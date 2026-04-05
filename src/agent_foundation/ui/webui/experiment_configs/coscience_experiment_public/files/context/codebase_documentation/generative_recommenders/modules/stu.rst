==============================
STU (Sequential Transduction Unit)
==============================

.. module:: generative_recommenders.modules.stu
   :synopsis: Core STU layer and stack implementations

The Sequential Transduction Unit (STU) is the fundamental building block of the HSTU architecture.

Overview
========

The STU module provides:

- Abstract base class for Sequential Transduction Units
- Single-layer STU implementation
- Multi-layer STU stacks
- Configuration dataclasses

Classes
=======

STU Base Class
--------------

.. class:: STU(HammerModule, abc.ABC)

   Abstract base class for Sequential Transduction Units.

   All STU implementations must inherit from this class and implement
   the required methods.

   .. method:: forward(x, x_lengths, x_offsets, max_seq_len, num_targets, max_kv_caching_len=0, kv_caching_lengths=None)

      Forward pass through the STU stack.

      :param x: Input embeddings ``[L, d]`` - jagged tensor of all tokens
      :param x_lengths: Sequence lengths ``[B]`` - length of each sequence in batch
      :param x_offsets: Cumulative offsets ``[B+1]`` - jagged tensor boundaries
      :param max_seq_len: Maximum sequence length in the batch
      :param num_targets: Number of target tokens ``[B]`` - candidates per sample
      :param max_kv_caching_len: Max KV cache length for inference (default: 0)
      :param kv_caching_lengths: KV cache lengths per sample (optional)
      :returns: Encoded embeddings ``[L, d]``

   .. method:: cached_forward(delta_x, num_targets, max_kv_caching_len=0, kv_caching_lengths=None)

      Cached inference with KV caching for incremental updates.

      :param delta_x: New query tokens to process
      :param num_targets: Number of targets per sample
      :param max_kv_caching_len: Maximum KV cache length
      :param kv_caching_lengths: Per-sample cache lengths
      :returns: Updated embeddings

STULayerConfig
--------------

.. class:: STULayerConfig

   Configuration dataclass for STU layers.

   :param embedding_dim: Model dimension (default: 512)
   :param num_heads: Number of attention heads (default: 4)
   :param hidden_dim: Linear hidden dimension (default: 128)
   :param attention_dim: QK attention dimension (default: 128)
   :param output_dropout_ratio: Output dropout ratio (default: 0.3)
   :param causal: Whether to use causal masking (default: True)
   :param target_aware: Enable target-aware attention (default: True)
   :param max_attn_len: Maximum attention length for sparsity (optional)
   :param attn_alpha: Attention scale factor (optional)

   **Example**:

   .. code-block:: python

      from generative_recommenders.modules.stu import STULayerConfig

      config = STULayerConfig(
          embedding_dim=512,
          num_heads=8,
          hidden_dim=256,
          attention_dim=128,
          output_dropout_ratio=0.2,
          causal=True,
          target_aware=True,
      )

STULayer
--------

.. class:: STULayer(STU)

   Single STU layer implementation.

   **Key Parameters**:

   - ``_uvqk_weight``: Unified projection for U, V, Q, K ``[d, 2h(d_hidden + d_attn)]``
   - ``_input_norm_weight/bias``: Layer norm for input
   - ``_output_weight``: Output transformation ``[3h*d_hidden, d]``
   - ``_output_norm_weight/bias``: Layer norm for output

   **Architecture**:

   .. code-block:: text

      Input X [L, d]
          ↓
      Layer Norm
          ↓
      UVQK Projection → U [L, h*d_h], V [L, h*d_h], Q [L, h*d_a], K [L, h*d_a]
          ↓
      Multi-Head Attention (Q, K, V) → Attention Output [L, h*d_h]
          ↓
      Gating: norm(attn·V) ⊙ U(X)
          ↓
      Output Projection: [U ⊙ attn; U; X] → [d]
          ↓
      Layer Norm + Residual + Dropout
          ↓
      Output [L, d]

   **Example**:

   .. code-block:: python

      from generative_recommenders.modules.stu import STULayer, STULayerConfig

      config = STULayerConfig(embedding_dim=512, num_heads=4)
      layer = STULayer(config)

      output = layer(
          x=embeddings,
          x_lengths=lengths,
          x_offsets=offsets,
          max_seq_len=max_len,
          num_targets=targets,
      )

STUStack
--------

.. class:: STUStack(STU)

   Stack of multiple STU layers.

   :param layers: List of STULayer instances
   :param is_inference: Whether in inference mode

   **Example**:

   .. code-block:: python

      from generative_recommenders.modules.stu import STUStack, STULayer, STULayerConfig

      config = STULayerConfig(embedding_dim=512, num_heads=4)
      layers = [STULayer(config) for _ in range(12)]
      stack = STUStack(layers=layers, is_inference=False)

      output = stack(
          x=embeddings,
          x_lengths=lengths,
          x_offsets=offsets,
          max_seq_len=max_len,
          num_targets=targets,
      )

Implementation Details
======================

Fused Operations
----------------

The STU layer uses fused operations for efficiency:

1. **hstu_preprocess_and_attention**: Combines layer norm, UVQK projection, and attention
2. **hstu_compute_output**: Combines gating, output projection, and residual

Memory Efficiency
-----------------

The STU layer achieves memory efficiency through:

- Fused kernels reducing intermediate tensor allocations
- Jagged tensor representation eliminating padding
- In-place operations where possible

Cross-References
================

- :doc:`/architecture/generative_recommenders` - Architecture overview
- :doc:`hstu_transducer` - Full HSTU pipeline
- :doc:`/generative_recommenders/ops/index` - Operations documentation
