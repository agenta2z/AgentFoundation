===================
Kernel Implementations
===================

.. module:: generative_recommenders.ops.triton
   :synopsis: Triton kernel implementations

This document describes the kernel implementations for HSTU operations.

Overview
========

The codebase provides multiple kernel backends:

- **PyTorch**: Reference implementation for correctness validation
- **Triton**: GPU-optimized kernels using OpenAI Triton
- **Triton CC**: Compiled Triton kernels for H100+ GPUs
- **CUDA**: Custom CUDA kernels for maximum performance

Kernel Selection
================

.. code-block:: python

   from generative_recommenders.ops import HammerKernel

   class HammerKernel(Enum):
       PYTORCH = "pytorch"       # Reference implementation
       TRITON = "triton"         # GPU-optimized Triton kernels
       TRITON_CC = "triton_cc"   # Compiled Triton (H100+)
       CUDA = "cuda"             # Custom CUDA kernels

Automatic Selection
-------------------

The ``HammerModule`` base class automatically selects kernels:

.. code-block:: python

   class HammerModule(torch.nn.Module):
       def hammer_kernel(self) -> HammerKernel:
           if self._is_inference and self._use_triton_cc:
               return HammerKernel.TRITON_CC
           return HammerKernel.TRITON

Triton Kernels
==============

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
       """Optimized attention kernel for GPU."""
       pass

**Performance Features**:

- Block-tiled computation for memory efficiency
- Shared memory utilization for key/value reuse
- Warp-level parallelism for maximum throughput
- Coalesced memory access patterns
- Autotuning for optimal block sizes

triton_delta_hstu_mha
---------------------

.. code-block:: python

   @triton.jit
   def triton_delta_hstu_mha_kernel(
       delta_q_ptr, k_ptr, v_ptr, output_ptr,
       seq_offsets_ptr, num_targets_ptr, kv_cache_lengths_ptr,
       L, bm, H, D_QK, D_V,
       alpha, max_attn_len,
       BLOCK_SIZE: tl.constexpr,
   ):
       """Incremental attention for M-FALCON inference."""
       pass

**Key Optimization**: Only computes attention for new queries against cached K/V.

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
       # Build attention mask
       mask = build_attention_mask(seq_offsets, causal, num_targets)

       # Compute attention scores
       scores = torch.einsum('lhd,mhd->lmh', q, k) * alpha

       # Apply mask
       scores = scores.masked_fill(~mask, 0.0)

       # Compute output (pointwise aggregation, not softmax)
       output = torch.einsum('lmh,mhd->lhd', scores, v)

       return output

Used for:

- CPU fallback when GPU is unavailable
- Numerical validation of optimized kernels
- Debugging and development

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

Kernel Development
==================

Adding New Kernels
------------------

To add a new kernel implementation:

1. Add implementation to appropriate directory (``triton/``, ``pytorch/``, ``cuda/``)
2. Register in dispatch function
3. Add tests comparing against reference implementation

.. code-block:: python

   # ops/triton/new_kernel.py
   @triton.jit
   def new_kernel(...):
       pass

   # ops/__init__.py
   def dispatch_kernel(kernel: HammerKernel):
       if kernel == HammerKernel.TRITON:
           return triton_implementation
       return pytorch_implementation

Testing Kernels
---------------

Kernels are validated against PyTorch reference:

.. code-block:: python

   def test_kernel_correctness():
       # Generate random inputs
       q = torch.randn(L, H, D)
       k = torch.randn(L, H, D)
       v = torch.randn(L, H, D)

       # Compute with reference
       ref_output = pytorch_hstu_mha(q, k, v, ...)

       # Compute with optimized kernel
       opt_output = triton_hstu_mha(q, k, v, ...)

       # Verify numerical equivalence
       torch.testing.assert_close(ref_output, opt_output, rtol=1e-3, atol=1e-3)

Cross-References
================

- :doc:`attention` - Attention operations
- :doc:`compute` - Fused compute operations
- :doc:`/architecture/generative_recommenders` - Architecture overview
