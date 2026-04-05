======================================
Generative Recommenders Architecture
======================================

.. module:: generative_recommenders
   :synopsis: Hierarchical Sequential Transduction Units for Generative Recommendations

This document provides an in-depth analysis of the Generative Recommenders architecture, which implements
"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (ICML'24).

Overview
========

The Generative Recommenders (GR) framework transforms traditional Deep Learning Recommendation Models (DLRMs)
into generative models by:

1. **Unifying Feature Spaces**: Sequentializing heterogeneous categorical and numerical features
2. **Sequential Transduction**: Casting ranking and retrieval as sequential transduction tasks
3. **Generative Training**: Training models in a sequential, generative fashion
4. **Efficient Inference**: Using M-FALCON for cost amortization during inference

Key Innovation: From DLRMs to GRs
=================================

Problems Addressed
------------------

1. **Scaling Bottleneck**: Traditional DLRMs saturate at ~200B parameters
2. **Computational Complexity**: Standard self-attention is O(N³d + N²d²)
3. **High-Cardinality Vocabularies**: Billion-scale dynamic vocabularies
4. **Long Sequences**: Traditional DLRM handles only 20-100 items

Solutions Provided
------------------

- **Linear Scaling**: GRs scale linearly with compute (power-law relationship)
- **Reduced Complexity**: O(N²d + Nd²) through generative training
- **Target-Aware Attention**: Handles dynamic vocabularies efficiently
- **Long Sequences**: Processes sequences up to 8,192 length

Core Architecture
=================

HSTU (Hierarchical Sequential Transduction Unit)
------------------------------------------------

The core building block is the HSTU, which consists of:

.. code-block:: text

    Input Embeddings [L, d]
            ↓
    ┌────────────────────────────────┐
    │         STULayer (×N)          │
    │  ┌──────────────────────────┐  │
    │  │ 1. Input Layer Norm      │  │
    │  │ 2. Fused UVQK Projection │  │
    │  │ 3. Multi-Head Attention  │  │
    │  │ 4. Gated Output          │  │
    │  │ 5. Output Layer Norm     │  │
    │  │ 6. Residual Connection   │  │
    │  └──────────────────────────┘  │
    └────────────────────────────────┘
            ↓
    Output Embeddings [L, d]

STULayer Architecture
---------------------

.. code-block:: python

    @dataclass
    class STULayerConfig:
        embedding_dim: int      # Model dimension (d=512)
        num_heads: int          # Attention heads (h=4-24)
        hidden_dim: int         # Linear dim (128-512)
        attention_dim: int      # QK dimension (128)
        output_dropout_ratio: float = 0.3
        causal: bool = True
        target_aware: bool = True
        max_attn_len: Optional[int] = None
        attn_alpha: Optional[float] = None

Key operations in STULayer:

1. **Input Normalization**: Layer norm on input
2. **Fused UVQK Projection**: Single linear projection to U, V, Q, K tensors
3. **Pointwise Aggregated Attention**: Non-softmax attention (preserves intensity)
4. **Gated Output**: ``norm(attn·V) ⊙ U(X)`` (similar to SwiGLU)
5. **Residual + Output Norm**: Standard transformer pattern

HSTUTransducer Pipeline
-----------------------

The full model pipeline implemented in ``HSTUTransducer``:

.. code-block:: python

    class HSTUTransducer(HammerModule):
        def __init__(
            self,
            stu_module: STU,
            input_preprocessor: InputPreprocessor,
            output_postprocessor: Optional[OutputPostprocessor],
            input_dropout_ratio: float = 0.0,
            positional_encoder: Optional[HSTUPositionalEncoder] = None,
            is_inference: bool = True,
            return_full_embeddings: bool = False,
            listwise: bool = False,
        ):
            pass

        def forward(
            self,
            max_uih_len: int,
            max_targets: int,
            total_uih_len: int,
            total_targets: int,
            seq_lengths: torch.Tensor,
            seq_embeddings: torch.Tensor,
            seq_timestamps: torch.Tensor,
            num_targets: torch.Tensor,
            seq_payloads: Dict[str, torch.Tensor],
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            pass

Data flow through HSTUTransducer:

1. ``_preprocess()``: Raw features → embeddings + positional encodings
2. ``_hstu_compute()``: STU stack → encoded embeddings
3. ``_postprocess()``: Split UIH/targets → candidate embeddings (L2 normalized)

Attention Mechanism
===================

Pointwise Aggregated Attention
------------------------------

Unlike standard softmax attention, HSTU uses pointwise aggregated attention that:

- **Preserves intensity**: User preferences are not diluted by normalization
- **Enables target-awareness**: Candidates attend only to user history
- **Reduces computation**: More efficient for long sequences

.. code-block:: python

    def hstu_mha(
        max_seq_len: int,
        alpha: float,                   # Scale (1/sqrt(d_attn))
        q: torch.Tensor,                # [L, H, d_qk]
        k: torch.Tensor,                # [L, H, d_qk]
        v: torch.Tensor,                # [L, H, d_v]
        seq_offsets: torch.Tensor,      # Jagged offsets [B+1]
        causal: bool = True,
        num_targets: Optional[torch.Tensor] = None,
        max_attn_len: int = 0,
        kernel: HammerKernel = HammerKernel.PYTORCH,
    ) -> torch.Tensor:
        pass

Target-Aware Attention
----------------------

When ``num_targets`` is provided, the attention mechanism ensures:

- Candidates **can** attend to user interaction history (UIH)
- Candidates **cannot** attend to each other
- UIH tokens use standard causal masking

This is crucial for inference with multiple candidate items.

M-FALCON Inference
==================

M-FALCON (Microbatched-Fast Attention Leveraging Cacheable OperatioNs) enables
efficient inference by:

1. **KV Caching**: Pre-compute K, V for user history (once per user)
2. **Microbatching**: Process candidates in batches of size ``m``
3. **Modified Attention**: Candidates attend to cached KV, not each other

.. code-block:: python

    def delta_hstu_mha(
        max_seq_len: int,
        alpha: float,
        delta_q: torch.Tensor,          # New queries [bm, H, d_qk]
        k: torch.Tensor,                # Cached K [full_seq, H, d_qk]
        v: torch.Tensor,                # Cached V [full_seq, H, d_v]
        seq_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor] = None,
        max_attn_len: int = 0,
        kernel: HammerKernel = HammerKernel.PYTORCH,
    ) -> torch.Tensor:
        pass

**Complexity Analysis**:

- Without M-FALCON: O(m × n² × d)
- With M-FALCON: O((n + bm)² × d) ≈ O(n² × d) when bm << n
- Achieves **285x model complexity at same compute budget**

Kernel Abstraction
==================

The ``HammerKernel`` enum provides multiple backend implementations:

.. code-block:: python

    class HammerKernel(Enum):
        PYTORCH = "pytorch"       # Reference implementation
        TRITON = "triton"         # GPU-optimized Triton kernels
        TRITON_CC = "triton_cc"   # Compiled Triton (H100+)
        CUDA = "cuda"             # Custom CUDA kernels

Kernel selection is automatic based on mode:

.. code-block:: python

    class HammerModule(torch.nn.Module):
        def hammer_kernel(self) -> HammerKernel:
            if self._is_inference and self._use_triton_cc:
                return HammerKernel.TRITON_CC
            return HammerKernel.TRITON

Fused Operations
================

For maximum performance, multiple operations are fused into single kernels:

hstu_preprocess_and_attention
-----------------------------

Combines:

1. Layer norm on input
2. Single linear projection → U, V, Q, K
3. Multi-head attention

.. code-block:: python

    def hstu_preprocess_and_attention(
        x: torch.Tensor,                # [L, d]
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        num_heads: int,
        attn_dim: int,
        hidden_dim: int,
        uvqk_weight: torch.Tensor,      # [d, 2h(d_h + d_a)]
        uvqk_bias: torch.Tensor,
        causal: bool = True,
        num_targets: Optional[torch.Tensor] = None,
        kernel: HammerKernel,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns: (U, attn_output, K, V)
        pass

hstu_compute_output
-------------------

Combines:

1. Normalize: attn·V
2. Gate: norm(attn·V) ⊙ U(X)
3. Project: [U ⊙ attn; U; X] → [d]
4. Residual + Dropout

.. code-block:: python

    def hstu_compute_output(
        attn: torch.Tensor,             # [L, h*d_h]
        u: torch.Tensor,                # [L, h*d_h]
        x: torch.Tensor,                # [L, d] (residual)
        norm_weight: torch.Tensor,
        norm_bias: torch.Tensor,
        output_weight: torch.Tensor,    # [3*h*d_h, d]
        dropout_ratio: float,
        num_heads: int,
        linear_dim: int,
        training: bool = True,
        kernel: HammerKernel,
    ) -> torch.Tensor:
        pass

Training Flow
=============

Generative Training Strategy
----------------------------

.. code-block:: text

    Raw Engagement Sequences (user, item, action, timestamp)
            ↓
    Sequentialize Features
    ├─ Main time series: Item interactions
    └─ Auxiliary: Demographics, followed creators
            ↓
    Unified Sequence: [Φ₀, a₀, Φ₁, a₁, ..., Φₙ₋₁, aₙ₋₁]
            ↓
    Emit Training Examples (at session end, ∝ 1/nᵢ sampling)
            ↓
    Forward through STU Stack
            ↓
    Extract Target Embeddings → Sampled Softmax Loss

Benefits of generative training:

- **Encoder Amortization**: Single encoder pass for multiple targets
- **Reduced Computation**: O(N) improvement over per-target encoding
- **Better Gradients**: Multiple targets provide richer supervision

DLRMv3 Integration
==================

The ``DlrmHSTU`` module integrates HSTU into the DLRMv3 framework:

.. code-block:: python

    class DlrmHSTU(HammerModule):
        def __init__(
            self,
            embedding_dim: int,
            num_layers: int,
            num_heads: int,
            attn_dim: int,
            hidden_dim: int,
            # ... additional config
        ):
            pass

        def forward(
            self,
            samples: Samples,
        ) -> torch.Tensor:
            pass

Configuration is managed through ``DlrmHSTUConfig``:

.. code-block:: python

    @dataclass
    class DlrmHSTUConfig:
        num_layers: int = 5
        num_heads: int = 4
        embedding_dim: int = 512
        attn_qk_dim: int = 128
        attn_linear_dim: int = 128
        max_seq_len: int = 8192
        max_num_candidates: int = 1000
        # ... additional config

Performance Characteristics
===========================

Computational Complexity
------------------------

+-------------------------+------------------+------------------+---------+
| Aspect                  | Traditional DLRM | GR (HSTU)        | Speedup |
+=========================+==================+==================+=========+
| Training (per sample)   | O(N³d + N²d²)    | O(N²d + Nd²)     | O(N)    |
+-------------------------+------------------+------------------+---------+
| Inference               | O(n²d)           | O((n+bm)²d)      | 285x*   |
+-------------------------+------------------+------------------+---------+
| Memory (activation)     | 33d per layer    | 14d per layer    | 2.3x    |
+-------------------------+------------------+------------------+---------+
| Sequence length         | 20-100           | 8,192            | 80-400x |
+-------------------------+------------------+------------------+---------+

*With M-FALCON microbatching

Benchmark Results
-----------------

**MovieLens-1M** (HR@10):

- SASRec: 0.2853
- HSTU-large: 0.3294 (**+15.5%**)

**MovieLens-20M** (HR@10):

- SASRec: 0.2906
- HSTU-large: 0.3556 (**+23.1%**)

**Amazon Books** (HR@10):

- SASRec: 0.0306
- HSTU-large: 0.0478 (**+56.7%**)

Cross-References
================

For related documentation:

- :doc:`/generative_recommenders/modules/index` - Module documentation
- :doc:`/generative_recommenders/ops/index` - Operations documentation
- :doc:`/generative_recommenders/dlrm_v3/index` - DLRMv3 integration
- :doc:`/workflows/training` - Training workflows
- :doc:`/workflows/inference` - Inference workflows
