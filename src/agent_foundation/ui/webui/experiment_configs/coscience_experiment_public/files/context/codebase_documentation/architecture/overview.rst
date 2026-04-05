=====================
Architecture Overview
=====================

This document provides a high-level overview of the architecture for both the Generative Recommenders and Wukong Recommendation systems.

System Landscape
================

The codebase consists of two related but distinct recommendation system implementations:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     Recommendation Systems Codebase                      │
    ├───────────────────────────────┬─────────────────────────────────────────┤
    │     Generative Recommenders   │         Wukong Recommendation           │
    │         (HSTU/GR)             │         (Stacked FMs)                   │
    ├───────────────────────────────┼─────────────────────────────────────────┤
    │  - Sequential Transduction    │  - Factorization Machines               │
    │  - Trillion-scale parameters  │  - Scaling Laws                         │
    │  - M-FALCON inference         │  - PyTorch + TensorFlow                 │
    │  - Triton GPU kernels         │  - Dual implementation                  │
    └───────────────────────────────┴─────────────────────────────────────────┘

Design Philosophy
=================

Generative Recommenders
-----------------------

The Generative Recommenders approach transforms traditional Deep Learning Recommendation Models (DLRMs)
into Generative Recommenders (GRs) through four key innovations:

1. **Unified Feature Spaces**: Sequentializes heterogeneous categorical and numerical features into a single time series
2. **Sequential Transduction**: Casts ranking and retrieval as sequential transduction tasks
3. **Generative Training**: Trains models generatively to amortize encoder costs
4. **M-FALCON Inference**: Uses microbatching for efficient cached inference

Wukong Recommendation
---------------------

The Wukong architecture achieves scaling laws through:

1. **Stacked Factorization Machines**: Captures progressively higher-order interactions
2. **Binary Exponentiation**: Layer i captures interactions of order 1 to 2^i
3. **Dense-Sparse Separation**: Separates interaction layer scaling from embedding scaling
4. **Low-Rank Approximation**: Reduces FM complexity from O(n²d) to O(nkd)

Key Architecture Patterns
=========================

HammerModule Pattern (Generative Recommenders)
----------------------------------------------

All neural modules inherit from ``HammerModule``, providing:

- Kernel selection (TRITON, TRITON_CC, PYTORCH, CUDA)
- Inference mode switching
- Recursive attribute setting

.. code-block:: python

    class HammerModule(torch.nn.Module, abc.ABC):
        def hammer_kernel(self) -> HammerKernel:
            if self._is_inference and self._use_triton_cc:
                return HammerKernel.TRITON_CC
            return HammerKernel.TRITON

Dual Framework Pattern (Wukong)
-------------------------------

Identical implementations in PyTorch and TensorFlow enable:

- Cross-framework validation
- Production flexibility
- Bit-level equivalence verification

.. code-block:: python

    # PyTorch
    class Wukong(nn.Module):
        def forward(self, sparse_inputs, dense_inputs): ...

    # TensorFlow
    class Wukong(Model):
        def call(self, inputs): ...  # [sparse, dense]

Data Flow Overview
==================

Generative Recommenders Data Flow
---------------------------------

.. code-block:: text

    Raw Engagement Sequences
            ↓
    Sequentialization (unified time series)
            ↓
    InputPreprocessor → seq_embeddings
            ↓
    HSTUPositionalEncoder → positional encodings
            ↓
    STUStack (N layers)
    ├── STULayer 1: Norm → UVQK → Attention → Output
    ├── STULayer 2: Residual + Output
    └── STULayer N: Final embeddings
            ↓
    OutputPostprocessor → candidate embeddings
            ↓
    Loss computation / Inference

Wukong Data Flow
----------------

.. code-block:: text

    Categorical Inputs    Dense Inputs
           ↓                    ↓
    ┌──────────────────────────────────┐
    │       Embedding Layer            │
    │  sparse → (bs, n_cat, d)         │
    │  dense → (bs, n_dense, d)        │
    │  concat → (bs, n_total, d)       │
    └──────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────┐
    │   Interaction Stack (l layers)   │
    │  ┌────────────────────────────┐  │
    │  │ WukongLayer i:             │  │
    │  │ ├─ FMB: Higher-order       │  │
    │  │ ├─ LCB: Linear compress    │  │
    │  │ ├─ Concat + Residual       │  │
    │  │ └─ LayerNorm               │  │
    │  └────────────────────────────┘  │
    └──────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────┐
    │       Projection Head (MLP)      │
    │  Flatten → MLP → Output logits   │
    └──────────────────────────────────┘

Component Relationships
=======================

Generative Recommenders Module Hierarchy
----------------------------------------

.. code-block:: text

    main.py
        └── research/trainer/train.py
            └── HSTUTransducer
                ├── InputPreprocessor
                ├── HSTUPositionalEncoder
                ├── STUStack
                │   └── STULayer (×N)
                │       ├── hstu_preprocess_and_attention()
                │       └── hstu_compute_output()
                └── OutputPostprocessor

Wukong Module Hierarchy
-----------------------

.. code-block:: text

    Wukong (Model)
        ├── Embedding
        │   ├── nn.Embedding (sparse)
        │   └── nn.Linear (dense)
        ├── WukongLayer (×l)
        │   ├── FactorizationMachineBlock
        │   │   ├── Low-rank projection
        │   │   ├── LayerNorm
        │   │   └── MLP
        │   ├── LinearCompressBlock
        │   │   └── Weight matrix
        │   └── ResidualProjection
        │       └── Weight matrix (if needed)
        └── MLP (Projection Head)

Cross-References
================

For detailed architecture documentation:

- :doc:`generative_recommenders` - HSTU architecture deep dive
- :doc:`wukong` - Wukong architecture deep dive

For module-specific documentation:

- :doc:`/generative_recommenders/modules/index` - GR modules
- :doc:`/wukong/pytorch/index` - Wukong PyTorch implementation
- :doc:`/wukong/tensorflow/index` - Wukong TensorFlow implementation
