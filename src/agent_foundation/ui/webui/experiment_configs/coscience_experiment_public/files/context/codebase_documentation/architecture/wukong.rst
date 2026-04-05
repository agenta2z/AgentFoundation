=====================
Wukong Architecture
=====================

.. module:: wukong
   :synopsis: Stacked Factorization Machines for Scaling Laws in Recommendation

This document provides an in-depth analysis of the Wukong Recommendation architecture, which implements
"Wukong: Towards a Scaling Law for Large-Scale Recommendation" (arXiv:2403.02545).

Overview
========

Wukong addresses a critical limitation in traditional recommendation systems: **poor scaling properties**.
While Large Language Models exhibit clear scaling laws (performance improves predictably with scale),
prior recommendation models plateau or show diminishing returns.

Key Innovation
--------------

Wukong achieves sustained quality improvement across **two orders of magnitude** in model complexity
(from ~1 to 100+ GFLOP/example), matching scales comparable to GPT-3/LLaMA-2.

The architecture follows the **principle of binary exponentiation** for capturing high-order feature interactions:

- Each layer captures progressively higher-order interactions at an exponential rate
- Layer ``i`` captures interactions of order 1 to 2^i
- Uses stacked Factorization Machines (FMs) as the core building block
- Separates dense scaling (interaction layers) from sparse scaling (embedding tables)

Core Architecture
=================

System Overview
---------------

.. code-block:: text

    Categorical Inputs → Embedding Layer ← Dense Inputs
           ↓
    Dense Embeddings (shape: bs × n × d)
           ↓
    ┌─────────────────────────────────────────┐
    │   Interaction Stack (l layers)          │
    │  ┌───────────────────────────────────┐  │
    │  │ Layer i:                          │  │
    │  │ ├─ FMB: Captures interactions     │  │
    │  │ ├─ LCB: Linear compression       │  │
    │  │ ├─ Concat FMB + LCB              │  │
    │  │ ├─ Residual Connection           │  │
    │  │ └─ LayerNorm                     │  │
    │  └───────────────────────────────────┘  │
    └─────────────────────────────────────────┘
           ↓
    Projection Head (MLP) → Output Logits

Mathematical Formulation
------------------------

Each WukongLayer implements:

.. math::

    X_{i+1} = LN(concat(FMB_i(X_i), LCB_i(X_i)) + ResidualProj(X_i))

Where:

- **FMB** (Factorization Machine Block): Captures higher-order interactions
- **LCB** (Linear Compress Block): Linear recombination preserving interaction order
- **LN**: Layer Normalization
- **ResidualProj**: Projects residuals when dimensions change

Components
==========

Embedding Layer
---------------

Transforms sparse categorical and dense features into unified embedding representations.

.. code-block:: python

    class Embedding(nn.Module):
        def __init__(
            self,
            num_sparse_emb: int,    # Size of embedding vocabulary
            dim_emb: int,           # Embedding dimension
            dim_input_dense: int,   # Number of dense input features
            bias: bool = True
        ):
            pass

        def forward(
            self,
            sparse_inputs: Tensor,  # (bs, num_cat_features)
            dense_inputs: Tensor    # (bs, num_dense_features)
        ) -> Tensor:
            # Returns: (bs, num_cat_features + num_dense_features, dim_emb)
            pass

Processing:

1. **Sparse inputs** → ``nn.Embedding`` lookup → ``(bs, num_cat, dim_emb)``
2. **Dense inputs** → ``nn.Linear`` → reshaped → ``(bs, num_dense, dim_emb)``
3. Both concatenated → ``(bs, num_cat + num_dense, dim_emb)``

**Key Design Decision**: Treats each embedding as a unit vector (embedding-level interactions), not element-wise.

FactorizationMachineBlock (FMB)
-------------------------------

Captures pairwise feature interactions using optimized low-rank factorization.

.. code-block:: python

    class FactorizationMachineBlock(nn.Module):
        def __init__(
            self,
            num_emb_in: int,
            num_emb_out: int,
            dim_emb: int,
            rank: int,              # Low-rank approximation
            num_hidden: int,        # MLP hidden layers
            dim_hidden: int,        # MLP hidden dimension
            dropout: float
        ):
            pass

        def forward(self, inputs: Tensor) -> Tensor:
            # inputs: (bs, num_emb_in, dim_emb)
            # returns: (bs, num_emb_out, dim_emb)
            pass

**Optimized FM Implementation** (Section 3.6 of paper):

1. Computes interaction matrix: **XXT** (dot products between all embedding pairs)
2. Reduces complexity from O(n²d) to O(nkd) using low-rank projection:

   - Computes ``XT @ Y`` first where ``Y ∈ ℝ^(n×k)``
   - Then does ``X @ (XT @ Y)`` to get n×n matrix
   - Result: n×n interaction matrix reduced to n×k

.. code-block:: python

    # Optimized computation:
    # X: (bs, n, d)
    # Goal: Compute XXT (dot products)

    # Step 1: Project embeddings to rank space
    outputs = inputs.permute(0, 2, 1)           # (bs, d, n)
    outputs = outputs @ self.weight              # (bs, d, rank)

    # Step 2: Compute interaction using associativity
    outputs = torch.bmm(inputs, outputs)         # (bs, n, rank)
    # Equivalent to: XXT·Y in reduced space

LinearCompressBlock (LCB)
-------------------------

Linearly recombines embeddings without increasing interaction orders.

.. code-block:: python

    class LinearCompressBlock(nn.Module):
        def __init__(
            self,
            num_emb_in: int,
            num_emb_out: int
        ):
            pass

        def forward(self, inputs: Tensor) -> Tensor:
            # inputs: (bs, num_emb_in, dim_emb)
            # returns: (bs, num_emb_out, dim_emb)
            pass

**Mathematical Operation**: ``LCB(X_i) = W_L @ X_i`` (weight matrix multiplication)

**Critical Role**: Ensures invariance of interaction order across layers, preserving lower-order interactions.

ResidualProjection
------------------

Projects residual connections when dimensions change between layers.

.. code-block:: python

    class ResidualProjection(nn.Module):
        def __init__(
            self,
            num_emb_in: int,
            num_emb_out: int
        ):
            pass

        def forward(self, inputs: Tensor) -> Tensor:
            pass

When ``num_emb_in == num_emb_out``, ``nn.Identity()`` is used instead.

WukongLayer
-----------

Complete interaction layer combining FMB, LCB, residual connection, and LayerNorm.

.. code-block:: python

    class WukongLayer(nn.Module):
        def __init__(
            self,
            num_emb_in: int,
            dim_emb: int,
            num_emb_lcb: int,      # LCB output embeddings
            num_emb_fmb: int,      # FMB output embeddings
            rank_fmb: int,         # FMB rank
            num_hidden: int,       # MLP hidden layers
            dim_hidden: int,       # MLP hidden dimension
            dropout: float
        ):
            pass

        def forward(self, inputs: Tensor) -> Tensor:
            fmb = self.fmb(inputs)
            lcb = self.lcb(inputs)
            outputs = torch.concat((fmb, lcb), dim=1)
            outputs = self.norm(outputs + self.residual_projection(inputs))
            return outputs

Wukong Model
------------

The complete model combining all components.

.. code-block:: python

    class Wukong(nn.Module):
        def __init__(
            self,
            num_layers: int,            # Number of interaction layers
            num_sparse_emb: int,        # Embedding vocabulary size
            dim_emb: int,               # Embedding dimension
            dim_input_sparse: int,      # Number of categorical features
            dim_input_dense: int,       # Number of dense features
            num_emb_lcb: int,           # LCB output embeddings
            num_emb_fmb: int,           # FMB output embeddings
            rank_fmb: int,              # FMB rank
            num_hidden_wukong: int,     # Wukong MLP hidden layers
            dim_hidden_wukong: int,     # Wukong MLP hidden dimension
            num_hidden_head: int,       # Projection head hidden layers
            dim_hidden_head: int,       # Projection head hidden dimension
            dim_output: int,            # Output dimension
            dropout: float = 0.0
        ):
            pass

        def forward(
            self,
            sparse_inputs: Tensor,      # (bs, dim_input_sparse)
            dense_inputs: Tensor        # (bs, dim_input_dense)
        ) -> Tensor:
            # Returns: (bs, dim_output)
            pass

Data Flow
=========

End-to-End Forward Pass
-----------------------

.. code-block:: text

    Input: sparse_inputs (bs, num_cat_features), dense_inputs (bs, num_dense_features)
    ↓
    [Embedding Layer]
       sparse: (bs, num_cat_features) → (bs, num_cat_features, dim_emb)
       dense:  (bs, num_dense_features) → (bs, num_dense_features, dim_emb)
       Concatenate → X_0: (bs, num_cat_features + num_dense_features, dim_emb)
    ↓
    [Interaction Stack - Layer 0]
       FMB(X_0) → (bs, num_emb_fmb, dim_emb)  [Higher-order interactions]
       LCB(X_0) → (bs, num_emb_lcb, dim_emb)  [Linear compression]
       Concat → (bs, num_emb_lcb + num_emb_fmb, dim_emb)
       Residual + LayerNorm → X_1
    ↓
    [Interaction Stack - Layers 1 to l-1]
       Similar pattern, num_emb_in = num_emb_lcb + num_emb_fmb (reduced from initial)
       Each layer captures 1 to 2^i order interactions
    ↓
    [Projection Head]
       Flatten: (bs, num_emb_lcb + num_emb_fmb, dim_emb) → (bs, total_dim)
       MLP: → (bs, dim_output)  [Typically 1 for binary classification]
    ↓
    Output: Logits (bs, dim_output) [Apply sigmoid for probability]

Interaction Order Dynamics
--------------------------

**Mathematical Proof** (Paper Section 3.3):

- **Input X_0**: Contains 1st order (individual embeddings)
- **Layer 1 (FMB)**: Second-order interactions (dot products)
- **Layer 2 (FMB)**: Up to 4th order (2² interactions)
- **Layer i (FMB)**: Up to 2^i order interactions
- **LCB Role**: Preserves lower-order interactions through linear recombination

Shape Evolution Example
-----------------------

With ``batch_size=1024``, typical configuration:

.. code-block:: text

    Initial sparse: (1024, 32)                    → (1024, 32, 128)
    Initial dense: (1024, 16)                     → (1024, 16, 128)
    After embedding: (1024, 48, 128)
    After layer 0 FMB: (1024, 16, 128)
    After layer 0 LCB: (1024, 16, 128)
    After layer 0 concat: (1024, 32, 128)
    After layer 0 norm: (1024, 32, 128)           ← Feed to layer 1
    After all layers: (1024, 32, 128)
    After flattening: (1024, 4096)                ← 32 × 128
    After head MLP: (1024, 1)                     ← Binary prediction

Dual Framework Support
======================

The Wukong codebase provides identical implementations in PyTorch and TensorFlow.

PyTorch Implementation
----------------------

.. code-block:: python

    # model/pytorch/wukong.py
    import torch
    from torch import nn

    class Wukong(nn.Module):
        def forward(self, sparse_inputs, dense_inputs):
            ...

TensorFlow Implementation
-------------------------

.. code-block:: python

    # model/tensorflow/wukong.py
    import tensorflow as tf
    from keras import Model, Layer

    class Wukong(Model):
        def call(self, inputs):  # [sparse, dense]
            ...

Key Differences
---------------

+----------------------+-----------------------+------------------------+
| Aspect               | PyTorch               | TensorFlow             |
+======================+=======================+========================+
| Module Definition    | ``__init__`` + ``forward()`` | ``build()`` + ``call()`` |
+----------------------+-----------------------+------------------------+
| Weight Creation      | Direct in ``__init__`` | Deferred in ``build()`` |
+----------------------+-----------------------+------------------------+
| Tensor Permutation   | ``.permute(0, 2, 1)`` | ``tf.transpose(x, (0, 2, 1))`` |
+----------------------+-----------------------+------------------------+
| Reshaping            | ``.view(bs, -1)``     | ``tf.reshape(x, (-1,))`` |
+----------------------+-----------------------+------------------------+
| Initialization       | ``kaiming_uniform_()`` | ``he_uniform``         |
+----------------------+-----------------------+------------------------+
| Model Base           | ``nn.Module``         | ``keras.Model``        |
+----------------------+-----------------------+------------------------+

Design Patterns
===============

Architectural Patterns
----------------------

+--------------------------+--------------------------------------------------+
| Pattern                  | Application                                      |
+==========================+==================================================+
| Stacked Architecture     | Multiple WukongLayers for exponential interaction|
+--------------------------+--------------------------------------------------+
| Residual Networks        | Enables deep stacking (tested up to 8 layers)    |
+--------------------------+--------------------------------------------------+
| Post-Norm                | LayerNorm after residual addition (vs pre-norm)  |
+--------------------------+--------------------------------------------------+
| Factorization Machines   | Efficient second-order interaction capture       |
+--------------------------+--------------------------------------------------+
| Low-Rank Approximation   | Reduces FM complexity from O(n²d) to O(nkd)      |
+--------------------------+--------------------------------------------------+
| Parallel Branches        | FMB and LCB process independently                |
+--------------------------+--------------------------------------------------+
| Embedding-Level          | Interactions at embedding granularity            |
+--------------------------+--------------------------------------------------+

Hyperparameter Scaling Strategy
-------------------------------

1. **Phase 1**: Increase ``l`` (number of layers) first
2. **Phase 2**: Then augment ``nF``, ``nL``, ``k`` (wider layers)

**Rationale**: Captures higher-order interactions efficiently before adding capacity.

Performance Characteristics
===========================

Scaling Law Results
-------------------

From the paper:

- Demonstrated across **2 orders of magnitude**: 1 to 100+ GFLOP/example
- **0.4% quality improvement** with 200-fold complexity increase
- ~0.1% improvement per quadrupling of complexity
- Baseline models plateau or show negative scaling

Public Dataset Performance
--------------------------

Best AUC on all 6 datasets:

- Frappe
- MicroVideo
- MovieLens
- KuaiVideo
- TaobaoAds
- Criteo Terabyte

Memory Efficiency
-----------------

**Low-Rank FM Impact**:

- Full FM stores n × n × d floats ≈ 96 × 96 × 128 = 1.18M floats
- Optimized FM stores n × rank × d floats ≈ 96 × 24 × 128 = 0.295M floats
- **4× reduction** with minimal quality loss

Usage Examples
==============

PyTorch Example
---------------

.. code-block:: python

    import torch
    from model.pytorch import Wukong

    # Configuration
    BATCH_SIZE = 1024
    NUM_EMBEDDING = 100
    NUM_CAT_FEATURES = 32
    NUM_DENSE_FEATURES = 16

    # Data
    sparse_inputs = torch.multinomial(
        torch.rand((BATCH_SIZE, NUM_EMBEDDING)),
        NUM_CAT_FEATURES,
        replacement=True,
    )  # (1024, 32)
    dense_inputs = torch.rand(BATCH_SIZE, NUM_DENSE_FEATURES)  # (1024, 16)

    # Model
    model = Wukong(
        num_layers=3,
        num_sparse_emb=NUM_EMBEDDING,
        dim_emb=128,
        dim_input_sparse=NUM_CAT_FEATURES,
        dim_input_dense=NUM_DENSE_FEATURES,
        num_emb_lcb=16,
        num_emb_fmb=16,
        rank_fmb=24,
        num_hidden_wukong=2,
        dim_hidden_wukong=512,
        num_hidden_head=2,
        dim_hidden_head=512,
        dim_output=1,
    )

    # Forward pass
    outputs = model(sparse_inputs, dense_inputs)  # (1024, 1)
    predictions = torch.sigmoid(outputs)          # Probability

TensorFlow Example
------------------

.. code-block:: python

    import tensorflow as tf
    from model.tensorflow import Wukong

    # Data
    inputs = [
        tf.random.categorical(..., NUM_CAT_FEATURES, dtype=tf.int32),
        tf.random.uniform((BATCH_SIZE, NUM_DENSE_FEATURES)),
    ]

    # Model
    model = Wukong(
        num_layers=2,
        num_sparse_emb=NUM_EMBEDDING,
        dim_emb=128,
        # ... other hyperparameters
    )

    # Forward pass
    outputs = model(inputs)  # (1024, 1)
    predictions = tf.nn.sigmoid(outputs)

Cross-References
================

For related documentation:

- :doc:`/wukong/pytorch/index` - PyTorch implementation details
- :doc:`/wukong/tensorflow/index` - TensorFlow implementation details
- :doc:`/workflows/training` - Training workflows
