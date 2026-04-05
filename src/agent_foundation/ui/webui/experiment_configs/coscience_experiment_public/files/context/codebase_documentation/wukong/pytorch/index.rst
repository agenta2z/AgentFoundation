=========================
Wukong PyTorch Implementation
=========================

This section documents the PyTorch implementation of the Wukong recommendation model.

.. toctree::
   :maxdepth: 2

   embedding
   mlp
   wukong

Module Overview
===============

The ``model/pytorch/`` directory contains the PyTorch implementation:

.. code-block:: text

    model/pytorch/
    ├── __init__.py
    ├── embedding.py       # Embedding layer
    ├── mlp.py             # Multi-layer perceptron
    └── wukong.py          # Core architecture components

Public Interface
================

The package exports the main ``Wukong`` model through ``__init__.py``:

.. code-block:: python

    from model.pytorch import Wukong

Embedding Module
================

.. module:: model.pytorch.embedding
   :synopsis: Embedding layer for sparse and dense features

.. class:: Embedding(nn.Module)

   Transforms sparse categorical and dense features into unified embedding representations.

   :param num_sparse_emb: Size of the embedding vocabulary
   :param dim_emb: Embedding dimension (standardized across all embeddings)
   :param dim_input_dense: Number of dense input features
   :param bias: Whether to use bias in dense transformation (default: True)

   **Example**:

   .. code-block:: python

       embedding = Embedding(
           num_sparse_emb=100,
           dim_emb=128,
           dim_input_dense=16,
           bias=True
       )

       sparse_inputs = torch.randint(0, 100, (1024, 32))
       dense_inputs = torch.rand(1024, 16)

       output = embedding(sparse_inputs, dense_inputs)
       # output.shape: (1024, 48, 128)

   .. method:: forward(sparse_inputs, dense_inputs)

      Process sparse and dense inputs into unified embeddings.

      :param sparse_inputs: Categorical feature indices ``(batch_size, num_cat_features)``
      :param dense_inputs: Dense feature values ``(batch_size, num_dense_features)``
      :returns: Combined embeddings ``(batch_size, num_cat + num_dense, dim_emb)``

MLP Module
==========

.. module:: model.pytorch.mlp
   :synopsis: Multi-layer perceptron building block

.. class:: MLP(nn.Sequential)

   Flexible multi-layer perceptron with batch normalization and dropout.

   :param dim_in: Input dimension
   :param num_hidden: Number of hidden layers
   :param dim_hidden: Hidden layer dimension
   :param dim_out: Output dimension (optional, defaults to dim_hidden)
   :param batch_norm: Enable batch normalization (default: True)
   :param dropout: Dropout probability (default: 0.0)

   **Architecture**: Linear → BatchNorm1d → ReLU → Dropout (repeated)

   **Example**:

   .. code-block:: python

       mlp = MLP(
           dim_in=512,
           num_hidden=2,
           dim_hidden=256,
           dim_out=1,
           batch_norm=True,
           dropout=0.1
       )

       x = torch.rand(1024, 512)
       output = mlp(x)  # (1024, 1)

Wukong Components
=================

.. module:: model.pytorch.wukong
   :synopsis: Core Wukong architecture components

LinearCompressBlock (LCB)
-------------------------

.. class:: LinearCompressBlock(nn.Module)

   Linearly recombines embeddings without increasing interaction orders.

   :param num_emb_in: Number of input embeddings
   :param num_emb_out: Number of output embeddings

   **Mathematical Operation**: ``LCB(X) = W @ X``

   .. method:: forward(inputs)

      :param inputs: Input embeddings ``(batch_size, num_emb_in, dim_emb)``
      :returns: Compressed embeddings ``(batch_size, num_emb_out, dim_emb)``

FactorizationMachineBlock (FMB)
-------------------------------

.. class:: FactorizationMachineBlock(nn.Module)

   Captures pairwise feature interactions using optimized low-rank factorization.

   :param num_emb_in: Number of input embeddings
   :param num_emb_out: Number of output embeddings
   :param dim_emb: Embedding dimension
   :param rank: Low-rank approximation rank
   :param num_hidden: Number of MLP hidden layers
   :param dim_hidden: MLP hidden dimension
   :param dropout: Dropout probability

   **Optimization**: Reduces complexity from O(n²d) to O(nkd).

   .. method:: forward(inputs)

      :param inputs: Input embeddings ``(batch_size, num_emb_in, dim_emb)``
      :returns: Interaction embeddings ``(batch_size, num_emb_out, dim_emb)``

ResidualProjection
------------------

.. class:: ResidualProjection(nn.Module)

   Projects residual connections when dimensions change between layers.

   :param num_emb_in: Number of input embeddings
   :param num_emb_out: Number of output embeddings

   When ``num_emb_in == num_emb_out``, an identity function is used.

WukongLayer
-----------

.. class:: WukongLayer(nn.Module)

   Complete interaction layer combining FMB, LCB, residual connection, and LayerNorm.

   :param num_emb_in: Number of input embeddings
   :param dim_emb: Embedding dimension
   :param num_emb_lcb: LCB output embeddings
   :param num_emb_fmb: FMB output embeddings
   :param rank_fmb: FMB rank
   :param num_hidden: MLP hidden layers
   :param dim_hidden: MLP hidden dimension
   :param dropout: Dropout probability

   **Architecture**:

   .. code-block:: text

       X_{i+1} = LN(concat(FMB(X_i), LCB(X_i)) + ResidualProj(X_i))

   .. method:: forward(inputs)

      :param inputs: Input embeddings ``(batch_size, num_emb_in, dim_emb)``
      :returns: Output embeddings ``(batch_size, num_emb_lcb + num_emb_fmb, dim_emb)``

Wukong Model
------------

.. class:: Wukong(nn.Module)

   Complete Wukong model for recommendation.

   :param num_layers: Number of interaction layers
   :param num_sparse_emb: Embedding vocabulary size
   :param dim_emb: Embedding dimension
   :param dim_input_sparse: Number of categorical features
   :param dim_input_dense: Number of dense features
   :param num_emb_lcb: LCB output embeddings
   :param num_emb_fmb: FMB output embeddings
   :param rank_fmb: FMB rank
   :param num_hidden_wukong: Wukong MLP hidden layers
   :param dim_hidden_wukong: Wukong MLP hidden dimension
   :param num_hidden_head: Projection head hidden layers
   :param dim_hidden_head: Projection head hidden dimension
   :param dim_output: Output dimension
   :param dropout: Dropout probability (default: 0.0)

   **Example**:

   .. code-block:: python

       from model.pytorch import Wukong

       model = Wukong(
           num_layers=3,
           num_sparse_emb=100,
           dim_emb=128,
           dim_input_sparse=32,
           dim_input_dense=16,
           num_emb_lcb=16,
           num_emb_fmb=16,
           rank_fmb=24,
           num_hidden_wukong=2,
           dim_hidden_wukong=512,
           num_hidden_head=2,
           dim_hidden_head=512,
           dim_output=1,
           dropout=0.1
       )

       sparse_inputs = torch.randint(0, 100, (1024, 32))
       dense_inputs = torch.rand(1024, 16)

       logits = model(sparse_inputs, dense_inputs)  # (1024, 1)
       predictions = torch.sigmoid(logits)

   .. method:: forward(sparse_inputs, dense_inputs)

      Full forward pass through the Wukong model.

      :param sparse_inputs: Categorical feature indices ``(batch_size, dim_input_sparse)``
      :param dense_inputs: Dense feature values ``(batch_size, dim_input_dense)``
      :returns: Output logits ``(batch_size, dim_output)``

Dependencies
============

- ``torch>=2.2``
- ``torch.nn``

Cross-References
================

- :doc:`/architecture/wukong` - Wukong architecture overview
- :doc:`/wukong/tensorflow/index` - TensorFlow implementation
