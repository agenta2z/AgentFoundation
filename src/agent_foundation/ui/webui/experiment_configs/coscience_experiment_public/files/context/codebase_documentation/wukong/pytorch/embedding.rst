================
Wukong Embedding
================

.. module:: model.pytorch.embedding
   :synopsis: Embedding layer for sparse and dense features

This document describes the embedding module in the Wukong PyTorch implementation.

Overview
========

The Embedding module transforms both sparse (categorical) and dense (numerical)
features into unified embedding representations.

Class Definition
================

.. class:: Embedding(nn.Module)

   Transforms sparse categorical and dense features into unified embedding representations.

   :param num_sparse_emb: Size of the embedding vocabulary
   :param dim_emb: Embedding dimension (standardized across all embeddings)
   :param dim_input_dense: Number of dense input features
   :param bias: Whether to use bias in dense transformation (default: True)

Constructor
-----------

.. method:: __init__(num_sparse_emb, dim_emb, dim_input_dense, bias=True)

   Initialize the embedding layer.

   **Internal Components**:

   - ``self._embedding``: ``nn.Embedding(num_sparse_emb, dim_emb)``
   - ``self._linear``: ``nn.Linear(1, dim_emb, bias=bias)``

Forward Method
--------------

.. method:: forward(sparse_inputs, dense_inputs)

   Process sparse and dense inputs into unified embeddings.

   :param sparse_inputs: Categorical feature indices ``(batch_size, num_cat_features)``
   :param dense_inputs: Dense feature values ``(batch_size, num_dense_features)``
   :returns: Combined embeddings ``(batch_size, num_cat + num_dense, dim_emb)``

   **Processing Steps**:

   1. Lookup sparse embeddings: ``sparse_outputs = self._embedding(sparse_inputs)``
   2. Transform dense features: ``dense_outputs = self._linear(dense_inputs.unsqueeze(-1))``
   3. Concatenate: ``outputs = torch.cat([sparse_outputs, dense_outputs], dim=1)``

Usage Example
=============

.. code-block:: python

   import torch
   from model.pytorch.embedding import Embedding

   # Create embedding layer
   embedding = Embedding(
       num_sparse_emb=100,      # Vocabulary size
       dim_emb=128,             # Embedding dimension
       dim_input_dense=16,      # Number of dense features
       bias=True
   )

   # Sample inputs
   batch_size = 1024
   num_cat_features = 32

   sparse_inputs = torch.randint(0, 100, (batch_size, num_cat_features))
   dense_inputs = torch.rand(batch_size, 16)

   # Forward pass
   output = embedding(sparse_inputs, dense_inputs)
   # output.shape: (1024, 48, 128) = (batch_size, 32 + 16, 128)

Shape Flow
==========

.. code-block:: text

   Sparse Inputs: (batch_size, num_cat_features)
       |
       v
   nn.Embedding lookup
       |
       v
   Sparse Embeddings: (batch_size, num_cat_features, dim_emb)

   Dense Inputs: (batch_size, num_dense_features)
       |
       v
   unsqueeze(-1) -> (batch_size, num_dense_features, 1)
       |
       v
   nn.Linear(1 -> dim_emb)
       |
       v
   Dense Embeddings: (batch_size, num_dense_features, dim_emb)

   Concatenate along dim=1
       |
       v
   Output: (batch_size, num_cat + num_dense, dim_emb)

Cross-References
================

- :doc:`/wukong/pytorch/index` - PyTorch implementation overview
- :doc:`wukong` - Wukong model
- :doc:`mlp` - MLP module
