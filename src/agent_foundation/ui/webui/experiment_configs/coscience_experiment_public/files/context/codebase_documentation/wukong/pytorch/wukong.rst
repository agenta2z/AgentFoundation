=====================
Wukong Model (PyTorch)
=====================

.. module:: model.pytorch.wukong
   :synopsis: Core Wukong architecture components

This document provides detailed documentation for the Wukong PyTorch model components.

For the main PyTorch module overview, see :doc:`/wukong/pytorch/index`.

Component Classes
=================

LinearCompressBlock
-------------------

.. class:: LinearCompressBlock(nn.Module)

   Linearly recombines embeddings without increasing interaction orders.

   **Mathematical Operation**: ``LCB(X) = W @ X``

   :param num_emb_in: Number of input embeddings
   :param num_emb_out: Number of output embeddings

   **Example**:

   .. code-block:: python

      lcb = LinearCompressBlock(num_emb_in=48, num_emb_out=16)
      output = lcb(inputs)  # (batch, 48, d) -> (batch, 16, d)

FactorizationMachineBlock
-------------------------

.. class:: FactorizationMachineBlock(nn.Module)

   Captures pairwise feature interactions using optimized low-rank factorization.

   :param num_emb_in: Number of input embeddings
   :param num_emb_out: Number of output embeddings
   :param dim_emb: Embedding dimension
   :param rank: Low-rank approximation rank
   :param num_hidden: Number of MLP hidden layers
   :param dim_hidden: MLP hidden dimension
   :param dropout: Dropout probability

   **Complexity**: O(nkd) instead of O(n^2d)

   **Example**:

   .. code-block:: python

      fmb = FactorizationMachineBlock(
          num_emb_in=48,
          num_emb_out=16,
          dim_emb=128,
          rank=24,
          num_hidden=2,
          dim_hidden=512,
          dropout=0.1
      )
      output = fmb(inputs)  # (batch, 48, 128) -> (batch, 16, 128)

ResidualProjection
------------------

.. class:: ResidualProjection(nn.Module)

   Projects residual connections when dimensions change between layers.

   :param num_emb_in: Number of input embeddings
   :param num_emb_out: Number of output embeddings

   When ``num_emb_in == num_emb_out``, uses ``nn.Identity()``.

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

   **Forward Pass**:

   .. code-block:: python

      def forward(self, inputs):
          fmb = self.fmb(inputs)
          lcb = self.lcb(inputs)
          outputs = torch.concat((fmb, lcb), dim=1)
          outputs = self.norm(outputs + self.residual_projection(inputs))
          return outputs

Wukong Model
------------

.. class:: Wukong(nn.Module)

   Complete Wukong model for recommendation.

   See :doc:`/wukong/pytorch/index` for full parameter documentation and examples.

Cross-References
================

- :doc:`/wukong/pytorch/index` - PyTorch implementation overview
- :doc:`/architecture/wukong` - Architecture overview
- :doc:`embedding` - Embedding module
- :doc:`mlp` - MLP module
