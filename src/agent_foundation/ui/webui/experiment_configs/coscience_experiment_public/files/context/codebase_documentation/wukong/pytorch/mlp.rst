==========
Wukong MLP
==========

.. module:: model.pytorch.mlp
   :synopsis: Multi-layer perceptron building block

This document describes the MLP module in the Wukong PyTorch implementation.

Overview
========

The MLP module provides a flexible multi-layer perceptron with batch normalization
and dropout, used throughout the Wukong architecture.

Class Definition
================

.. class:: MLP(nn.Sequential)

   Flexible multi-layer perceptron with batch normalization and dropout.

   :param dim_in: Input dimension
   :param num_hidden: Number of hidden layers
   :param dim_hidden: Hidden layer dimension
   :param dim_out: Output dimension (optional, defaults to dim_hidden)
   :param batch_norm: Enable batch normalization (default: True)
   :param dropout: Dropout probability (default: 0.0)

Architecture
------------

Each layer in the MLP consists of:

.. code-block:: text

   Linear(dim_in, dim_hidden)
       |
       v
   BatchNorm1d (if enabled)
       |
       v
   ReLU
       |
       v
   Dropout (if dropout > 0)
       |
       v
   [Repeat for num_hidden layers]
       |
       v
   Linear(dim_hidden, dim_out) [Final layer]

Constructor
-----------

.. method:: __init__(dim_in, num_hidden, dim_hidden, dim_out=None, batch_norm=True, dropout=0.0)

   Initialize the MLP.

   **Parameters**:

   - ``dim_in``: Input dimension
   - ``num_hidden``: Number of hidden layers (before output layer)
   - ``dim_hidden``: Hidden layer dimension
   - ``dim_out``: Output dimension (defaults to ``dim_hidden`` if None)
   - ``batch_norm``: Enable BatchNorm1d after each linear layer
   - ``dropout``: Dropout probability after each activation

Usage Example
=============

.. code-block:: python

   import torch
   from model.pytorch.mlp import MLP

   # Create MLP
   mlp = MLP(
       dim_in=512,
       num_hidden=2,
       dim_hidden=256,
       dim_out=1,
       batch_norm=True,
       dropout=0.1
   )

   # Forward pass
   x = torch.rand(1024, 512)
   output = mlp(x)  # (1024, 1)

Layer Breakdown
===============

For ``MLP(dim_in=512, num_hidden=2, dim_hidden=256, dim_out=1)``:

.. code-block:: text

   Layer 0: Linear(512, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.1)
   Layer 1: Linear(256, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.1)
   Layer 2: Linear(256, 1)   [Output layer, no BN/ReLU/Dropout]

Configuration Options
=====================

+------------------+-------------+------------------------------------------+
| Parameter        | Default     | Description                              |
+==================+=============+==========================================+
| ``dim_in``       | Required    | Input dimension                          |
+------------------+-------------+------------------------------------------+
| ``num_hidden``   | Required    | Number of hidden layers                  |
+------------------+-------------+------------------------------------------+
| ``dim_hidden``   | Required    | Hidden layer dimension                   |
+------------------+-------------+------------------------------------------+
| ``dim_out``      | dim_hidden  | Output dimension                         |
+------------------+-------------+------------------------------------------+
| ``batch_norm``   | True        | Enable batch normalization               |
+------------------+-------------+------------------------------------------+
| ``dropout``      | 0.0         | Dropout probability                      |
+------------------+-------------+------------------------------------------+

Cross-References
================

- :doc:`/wukong/pytorch/index` - PyTorch implementation overview
- :doc:`wukong` - Wukong model
- :doc:`embedding` - Embedding module
