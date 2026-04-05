======================
Wukong MLP (TensorFlow)
======================

.. module:: model.tensorflow.mlp
   :synopsis: Multi-layer perceptron building block (TensorFlow)

This document describes the MLP module in the Wukong TensorFlow implementation.

Overview
========

The MLP module provides a flexible multi-layer perceptron using Keras Sequential API.

Class Definition
================

.. class:: MLP(Sequential)

   Flexible multi-layer perceptron with batch normalization and dropout.

   :param num_hidden: Number of hidden layers
   :param dim_hidden: Hidden layer dimension
   :param dim_out: Output dimension (optional)
   :param dropout: Dropout probability (default: 0.0)
   :param name: Layer name (default: "MLP")

   **Note**: Unlike PyTorch, ``dim_in`` is inferred at runtime.

Architecture
------------

.. code-block:: text

   Dense(dim_hidden)
       |
       v
   BatchNormalization
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
   Dense(dim_out) [Final layer]

Get Config Method
-----------------

.. method:: get_config()

   Return configuration for serialization.

   :returns: Dictionary with layer parameters

   .. code-block:: python

      {
          "num_hidden": self._num_hidden,
          "dim_hidden": self._dim_hidden,
          "dim_out": self._dim_out,
          "dropout": self._dropout,
      }

Usage Example
=============

.. code-block:: python

   import tensorflow as tf
   from model.tensorflow.mlp import MLP

   # Create MLP
   mlp = MLP(
       num_hidden=2,
       dim_hidden=256,
       dim_out=1,
       dropout=0.1,
       name="projection_head"
   )

   # Forward pass
   x = tf.random.uniform((1024, 512))
   output = mlp(x)  # (1024, 1)

Key Differences from PyTorch
============================

+----------------------+-----------------------+------------------------+
| Aspect               | PyTorch               | TensorFlow             |
+======================+=======================+========================+
| Base Class           | ``nn.Sequential``     | ``keras.Sequential``   |
+----------------------+-----------------------+------------------------+
| Input dim            | Required in init      | Inferred at runtime    |
+----------------------+-----------------------+------------------------+
| BatchNorm            | ``BatchNorm1d``       | ``BatchNormalization`` |
+----------------------+-----------------------+------------------------+
| Activation           | ``nn.ReLU()``         | ``Activation('relu')`` |
+----------------------+-----------------------+------------------------+

Cross-References
================

- :doc:`/wukong/tensorflow/index` - TensorFlow implementation overview
- :doc:`wukong` - Wukong model
- :doc:`embedding` - Embedding module
