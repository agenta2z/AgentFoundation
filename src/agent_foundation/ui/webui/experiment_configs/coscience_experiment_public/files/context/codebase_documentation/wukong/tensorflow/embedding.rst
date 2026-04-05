===========================
Wukong Embedding (TensorFlow)
===========================

.. module:: model.tensorflow.embedding
   :synopsis: Embedding layer for sparse and dense features (TensorFlow)

This document describes the embedding module in the Wukong TensorFlow implementation.

Overview
========

The Embedding layer (Keras Layer) transforms both sparse (categorical) and dense
(numerical) features into unified embedding representations.

Class Definition
================

.. class:: Embedding(Layer)

   Transforms sparse categorical and dense features into unified embedding representations.

   :param num_sparse_emb: Size of the embedding vocabulary
   :param dim_emb: Embedding dimension
   :param bias: Whether to use bias in dense transformation (default: True)

   **Note**: Unlike PyTorch, ``dim_input_dense`` is inferred from input shape in ``build()``.

Build Method
------------

.. method:: build(inputs_shape)

   Create weights based on input shape.

   :param inputs_shape: List of [sparse_shape, dense_shape]

   **Created Variables**:

   - ``_embedding``: Embedding weights ``(num_sparse_emb, dim_emb)``
   - ``_dense_weight``: Dense transformation weights ``(1, dim_emb)``
   - ``_dense_bias``: Dense transformation bias ``(dim_emb,)`` (if bias=True)

Call Method
-----------

.. method:: call(inputs)

   Process inputs through the embedding layer.

   :param inputs: List of [sparse_inputs, dense_inputs]
   :returns: Combined embeddings ``(batch_size, num_cat + num_dense, dim_emb)``

Get Config Method
-----------------

.. method:: get_config()

   Return layer configuration for serialization.

   :returns: Dictionary with layer parameters

   .. code-block:: python

      {
          "num_sparse_emb": self._num_sparse_emb,
          "dim_emb": self._dim_emb,
          "bias": self._bias,
      }

Usage Example
=============

.. code-block:: python

   import tensorflow as tf
   from model.tensorflow.embedding import Embedding

   # Create embedding layer
   embedding = Embedding(
       num_sparse_emb=100,
       dim_emb=128,
       bias=True
   )

   # Sample inputs (note: list format for TensorFlow)
   batch_size = 1024
   sparse_inputs = tf.random.uniform(
       (batch_size, 32), minval=0, maxval=100, dtype=tf.int32
   )
   dense_inputs = tf.random.uniform((batch_size, 16))

   # Forward pass
   output = embedding([sparse_inputs, dense_inputs])
   # output.shape: (1024, 48, 128)

Key Differences from PyTorch
============================

+----------------------+-----------------------+------------------------+
| Aspect               | PyTorch               | TensorFlow             |
+======================+=======================+========================+
| Input Format         | Two separate args     | List of two tensors    |
+----------------------+-----------------------+------------------------+
| Weight Creation      | In ``__init__``       | In ``build()``         |
+----------------------+-----------------------+------------------------+
| Dense dim inference  | Constructor param     | From input shape       |
+----------------------+-----------------------+------------------------+
| Embedding lookup     | ``self._embedding()`` | ``tf.nn.embedding_lookup`` |
+----------------------+-----------------------+------------------------+

Cross-References
================

- :doc:`/wukong/tensorflow/index` - TensorFlow implementation overview
- :doc:`wukong` - Wukong model
- :doc:`mlp` - MLP module
