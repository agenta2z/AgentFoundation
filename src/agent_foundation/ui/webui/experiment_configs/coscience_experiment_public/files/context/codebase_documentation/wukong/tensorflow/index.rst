=============================
Wukong TensorFlow Implementation
=============================

This section documents the TensorFlow implementation of the Wukong recommendation model.

.. toctree::
   :maxdepth: 2

   embedding
   mlp
   wukong

Module Overview
===============

The ``model/tensorflow/`` directory contains the TensorFlow/Keras implementation:

.. code-block:: text

    model/tensorflow/
    ├── __init__.py
    ├── embedding.py       # Embedding layer (Keras Layer)
    ├── mlp.py             # Multi-layer perceptron (Keras Sequential)
    └── wukong.py          # Core architecture components

Public Interface
================

The package exports the main ``Wukong`` model through ``__init__.py``:

.. code-block:: python

    from model.tensorflow import Wukong

Key Differences from PyTorch
============================

The TensorFlow implementation uses Keras conventions:

+----------------------+-----------------------+------------------------+
| Aspect               | PyTorch               | TensorFlow             |
+======================+=======================+========================+
| Module Base          | ``nn.Module``         | ``keras.Layer``        |
+----------------------+-----------------------+------------------------+
| Model Base           | ``nn.Module``         | ``keras.Model``        |
+----------------------+-----------------------+------------------------+
| Weight Creation      | ``__init__()``        | ``build()``            |
+----------------------+-----------------------+------------------------+
| Forward Pass         | ``forward()``         | ``call()``             |
+----------------------+-----------------------+------------------------+
| Config Export        | Checkpoint files      | ``get_config()``       |
+----------------------+-----------------------+------------------------+

Embedding Module
================

.. module:: model.tensorflow.embedding
   :synopsis: Embedding layer for sparse and dense features

.. class:: Embedding(Layer)

   Transforms sparse categorical and dense features into unified embedding representations.

   :param num_sparse_emb: Size of the embedding vocabulary
   :param dim_emb: Embedding dimension
   :param bias: Whether to use bias in dense transformation (default: True)

   **Note**: Unlike PyTorch, ``dim_input_dense`` is inferred from input shape in ``build()``.

   .. method:: build(inputs_shape)

      Create weights based on input shape.

      :param inputs_shape: List of [sparse_shape, dense_shape]

   .. method:: call(inputs)

      Process inputs through the embedding layer.

      :param inputs: List of [sparse_inputs, dense_inputs]
      :returns: Combined embeddings

   .. method:: get_config()

      Return layer configuration for serialization.

      :returns: Dictionary with layer parameters

MLP Module
==========

.. module:: model.tensorflow.mlp
   :synopsis: Multi-layer perceptron building block

.. class:: MLP(Sequential)

   Flexible multi-layer perceptron with batch normalization and dropout.

   :param num_hidden: Number of hidden layers
   :param dim_hidden: Hidden layer dimension
   :param dim_out: Output dimension (optional)
   :param dropout: Dropout probability (default: 0.0)
   :param name: Layer name (default: "MLP")

   **Note**: Unlike PyTorch, ``dim_in`` is inferred at runtime.

   **Architecture**: Dense → BatchNormalization → ReLU → Dropout (repeated)

   .. method:: get_config()

      Return configuration for serialization.

Wukong Components
=================

.. module:: model.tensorflow.wukong
   :synopsis: Core Wukong architecture components

LinearCompressBlock (LCB)
-------------------------

.. class:: LinearCompressBlock(Layer)

   Linearly recombines embeddings without increasing interaction orders.

   :param num_emb_out: Number of output embeddings
   :param weights_initializer: Weight initializer (default: "he_uniform")
   :param name: Layer name (default: "lcb")

   .. method:: build(input_shape)

      Create weights. Infers ``num_emb_in`` from input shape.

   .. method:: call(inputs)

      :param inputs: Input embeddings ``(batch_size, num_emb_in, dim_emb)``
      :returns: Compressed embeddings ``(batch_size, num_emb_out, dim_emb)``

FactorizationMachineBlock (FMB)
-------------------------------

.. class:: FactorizationMachineBlock(Layer)

   Captures pairwise feature interactions using optimized low-rank factorization.

   :param num_emb_out: Number of output embeddings
   :param dim_emb: Embedding dimension
   :param rank: Low-rank approximation rank
   :param num_hidden: Number of MLP hidden layers
   :param dim_hidden: MLP hidden dimension
   :param dropout: Dropout probability
   :param weights_initializer: Weight initializer (default: "he_uniform")
   :param name: Layer name (default: "fmb")

   .. method:: build(input_shape)

      Create weights. Infers ``num_emb_in`` from input shape.

   .. method:: call(inputs)

      :param inputs: Input embeddings
      :returns: Interaction embeddings

ResidualProjection
------------------

.. class:: ResidualProjection(Layer)

   Projects residual connections when dimensions change.

   :param num_emb_out: Number of output embeddings
   :param weights_initializer: Weight initializer (default: "he_uniform")
   :param name: Layer name (default: "residual_projection")

WukongLayer
-----------

.. class:: WukongLayer(Layer)

   Complete interaction layer combining FMB, LCB, residual connection, and LayerNorm.

   :param num_emb_lcb: LCB output embeddings
   :param num_emb_fmb: FMB output embeddings
   :param rank_fmb: FMB rank
   :param num_hidden: MLP hidden layers
   :param dim_hidden: MLP hidden dimension
   :param dropout: Dropout probability
   :param name: Layer name (default: "wukong")

   .. method:: build(input_shape)

      Create FMB, LCB, and residual projection layers.
      Infers ``num_emb_in`` and ``dim_emb`` from input shape.

   .. method:: call(inputs)

      :param inputs: Input embeddings
      :returns: Output embeddings

Wukong Model
------------

.. class:: Wukong(Model)

   Complete Wukong model for recommendation.

   :param num_layers: Number of interaction layers
   :param num_sparse_emb: Embedding vocabulary size
   :param dim_emb: Embedding dimension
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

       import tensorflow as tf
       from model.tensorflow import Wukong

       model = Wukong(
           num_layers=3,
           num_sparse_emb=100,
           dim_emb=128,
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

       # Note: TensorFlow uses list input format
       sparse_inputs = tf.random.uniform(
           (1024, 32), minval=0, maxval=100, dtype=tf.int32
       )
       dense_inputs = tf.random.uniform((1024, 16))

       logits = model([sparse_inputs, dense_inputs])  # (1024, 1)
       predictions = tf.nn.sigmoid(logits)

   .. method:: call(inputs)

      Full forward pass through the Wukong model.

      :param inputs: List of [sparse_inputs, dense_inputs]
      :returns: Output logits ``(batch_size, dim_output)``

Lazy Initialization Pattern
===========================

TensorFlow/Keras uses lazy initialization through the ``build()`` method:

.. code-block:: python

    class Component(Layer):
        def __init__(self, num_out):
            super().__init__()
            self.num_out = num_out

        def build(self, input_shape):
            num_in = input_shape[-1]
            self.weight = self.add_weight(
                shape=(num_in, self.num_out),
                initializer="he_uniform",
                name="weight"
            )

        def call(self, x):
            return x @ self.weight

This pattern:

- Defers weight creation until input shapes are known
- Enables more flexible model composition
- Supports dynamic input shapes

Serialization Support
=====================

All components implement ``get_config()`` for model saving/loading:

.. code-block:: python

    def get_config(self):
        return {
            "num_emb_out": self.num_emb_out,
            "weights_initializer": self._weights_initializer,
        }

This enables:

- Model serialization with ``model.save()``
- Layer reconstruction from config
- Model export for deployment

Dependencies
============

- ``tensorflow>=2.16``
- ``keras`` (bundled with TensorFlow)

Cross-References
================

- :doc:`/architecture/wukong` - Wukong architecture overview
- :doc:`/wukong/pytorch/index` - PyTorch implementation
