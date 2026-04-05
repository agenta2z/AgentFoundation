=========================
Wukong Model (TensorFlow)
=========================

.. module:: model.tensorflow.wukong
   :synopsis: Core Wukong architecture components (TensorFlow)

This document provides detailed documentation for the Wukong TensorFlow model components.

For the main TensorFlow module overview, see :doc:`/wukong/tensorflow/index`.

Component Classes
=================

LinearCompressBlock
-------------------

.. class:: LinearCompressBlock(Layer)

   Linearly recombines embeddings without increasing interaction orders.

   :param num_emb_out: Number of output embeddings
   :param weights_initializer: Weight initializer (default: "he_uniform")
   :param name: Layer name (default: "lcb")

   **Build Method**: Creates weight from ``input_shape[-2]`` (num_emb_in).

FactorizationMachineBlock
-------------------------

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

   **Build Method**: Infers ``num_emb_in`` and ``dim_emb`` from input shape.

Wukong Model
------------

.. class:: Wukong(Model)

   Complete Wukong model for recommendation.

   See :doc:`/wukong/tensorflow/index` for full parameter documentation and examples.

Lazy Initialization Pattern
===========================

TensorFlow uses lazy initialization through ``build()``:

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

Benefits:

- Defers weight creation until input shapes are known
- Enables more flexible model composition
- Supports dynamic input shapes

Cross-References
================

- :doc:`/wukong/tensorflow/index` - TensorFlow implementation overview
- :doc:`/architecture/wukong` - Architecture overview
- :doc:`embedding` - Embedding module
- :doc:`mlp` - MLP module
