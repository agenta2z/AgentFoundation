=============
Preprocessors
=============

.. module:: generative_recommenders.modules.preprocessors
   :synopsis: Input feature preprocessing

The preprocessor module transforms raw features into embeddings suitable for HSTU processing.

Overview
========

Preprocessors handle:

- Sparse (categorical) feature embedding lookup
- Dense (numerical) feature transformation
- Feature combination and normalization
- Multi-feature fusion strategies

InputPreprocessor
=================

.. class:: InputPreprocessor(HammerModule)

   Transforms raw features into embeddings suitable for HSTU.

   **Responsibilities**:

   - Feature embedding lookup for categorical features
   - Linear transformation for dense features
   - Feature combination (concatenation, addition, or attention)
   - Layer normalization

   :param embedding_configs: Dictionary of feature name to EmbeddingConfig
   :param embedding_dim: Output embedding dimension
   :param combination_method: How to combine multiple features ("concat", "add", "attention")

   .. method:: forward(features)

      Transform features into embeddings.

      :param features: Dictionary of feature name to tensor
      :returns: Combined embeddings ``[L, d]``

Feature Processing Pipeline
===========================

.. code-block:: text

   Raw Features
       |
   +-------------------------------------+
   |    Sparse Features                  |
   |    (user_id, item_id, ...)         |
   |         |                           |
   |    EmbeddingBag lookup              |
   |         |                           |
   |    Sparse Embeddings [L, d]         |
   +-------------------------------------+
       |
   +-------------------------------------+
   |    Dense Features                   |
   |    (price, ratings, ...)           |
   |         |                           |
   |    Linear Transformation            |
   |         |                           |
   |    Dense Embeddings [L, d]          |
   +-------------------------------------+
       |
   +-------------------------------------+
   |    Feature Combination              |
   |    concat / add / attention         |
   |         |                           |
   |    Combined Embeddings [L, d]       |
   +-------------------------------------+
       |
   +-------------------------------------+
   |    Layer Normalization              |
   |         |                           |
   |    Normalized Embeddings [L, d]     |
   +-------------------------------------+

EmbeddingConfig
===============

.. class:: EmbeddingConfig

   Configuration for a single embedding table.

   :param num_embeddings: Vocabulary size
   :param embedding_dim: Embedding dimension
   :param feature_names: List of feature names using this table
   :param combiner: How to combine multi-hot features ("sum", "mean")

   **Example**:

   .. code-block:: python

      item_embedding_config = EmbeddingConfig(
          num_embeddings=1000000,
          embedding_dim=512,
          feature_names=["item_id"],
          combiner="sum",
      )

Usage Example
=============

.. code-block:: python

   from generative_recommenders.modules.preprocessors import InputPreprocessor

   # Define embedding configurations
   embedding_configs = {
       "user_id": EmbeddingConfig(num_embeddings=1000000, embedding_dim=512),
       "item_id": EmbeddingConfig(num_embeddings=500000, embedding_dim=512),
       "action_type": EmbeddingConfig(num_embeddings=10, embedding_dim=512),
   }

   # Create preprocessor
   preprocessor = InputPreprocessor(
       embedding_configs=embedding_configs,
       embedding_dim=512,
       combination_method="add",
   )

   # Process features
   features = {
       "user_id": user_ids,
       "item_id": item_ids,
       "action_type": action_types,
   }
   embeddings = preprocessor(features)

Feature Types
=============

Categorical Features
--------------------

Processed using embedding lookup:

- User IDs
- Item IDs
- Action types
- Category IDs

Numerical Features
------------------

Processed using linear transformation:

- Timestamps (after encoding)
- Prices
- Ratings
- Engagement metrics

Sequence Features
-----------------

Processed using EmbeddingBag with combiner:

- Multi-valued attributes
- Historical sequences

Cross-References
================

- :doc:`hstu_transducer` - Full HSTU pipeline
- :doc:`encoders` - Feature encoders
- :doc:`/generative_recommenders/dlrm_v3/index` - DLRMv3 integration
