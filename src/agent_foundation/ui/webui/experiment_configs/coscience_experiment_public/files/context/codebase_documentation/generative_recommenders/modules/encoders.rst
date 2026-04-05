========
Encoders
========

This section documents the encoder modules used for feature encoding in Generative Recommenders.

Action Encoder
==============

.. module:: generative_recommenders.modules.action_encoder
   :synopsis: Action sequence encoding

Encodes user actions (clicks, views, purchases) into embeddings.

.. class:: ActionEncoder(HammerModule)

   Encodes action types into dense embeddings.

   :param num_action_types: Number of distinct action types
   :param embedding_dim: Output embedding dimension

   .. method:: forward(action_ids)

      Encode action IDs into embeddings.

      :param action_ids: Action type IDs ``[L]``
      :returns: Action embeddings ``[L, d]``

   **Example**:

   .. code-block:: python

      action_encoder = ActionEncoder(
          num_action_types=5,  # click, view, add_to_cart, purchase, share
          embedding_dim=512,
      )

      action_embeddings = action_encoder(action_ids)

Content Encoder
===============

.. module:: generative_recommenders.modules.content_encoder
   :synopsis: Content/item encoding

Encodes item content features into embeddings.

.. class:: ContentEncoder(HammerModule)

   Encodes item content features (text, categories, attributes).

   :param feature_configs: Dictionary of feature configurations
   :param embedding_dim: Output embedding dimension
   :param fusion_method: How to fuse multiple features ("concat", "add", "attention")

   .. method:: forward(content_features)

      Encode content features into embeddings.

      :param content_features: Dictionary of feature tensors
      :returns: Content embeddings ``[num_items, d]``

   **Example**:

   .. code-block:: python

      content_encoder = ContentEncoder(
          feature_configs={
              "category": {"num_categories": 1000, "dim": 64},
              "brand": {"num_brands": 5000, "dim": 64},
          },
          embedding_dim=512,
          fusion_method="add",
      )

      item_embeddings = content_encoder(content_features)

Positional Encoder
==================

.. module:: generative_recommenders.modules.positional_encoder
   :synopsis: Positional and temporal encodings

.. class:: HSTUPositionalEncoder(HammerModule)

   Adds positional and temporal information to embeddings.

   **Supported encoding types**:

   - Absolute position encoding
   - Relative position encoding
   - Temporal (timestamp) encoding
   - Combined position + temporal encoding

   :param embedding_dim: Embedding dimension
   :param max_seq_len: Maximum sequence length
   :param encoding_type: Type of encoding ("absolute", "relative", "temporal", "combined")
   :param time_unit: Time unit for temporal encoding ("second", "minute", "hour", "day")

   .. method:: forward(embeddings, positions, timestamps)

      Add positional/temporal encodings to embeddings.

      :param embeddings: Input embeddings ``[L, d]``
      :param positions: Position indices ``[L]``
      :param timestamps: Timestamp values ``[L]``
      :returns: Embeddings with positional information ``[L, d]``

Encoding Types
--------------

Absolute Position Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~

Standard sinusoidal position encoding:

.. math::

   PE_{pos,2i} = \sin(pos / 10000^{2i/d})
   PE_{pos,2i+1} = \cos(pos / 10000^{2i/d})

Relative Position Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~

Encodes relative distances between tokens.

Temporal Encoding
~~~~~~~~~~~~~~~~~

Encodes timestamps using learned embeddings:

.. code-block:: python

   def temporal_encoding(timestamps, time_unit="hour"):
       # Bucket timestamps into time units
       time_buckets = timestamps // TIME_UNIT_SECONDS[time_unit]
       # Lookup temporal embeddings
       return temporal_embedding(time_buckets)

Combined Encoding
~~~~~~~~~~~~~~~~~

Combines position and temporal information:

.. math::

   E_{combined} = E_{position} + E_{temporal}

Usage Example
=============

.. code-block:: python

   from generative_recommenders.modules.positional_encoder import HSTUPositionalEncoder

   encoder = HSTUPositionalEncoder(
       embedding_dim=512,
       max_seq_len=8192,
       encoding_type="combined",
       time_unit="hour",
   )

   # Add positional/temporal encodings
   encoded_embeddings = encoder(
       embeddings=seq_embeddings,
       positions=positions,
       timestamps=timestamps,
   )

Cross-References
================

- :doc:`preprocessors` - Input preprocessing
- :doc:`hstu_transducer` - Full HSTU pipeline
- :doc:`/architecture/generative_recommenders` - Architecture overview
