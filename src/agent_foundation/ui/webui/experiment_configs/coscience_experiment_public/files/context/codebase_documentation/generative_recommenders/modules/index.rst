==============================
Generative Recommenders Modules
==============================

This section documents the core neural network modules in the Generative Recommenders framework.

.. toctree::
   :maxdepth: 2

   stu
   hstu_transducer
   preprocessors
   postprocessors
   encoders

Module Overview
===============

The ``generative_recommenders/modules/`` directory contains the neural architecture components:

.. code-block:: text

    modules/
    ├── stu.py                    # Sequential Transduction Unit
    ├── hstu_transducer.py        # Full HSTU pipeline
    ├── dlrm_hstu.py              # DLRMv3 integration
    ├── action_encoder.py         # Action sequence encoding
    ├── content_encoder.py        # Content/item encoding
    ├── positional_encoder.py     # Positional + temporal encodings
    ├── preprocessors.py          # Input feature preprocessing
    ├── postprocessors.py         # Output embedding postprocessing
    └── multitask_module.py       # Multi-task learning heads

Core Components
===============

STU (Sequential Transduction Unit)
----------------------------------

.. module:: generative_recommenders.modules.stu
   :synopsis: Core STU layer and stack implementations

The STU is the fundamental building block of the HSTU architecture.

**STU Base Class**:

.. class:: STU(HammerModule, abc.ABC)

   Abstract base class for Sequential Transduction Units.

   .. method:: forward(x, x_lengths, x_offsets, max_seq_len, num_targets, max_kv_caching_len=0, kv_caching_lengths=None)

      Forward pass through the STU stack.

      :param x: Input embeddings ``[L, d]``
      :param x_lengths: Sequence lengths ``[B]``
      :param x_offsets: Cumulative offsets ``[B+1]``
      :param max_seq_len: Maximum sequence length
      :param num_targets: Number of target tokens ``[B]``
      :param max_kv_caching_len: Max KV cache length for inference
      :param kv_caching_lengths: KV cache lengths per sample
      :returns: Encoded embeddings ``[L, d]``

   .. method:: cached_forward(delta_x, num_targets, max_kv_caching_len=0, kv_caching_lengths=None)

      Cached inference with KV caching for incremental updates.

      :param delta_x: New query tokens
      :param num_targets: Number of targets per sample
      :returns: Updated embeddings

**STULayerConfig**:

.. class:: STULayerConfig

   Configuration dataclass for STU layers.

   :param embedding_dim: Model dimension (default: 512)
   :param num_heads: Number of attention heads (default: 4)
   :param hidden_dim: Linear hidden dimension (default: 128)
   :param attention_dim: QK attention dimension (default: 128)
   :param output_dropout_ratio: Output dropout ratio (default: 0.3)
   :param causal: Whether to use causal masking (default: True)
   :param target_aware: Enable target-aware attention (default: True)
   :param max_attn_len: Maximum attention length for sparsity
   :param attn_alpha: Attention scale factor

**STULayer**:

.. class:: STULayer(STU)

   Single STU layer implementation.

   Key parameters:

   - ``_uvqk_weight``: Unified projection for U, V, Q, K ``[d, 2h(d_hidden + d_attn)]``
   - ``_input_norm_weight/bias``: Layer norm for input
   - ``_output_weight``: Output transformation ``[3h*d_hidden, d]``
   - ``_output_norm_weight/bias``: Layer norm for output

**STUStack**:

.. class:: STUStack(STU)

   Stack of multiple STU layers.

   :param layers: List of STULayer instances
   :param is_inference: Whether in inference mode

HSTUTransducer
--------------

.. module:: generative_recommenders.modules.hstu_transducer
   :synopsis: Full HSTU model pipeline

.. class:: HSTUTransducer(HammerModule)

   Full HSTU model combining preprocessing, computation, and postprocessing.

   :param stu_module: STU stack module
   :param input_preprocessor: Input feature preprocessor
   :param output_postprocessor: Output postprocessor (optional)
   :param input_dropout_ratio: Dropout on input embeddings
   :param positional_encoder: Positional encoder (optional)
   :param is_inference: Whether in inference mode
   :param return_full_embeddings: Return all embeddings, not just targets
   :param listwise: Enable listwise mode

   .. method:: forward(max_uih_len, max_targets, total_uih_len, total_targets, seq_lengths, seq_embeddings, seq_timestamps, num_targets, seq_payloads)

      Full forward pass through the HSTU pipeline.

      :param max_uih_len: Maximum user interaction history length
      :param max_targets: Maximum number of target candidates
      :param total_uih_len: Total UIH tokens across batch
      :param total_targets: Total target tokens across batch
      :param seq_lengths: Per-sample sequence lengths
      :param seq_embeddings: Input embeddings
      :param seq_timestamps: Temporal information
      :param num_targets: Per-sample target counts
      :param seq_payloads: Optional metadata dictionary
      :returns: Tuple of (candidate_embeddings, full_embeddings_optional)

Preprocessors
-------------

.. module:: generative_recommenders.modules.preprocessors
   :synopsis: Input feature preprocessing

.. class:: InputPreprocessor(HammerModule)

   Transforms raw features into embeddings suitable for HSTU.

   Responsibilities:

   - Feature embedding lookup
   - Feature combination
   - Normalization

Postprocessors
--------------

.. module:: generative_recommenders.modules.postprocessors
   :synopsis: Output embedding postprocessing

.. class:: OutputPostprocessor(HammerModule)

   Processes HSTU output embeddings for final prediction.

   Default behavior: L2 normalization of candidate embeddings.

Encoders
--------

Action Encoder
~~~~~~~~~~~~~~

.. module:: generative_recommenders.modules.action_encoder
   :synopsis: Action sequence encoding

Encodes user actions (clicks, views, purchases) into embeddings.

Content Encoder
~~~~~~~~~~~~~~~

.. module:: generative_recommenders.modules.content_encoder
   :synopsis: Content/item encoding

Encodes item content features into embeddings.

Positional Encoder
~~~~~~~~~~~~~~~~~~

.. module:: generative_recommenders.modules.positional_encoder
   :synopsis: Positional and temporal encodings

.. class:: HSTUPositionalEncoder(HammerModule)

   Adds positional and temporal information to embeddings.

   Supports:

   - Absolute position encoding
   - Relative position encoding
   - Temporal (timestamp) encoding
   - Combined position + temporal encoding

Cross-References
================

- :doc:`/architecture/generative_recommenders` - Architecture overview
- :doc:`/generative_recommenders/ops/index` - Operations documentation
- :doc:`/workflows/training` - Training workflows
