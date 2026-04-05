==============
HSTUTransducer
==============

.. module:: generative_recommenders.modules.hstu_transducer
   :synopsis: Full HSTU model pipeline

The HSTUTransducer combines preprocessing, STU computation, and postprocessing
into a complete sequential recommendation model.

Overview
========

HSTUTransducer is the main model class that orchestrates:

1. Input preprocessing (feature embedding)
2. Positional/temporal encoding
3. STU stack computation
4. Output postprocessing (candidate embedding extraction)

Class Definition
================

.. class:: HSTUTransducer(HammerModule)

   Full HSTU model combining preprocessing, computation, and postprocessing.

   :param stu_module: STU stack module (STUStack instance)
   :param input_preprocessor: Input feature preprocessor
   :param output_postprocessor: Output postprocessor (optional)
   :param input_dropout_ratio: Dropout on input embeddings (default: 0.0)
   :param positional_encoder: Positional encoder (optional)
   :param is_inference: Whether in inference mode (default: True)
   :param return_full_embeddings: Return all embeddings, not just targets (default: False)
   :param listwise: Enable listwise mode (default: False)

Forward Method
--------------

.. method:: forward(max_uih_len, max_targets, total_uih_len, total_targets, seq_lengths, seq_embeddings, seq_timestamps, num_targets, seq_payloads)

   Full forward pass through the HSTU pipeline.

   :param max_uih_len: Maximum user interaction history length
   :param max_targets: Maximum number of target candidates
   :param total_uih_len: Total UIH tokens across batch
   :param total_targets: Total target tokens across batch
   :param seq_lengths: Per-sample sequence lengths ``[B]``
   :param seq_embeddings: Input embeddings ``[L, d]``
   :param seq_timestamps: Temporal information ``[L]``
   :param num_targets: Per-sample target counts ``[B]``
   :param seq_payloads: Optional metadata dictionary
   :returns: Tuple of (candidate_embeddings, full_embeddings_optional)

Cached Forward Method
---------------------

.. method:: cached_forward(delta_embeddings, num_targets, max_kv_caching_len, kv_caching_lengths)

   Incremental forward pass using KV cache (M-FALCON inference).

   :param delta_embeddings: New candidate embeddings to process
   :param num_targets: Number of targets per sample
   :param max_kv_caching_len: Maximum KV cache length
   :param kv_caching_lengths: Per-sample KV cache lengths
   :returns: Candidate embeddings for new items

Architecture
============

.. code-block:: text

   Input Features (UIH + Candidates)
           ↓
   ┌─────────────────────────────────────┐
   │      InputPreprocessor              │
   │  Features → Embeddings [L, d]       │
   └─────────────────────────────────────┘
           ↓
   ┌─────────────────────────────────────┐
   │   HSTUPositionalEncoder (optional)  │
   │  Add positional/temporal encodings  │
   └─────────────────────────────────────┘
           ↓
   ┌─────────────────────────────────────┐
   │         Input Dropout               │
   └─────────────────────────────────────┘
           ↓
   ┌─────────────────────────────────────┐
   │          STUStack                   │
   │   STULayer 1 → STULayer 2 → ...    │
   │           → STULayer N             │
   └─────────────────────────────────────┘
           ↓
   ┌─────────────────────────────────────┐
   │    OutputPostprocessor (optional)   │
   │  Extract candidate embeddings       │
   │  Apply L2 normalization             │
   └─────────────────────────────────────┘
           ↓
   Candidate Embeddings [total_targets, d]

Usage Example
=============

Training Mode
-------------

.. code-block:: python

   from generative_recommenders.modules import HSTUTransducer, STUStack, STULayer
   from generative_recommenders.modules import InputPreprocessor, OutputPostprocessor

   # Create components
   config = STULayerConfig(embedding_dim=512, num_heads=4)
   stu_stack = STUStack([STULayer(config) for _ in range(12)])
   preprocessor = InputPreprocessor(...)
   postprocessor = OutputPostprocessor()

   # Create transducer
   model = HSTUTransducer(
       stu_module=stu_stack,
       input_preprocessor=preprocessor,
       output_postprocessor=postprocessor,
       input_dropout_ratio=0.1,
       is_inference=False,
   )

   # Forward pass
   candidate_embeddings, _ = model(
       max_uih_len=100,
       max_targets=10,
       total_uih_len=batch_total_uih,
       total_targets=batch_total_targets,
       seq_lengths=seq_lengths,
       seq_embeddings=seq_embeddings,
       seq_timestamps=seq_timestamps,
       num_targets=num_targets,
       seq_payloads={},
   )

Inference Mode (M-FALCON)
-------------------------

.. code-block:: python

   # Switch to inference mode
   model.set_is_inference(True)

   # Phase 1: Prefill KV cache with user history
   model.prefill_kv_cache(user_history_embeddings)

   # Phase 2: Process candidate microbatches
   for candidate_batch in candidate_batches:
       scores = model.cached_forward(
           delta_embeddings=candidate_batch,
           num_targets=batch_targets,
           max_kv_caching_len=max_cache_len,
           kv_caching_lengths=cache_lengths,
       )
       yield scores

Internal Methods
================

_preprocess
-----------

.. method:: _preprocess(seq_embeddings, seq_timestamps, seq_payloads)

   Preprocess raw features into embeddings.

   1. Apply input preprocessor to convert features to embeddings
   2. Add positional/temporal encodings if encoder is provided
   3. Apply input dropout during training

_hstu_compute
-------------

.. method:: _hstu_compute(embeddings, seq_lengths, seq_offsets, max_seq_len, num_targets)

   Run STU stack computation.

   1. Forward through STU layers
   2. Handle target-aware attention masking
   3. Return encoded embeddings

_postprocess
------------

.. method:: _postprocess(embeddings, seq_offsets, num_targets)

   Extract and normalize candidate embeddings.

   1. Split UIH and target embeddings
   2. Apply L2 normalization to candidate embeddings
   3. Return formatted output

Configuration
=============

Key configuration options:

+-------------------------+-------------+------------------------------------------+
| Parameter               | Default     | Description                              |
+=========================+=============+==========================================+
| ``input_dropout_ratio`` | 0.0         | Dropout probability for input embeddings |
+-------------------------+-------------+------------------------------------------+
| ``is_inference``        | True        | Enable inference optimizations           |
+-------------------------+-------------+------------------------------------------+
| ``return_full_embeddings`` | False    | Return all embeddings (not just targets) |
+-------------------------+-------------+------------------------------------------+
| ``listwise``            | False       | Enable listwise ranking mode             |
+-------------------------+-------------+------------------------------------------+

Cross-References
================

- :doc:`stu` - STU layer documentation
- :doc:`preprocessors` - Input preprocessing
- :doc:`postprocessors` - Output postprocessing
- :doc:`/architecture/generative_recommenders` - Architecture overview
