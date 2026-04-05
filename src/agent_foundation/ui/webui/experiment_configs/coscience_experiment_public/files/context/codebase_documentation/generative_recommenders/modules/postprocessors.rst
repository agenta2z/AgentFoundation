==============
Postprocessors
==============

.. module:: generative_recommenders.modules.postprocessors
   :synopsis: Output embedding postprocessing

The postprocessor module handles output embedding transformation and candidate extraction.

Overview
========

Postprocessors handle:

- Candidate embedding extraction from full sequence
- L2 normalization for similarity computation
- Score computation for ranking

OutputPostprocessor
===================

.. class:: OutputPostprocessor(HammerModule)

   Processes HSTU output embeddings for final prediction.

   **Default behavior**: L2 normalization of candidate embeddings.

   :param normalize: Whether to apply L2 normalization (default: True)
   :param temperature: Temperature for similarity computation (default: 1.0)

   .. method:: forward(embeddings, seq_offsets, num_targets)

      Process output embeddings.

      :param embeddings: Full sequence embeddings ``[L, d]``
      :param seq_offsets: Sequence boundary offsets ``[B+1]``
      :param num_targets: Number of targets per sample ``[B]``
      :returns: Processed candidate embeddings ``[total_targets, d]``

Processing Pipeline
===================

.. code-block:: text

   Full Sequence Embeddings [L, d]
       ↓
   ┌─────────────────────────────────────┐
   │    Extract Target Embeddings        │
   │    Split by seq_offsets + num_targets│
   │         ↓                           │
   │    Target Embeddings [T, d]         │
   └─────────────────────────────────────┘
       ↓
   ┌─────────────────────────────────────┐
   │    L2 Normalization                 │
   │    ||e|| = 1 for each embedding     │
   │         ↓                           │
   │    Normalized Embeddings [T, d]     │
   └─────────────────────────────────────┘
       ↓
   ┌─────────────────────────────────────┐
   │    Score Computation (optional)     │
   │    dot(candidate, user) / temp      │
   │         ↓                           │
   │    Scores [T]                       │
   └─────────────────────────────────────┘

Implementation Details
======================

Target Extraction
-----------------

The postprocessor uses jagged tensor offsets to extract target embeddings:

.. code-block:: python

   def extract_targets(embeddings, seq_offsets, num_targets):
       target_embeddings = []
       for i in range(len(num_targets)):
           start = seq_offsets[i + 1] - num_targets[i]
           end = seq_offsets[i + 1]
           target_embeddings.append(embeddings[start:end])
       return torch.cat(target_embeddings, dim=0)

L2 Normalization
----------------

L2 normalization ensures embeddings have unit norm, enabling cosine similarity via dot product:

.. code-block:: python

   def l2_normalize(embeddings):
       return embeddings / (embeddings.norm(dim=-1, keepdim=True) + 1e-8)

Usage Example
=============

.. code-block:: python

   from generative_recommenders.modules.postprocessors import OutputPostprocessor

   postprocessor = OutputPostprocessor(
       normalize=True,
       temperature=0.1,
   )

   # Extract and normalize candidate embeddings
   candidate_embeddings = postprocessor(
       embeddings=full_embeddings,
       seq_offsets=seq_offsets,
       num_targets=num_targets,
   )

   # Compute similarity scores
   scores = torch.matmul(candidate_embeddings, user_embedding.T)

Custom Postprocessors
=====================

For custom output processing, inherit from OutputPostprocessor:

.. code-block:: python

   class CustomPostprocessor(OutputPostprocessor):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           self.output_projection = nn.Linear(512, 128)

       def forward(self, embeddings, seq_offsets, num_targets):
           # Extract targets
           targets = self._extract_targets(embeddings, seq_offsets, num_targets)
           # Project to lower dimension
           projected = self.output_projection(targets)
           # Normalize
           return F.normalize(projected, dim=-1)

Cross-References
================

- :doc:`hstu_transducer` - Full HSTU pipeline
- :doc:`preprocessors` - Input preprocessing
- :doc:`/workflows/inference` - Inference workflows
