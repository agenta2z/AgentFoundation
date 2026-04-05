.. _cfr-hstu-datafm:

====================================
HSTU & DataFM Integration
====================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

HSTU (Hierarchical Sequential Transformer Units) processes user behavior
sequences using transformer architecture with DataFM foundation model features.

Why Sequential Modeling?
========================

User behavior has temporal patterns:

1. **Recency Matters**: Recent interactions are more relevant
2. **Sequence Reveals Intent**: 5 cooking videos → show more recipes
3. **In-Session Behavior**: Actions within a session are correlated

HSTU & DataFM Pipeline
=======================

.. figure:: ../../_static/cfr_hstu_datafm_pipeline.svg
   :alt: HSTU & DataFM Pipeline
   :align: center
   :width: 100%

   Complete pipeline from raw user history to HSTU output

HSTU Architecture
=================

.. code-block:: python

   class HstuCintModule(nn.Module):
       def __init__(self, embedding_tables, hstu_config_dict):
           self.hash_size = 40_000_000
           self.embedding_dim = 64
           self.encoder_dim = 192
           self.max_seq_len = 512 - 50 - 2

           self.contextual_features = [
               ("viewer_country_id_dup", 1),
               ("viewer_age_group", 1)
           ]

DataFM Features
===============

**Public Traits:**

.. code-block:: python

   DATAFM_PUBLIC_FEATURES = [
       "datafm_fb_public_trait_surface",
       "datafm_fb_public_trait_content_type",
       "datafm_fb_public_trait_vpvd_msec",
       "datafm_fb_public_trait_server_time",
   ]

**Merged Features:**

.. code-block:: python

   DATAFM_MERGED_FEATURES = [
       "datafm_fb_trait_post_id_unhashed",
       "datafm_fb_trait_presence_weights",
       "datafm_fb_trait_weights",
       "datafm_fb_trait_server_time"
   ]

Event Weight Encoding
=====================

.. code-block:: python

   PRESENCE_WEIGHTS = {
       'like': 1,        # 2^0
       'comment': 2,     # 2^1
       'share': 4,       # 2^2
       'photo_click': 8, # 2^3
       'vpvd_5s': 16,    # 2^4
   }

   # Example: User liked AND commented
   # presence_weight = 1 | 2 = 3 (binary: 11)

HSTU Forward Pass
=================

.. code-block:: python

   def forward(self, datafm_features, contextual_features, labels):
       # 1. Merge public + private traits
       merged_seq = self.merge_and_dedup(datafm_features)

       # 2. Apply sampling (for efficiency)
       if self.training:
           sampled_seq = self.sample_sequence(merged_seq, alpha=1.96)

       # 3. Embed sequence items
       seq_embeddings = self.embed_sequence(sampled_seq)

       # 4. Add contextual features
       context_emb = self.embed_context(contextual_features)
       combined = torch.cat([seq_embeddings, context_emb], dim=1)

       # 5. Transformer encoding
       encoded = self.transformer_encoder(combined)

       # 6. Pool to fixed output size
       pooled = self.pma_pooling(encoded, num_output=50)

       return pooled.view(B, -1)  # [B, 50*192]

Sequence Length Sampling
========================

.. code-block:: python

   SEQ_LEN_SAMPLING_CONFIGS = (
       DistributionTypes.TS_WEIGHTED,
       SequenceLengthSamplingStrategy(
           weighted_sample=True,
           alpha=1.96  # Recent items more likely sampled
       )
   )

Label Formulas
==============

.. code-block:: python

   # MSI (Multi-Signal Indicator)
   MSI_DEF = "msi_score_2020H2_no_friend_multiplier"

   # Video/Non-video thresholds
   VIDEO_THRESHOLD = "2000"  # ms
   NON_VIDEO_THRESHOLD = "1000"  # ms

   # Skip label
   SKIP_LABEL = """((1 - has_conversions) * (
       (is_video * Lt(time_spent, 2000)) +
       ((1 - is_video) * Lt(time_spent, 1000))
   ))"""

Configuration Summary
=====================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Value
   * - hash_size
     - 40,000,000
   * - embedding_dim
     - 64
   * - encoder_dim
     - 192
   * - max_seq_len
     - 460 (512 - 50 - 2)
   * - num_layers
     - 2
   * - num_heads
     - 4
   * - max_output_len
     - 50

HSTU Kernel Integration
=======================

HSTU leverages several GPU kernel optimizations:

Scaled Dot-Product Attention (SDPA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transformer encoder uses PyTorch's SDPA which dispatches to Flash Attention
when available (``pytorch_modules_roo.py:879``):

.. code-block:: python

   attn_output = torch.nn.functional.scaled_dot_product_attention(
       q, k, v, attn_mask=None, dropout_p=0.0,
   )

SDPA automatically selects the optimal backend (Flash Attention, Memory-Efficient
Attention, or Math) based on input characteristics.

Spindle with Triton Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~

HSTU output processing uses Spindle with Triton kernels
(``pytorch_modules_roo.py:1436-1448``):

.. code-block:: python

   self.spindle = Spindle(
       in_dim=hstu_encoder_dim * 3 * num_hstu,
       hidden_dim=max(hstu_encoder_dim * 3 * num_hstu, skip_conn_dim),
       out_dim=skip_conn_dim,
       kernel=HammerKernel.TRITON if is_train else HammerKernel.TRITON.TRITON_CC,
   )

- **Training**: ``HammerKernel.TRITON`` - Standard Triton kernel
- **Inference**: ``HammerKernel.TRITON.TRITON_CC`` - Triton CudaCompiled variant

PT2 HSTU Loss Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

PT2 compilation on HSTU Loss achieves significant kernel fusion:

- **Before**: 221 kernel launches
- **After**: 30 kernel launches
- **Commit**: ``1bed91e0``

See :doc:`kernel_optimizations` for complete kernel infrastructure documentation.

Next Steps
==========

- :doc:`kernel_optimizations` - GPU kernel optimizations
- :doc:`training_workflow` - Training guide
- :doc:`configuration` - Configuration reference
