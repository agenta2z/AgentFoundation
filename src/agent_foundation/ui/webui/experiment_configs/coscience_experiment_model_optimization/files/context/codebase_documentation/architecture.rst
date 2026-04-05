.. _cfr-architecture:

=================================
CFR Core Architecture
=================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Executive Summary
=================

The CFR (Content Feed Ranking) model is Meta's production ranking system for the
Facebook Main Feed. This document provides a comprehensive deep dive into the
model architecture, key components, and design decisions.

**Key Highlights:**

- **5.1B parameters** across embedding tables and model weights
- **65+ prediction tasks** including engagement, quality, and safety signals
- **Dual-path architecture** (RO/NRO) for efficient pairwise ranking
- **~6 hours training time** on 512 A100 GPUs

Model Architecture Visualization
================================

.. figure:: ../../_static/cfr_roo_architecture.svg
   :alt: CFR ROO Architecture Detailed View
   :align: center
   :width: 100%

   Detailed CFR ROO architecture showing all major components

.. raw:: html

   <details>
   <summary>Text version (for accessibility)</summary>
   <pre>
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │                              INPUT LAYER                                    │
   ├───────────────────────────────┬─────────────────────────────────────────────┤
   │      RO (Request-Only)        │       NRO (Non-Request-Only)                │
   │  ┌─────────┬─────────┬─────┐  │  ┌─────────┬─────────┬─────────┐           │
   │  │RO Float │RO ID    │RO ID│  │  │NRO Float│NRO ID   │NRO ID   │           │
   │  │Features │List     │Score│  │  │Features │List     │Score    │           │
   │  └────┬────┴────┬────┴──┬──┘  │  └────┬────┴────┬────┴────┬────┘           │
   └───────┼─────────┼───────┼─────┴───────┼─────────┼─────────┼────────────────┘
           │         │       │             │         │         │
           ▼         ▼       ▼             ▼         ▼         ▼
   ┌─────────────────────────────┐ ┌─────────────────────────────────┐
   │      EMBEDDING LAYER        │ │        EMBEDDING LAYER          │
   │    RO Sparse Architecture   │ │     NRO Sparse Architecture     │
   └──────────────┬──────────────┘ └──────────────┬──────────────────┘
                  │                               │
                  ▼                               ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                    DENSE PROCESSING                              │
   │  ┌──────────────────────┐  ┌──────────────────────┐             │
   │  │  RO Dense Heads (6)  │  │  NRO Dense Heads (6) │             │
   │  └──────────┬───────────┘  └───────────┬──────────┘             │
   └─────────────┼──────────────────────────┼─────────────────────────┘
                 │                          │
                 ▼                          │
   ┌──────────────────────────────────────────────────────────────────┐
   │                 HSTU SEQUENTIAL MODELING                         │
   │  ┌──────────────┐  ┌────────────────────┐  ┌──────────────┐     │
   │  │DataFM Feats  │─▶│Transformer Encoder │─▶│ PMA Pooling  │     │
   │  │(512 seq)     │  │(2 layers, 192 dim) │  │ (50 outputs) │     │
   │  └──────────────┘  └────────────────────┘  └──────────────┘     │
   └──────────────────────────────────┬───────────────────────────────┘
                                      │
   ┌──────────────────────────────────▼───────────────────────────────┐
   │                   SHARED INTERACTION                             │
   │  ┌────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐   │
   │  │ Embedding  │─▶│   DCFN   │─▶│ Attention FM │─▶│ DeepFM   │   │
   │  │ Generation │  │(k=16)    │  │              │  │[256,256] │   │
   │  └────────────┘  └──────────┘  └──────────────┘  └──────────┘   │
   └──────────────────────────────────┬───────────────────────────────┘
                                      │
   ┌──────────────────────────────────▼───────────────────────────────┐
   │              MULTI-HEAD TALKING ATTENTION (MHTA)                 │
   │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌────────────────┐     │
   │  │Head 1│  │Head 2│  │ ...  │  │Head 8│─▶│ Gating Network │     │
   │  │ DCN  │  │ DCN  │  │      │  │ DCN  │  │                │     │
   │  └──────┘  └──────┘  └──────┘  └──────┘  └────────────────┘     │
   └──────────────────────────────────┬───────────────────────────────┘
                                      │
   ┌──────────────────────────────────▼───────────────────────────────┐
   │              CONTENT INTELLIGENCE MODULE                         │
   │  ┌────────────────┐  ┌───────────┐  ┌───────────┐               │
   │  │ CI Embeddings  │─▶│ SlimDHEN  │─▶│ CI Output │               │
   │  │(IFM Mini,KGID) │  │ (128 dim) │  │           │               │
   │  └────────────────┘  └───────────┘  └───────────┘               │
   └──────────────────────────────────┬───────────────────────────────┘
                                      │
   ┌──────────────────────────────────▼───────────────────────────────┐
   │                    PLE EXPERT NETWORK                            │
   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
   │  │Shared Experts│  │Task Experts  │  │Gating Network│           │
   │  │(2 x 256 dim) │  │(1 per task)  │  │(per task)    │           │
   │  └──────────────┘  └──────────────┘  └──────────────┘           │
   └──────────────────────────────────┬───────────────────────────────┘
                                      │
   ┌──────────────────────────────────▼───────────────────────────────┐
   │                      TASK HEADS (65+)                            │
   │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐│
   │  │ Like │ │Comment│ │Share │ │VPVD  │ │Hide  │ │ KD   │ │Pair- ││
   │  │ BCE  │ │ BCE   │ │ BCE  │ │ MSE  │ │ BCE  │ │ KDLM │ │wise  ││
   │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘│
   │           23 Prediction + 13 Auxiliary + 29 Bias = 65 Tasks     │
   └─────────────────────────────────────────────────────────────────┘
   </pre>
   </details>
   <br><br>

Key Components Deep Dive
========================

1. ROO Architecture Overview
-----------------------------

The ROO (Rank-Ordered Objectives) architecture introduces a fundamental split
between **Request-Only (RO)** and **Non-Request-Only (NRO)** features.

**RO Features** are fixed per request and shared across all candidates:
   - User demographics (age, gender, country)
   - User history (engagement rates, session context)
   - Session information (time of day, device type)

**NRO Features** are unique per candidate item:
   - Post content (text, media, length)
   - Author information (follower count, posting frequency)
   - Engagement statistics (likes, comments, shares)

See :doc:`roo_architecture` for detailed ROO implementation.

2. HSTU Integration
--------------------

The HSTU (Hierarchical Sequential Transformer Units) module processes user
behavior sequences using transformer architecture with DataFM foundation model
features.

.. code-block:: python

   class HstuCintModule(nn.Module):
       """HSTU with Content Intelligence integration"""

       def __init__(
           self,
           embedding_tables: Dict[str, EmbeddingTable],
           hstu_config_dict: Dict[str, HstuConfigs],
           base_embedding_dim: int = 160
       ):
           # HSTU parameters
           self.hash_size = 40_000_000
           self.embedding_dim = 64
           self.encoder_dim = 192
           self.max_seq_len = 512 - 50 - 2  # max - output - contextual

           # Contextual features
           self.contextual_features = [
               ("viewer_country_id_dup", 1),
               ("viewer_age_group", 1)
           ]

See :doc:`hstu_datafm` for complete HSTU documentation.

3. Content Intelligence Module
-------------------------------

The Content Intelligence module processes content-specific embeddings through
a SlimDHEN architecture for better feature learning.

.. code-block:: python

   ci_arch = SlimDHEN(
       layers=[
           SlimDHENLayer(
               dcpp=DotCompressPP(
                   embedding_num=ci_num_embs,
                   embedding_dim=160,
                   k=16,  # Compression factor
                   weights_arch=[32, 32],
                   resnet_arch=[32, 32]
               ),
               attention_fm=None,
               dcn=None,
               mlp=MLP(
                   input_dim=ci_num_embs * 16,
                   mlp_arch=[128],
                   activation=SWISH_LAYER_NORM
               ),
               ln=SwishLayerNorm(input_dims=[128])
           )
       ],
       in_features=160 * ci_num_embs,
       out_features=128
   )

**Content Intelligence Features:**

.. code-block:: python

   CONTENT_INTELLIGENCE_FEATURES = [
       # IFM Mini features (Instagram Foundation Model)
       "ifm_mini_hashtag_embedding",
       "ifm_mini_entity_embedding",
       "ifm_mini_open_vocabulary_embedding",

       # Cross-surface features
       "matcha_kgid_embedding",           # Knowledge graph IDs
       "xray_visual_cluster_embedding",   # Visual similarity clusters

       # UIH features (User In-session History)
       "uih_recent_click_embeddings",
       "uih_recent_like_embeddings"
   ]

4. Multi-Task Learning Configuration
-------------------------------------

The CFR model supports 65+ tasks across three categories:

**Prediction Tasks (23 tasks)**

.. code-block:: python

   PREDICTION_TASKS = [
       "comment", "like", "share", "photo_click", "linear_vpvd",
       "comment_surface_click", "hide", "report", "video_share",
       "comment_vpvd", "photo_vpvd", "dislike", "hide_v2",
       "show_more", "show_less", "comment_tap", "like_reaction",
       "love_reaction", "wow_reaction", "haha_reaction",
       "sad_reaction", "angry_reaction", "support_reaction"
   ]

**Auxiliary Tasks (13 tasks)**

.. code-block:: python

   AUXILIARY_TASKS = [
       "comment_surface_click_x_num_comment_vpv",
       "comment_surface_click_x_num_comment",
       "like_x_num_like",
       "share_x_num_share",
       "photo_click_x_num_photo_click",
       # ... more combined objectives
   ]

**Bias Correction Tasks (29 tasks)**

.. code-block:: python

   BIAS_TASKS = [
       # Knowledge distillation
       "kd_comment", "kd_like", "kd_share",
       # Bias correction
       "bias_comment", "bias_like", "bias_share",
       # Pairwise ranking
       "comment_pairwise", "like_pairwise", "share_pairwise",
       # ... more bias tasks
   ]

5. Loss Functions
------------------

The model uses different loss functions for different task types:

**Binary Cross-Entropy (BCE)**

.. code-block:: python

   def bce_loss(logits, labels, weights):
       """Standard classification loss for binary tasks"""
       loss = F.binary_cross_entropy_with_logits(
           logits, labels, reduction='none'
       ) * weights
       return loss.mean()

**Mean Squared Error (MSE)**

.. code-block:: python

   def mse_loss(predictions, labels, weights):
       """For continuous outputs like view duration"""
       loss = F.mse_loss(predictions, labels, reduction='none') * weights / 2.0
       return loss.mean()

**Pairwise Ranking Loss (RankNet)**

.. code-block:: python

   def ranknet_loss(B, logits, labels, weights):
       """Pairwise ranking loss for ROO"""
       scores_i = logits[:B]      # Positive items
       scores_j = logits[B:2*B]   # Negative items

       # Compute pairwise probability
       P_ij = torch.sigmoid(scores_i - scores_j)

       # Binary cross-entropy on pairs
       loss = -torch.log(P_ij + 1e-8)
       return (loss * weights).sum() / weights.sum()

**Knowledge Distillation Logits Matching (KDLM)**

.. code-block:: python

   def kdlm_loss(logits, soft_labels, weights, kd_lm_pred_cap=20.0):
       """Match logits from teacher model"""
       soft_labels = soft_labels.clamp(min=0, max=1)
       soft_label_logits = torch.logit(soft_labels).clamp(
           min=-kd_lm_pred_cap,
           max=kd_lm_pred_cap
       )
       loss = F.mse_loss(logits, soft_label_logits, reduction='none') * weights / 2.0
       return loss.mean()

PLE Expert-Gating Architecture
==============================

.. figure:: ../../_static/cfr_ple_expert_gating.svg
   :alt: PLE Expert-Gating Architecture
   :align: center
   :width: 100%

   Progressive Layered Extraction with shared and task-specific experts

.. raw:: html

   <details>
   <summary>Text version (for accessibility)</summary>
   <pre>
   ┌─────────────────────────────────────────────────────────────────┐
   │                     INPUT FANOUT LAYER                          │
   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
   │  │ Shared   │ │ Task 1   │ │ Task 2   │ │ Task N   │          │
   │  │ Input    │ │ Input    │ │ Input    │ │ Input    │          │
   │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘          │
   └───────┼────────────┼────────────┼────────────┼─────────────────┘
           │            │            │            │
           ▼            ▼            ▼            ▼
   ┌─────────────────┐ ┌─────────────────────────────────────────────┐
   │ SHARED EXPERTS  │ │          TASK-SPECIFIC EXPERTS              │
   │ ┌─────┐ ┌─────┐ │ │ ┌─────┐ ┌─────┐ ┌─────┐      ┌─────┐      │
   │ │ SE1 │ │ SE2 │ │ │ │ TE1 │ │ TE2 │ │ TE3 │ .... │ TEN │      │
   │ └──┬──┘ └──┬──┘ │ │ └──┬──┘ └──┬──┘ └──┬──┘      └──┬──┘      │
   └────┼───────┼────┘ └────┼───────┼───────┼────────────┼──────────┘
        │       │           │       │       │            │
        │       │           │       │       │            │
        ▼       ▼           ▼       ▼       ▼            ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    GATING NETWORKS                              │
   │  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐       │
   │  │ Gate 0  │  │ Gate 1  │  │ Gate 2  │ ..... │ Gate N  │       │
   │  │(Shared) │  │(Task 1) │  │(Task 2) │       │(Task N) │       │
   │  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘       │
   └───────┼────────────┼────────────┼─────────────────┼────────────┘
           │            │            │                 │
           ▼            ▼            ▼                 ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                      TASK OUTPUTS                               │
   │        ┌──────────┐  ┌──────────┐       ┌──────────┐           │
   │        │ Task 1   │  │ Task 2   │ ..... │ Task N   │           │
   │        │ Output   │  │ Output   │       │ Output   │           │
   │        └──────────┘  └──────────┘       └──────────┘           │
   └─────────────────────────────────────────────────────────────────┘

   Legend:
   ─────── Green arrows: Shared expert outputs (go to ALL gates)
   ─────── Orange arrows: Task expert outputs (go ONLY to their gate)
   </pre>
   </details>
   <br><br>

**PLE Configuration:**

.. code-block:: python

   ple_config = {
       "num_expert_groups": 4,
       "num_task_experts": 1,      # Per task
       "num_shared_experts": 2,    # Shared across all tasks
       "expert_layer_size": 256,
   }

Gradient Scaling on Shared Parameters
=====================================

To balance task learning, gradient scaling is applied to shared parameters:

.. code-block:: python

   task_gradients_on_shared = {
       # Core engagement tasks: full gradient
       "like": 1.0,
       "comment": 1.0,
       "share": 1.0,

       # Negative signals: reduced gradient (noisy)
       "hide": 0.001,
       "report": 0.001,

       # Auxiliary tasks: medium gradient
       "linear_vpvd": 0.5,

       # Knowledge distillation: medium gradient
       "kd_like": 0.5,
       "kd_comment": 0.5
   }

   # Implementation in PLE forward pass
   for task_name, task_output in task_outputs.items():
       gradient_scale = task_gradients_on_shared.get(task_name, 1.0)

       # Scale gradients flowing to shared experts
       if gradient_scale != 1.0:
           task_output = (task_output * gradient_scale +
                         task_output.detach() * (1 - gradient_scale))

Task Architecture Configuration
===============================

Each task has its own architecture configuration:

.. code-block:: python

   task_arch_configs = {
       # Binary classification task
       "like": PredictionTaskArchConfig(
           input_dim=256,           # From PLE output
           layer_sizes=[128, 64],   # MLP architecture
           output_dim=1,
           activation="swish_layer_norm",
           loss_fn=LossFnEnum.BCE,
           task_weight=1.0
       ),

       # Regression task
       "linear_vpvd": PredictionTaskArchConfig(
           input_dim=256,
           layer_sizes=[128, 64],
           output_dim=1,
           activation="swish_layer_norm",
           loss_fn=LossFnEnum.MSE,
           task_weight=10.3  # Higher weight
       ),

       # Pairwise ranking task
       "comment_pairwise": PredictionTaskArchConfig(
           input_dim=256,
           layer_sizes=[128, 64],
           output_dim=1,
           activation="swish_layer_norm",
           loss_fn=LossFnEnum.PAIRWISE,
           use_pairwise_loss=True,
           task_weight=10.0
       ),

       # Knowledge distillation task
       "kd_linear_vpvd": BiasTaskArchConfig(
           input_dim=256,
           layer_sizes=[128, 64],
           output_dim=1,
           activation="swish_layer_norm",
           loss_fn=LossFnEnum.KDLM,
           kd_lm_pred_cap=20.0,  # Clamp logits
           task_weight=10.3
       )
   }

Model Forward Pass
==================

The complete forward pass through the CFR model:

.. code-block:: python

   class MainFeedMTMLROO(nn.Module):
       """Main Feed MTML with ROO architecture"""

       def forward(
           self,
           ro_float_features: Tensor,
           nro_float_features: Tensor,
           ro_id_list_features: KeyedJaggedTensor,
           nro_id_list_features: KeyedJaggedTensor,
           ro_id_score_list_features: KeyedJaggedTensor,
           nro_id_score_list_features: KeyedJaggedTensor,
           num_candidates: Tensor,
           num_candidates_sum: int
       ):
           # 1. Process sparse features through embedding layers
           ro_sparse_emb = self.ro_sparse_arch(ro_id_list_features)
           nro_sparse_emb = self.nro_sparse_arch(nro_id_list_features)

           # 2. Process dense features through dense architecture
           ro_dense_out = self.ro_dense_arch(ro_float_features)
           nro_dense_out = self.nro_dense_arch(nro_float_features)

           # 3. HSTU sequential modeling (if enabled)
           if self.hstu_enabled:
               hstu_out = self.hstu_module(
                   ro_sparse_emb, nro_sparse_emb,
                   contextual_features
               )

           # 4. Combine in shared module
           shared_out = self.shared_module(
               ro_dense_out, nro_dense_out,
               ro_sparse_emb, nro_sparse_emb,
               hstu_out
           )

           # 5. Interaction architecture (DCFN, DeepFM, etc.)
           interaction_out = self.interaction_arch(shared_out)

           # 6. Multi-head talking attention
           mhta_out = self.mhta(interaction_out)

           # 7. Content Intelligence (if enabled)
           if self.ci_enabled:
               ci_out = self.ci_module(nro_sparse_emb)
               combined = torch.cat([mhta_out, ci_out], dim=-1)
           else:
               combined = mhta_out

           # 8. Over architecture
           over_out = self.over_arch(combined)

           # 9. PLE layer
           ple_out = self.ple(over_out)

           # 10. Task-specific heads
           predictions = {}
           logits = {}
           for task_name, task_arch in self.task_archs.items():
               task_in = ple_out[task_name]
               predictions[task_name], logits[task_name] = task_arch(task_in)

           return logits, predictions

Key Files Reference
===================

**Model Implementation:**

- ``minimal_viable_ai/models/main_feed_mtml/model_roo_v0.py``

**Configuration:**

- ``minimal_viable_ai/models/main_feed_mtml/conf/flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py``

**HSTU Module:**

- ``hammer/modules/sequential/`` (referenced)

GPU Kernel Infrastructure
========================

The model utilizes several GPU kernel technologies for optimization:

- **Triton Kernels**: Via Spindle for HSTU interaction processing
- **Flash Attention**: Through SDPA dispatch for efficient attention computation
- **FBGEMM**: GPU-optimized embedding operations
- **torch._inductor**: PyTorch 2.0 kernel compilation and fusion

See :doc:`kernel_optimizations` for complete kernel infrastructure documentation,
including already-implemented optimizations and future opportunities.

Next Steps
==========

- :doc:`roo_architecture` - Detailed ROO implementation
- :doc:`hstu_datafm` - HSTU and DataFM integration
- :doc:`kernel_optimizations` - GPU kernel optimizations
- :doc:`training_workflow` - Training procedures
- :doc:`configuration` - Configuration reference
