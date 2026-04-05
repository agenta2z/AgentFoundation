.. _cfr-configuration:

====================================
Configuration Reference
====================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

This document provides a comprehensive reference for CFR model configuration
including features, tasks, and hyperparameters.

Dense Feature Processing
========================

**RO (Request-Only) Dense Features:**

.. code-block:: python

   RO_DENSE_FEATURES = [
       # User demographics
       "viewer_age_group",
       "viewer_gender",
       "viewer_country_id",

       # User activity
       "user_feed_engagement_rate_7d",
       "user_comment_rate_7d",
       "user_share_rate_7d",

       # Session context
       "session_position",
       "session_duration_sec",
       "time_of_day_encoded",
       "day_of_week_encoded"
   ]

**NRO (Non-Request-Only) Dense Features:**

.. code-block:: python

   NRO_DENSE_FEATURES = [
       # Content features
       "post_age_hours",
       "post_length_chars",
       "post_media_count",
       "is_video_story_v1",

       # Author features
       "author_follower_count",
       "author_post_frequency",

       # Engagement features
       "post_like_count",
       "post_comment_count",
       "post_share_count",

       # Quality scores
       "post_quality_score",
       "msi_score_2020H2_no_friend_multiplier"
   ]

Compute Meta
============

Feature normalization statistics:

.. code-block:: python

   COMPUTE_META_ID = 749856691  # Latest run ID

   feature_stats = {
       "post_age_hours": {
           "mean": 24.5, "std": 48.2,
           "min": 0.0, "max": 720.0
       },
       "user_feed_engagement_rate_7d": {
           "mean": 0.15, "std": 0.12,
           "min": 0.0, "max": 1.0
       }
   }

   def normalize_feature(value, feature_name):
       stats = feature_stats[feature_name]
       normalized = (value - stats['mean']) / (stats['std'] + 1e-8)
       return np.clip(normalized, -3.0, 3.0)

Sparse Feature Embedding Tables
===============================

.. code-block:: python

   embedding_tables = {
       "user_id": EmbeddingTable(
           num_embeddings=1_000_000_000,
           embedding_dim=160,
           data_type=DataType.FP16,
           pooling=PoolingType.NONE,
           sharding=ShardingType.TABLE_ROW_WISE,
           location=LocationType.HBM
       ),
       "post_id": EmbeddingTable(
           num_embeddings=500_000_000,
           embedding_dim=160,
           data_type=DataType.FP16,
           pooling=PoolingType.NONE,
           sharding=ShardingType.TABLE_ROW_WISE,
           location=LocationType.HBM
       ),
       "post_topic_ids": EmbeddingTable(
           num_embeddings=50_000,
           embedding_dim=64,
           pooling=PoolingType.SUM,
           sharding=ShardingType.DATA_PARALLEL,
           location=LocationType.DDR
       ),
   }

ID Score List Features
======================

.. code-block:: python

   id_score_list_features = {
       "recent_viewed_post_ids": IdScoreListFeature(
           hash_size=50_000_000,
           pooling_factor_mean=0.14,
           pooling_factor_std=1.0,
           time_gap_pooling=True,
           decay_pow=0.5
       ),
       "user_author_interaction_history": IdScoreListFeature(
           hash_size=100_000_000,
           pooling_factor_mean=0.14,
           pooling_factor_std=1.0,
           time_gap_pooling=True,
           decay_pow=1.0
       )
   }

Task Architecture Configuration
===============================

.. code-block:: python

   task_arch_configs = {
       "like": PredictionTaskArchConfig(
           input_dim=256,
           layer_sizes=[128, 64],
           output_dim=1,
           activation="swish_layer_norm",
           loss_fn=LossFnEnum.BCE,
           task_weight=1.0
       ),
       "linear_vpvd": PredictionTaskArchConfig(
           input_dim=256,
           layer_sizes=[128, 64],
           output_dim=1,
           activation="swish_layer_norm",
           loss_fn=LossFnEnum.MSE,
           task_weight=10.3
       ),
       "kd_linear_vpvd": BiasTaskArchConfig(
           input_dim=256,
           layer_sizes=[128, 64],
           output_dim=1,
           loss_fn=LossFnEnum.KDLM,
           kd_lm_pred_cap=20.0,
           task_weight=10.3
       )
   }

Task Weights and Gradient Scaling
=================================

.. code-block:: python

   task_gradients_on_shared = {
       # Core engagement: full gradient
       "like": 1.0,
       "comment": 1.0,
       "share": 1.0,

       # Negative signals: reduced (noisy)
       "hide": 0.001,
       "report": 0.001,

       # Auxiliary: medium gradient
       "linear_vpvd": 0.5,
       "kd_like": 0.5,
       "kd_comment": 0.5
   }

Hyperparameter Reference
========================

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Parameter
     - Value
     - Description
   * - embedding_dim
     - 160
     - Base embedding dimension
   * - hstu_embedding_dim
     - 64
     - HSTU embedding dimension
   * - hstu_encoder_dim
     - 192
     - HSTU transformer dimension
   * - hstu_max_seq_len
     - 460
     - Max sequence length
   * - dense_num_heads
     - 8
     - MHTA heads
   * - dense_head_dim
     - 1024
     - MHTA head dimension
   * - num_expert_groups
     - 4
     - PLE expert groups
   * - num_shared_experts
     - 2
     - Shared experts per group
   * - dcfn_k
     - 16
     - DCFN compression factor
   * - learning_rate_dense
     - 0.05
     - Dense parameter LR
   * - learning_rate_sparse
     - 0.05
     - Sparse parameter LR
   * - batch_size_per_gpu
     - 512
     - Per-GPU batch size
   * - gradient_clip_norm
     - 1.0
     - Gradient clipping
   * - warmup_steps
     - 1000
     - LR warmup steps

Optimizer Configuration
=======================

**Dense Optimizer (Shampoo V2):**

.. code-block:: python

   dense_optimizer = create_shampoo_v2_optimizer_config(
       lr=0.05,
       betas=(0.9, 0.999),
       epsilon=1e-8,
       weight_decay=0.01,
       preconditioning_compute_steps=10,
       statistics_compute_steps=1,
   )

**Sparse Optimizer (Row-wise AdaGrad):**

.. code-block:: python

   sparse_optimizer = SparseOptimConfig(
       optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
       learning_rate=0.05,
       weight_decay=1e-8,
       weight_decay_mode=WeightDecayMode.DECOUPLE,
   )

Next Steps
==========

- :doc:`appendices` - Task lists and troubleshooting
- :doc:`architecture` - Architecture overview
