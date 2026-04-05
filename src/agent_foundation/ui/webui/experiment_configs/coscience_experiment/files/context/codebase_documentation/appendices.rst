.. _cfr-appendices:

====================================
Appendices
====================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Appendix A: Complete Task List
==============================

**Prediction Tasks (23 tasks):**

.. code-block:: python

   PREDICTION_TASKS = [
       "comment", "like", "share", "photo_click", "linear_vpvd",
       "comment_surface_click", "hide", "report", "video_share",
       "comment_vpvd", "photo_vpvd", "dislike", "hide_v2",
       "show_more", "show_less", "comment_tap", "like_reaction",
       "love_reaction", "wow_reaction", "haha_reaction",
       "sad_reaction", "angry_reaction", "support_reaction"
   ]

**Auxiliary Tasks (13 tasks):**

.. code-block:: python

   AUXILIARY_TASKS = [
       "comment_surface_click_x_num_comment_vpv",
       "comment_surface_click_x_num_comment",
       "like_x_num_like",
       "share_x_num_share",
       "photo_click_x_num_photo_click",
       "linear_vpvd_x_time_spent",
       "hide_x_num_hide",
       "comment_tap_x_num_comment_tap",
       "like_reaction_x_num_like_reaction",
       "love_reaction_x_num_love_reaction",
       "wow_reaction_x_num_wow_reaction",
       "haha_reaction_x_num_haha_reaction",
       "sad_reaction_x_num_sad_reaction"
   ]

**Bias Correction Tasks (29 tasks):**

.. code-block:: python

   BIAS_TASKS = [
       # Knowledge distillation
       "kd_comment", "kd_like", "kd_share", "kd_photo_click",
       "kd_linear_vpvd", "kd_comment_surface_click", "kd_hide",
       "kd_bce_linear_vpvd",
       # Bias correction
       "bias_comment", "bias_like", "bias_share", "bias_photo_click",
       "bias_linear_vpvd", "bias_comment_surface_click", "bias_hide",
       # Pairwise ranking
       "comment_pairwise", "like_pairwise", "share_pairwise",
       "photo_click_pairwise", "comment_surface_click_pairwise",
       # Additional
       "bias_hide_v2", "bias_show_more", "bias_show_less",
       "bias_comment_tap", "bias_dislike",
       "comment_x_num_comment", "like_x_num_like", "share_x_num_share"
   ]

   # Total: 23 + 13 + 29 = 65 tasks

Appendix B: Quick Start Guide
=============================

**1. Clone Training Notebook:**

.. code-block:: python

   # Notebook: N5780651
   config = {
       "test_arm": "My Feature Test",
       "app_layer": "cfr_main_feed_mtml_roo_hstu:YOUR_HASH",
       "hardware": "A100_80GB",
       "last_train_ds": "2024-12-01",
       "training_days": 7
   }

**2. Monitor Learning Curve:**

.. code-block:: text

   Notebook: N7726803
   Compare: Candidate vs Retrain
   Key metrics: like_auc, comment_auc, share_auc, linear_vpvd_mse

**3. Setup Recurring:**

.. code-block:: text

   Frequency: 0 1 * * * (daily at 1 AM)
   Start Date: Initial train end + 1 day
   Max Concurrent: All set to 1

**4. Feature Importance:**

.. code-block:: text

   Notebook: N6081697
   Get checkpoint: bunnylol mlhub model <model_id>

**5. Evaluation:**

.. code-block:: text

   Notebook: N5807788 (start evals)
   Notebook: N6075816 (plot results)

Appendix C: Troubleshooting
===========================

Issue 1: OOM During Training
-----------------------------

.. code-block:: python

   # Solution 1: Reduce batch size
   batch_size_per_gpu = 256  # Instead of 512

   # Solution 2: Enable gradient checkpointing
   model_config.activation_checkpointing = True

   # Solution 3: Use FP16 mixed precision
   model_config.use_fp16 = True

Issue 2: Poor AUC on Specific Task
-----------------------------------

.. code-block:: python

   # Solution 1: Increase task weight
   task_weights["share"] = 3.0  # From 2.0

   # Solution 2: Adjust gradient scaling
   task_gradients_on_shared["share"] = 1.0  # Full gradient

   # Solution 3: Add auxiliary task
   auxiliary_tasks.append("share_x_num_share")

Issue 3: Training Divergence
-----------------------------

.. code-block:: python

   # Solution 1: Lower learning rate
   lr_dense = 0.03  # From 0.05

   # Solution 2: Increase warmup
   warmup_steps = 2000  # From 1000

   # Solution 3: Check for bad data
   # Review learning curve for spikes

Appendix D: Model Comparison - CFR vs IFR
=========================================

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - CFR (Main Feed)
     - IFR (Instagram Feed)
   * - Base Architecture
     - MainFeedMTMLROO
     - IFRMTMLCintLite
   * - ROO Support
     - Full ROO with RO/NRO split
     - Limited/Experimental
   * - HSTU
     - Full integration with DataFM
     - Similar but different features
   * - PLE
     - 4-6 expert groups
     - 3-4 expert groups
   * - Tasks
     - 65+ tasks
     - 40+ tasks
   * - Dense Features
     - ~172 (RO + NRO split)
     - ~150 (no RO/NRO split)
   * - Parameters
     - ~5.1B
     - ~3.8B
   * - Training GPUs
     - 512
     - 256
   * - Optimizer
     - ShampooV2
     - AdamW

Key Files Reference
===================

**Model Implementation:**

- ``minimal_viable_ai/models/main_feed_mtml/model_roo_v0.py``

**Configuration:**

- ``minimal_viable_ai/models/main_feed_mtml/conf/flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py``

**Feature Config:**

- ``minimal_viable_ai/models/main_feed_mtml/conf/float_features_config_2024_aug_cargo_roo.py``

**Training Notebooks:**

- `N5780651 <https://www.internalfb.com/intern/anp/view/?id=5780651>`_ - Initial Training
- `N7726803 <https://www.internalfb.com/intern/anp/view/?id=7726803>`_ - Learning Curve
- `N5814221 <https://www.internalfb.com/intern/anp/view/?id=5814221>`_ - Compute Meta
- `N6081697 <https://www.internalfb.com/intern/anp/view/?id=6081697>`_ - Feature Importance
- `N5807788 <https://www.internalfb.com/intern/anp/view/?id=5807788>`_ - Model Evaluation
- `N6075816 <https://www.internalfb.com/intern/anp/view/?id=6075816>`_ - Extract Results

Document Version
================

- **Version**: 1.0
- **Last Updated**: 2025-12-02
- **Status**: Complete

Back to Index
=============

- :doc:`index` - Main documentation index
- :doc:`architecture` - Architecture overview
