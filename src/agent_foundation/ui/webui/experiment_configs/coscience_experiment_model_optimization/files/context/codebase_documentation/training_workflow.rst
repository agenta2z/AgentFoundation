.. _cfr-training-workflow:

====================================
CFR Training Workflow
====================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

This document covers the complete training workflow for CFR models, from
development to production deployment.

Training Workflow Diagram
=========================

.. figure:: ../../_static/cfr_training_workflow.svg
   :alt: CFR Training Workflow
   :align: center
   :width: 100%

   Complete model development to production pipeline

Step 1: Model Development
=========================

**Local Development:**

.. code-block:: bash

   # 1. Rebase development diff
   arc patch D74835391

   # 2. Get correct light_cli version
   fbpkg fetch light_cli:1559 -d ~

   # 3. Set breakpoints for debugging
   from fbvscode import set_trace
   set_trace()

   # 4. Run local training
   cd minimal_viable_ai/models/main_feed_mtml/scripts
   ./local_run.sh

Step 2: Compute Meta
====================

**Notebook:** `N5814221 <https://www.internalfb.com/intern/anp/view/?id=5814221>`_

When adding new dense features, compute their mean/std statistics:

.. code-block:: python

   # Update in your model keeper diff:
   COMPUTE_META_ID = 749856691  # Your compute meta run ID

   # Example output
   new_feature_stats = {
       "my_new_feature": {
           "mean": 12.5,
           "std": 8.3,
           "min": 0.0,
           "max": 100.0
       }
   }

Step 3: Build App Layer
=======================

.. code-block:: bash

   # Build custom model keeper package
   fbpkg build --ephemeral cfr_main_feed_mtml_roo_hstu_v2 --expire 28d

   # Example output:
   # cfr_main_feed_mtml_roo_hstu:2c4faab  <-- Your app layer hash

Step 4: Initial Training
========================

**Notebook:** `N5780651 <https://www.internalfb.com/intern/anp/view/?id=5780651>`_

.. code-block:: python

   config = {
       "test_arm": "UIP User Features",
       "app_layer": "cfr_main_feed_mtml_roo_hstu:2c4faab",
       "hardware": "A100_80GB",  # or H100, MI250
       "last_train_ds": "2024-11-21",
       "training_days": 7,  # Must be 7 for MC proposals
       "learning_rate": 0.05,
       "batch_size_per_gpu": 512,
       "num_trainers": 64,
       "num_gpus_per_trainer": 8
   }

.. warning::

   Only run **1 job per hardware type** at any time to avoid resource deadlock!

Step 5: Monitor Learning Curve
==============================

**Notebook:** `N7726803 <https://www.internalfb.com/intern/anp/view/?id=7726803>`_

.. code-block:: python

   metrics_to_monitor = [
       "like_auc",
       "comment_auc",
       "share_auc",
       "linear_vpvd_mse",
       "overall_loss"
   ]

**Good signs:**

- Candidate AUC > Retrain AUC
- Loss curves converging
- No significant degradation in any task

**Bad signs:**

- Diverging loss
- Candidate AUC < Retrain AUC
- One task improving at expense of others

Step 6: Recurring Training
==========================

.. code-block:: python

   recurring_config = {
       "frequency": "0 1 * * *",  # Daily at 1 AM
       "start_date": "2024-11-22T01:00:00",
       "max_concurrent_runs": 1,
       "max_retries": 3,
       "retry_delay": "1h",
       "timeout": "24h"
   }

Step 7: Resume from Checkpoint
==============================

If training fails due to transient errors:

.. code-block:: python

   resume_config = {
       "model_id": "your_model_id",
       "checkpoint_id": "latest",
       "resume_from_checkpoint": True,
       "copy_successful_operators": True
   }

Step 8: Feature Importance
==========================

**Notebook:** `N6081697 <https://www.internalfb.com/intern/anp/view/?id=6081697>`_

.. code-block:: bash

   app-layer main fire-app \
       -d ~/fbsource/fbcode/pyper_models/feed_ranking \
       -d ~/fbsource/fbcode/minimal_viable_ai \
       ~/fbsource/fbcode/minimal_viable_ai/models/main_feed_mtml/launch_roo.py

   # Output: fire-app:9f4c237

Step 9: Model Evaluation
========================

**Start Evals:** `N5807788 <https://www.internalfb.com/intern/anp/view/?id=5807788>`_

**Extract Results:** `N6075816 <https://www.internalfb.com/intern/anp/view/?id=6075816>`_

Step 10: Cargo Acceptance
=========================

.. code-block:: text

   CFR Cargo Process:
   Week 1: Submit proposals
   Week 2: Cargo captain incorporates proposals
   Week 3-4: Online QE (Quality Evaluation)
   Week 5: Launch if s.s. topline gains observed

Permissions Required
====================

Contact Silvester Yao (silyao@meta.com) to apply:

1. **"users"** permission in team_public_content_ranking ACL
2. **"write"** permission in:
   - FBPKG:cfr_main_feed_mtml
   - FBPKG:team_public_content_ranking

Hardware Options
================

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Hardware
     - Notes
   * - **A100_80GB**
     - Standard production hardware
   * - **H100**
     - Faster, limited availability
   * - **MI250**
     - AMD GPUs, experimental

Next Steps
==========

- :doc:`configuration` - Configuration reference
- :doc:`appendices` - Troubleshooting guide
