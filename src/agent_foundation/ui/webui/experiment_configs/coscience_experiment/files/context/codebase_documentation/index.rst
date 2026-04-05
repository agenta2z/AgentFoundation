.. _cfr-index:

============================
CFR Models Documentation
============================

The **CFR (Content Feed Ranking)** system powers Facebook's Main Feed, ranking
billions of posts daily for personalized content delivery.

The Main Feed MTML (Multi-Task Multi-Label) model combines:

- **ROO (Rank-Ordered Objectives)**: Pairwise ranking optimization
- **PLE (Progressive Layered Extraction)**: Multi-task learning with shared + task-specific experts
- **HSTU (Hierarchical Sequential Transformer Units)**: User behavior sequence modeling
- **DataFM**: Foundation model integration for rich feature representations

Key Statistics
==============

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Value
   * - **Scale**
     - 5.1B parameters, 100-200 features, 65+ tasks
   * - **Training**
     - 512 GPUs, 256K batch size, ~6 hours to converge
   * - **Inference**
     - ~10ms p50 latency, 10K QPS per instance

Getting Started
===============

For new engineers, read the documentation in this order:

1. :doc:`architecture` - Understand the overall model structure
2. :doc:`roo_architecture` - Learn about pairwise ranking optimization
3. :doc:`hstu_datafm` - Understand sequential user modeling
4. :doc:`training_workflow` - Follow the training guide
5. :doc:`configuration` - Reference for feature and task configuration
6. :doc:`appendices` - Quick reference and troubleshooting

Key Files
=========

**Model Implementation**

- ``minimal_viable_ai/models/main_feed_mtml/model_roo_v0.py`` - Main ROO model
- ``minimal_viable_ai/models/main_feed_mtml/conf/flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py`` - Production config

**Training Notebooks**

- `N5780651 <https://www.internalfb.com/intern/anp/view/?id=5780651>`_ - Initial Training
- `N7726803 <https://www.internalfb.com/intern/anp/view/?id=7726803>`_ - Learning Curve Monitoring
- `N5814221 <https://www.internalfb.com/intern/anp/view/?id=5814221>`_ - Compute Meta
- `N6081697 <https://www.internalfb.com/intern/anp/view/?id=6081697>`_ - Feature Importance
- `N5807788 <https://www.internalfb.com/intern/anp/view/?id=5807788>`_ - Model Evaluation

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Contents

   architecture
   roo_architecture
   hstu_datafm
   training_workflow
   configuration
   appendices

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
