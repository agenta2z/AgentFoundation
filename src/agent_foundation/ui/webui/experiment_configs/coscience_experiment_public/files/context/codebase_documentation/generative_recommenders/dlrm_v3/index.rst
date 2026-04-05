=============
DLRMv3 Module
=============

This section documents the DLRMv3 integration with HSTU for production-ready recommendation.

.. toctree::
   :maxdepth: 2

   datasets
   configs
   training
   inference

DLRMv3 Overview
===============

The ``generative_recommenders/dlrm_v3/`` directory contains production-ready model variants:

.. code-block:: text

    dlrm_v3/
    ├── train/                    # Training loop and utilities
    │   ├── train_ranker.py       # Main training entry point
    │   ├── utils.py              # Training utilities
    │   └── gin/                  # GIN configuration files
    ├── inference/                # Inference with MLPerf loadgen
    │   └── main.py
    ├── datasets/                 # Dataset implementations
    │   └── dataset.py
    ├── configs.py                # Model configuration factory
    └── checkpoint.py             # Checkpointing utilities

DlrmHSTU Model
==============

.. module:: generative_recommenders.modules.dlrm_hstu
   :synopsis: DLRMv3 model integrating HSTU

.. class:: DlrmHSTU(HammerModule)

   Production-ready DLRMv3 model using HSTU encoder.

   :param embedding_dim: Model embedding dimension
   :param num_layers: Number of STU layers
   :param num_heads: Number of attention heads
   :param attn_dim: Attention dimension
   :param hidden_dim: Hidden dimension
   :param additional params: See DlrmHSTUConfig

   .. method:: forward(samples)

      Forward pass through the DLRMv3 model.

      :param samples: Samples container with UIH and candidate features
      :returns: Model outputs for loss computation

DlrmHSTUConfig
--------------

.. class:: DlrmHSTUConfig

   Configuration dataclass for DLRMv3 HSTU model.

   :param num_layers: Number of STU layers (default: 5)
   :param num_heads: Number of attention heads (default: 4)
   :param embedding_dim: Embedding dimension (default: 512)
   :param attn_qk_dim: Attention QK dimension (default: 128)
   :param attn_linear_dim: Attention linear dimension (default: 128)
   :param max_seq_len: Maximum sequence length (default: 8192)
   :param max_num_candidates: Maximum candidates (default: 1000)

Datasets
========

Samples Container
-----------------

.. class:: Samples

   Batched sample container for DLRMv3.

   :param uih_features_kjt: User interaction history features (KeyedJaggedTensor)
   :param candidates_features_kjt: Candidate item features (KeyedJaggedTensor)

   .. method:: batch_size()

      Returns the batch size.

Dataset Base Class
------------------

.. class:: Dataset

   Base class for DLRMv3 datasets.

   .. method:: preprocess(use_cache=True)

      Preprocess the dataset.

   .. method:: load_query_samples(sample_list)

      Load samples into memory.

   .. method:: get_samples(id_list)

      Get multiple samples and collate into batch.

   .. method:: get_item_count()

      Return total number of samples.

DLRMv3RandomDataset
-------------------

.. class:: DLRMv3RandomDataset(Dataset)

   Synthetic dataset for testing and benchmarking.

   :param hstu_config: DlrmHSTUConfig instance
   :param num_aggregated_samples: Number of samples to generate
   :param is_inference: Whether for inference mode

Configuration Factory
=====================

.. function:: get_hstu_configs(dataset="debug")

   Factory function to create dataset-specific configurations.

   :param dataset: Dataset name (debug, ml-1m, ml-20m, ml-3b, etc.)
   :returns: DlrmHSTUConfig instance

   Supported datasets:

   - ``debug``: Quick testing configuration
   - ``movielens_1m``: MovieLens-1M
   - ``movielens_20m``: MovieLens-20M
   - ``movielens_13b``: MovieLens-3B (expanded)
   - ``kuairand_1k``: KuaiRand 1K subset
   - ``streaming_*``: Synthetic streaming datasets

.. function:: get_embedding_configs(dataset)

   Get embedding table configurations for a dataset.

   :param dataset: Dataset name
   :returns: Dictionary of feature name to EmbeddingConfig

Training
========

Entry Point
-----------

.. code-block:: bash

   LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \
       generative_recommenders/dlrm_v3/train/train_ranker.py \
       --dataset debug \
       --mode train

Training Modes
--------------

- ``train``: Standard training
- ``eval``: Evaluation only
- ``train-eval``: Training with periodic evaluation
- ``streaming-train-eval``: Streaming training with evaluation

Inference
=========

MLPerf Integration
------------------

The inference module integrates with MLPerf LoadGen for standardized benchmarking.

.. code-block:: bash

   python3 generative_recommenders/dlrm_v3/inference/main.py \
       --config configs/inference.gin

Cross-References
================

- :doc:`/architecture/generative_recommenders` - Architecture overview
- :doc:`/generative_recommenders/modules/index` - Module documentation
- :doc:`/workflows/training` - Training workflows
