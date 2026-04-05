==================
DLRMv3 Configs
==================

.. module:: generative_recommenders.dlrm_v3.configs
   :synopsis: Configuration for DLRMv3

This document describes the configuration system for DLRMv3.

Configuration Factory
=====================

.. function:: get_hstu_configs(dataset="debug")

   Factory function to create dataset-specific configurations.

   :param dataset: Dataset name
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

Cross-References
================

- :doc:`/generative_recommenders/dlrm_v3/index` - DLRMv3 overview
- :doc:`/configuration/model_configs` - Model configuration
