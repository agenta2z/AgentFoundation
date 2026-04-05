================
DLRMv3 Datasets
================

.. module:: generative_recommenders.dlrm_v3.datasets
   :synopsis: Dataset implementations for DLRMv3

This document describes the dataset implementations for DLRMv3.

Samples Container
=================

.. class:: Samples

   Batched sample container for DLRMv3.

   :param uih_features_kjt: User interaction history features (KeyedJaggedTensor)
   :param candidates_features_kjt: Candidate item features (KeyedJaggedTensor)

   .. method:: batch_size()

      Returns the batch size.

Dataset Base Class
==================

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
===================

.. class:: DLRMv3RandomDataset(Dataset)

   Synthetic dataset for testing and benchmarking.

   :param hstu_config: DlrmHSTUConfig instance
   :param num_aggregated_samples: Number of samples to generate
   :param is_inference: Whether for inference mode

Cross-References
================

- :doc:`/generative_recommenders/dlrm_v3/index` - DLRMv3 overview
- :doc:`configs` - Configuration
- :doc:`training` - Training
