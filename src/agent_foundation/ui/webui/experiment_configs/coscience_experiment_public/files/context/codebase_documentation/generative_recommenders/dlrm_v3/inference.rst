=================
DLRMv3 Inference
=================

.. module:: generative_recommenders.dlrm_v3.inference
   :synopsis: Inference for DLRMv3

This document describes the inference system for DLRMv3.

MLPerf Integration
==================

The inference module integrates with MLPerf LoadGen for standardized benchmarking:

.. code-block:: bash

   python3 generative_recommenders/dlrm_v3/inference/main.py \
       --config configs/inference.gin

Cross-References
================

- :doc:`/generative_recommenders/dlrm_v3/index` - DLRMv3 overview
- :doc:`/workflows/inference` - Inference workflows
