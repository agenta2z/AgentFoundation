================================================
Generative Recommenders & Wukong Documentation
================================================

.. highlight:: python

Welcome to the comprehensive documentation for the Generative Recommenders and Wukong Recommendation Systems codebase.
This documentation covers two related repositories that advance the state-of-the-art in large-scale recommendation systems.

Overview
========

This codebase contains implementations of cutting-edge recommendation system architectures:

**Generative Recommenders (ICML'24)**
   Implements "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
   - Reformulates recommendation systems as generative modeling problems
   - Uses Hierarchical Sequential Transduction Units (HSTU)
   - Achieves 1.5 trillion parameters with 5.3x-15.2x speedup over FlashAttention2

**Wukong Recommendation System**
   Implements "Wukong: Towards a Scaling Law for Large-Scale Recommendation"
   - Establishes scaling laws for recommendation models
   - Uses stacked Factorization Machines for efficient high-order interactions
   - Provides both PyTorch and TensorFlow implementations

Quick Start
===========

Generative Recommenders
-----------------------

.. code-block:: bash

   # Install dependencies
   pip3 install -r requirements.txt

   # Preprocess data
   mkdir -p tmp/
   python3 preprocess_public_data.py

   # Train model
   CUDA_VISIBLE_DEVICES=0 python3 main.py \
       --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
       --master_port=12345

Wukong Recommendation
---------------------

.. code-block:: python

   import torch
   from model.pytorch import Wukong

   model = Wukong(
       num_layers=3,
       num_sparse_emb=100,
       dim_emb=128,
       dim_input_sparse=32,
       dim_input_dense=16,
       num_emb_lcb=16,
       num_emb_fmb=16,
       rank_fmb=24,
       num_hidden_wukong=2,
       dim_hidden_wukong=512,
       num_hidden_head=2,
       dim_hidden_head=512,
       dim_output=1,
   )

   outputs = model(sparse_inputs, dense_inputs)

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/overview
   architecture/generative_recommenders
   architecture/wukong

.. toctree::
   :maxdepth: 2
   :caption: Generative Recommenders

   generative_recommenders/modules/index
   generative_recommenders/ops/index
   generative_recommenders/dlrm_v3/index
   generative_recommenders/research/index

.. toctree::
   :maxdepth: 2
   :caption: Wukong Recommendation

   wukong/pytorch/index
   wukong/tensorflow/index

.. toctree::
   :maxdepth: 2
   :caption: Datasets

   datasets/index

.. toctree::
   :maxdepth: 2
   :caption: Workflows

   workflows/training
   workflows/inference
   workflows/data_preprocessing

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   configuration/gin_configs
   configuration/model_configs

.. toctree::
   :maxdepth: 1
   :caption: Appendices

   appendices/glossary
   appendices/api_reference

Key Features
============

Generative Recommenders
-----------------------

- **HSTU Architecture**: Hierarchical Sequential Transduction Units for efficient sequential modeling
- **M-FALCON Inference**: Microbatched-Fast Attention Leveraging Cacheable OperatioNs for 285x model complexity
- **Triton Kernels**: Optimized GPU kernels for training and inference
- **Distributed Training**: Multi-GPU training with PyTorch DDP

Wukong
------

- **Stacked Factorization Machines**: Captures up to 2^l order interactions with l layers
- **Low-Rank Approximation**: Reduces FM complexity from O(n²d) to O(nkd)
- **Dual Framework Support**: Identical PyTorch and TensorFlow implementations
- **Scaling Laws**: Demonstrates sustained quality improvement across two orders of magnitude

Research Papers
===============

1. **HSTU/Generative Recommenders**: "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" (ICML 2024)

2. **Wukong**: "Wukong: Towards a Scaling Law for Large-Scale Recommendation" (arXiv:2403.02545)

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
