=================
Research Modules
=================

This section documents the research code used for the HSTU paper experiments.

Research Overview
=================

The ``generative_recommenders/research/`` directory contains the original research implementation:

.. code-block:: text

    research/
    ├── trainer/                  # Training loop implementation
    │   └── train.py              # Main training function
    ├── modeling/sequential/      # HSTU model definitions
    │   ├── hstu.py               # HSTU model
    │   ├── sasrec.py             # SASRec baseline
    │   ├── input_features_preprocessors.py
    │   └── embedding_modules.py
    ├── data/                     # Data loading
    │   ├── preprocessor.py       # Dataset preprocessors
    │   ├── dataset.py            # Dataset implementations
    │   └── item_features.py
    └── indexing/                 # Candidate indexing
        └── candidate_index.py

Training Module
===============

train_fn
--------

.. function:: train_fn(rank, world_size, master_port)

   Main training function for distributed HSTU training.

   :param rank: Current process rank
   :param world_size: Total number of processes
   :param master_port: Master process port for communication

   The training loop includes:

   1. Distributed training setup (PyTorch DDP)
   2. Model and optimizer creation
   3. Data loading with sampled softmax
   4. Training loop with gradient accumulation
   5. Periodic checkpointing and evaluation

Sequential Modeling
===================

HSTU Model
----------

.. module:: generative_recommenders.research.modeling.sequential.hstu
   :synopsis: Original HSTU model implementation

The research HSTU implementation used for paper experiments.

SASRec Baseline
---------------

.. module:: generative_recommenders.research.modeling.sequential.sasrec
   :synopsis: SASRec transformer baseline

SASRec (Self-Attentive Sequential Recommendation) baseline implementation
for comparison experiments.

Data Processing
===============

Dataset Preprocessors
---------------------

.. function:: get_common_preprocessors()

   Factory function returning preprocessors for common datasets.

   :returns: Dictionary mapping dataset names to preprocessor instances

   Supported datasets:

   - ``ml-1m``: MovieLens-1M
   - ``ml-20m``: MovieLens-20M
   - ``amzn-books``: Amazon Books

Cross-References
================

- :doc:`/architecture/generative_recommenders` - Architecture overview
- :doc:`/generative_recommenders/modules/index` - Production modules
- :doc:`/workflows/training` - Training workflows
