=================
DLRMv3 Training
=================

.. module:: generative_recommenders.dlrm_v3.train
   :synopsis: Training for DLRMv3

This document describes the training system for DLRMv3.

Entry Point
===========

.. code-block:: bash

   LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \
       generative_recommenders/dlrm_v3/train/train_ranker.py \
       --dataset debug \
       --mode train

Training Modes
==============

- ``train``: Standard training
- ``eval``: Evaluation only
- ``train-eval``: Training with periodic evaluation
- ``streaming-train-eval``: Streaming training with evaluation

Cross-References
================

- :doc:`/generative_recommenders/dlrm_v3/index` - DLRMv3 overview
- :doc:`/workflows/training` - Training workflows
