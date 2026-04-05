=================
Training Workflows
=================

This document describes the training workflows for both Generative Recommenders and Wukong models.

Generative Recommenders Training
================================

Prerequisites
-------------

1. Install dependencies:

   .. code-block:: bash

       pip3 install -r requirements.txt

2. Prepare data:

   .. code-block:: bash

       mkdir -p tmp/
       python3 preprocess_public_data.py

   This downloads and preprocesses:

   - MovieLens-1M
   - MovieLens-20M
   - Amazon Books

Research Training (Paper Experiments)
-------------------------------------

For reproducing paper experiments, use the research trainer:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python3 main.py \
       --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
       --master_port=12345

Available configurations:

+----------------+--------------------------------------------------+
| Dataset        | Config Files                                     |
+================+==================================================+
| MovieLens-1M   | ``configs/ml-1m/hstu-sampled-softmax-*.gin``     |
+----------------+--------------------------------------------------+
| MovieLens-20M  | ``configs/ml-20m/hstu-sampled-softmax-*.gin``    |
+----------------+--------------------------------------------------+
| MovieLens-3B   | ``configs/ml-3b/hstu-sampled-softmax-*.gin``     |
+----------------+--------------------------------------------------+
| Amazon Books   | ``configs/amzn-books/hstu-sampled-softmax-*.gin``|
+----------------+--------------------------------------------------+

Monitoring with TensorBoard:

.. code-block:: bash

   tensorboard --logdir exps/ml-1m-l200/ --port 24001 --bind_all

DLRMv3 Training (Production)
----------------------------

For production-ready training with DLRMv3:

.. code-block:: bash

   LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 \
       generative_recommenders/dlrm_v3/train/train_ranker.py \
       --dataset movielens_1m \
       --mode train

Training Modes
~~~~~~~~~~~~~~

- ``train``: Standard training loop
- ``eval``: Evaluation only (requires checkpoint)
- ``train-eval``: Training with periodic evaluation
- ``streaming-train-eval``: Streaming training (continuous data)

Multi-GPU Training
~~~~~~~~~~~~~~~~~~

Set environment variables for distributed training:

.. code-block:: bash

   # 4 GPUs on single machine
   LOCAL_WORLD_SIZE=4 WORLD_SIZE=4 python3 train_ranker.py ...

   # 8 GPUs across 2 machines
   LOCAL_WORLD_SIZE=4 WORLD_SIZE=8 RANK=0 python3 train_ranker.py ...  # Machine 1
   LOCAL_WORLD_SIZE=4 WORLD_SIZE=8 RANK=4 python3 train_ranker.py ...  # Machine 2

Synthetic Data Generation
-------------------------

Generate large-scale synthetic data using fractal expansion:

.. code-block:: bash

   python3 run_fractal_expansion.py \
       --input-csv-file tmp/ml-20m/ratings.csv \
       --write-dataset True \
       --output-prefix tmp/ml-3b/

This expands MovieLens-20M (20M ratings) to MovieLens-3B (3B ratings).

Wukong Training
===============

PyTorch Training
----------------

.. code-block:: python

   import torch
   from model.pytorch import Wukong

   # Create model
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

   # Training setup
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   criterion = torch.nn.BCEWithLogitsLoss()

   # Training loop
   for epoch in range(num_epochs):
       for sparse_batch, dense_batch, labels in dataloader:
           optimizer.zero_grad()
           outputs = model(sparse_batch, dense_batch)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()

TensorFlow Training
-------------------

.. code-block:: python

   import tensorflow as tf
   from model.tensorflow import Wukong

   # Create model
   model = Wukong(
       num_layers=3,
       num_sparse_emb=100,
       dim_emb=128,
       num_emb_lcb=16,
       num_emb_fmb=16,
       rank_fmb=24,
       num_hidden_wukong=2,
       dim_hidden_wukong=512,
       num_hidden_head=2,
       dim_hidden_head=512,
       dim_output=1,
   )

   # Compile model
   model.compile(
       optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
       metrics=['accuracy']
   )

   # Train
   model.fit(dataset, epochs=num_epochs)

Hyperparameter Recommendations
==============================

Generative Recommenders
-----------------------

+--------------------+------------------+-------------------+
| Hyperparameter     | Small Model      | Large Model       |
+====================+==================+===================+
| num_layers         | 4-6              | 12-24             |
+--------------------+------------------+-------------------+
| num_heads          | 4                | 4-8               |
+--------------------+------------------+-------------------+
| embedding_dim      | 256-512          | 512-1024          |
+--------------------+------------------+-------------------+
| attn_dim           | 128              | 128               |
+--------------------+------------------+-------------------+
| hidden_dim         | 128              | 128-512           |
+--------------------+------------------+-------------------+
| learning_rate      | 1e-3             | 1e-4              |
+--------------------+------------------+-------------------+

Wukong
------

+--------------------+------------------+-------------------+
| Hyperparameter     | Small Model      | Large Model       |
+====================+==================+===================+
| num_layers         | 2-3              | 6-8               |
+--------------------+------------------+-------------------+
| dim_emb            | 64-128           | 256-512           |
+--------------------+------------------+-------------------+
| num_emb_lcb        | 8-16             | 32-64             |
+--------------------+------------------+-------------------+
| num_emb_fmb        | 8-16             | 32-64             |
+--------------------+------------------+-------------------+
| rank_fmb           | 16-24            | 32-64             |
+--------------------+------------------+-------------------+
| learning_rate      | 1e-3             | 1e-4              |
+--------------------+------------------+-------------------+

Cross-References
================

- :doc:`/architecture/generative_recommenders` - GR architecture
- :doc:`/architecture/wukong` - Wukong architecture
- :doc:`/configuration/gin_configs` - GIN configuration
- :doc:`inference` - Inference workflows
