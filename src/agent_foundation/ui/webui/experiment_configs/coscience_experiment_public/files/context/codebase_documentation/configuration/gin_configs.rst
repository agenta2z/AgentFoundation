==================
GIN Configuration
==================

This document describes the GIN configuration system used by Generative Recommenders.

GIN Overview
============

GIN (Gin Is Not a configuration format) is a configuration system that allows
binding values to functions/classes at runtime. It provides a clean way to
configure hyperparameters across the codebase.

Configuration Files
===================

Location
--------

Configuration files are located in the ``configs/`` directory:

.. code-block:: text

    configs/
    ├── ml-1m/
    │   ├── hstu-sampled-softmax-n128-final.gin
    │   ├── hstu-sampled-softmax-n128-large-final.gin
    │   └── sasrec-sampled-softmax-n128-final.gin
    ├── ml-20m/
    │   ├── hstu-sampled-softmax-n128-final.gin
    │   ├── hstu-sampled-softmax-n128-large-final.gin
    │   └── sasrec-sampled-softmax-n128-final.gin
    ├── ml-3b/
    │   └── hstu-sampled-softmax-n96-seqlen500-*.gin
    └── amzn-books/
        └── hstu-sampled-softmax-n512-*.gin

Naming Convention
-----------------

Config filenames follow the pattern:

``{model}-{loss}-n{negative_samples}-{variant}-final.gin``

- **model**: ``hstu`` or ``sasrec``
- **loss**: ``sampled-softmax``
- **n{X}**: Number of negative samples
- **variant**: ``large`` for larger models, or specific variants

Configuration Structure
=======================

A typical GIN config file looks like:

.. code-block:: text

    # Model Configuration
    train_fn.num_layers = 4
    train_fn.num_heads = 4
    train_fn.embedding_dim = 512
    train_fn.attention_dim = 128
    train_fn.hidden_dim = 128

    # Training Configuration
    train_fn.learning_rate = 1e-3
    train_fn.batch_size = 64
    train_fn.num_epochs = 200

    # Loss Configuration
    train_fn.num_negatives = 128
    train_fn.loss_type = "sampled_softmax"

Key Parameters
==============

Model Architecture
------------------

+----------------------+-------------+------------------------------------------+
| Parameter            | Default     | Description                              |
+======================+=============+==========================================+
| ``num_layers``       | 4-24        | Number of STU layers                     |
+----------------------+-------------+------------------------------------------+
| ``num_heads``        | 4           | Number of attention heads                |
+----------------------+-------------+------------------------------------------+
| ``embedding_dim``    | 512         | Model embedding dimension                |
+----------------------+-------------+------------------------------------------+
| ``attention_dim``    | 128         | Attention QK dimension                   |
+----------------------+-------------+------------------------------------------+
| ``hidden_dim``       | 128         | FFN hidden dimension                     |
+----------------------+-------------+------------------------------------------+

Training
--------

+----------------------+-------------+------------------------------------------+
| Parameter            | Default     | Description                              |
+======================+=============+==========================================+
| ``learning_rate``    | 1e-3        | Initial learning rate                    |
+----------------------+-------------+------------------------------------------+
| ``batch_size``       | 64          | Training batch size                      |
+----------------------+-------------+------------------------------------------+
| ``num_epochs``       | 200         | Number of training epochs                |
+----------------------+-------------+------------------------------------------+
| ``dropout``          | 0.3         | Dropout probability                      |
+----------------------+-------------+------------------------------------------+

Loss
----

+----------------------+-------------+------------------------------------------+
| Parameter            | Default     | Description                              |
+======================+=============+==========================================+
| ``num_negatives``    | 128         | Number of negative samples               |
+----------------------+-------------+------------------------------------------+
| ``loss_type``        | sampled_softmax | Loss function type                   |
+----------------------+-------------+------------------------------------------+

Usage
=====

Loading Configuration
---------------------

.. code-block:: python

    import gin

    # Load configuration file
    gin.parse_config_file("configs/ml-1m/hstu-sampled-softmax-n128-final.gin")

    # Access configured values
    @gin.configurable
    def train_fn(
        num_layers=gin.REQUIRED,
        num_heads=gin.REQUIRED,
        embedding_dim=gin.REQUIRED,
        ...
    ):
        pass

Command Line Override
---------------------

Override configuration values from command line:

.. code-block:: bash

   python3 main.py \
       --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-final.gin \
       --gin_bindings="train_fn.learning_rate=0.0001"

Cross-References
================

- :doc:`/workflows/training` - Training workflows
- :doc:`model_configs` - Model configuration details
