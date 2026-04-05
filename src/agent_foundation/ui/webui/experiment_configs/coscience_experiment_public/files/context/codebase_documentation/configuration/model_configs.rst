===================
Model Configuration
===================

This document describes the model configuration options for both systems.

Generative Recommenders Configuration
=====================================

DlrmHSTUConfig
--------------

The main configuration dataclass for DLRMv3 models:

.. code-block:: python

    @dataclass
    class DlrmHSTUConfig:
        # Model Architecture
        num_layers: int = 5
        num_heads: int = 4
        embedding_dim: int = 512
        attn_qk_dim: int = 128
        attn_linear_dim: int = 128

        # Sequence Configuration
        max_seq_len: int = 8192
        max_num_candidates: int = 1000

        # Feature Configuration
        user_embedding_feature_names: List[str] = field(default_factory=list)
        item_embedding_feature_names: List[str] = field(default_factory=list)

        # Dropout
        input_dropout_ratio: float = 0.0
        output_dropout_ratio: float = 0.3

        # Attention
        causal: bool = True
        target_aware: bool = True

Configuration Factory
---------------------

Use ``get_hstu_configs()`` to get pre-configured settings:

.. code-block:: python

    from generative_recommenders.dlrm_v3.configs import get_hstu_configs

    config = get_hstu_configs("movielens_1m")

Wukong Configuration
====================

Wukong models are configured via constructor parameters:

.. code-block:: python

    model = Wukong(
        # Layer Configuration
        num_layers=3,               # Number of WukongLayers

        # Embedding Configuration
        num_sparse_emb=100,         # Vocabulary size
        dim_emb=128,                # Embedding dimension
        dim_input_sparse=32,        # Number of categorical features
        dim_input_dense=16,         # Number of dense features

        # Interaction Layer Configuration
        num_emb_lcb=16,             # LCB output embeddings
        num_emb_fmb=16,             # FMB output embeddings
        rank_fmb=24,                # FM low-rank approximation

        # MLP Configuration
        num_hidden_wukong=2,        # Hidden layers in WukongLayer MLP
        dim_hidden_wukong=512,      # Hidden dimension
        num_hidden_head=2,          # Hidden layers in projection head
        dim_hidden_head=512,        # Hidden dimension

        # Output
        dim_output=1,               # Output dimension

        # Regularization
        dropout=0.0,                # Dropout probability
    )

Hyperparameter Selection Guide
==============================

Based on Model Size
-------------------

+------------------+------------------+------------------+------------------+
| Parameter        | Small (~1 GFLOP) | Medium (~10 GFLOP) | Large (~100 GFLOP) |
+==================+==================+==================+==================+
| ``num_layers``   | 2                | 4                | 8                |
+------------------+------------------+------------------+------------------+
| ``dim_emb``      | 64               | 128              | 256              |
+------------------+------------------+------------------+------------------+
| ``num_emb_lcb``  | 8                | 16               | 32               |
+------------------+------------------+------------------+------------------+
| ``num_emb_fmb``  | 8                | 16               | 32               |
+------------------+------------------+------------------+------------------+
| ``rank_fmb``     | 16               | 24               | 48               |
+------------------+------------------+------------------+------------------+

Based on Dataset Size
---------------------

+------------------+------------------+------------------+
| Parameter        | Small (<1M)      | Large (>100M)    |
+==================+==================+==================+
| ``dropout``      | 0.2-0.3          | 0.0-0.1          |
+------------------+------------------+------------------+
| ``learning_rate``| 1e-3             | 1e-4             |
+------------------+------------------+------------------+

Cross-References
================

- :doc:`gin_configs` - GIN configuration system
- :doc:`/workflows/training` - Training workflows
