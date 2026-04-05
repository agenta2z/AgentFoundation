Datasets
========

This section provides comprehensive documentation for all datasets used in the
Generative Recommenders and Wukong recommendation systems, including data sources,
preprocessing pipelines, and instructions for custom dataset integration.

.. toctree::
   :maxdepth: 2
   :caption: Dataset Documentation

   public_datasets
   preprocessing_pipeline
   synthetic_data
   custom_datasets
   data_formats
   examples/index

Overview
--------

The codebase supports three categories of datasets:

1. **Public Benchmark Datasets**: MovieLens-1M, MovieLens-20M, Amazon Books
2. **Synthetic Large-Scale Datasets**: ML-3B, ML-13B, ML-18B (fractal expansion)
3. **Custom Datasets**: User-provided data following the standard format

Quick Start
-----------

To prepare all public datasets with a single command::

    python preprocess_public_data.py

This will:

1. Download datasets from official sources (if not already cached)
2. Apply standard preprocessing (user filtering, train/test splits)
3. Output SASRec-compatible format ready for training

Dataset Statistics
------------------

+----------------+------------+------------+------------+------------------+
| Dataset        | Users      | Items      | Ratings    | Density          |
+================+============+============+============+==================+
| MovieLens-1M   | ~6,040     | 3,706      | ~1M        | 4.47%            |
+----------------+------------+------------+------------+------------------+
| MovieLens-20M  | ~138,493   | 26,744     | ~20M       | 0.54%            |
+----------------+------------+------------+------------+------------------+
| Amazon Books   | ~8M        | 695,762    | ~22M       | 0.0004%          |
+----------------+------------+------------+------------+------------------+
| ML-3B (synth)  | ~150M      | ~5M        | ~3B        | Synthetic        |
+----------------+------------+------------+------------+------------------+

Benchmark Results
-----------------

Reference results from the original papers:

**MovieLens-20M NDCG@10**:

- HSTU-large: 0.3813
- SASRec: 0.377
- BERT4Rec: 0.368

**Amazon Books NDCG@10**:

- HSTU-large: 0.0709
- SASRec: 0.054
- GRU4Rec: 0.048
