===================
Data Preprocessing
===================

This document describes the data preprocessing workflows for both systems.

Generative Recommenders Data Preprocessing
==========================================

Automated Preprocessing
-----------------------

Run the preprocessing script to download and prepare datasets:

.. code-block:: bash

   mkdir -p tmp/
   python3 preprocess_public_data.py

This script processes:

- **MovieLens-1M**: 1 million ratings
- **MovieLens-20M**: 20 million ratings
- **Amazon Books**: Product reviews

Processing Steps
----------------

1. **Download**: Automatically downloads raw data files
2. **Filter**: Removes users with fewer than 3 distinct timestamps
3. **Create Sparse Matrix**: Converts to sparse rating matrix format
4. **Save**: Outputs processed files to ``tmp/`` directory

Custom Dataset Processing
-------------------------

For custom datasets, implement a preprocessor:

.. code-block:: python

   from generative_recommenders.research.data.preprocessor import DatasetPreprocessor

   class MyDatasetPreprocessor(DatasetPreprocessor):
       def preprocess_rating(self):
           # Load raw data
           df = pd.read_csv("my_data.csv")

           # Filter users
           user_counts = df.groupby("user_id").size()
           valid_users = user_counts[user_counts >= 3].index
           df = df[df["user_id"].isin(valid_users)]

           # Create sparse matrix
           # ...

Fractal Expansion
=================

Generate large-scale synthetic data using fractal expansion:

.. code-block:: bash

   python3 run_fractal_expansion.py \
       --input-csv-file tmp/ml-20m/ratings.csv \
       --write-dataset True \
       --output-prefix tmp/ml-3b/

Algorithm Overview
------------------

Based on arXiv:1901.08910, fractal expansion:

1. **Randomized Kronecker Product**: Expands the rating matrix
2. **SVD Reduction**: Applies dimensionality reduction
3. **Shuffling**: Adds stochasticity for realistic patterns

Expansion Factors
-----------------

- ML-20M (20M ratings) → ML-3B (3B ratings)
- Preserves statistical properties of original data
- Enables large-scale training experiments

Cross-References
================

- :doc:`training` - Training workflows
- :doc:`inference` - Inference workflows
