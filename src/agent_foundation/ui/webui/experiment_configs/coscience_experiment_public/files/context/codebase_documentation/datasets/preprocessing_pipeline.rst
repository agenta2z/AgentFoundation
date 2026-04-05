Preprocessing Pipeline
======================

This page documents the data preprocessing pipeline that transforms raw datasets
into the format required for training sequential recommendation models.

Entry Point
-----------

The main preprocessing script is ``preprocess_public_data.py``::

    python preprocess_public_data.py

This single command processes all three public datasets (ML-1M, ML-20M, Amazon Books).

Pipeline Architecture
---------------------

The preprocessing pipeline is implemented in
``generative_recommenders/research/data/preprocessor.py``.

.. code-block:: text

    Raw Data (CSV/DAT)
           |
           v
    +------------------+
    | Download & Cache |  <-- Auto-download from official URLs
    +------------------+
           |
           v
    +------------------+
    | Load Ratings     |  <-- Parse user_id, item_id, rating, timestamp
    +------------------+
           |
           v
    +------------------+
    | Filter Users     |  <-- Remove users with < 3 interactions
    +------------------+
           |
           v
    +------------------+
    | Sort by Time     |  <-- Chronological ordering per user
    +------------------+
           |
           v
    +------------------+
    | Category Mapping |  <-- Convert IDs to contiguous integers
    +------------------+
           |
           v
    +------------------+
    | Train/Val/Test   |  <-- Last item = test, second-to-last = val
    +------------------+
           |
           v
    SASRec Format Output

Preprocessing Steps in Detail
-----------------------------

1. **Download and Caching**

   Datasets are automatically downloaded on first run::

       def _download_ml_1m(self) -> None:
           url = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
           # Downloads to tmp_dir, extracts, caches for future runs

2. **User Filtering**

   Users with fewer than 3 timestamps are filtered out to ensure meaningful
   sequences::

       user_count = ratings.groupby("user_id")["timestamp"].nunique()
       valid_users = user_count[user_count >= 3].index
       ratings = ratings[ratings["user_id"].isin(valid_users)]

3. **Chronological Sorting**

   Interactions are sorted by timestamp within each user::

       ratings = ratings.sort_values(["user_id", "timestamp"])

4. **ID Mapping**

   Original IDs are converted to contiguous integers starting from 1::

       user_ids = ratings["user_id"].astype("category").cat.codes + 1
       item_ids = ratings["item_id"].astype("category").cat.codes + 1

5. **Train/Validation/Test Split**

   Uses leave-one-out evaluation protocol:

   - **Test set**: Last item in each user's sequence
   - **Validation set**: Second-to-last item
   - **Training set**: All preceding items

Output Format
-------------

The output follows the SASRec format with separate files for train/val/test::

    data/
    ├── ml-1m/
    │   ├── ml-1m.txt          # Full sequences
    │   ├── ml-1m_train.txt    # Training data
    │   ├── ml-1m_valid.txt    # Validation data
    │   └── ml-1m_test.txt     # Test data
    ├── ml-20m/
    │   └── ...
    └── amzn-books/
        └── ...

**File Format** (space-separated)::

    user_id item_id_1 item_id_2 item_id_3 ...

Example::

    1 45 67 89 123 456
    2 12 34 56 78 90 112

Preprocessor API
----------------

.. py:class:: PublicDataPreprocessor

   Base class for dataset-specific preprocessors.

   .. py:method:: preprocess_rating()

      Main entry point for preprocessing.

   .. py:method:: _download()

      Download dataset from official URL.

   .. py:method:: _load_ratings() -> pd.DataFrame

      Load raw ratings into DataFrame.

   .. py:method:: _create_sasrec_format(ratings: pd.DataFrame)

      Convert to SASRec output format.

Common Preprocessors Registry
-----------------------------

Access preprocessors via the registry::

    from generative_recommenders.research.data.preprocessor import get_common_preprocessors

    preprocessors = get_common_preprocessors()

    # Available keys: "ml-1m", "ml-20m", "amzn-books"
    preprocessors["ml-1m"].preprocess_rating()
    preprocessors["ml-20m"].preprocess_rating()
    preprocessors["amzn-books"].preprocess_rating()

Validation
----------

Each preprocessor validates expected item counts::

    Expected items:
    - ml-1m: 3,706
    - ml-20m: 26,744
    - amzn-books: 695,762

If counts don't match, a warning is logged but processing continues.
