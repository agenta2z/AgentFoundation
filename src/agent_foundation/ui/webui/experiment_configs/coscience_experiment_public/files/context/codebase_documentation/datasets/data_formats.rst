Data Formats
============

This page provides detailed specifications for all data formats used in the
Generative Recommenders codebase.

SASRec Format (Primary)
-----------------------

The SASRec format is the primary data format used for training and evaluation.

**Specification**:

- Plain text file, one user per line
- Space-separated values
- First value: user ID (integer >= 1)
- Remaining values: item IDs in chronological order (oldest to newest)

**Example**::

    1 45 67 89 123 456 789
    2 12 34
    3 56 78 90 112 134 156 178

**Formal Grammar**::

    file     := line*
    line     := user_id (SP item_id)+ NL
    user_id  := INTEGER (>= 1)
    item_id  := INTEGER (>= 1)
    SP       := ' '
    NL       := '\n'

File Naming Conventions
-----------------------

**Standard Dataset Files**::

    {dataset_name}/
    ├── {dataset_name}.txt        # Full sequences (all data)
    ├── {dataset_name}_train.txt  # Training sequences
    ├── {dataset_name}_valid.txt  # Validation (single item per user)
    └── {dataset_name}_test.txt   # Test (single item per user)

**Validation/Test Format**:

For validation and test files, each line contains exactly one target item::

    # valid.txt / test.txt format
    user_id target_item_id

    # Example:
    1 789
    2 34
    3 178

**Training Format**:

Training file contains the sequence up to (but not including) validation item::

    # train.txt format
    user_id item_1 item_2 ... item_n

    # Example (corresponds to above valid/test):
    1 45 67 89 123 456    # Excludes 789 (valid) and one more for test
    2 12                   # Excludes 34
    3 56 78 90 112 134 156 # Excludes 178

Raw Data Formats
----------------

**MovieLens DAT Format** (ML-1M)::

    UserID::MovieID::Rating::Timestamp

    # Example:
    1::1193::5::978300760
    1::661::3::978302109

**MovieLens CSV Format** (ML-20M)::

    userId,movieId,rating,timestamp

    # Example:
    1,2,3.5,1112486027
    1,29,3.5,1112484676

**Amazon CSV Format**::

    user_id,item_id,rating,timestamp

    # Example (no header):
    A2SUAM1J3GNN3B,0000013714,5.0,1400529600

Intermediate Formats
--------------------

**Pandas DataFrame Schema**:

During preprocessing, data is held in DataFrames with this schema::

    Column      Type      Description
    --------    -------   -----------
    user_id     int64     Original user identifier
    item_id     int64     Original item identifier
    rating      float64   Rating value (optional)
    timestamp   int64     Unix timestamp

**Categorical ID Mapping**:

IDs are converted to contiguous integers::

    # Original -> Mapped
    "user_abc" -> 1
    "user_xyz" -> 2
    "item_123" -> 1
    "item_456" -> 2

Model Input Format
------------------

**Batch Format** (during training)::

    {
        "user_ids": Tensor[batch_size],           # User identifiers
        "input_ids": Tensor[batch_size, seq_len], # Historical item sequence
        "target_ids": Tensor[batch_size],         # Target item to predict
        "seq_lengths": Tensor[batch_size],        # Actual sequence lengths
        "timestamps": Tensor[batch_size, seq_len] # Optional temporal info
    }

**Embedding Input**::

    # Item embeddings: [num_items + 1, embed_dim]
    # Index 0 reserved for padding
    # Indices 1 to num_items for actual items

Negative Sampling Format
------------------------

For training with negative sampling::

    {
        "positive_ids": Tensor[batch_size],        # Ground truth items
        "negative_ids": Tensor[batch_size, num_neg] # Sampled negatives
    }

**Sampling Strategies**:

1. **Uniform**: Random items from catalog
2. **Popularity-based**: Weighted by item frequency
3. **In-batch**: Other positives in same batch

Output Format (Predictions)
---------------------------

**Ranking Output**::

    {
        "user_id": int,
        "ranked_items": List[int],      # Ordered by predicted relevance
        "scores": List[float],          # Corresponding scores
        "ground_truth": int             # Actual target item
    }

**Metric Computation Input**::

    predictions: Tensor[num_users, num_items]  # Score matrix
    targets: Tensor[num_users]                 # Ground truth item per user

Checkpoint Format
-----------------

Model checkpoints follow PyTorch conventions::

    checkpoint.pt:
    {
        "model_state_dict": OrderedDict,
        "optimizer_state_dict": dict,
        "epoch": int,
        "best_metric": float,
        "config": dict
    }

Binary Formats (Future)
-----------------------

For very large datasets, binary formats may be used:

**NumPy Memory-Mapped**::

    # Sequences stored as variable-length arrays
    sequences.npy  # Item IDs
    offsets.npy    # Start index for each user
    lengths.npy    # Sequence length per user

**Apache Parquet**::

    # Columnar format for distributed processing
    data.parquet
    Schema:
    - user_id: INT64
    - item_sequence: LIST<INT64>
    - timestamps: LIST<INT64>

Format Conversion Utilities
---------------------------

Convert between formats::

    from generative_recommenders.research.data import format_utils

    # CSV to SASRec
    format_utils.csv_to_sasrec(
        input_path="data.csv",
        output_path="data_sasrec/",
        user_col="user_id",
        item_col="item_id",
        time_col="timestamp"
    )

    # SASRec to DataFrame
    df = format_utils.sasrec_to_dataframe("data.txt")

    # Validate format
    format_utils.validate_sasrec("data.txt")
