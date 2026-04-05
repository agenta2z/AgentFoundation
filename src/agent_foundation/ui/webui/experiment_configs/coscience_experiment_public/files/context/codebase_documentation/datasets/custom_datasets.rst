Custom Datasets
===============

This page documents how to integrate your own datasets into the Generative
Recommenders framework for experimentation with novel data sources.

Required Format
---------------

Custom datasets must be converted to the **SASRec format**:

**File Structure**::

    your_dataset/
    ├── your_dataset.txt       # Full sequences (optional)
    ├── your_dataset_train.txt # Training sequences
    ├── your_dataset_valid.txt # Validation data
    └── your_dataset_test.txt  # Test data

**Line Format** (space-separated)::

    user_id item_id_1 item_id_2 item_id_3 ...

**Requirements**:

1. User IDs: Contiguous integers starting from 1
2. Item IDs: Contiguous integers starting from 1
3. Sequences: Chronologically ordered (oldest to newest)
4. Minimum length: 3 items per user (train + valid + test)

Creating a Custom Preprocessor
------------------------------

Extend the base preprocessor class::

    from generative_recommenders.research.data.preprocessor import PublicDataPreprocessor

    class MyDatasetPreprocessor(PublicDataPreprocessor):
        def __init__(self):
            super().__init__(
                name="my-dataset",
                expected_items=None,  # Set if known
                tmp_dir="./tmp/my-dataset"
            )

        def _download(self) -> None:
            # Implement download logic if needed
            # Or skip if data is already local
            pass

        def _load_ratings(self) -> pd.DataFrame:
            # Load your data into DataFrame with columns:
            # user_id, item_id, rating (optional), timestamp
            df = pd.read_csv("path/to/your/data.csv")
            return df[["user_id", "item_id", "timestamp"]]

    # Use your preprocessor
    preprocessor = MyDatasetPreprocessor()
    preprocessor.preprocess_rating()

Preprocessing Best Practices
----------------------------

**1. User Filtering**

Remove users with too few interactions::

    min_interactions = 5  # Adjust based on your data
    user_counts = df.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df["user_id"].isin(valid_users)]

**2. Item Filtering**

Remove very rare items for cleaner training::

    min_item_freq = 5
    item_counts = df.groupby("item_id").size()
    valid_items = item_counts[item_counts >= min_item_freq].index
    df = df[df["item_id"].isin(valid_items)]

**3. Deduplication**

Handle repeated user-item pairs::

    # Keep only first interaction (or last, depending on use case)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")

**4. Timestamp Handling**

If timestamps are missing, create artificial ordering::

    # If no timestamps, use interaction order
    df["timestamp"] = range(len(df))

    # Or use rating timestamps if available
    df["timestamp"] = pd.to_datetime(df["date"]).astype(int)

Validation Checklist
--------------------

Before training, verify your dataset::

    import pandas as pd

    def validate_dataset(train_path, valid_path, test_path):
        train = pd.read_csv(train_path, sep=" ", header=None)
        valid = pd.read_csv(valid_path, sep=" ", header=None)
        test = pd.read_csv(test_path, sep=" ", header=None)

        # Check user ID consistency
        train_users = set(train[0])
        valid_users = set(valid[0])
        test_users = set(test[0])

        assert valid_users.issubset(train_users), "Valid users not in train"
        assert test_users.issubset(train_users), "Test users not in train"

        # Check ID ranges start from 1
        all_items = set()
        for df in [train, valid, test]:
            for col in df.columns[1:]:
                all_items.update(df[col].dropna().astype(int))

        assert min(all_items) >= 1, "Item IDs should start from 1"

        # Check for contiguity
        max_item = max(all_items)
        missing = set(range(1, max_item + 1)) - all_items
        if missing:
            print(f"Warning: {len(missing)} missing item IDs")

        print("Validation passed!")

    validate_dataset("train.txt", "valid.txt", "test.txt")

Integration with Training
-------------------------

Update the data config to point to your dataset::

    # In your gin config or training script
    dataset_name = "my-dataset"
    data_dir = "data/my-dataset/"

    # The trainer will look for:
    # - data/my-dataset/my-dataset_train.txt
    # - data/my-dataset/my-dataset_valid.txt
    # - data/my-dataset/my-dataset_test.txt

Common Data Sources
-------------------

**E-commerce Transactions**::

    # Typical format: order_id, user_id, product_id, timestamp
    df = pd.read_csv("transactions.csv")
    df = df.rename(columns={"product_id": "item_id"})

**Click Streams**::

    # Typical format: session_id, user_id, page_id, click_time
    df = pd.read_csv("clicks.csv")
    df = df.rename(columns={"page_id": "item_id", "click_time": "timestamp"})

**Music Listening History**::

    # Typical format: user_id, track_id, play_count, last_played
    df = pd.read_csv("listens.csv")
    df = df.rename(columns={"track_id": "item_id", "last_played": "timestamp"})

Troubleshooting
---------------

**Problem**: Out of memory during preprocessing

**Solution**: Process in chunks::

    chunk_size = 1_000_000
    chunks = []
    for chunk in pd.read_csv("large_data.csv", chunksize=chunk_size):
        processed = preprocess_chunk(chunk)
        chunks.append(processed)
    df = pd.concat(chunks)

**Problem**: Training fails with dimension mismatch

**Solution**: Verify num_items matches your data::

    # Count unique items across all files
    all_items = set()
    for f in ["train.txt", "valid.txt", "test.txt"]:
        with open(f) as file:
            for line in file:
                items = line.strip().split()[1:]
                all_items.update(map(int, items))

    num_items = max(all_items)  # Use this in config

**Problem**: Poor performance on custom data

**Solution**: Check data quality:

1. Verify chronological ordering
2. Remove bot/spam users
3. Ensure sufficient sequence lengths
4. Check for data leakage between splits
