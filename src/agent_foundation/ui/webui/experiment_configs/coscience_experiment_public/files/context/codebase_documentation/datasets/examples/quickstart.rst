Quick Start: Loading Datasets
=============================

This example demonstrates how to load and use the preprocessed datasets in under
5 minutes. Perfect for getting started with model training.

.. contents:: Contents
   :local:
   :depth: 2

Setup and Prerequisites
-----------------------

First, ensure you have preprocessed the data::

    python preprocess_public_data.py

Example 1: Loading MovieLens-1M
-------------------------------

The simplest example - loading a small dataset for quick experimentation.

.. code-block:: python
   :caption: load_ml1m.py

    """
    Quick Start Example: Loading MovieLens-1M Dataset

    This notebook demonstrates how to load the MovieLens-1M dataset
    and prepare it for training with the Generative Recommenders framework.
    """

    # Cell 1: Import required libraries
    import os
    from pathlib import Path

    # Cell 2: Define paths
    DATA_DIR = Path("data/ml-1m")
    TRAIN_FILE = DATA_DIR / "ml-1m_train.txt"
    VALID_FILE = DATA_DIR / "ml-1m_valid.txt"
    TEST_FILE = DATA_DIR / "ml-1m_test.txt"

    print(f"Data directory: {DATA_DIR}")
    print(f"Files exist: {TRAIN_FILE.exists()}, {VALID_FILE.exists()}, {TEST_FILE.exists()}")

    # Cell 3: Load training data
    def load_sasrec_data(filepath):
        """Load SASRec format data into a dictionary of user sequences."""
        user_sequences = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    item_ids = [int(x) for x in parts[1:]]
                    user_sequences[user_id] = item_ids
        return user_sequences

    train_data = load_sasrec_data(TRAIN_FILE)
    print(f"Loaded {len(train_data)} users from training set")

    # Cell 4: Examine data structure
    # Get first 5 users
    sample_users = list(train_data.keys())[:5]
    for user_id in sample_users:
        seq = train_data[user_id]
        print(f"User {user_id}: {len(seq)} items - {seq[:5]}...")

    # Cell 5: Calculate basic statistics
    sequence_lengths = [len(seq) for seq in train_data.values()]
    print(f"\nDataset Statistics:")
    print(f"  Total users: {len(train_data)}")
    print(f"  Total interactions: {sum(sequence_lengths)}")
    print(f"  Avg sequence length: {sum(sequence_lengths)/len(sequence_lengths):.2f}")
    print(f"  Min sequence length: {min(sequence_lengths)}")
    print(f"  Max sequence length: {max(sequence_lengths)}")

    # Cell 6: Count unique items
    all_items = set()
    for seq in train_data.values():
        all_items.update(seq)
    print(f"  Unique items: {len(all_items)}")

Expected Output
^^^^^^^^^^^^^^^

When you run this example, you should see::

    Data directory: data/ml-1m
    Files exist: True, True, True
    Loaded 6040 users from training set

    User 1: 48 items - [1193, 661, 914, 3408, 2355]...
    User 2: 127 items - [1357, 3068, 1537, 647, 2194]...
    ...

    Dataset Statistics:
      Total users: 6040
      Total interactions: ~1,000,000
      Avg sequence length: ~165
      Min sequence length: 1
      Max sequence length: ~2000
      Unique items: 3706

Example 2: Loading for PyTorch Training
---------------------------------------

This example shows how to create a PyTorch DataLoader for training.

.. code-block:: python
   :caption: pytorch_dataloader.py

    """
    Example: Creating a PyTorch DataLoader for Sequential Recommendation
    """

    import torch
    from torch.utils.data import Dataset, DataLoader
    from pathlib import Path

    class SequentialRecDataset(Dataset):
        """PyTorch Dataset for sequential recommendation."""

        def __init__(self, data_path, max_seq_len=50):
            self.max_seq_len = max_seq_len
            self.user_sequences = []
            self.targets = []

            # Load data
            with open(data_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        items = [int(x) for x in parts[1:]]
                        # Use all but last item as input, last as target
                        if len(items) >= 2:
                            self.user_sequences.append(items[:-1])
                            self.targets.append(items[-1])

        def __len__(self):
            return len(self.user_sequences)

        def __getitem__(self, idx):
            seq = self.user_sequences[idx]
            target = self.targets[idx]

            # Truncate or pad sequence
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]  # Keep most recent
            else:
                seq = [0] * (self.max_seq_len - len(seq)) + seq  # Pad with 0

            return {
                'input_ids': torch.tensor(seq, dtype=torch.long),
                'target_id': torch.tensor(target, dtype=torch.long),
                'seq_len': torch.tensor(min(len(self.user_sequences[idx]), self.max_seq_len))
            }


    # Create dataset and dataloader
    train_dataset = SequentialRecDataset('data/ml-1m/ml-1m_train.txt', max_seq_len=50)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    # Verify the dataloader
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of batches: {len(train_loader)}")

    # Get a sample batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  target_id: {batch['target_id'].shape}")
    print(f"  seq_len: {batch['seq_len'].shape}")

Example 3: Quick Loading with Framework Utilities
-------------------------------------------------

Using the built-in preprocessor utilities for convenience.

.. code-block:: python
   :caption: framework_loading.py

    """
    Example: Using Framework Utilities for Data Loading
    """

    from generative_recommenders.research.data.preprocessor import get_common_preprocessors

    # Get preprocessor registry
    preprocessors = get_common_preprocessors()

    # Available datasets
    print("Available datasets:")
    for name in preprocessors.keys():
        print(f"  - {name}")

    # Preprocess if not already done
    # preprocessors["ml-1m"].preprocess_rating()

    # The preprocessor handles:
    # 1. Downloading from official sources
    # 2. User filtering (min 3 interactions)
    # 3. Train/val/test splitting
    # 4. SASRec format output

Complete Notebook Code
----------------------

Here's the complete Jupyter notebook cells you can copy directly:

**Cell 1: Imports and Setup**

.. code-block:: python

    import os
    from pathlib import Path
    import torch
    from torch.utils.data import Dataset, DataLoader

    # Set data directory
    DATA_DIR = Path("data/ml-1m")

**Cell 2: Load Function**

.. code-block:: python

    def load_sasrec_data(filepath):
        """Load SASRec format data into dictionary."""
        user_sequences = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    item_ids = [int(x) for x in parts[1:]]
                    user_sequences[user_id] = item_ids
        return user_sequences

**Cell 3: Load All Splits**

.. code-block:: python

    train_data = load_sasrec_data(DATA_DIR / "ml-1m_train.txt")
    valid_data = load_sasrec_data(DATA_DIR / "ml-1m_valid.txt")
    test_data = load_sasrec_data(DATA_DIR / "ml-1m_test.txt")

    print(f"Train: {len(train_data)} users")
    print(f"Valid: {len(valid_data)} users")
    print(f"Test: {len(test_data)} users")

**Cell 4: Ready for Training**

.. code-block:: python

    # Now use train_data, valid_data, test_data in your training loop
    print("Data loaded and ready for training!")
    print(f"Example user sequence: {train_data[1][:10]}...")

Next Steps
----------

- :doc:`data_exploration` - Analyze your data in detail
- :doc:`custom_integration` - Add your own datasets
- :doc:`training_workflow` - Complete training pipeline
