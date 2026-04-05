Custom Dataset Integration
==========================

This example demonstrates how to integrate your own datasets into the Generative
Recommenders framework, from raw data to training-ready format.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

Integrating a custom dataset involves:

1. Converting raw data to SASRec format
2. Validating the converted data
3. Creating train/valid/test splits
4. Registering with the preprocessing framework

Example 1: E-commerce Transaction Data
--------------------------------------

Convert typical e-commerce transaction logs.

.. code-block:: python
   :caption: ecommerce_integration.py

    """
    Custom Dataset Integration: E-commerce Transactions

    This notebook demonstrates converting e-commerce transaction data
    to the SASRec format used by Generative Recommenders.
    """

    # Cell 1: Imports
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    # Cell 2: Load Raw Data
    # Example: Your e-commerce data might look like this
    raw_data = """
    order_id,user_id,product_id,purchase_time,price
    1001,user_a,prod_1,2024-01-15 10:30:00,29.99
    1002,user_a,prod_5,2024-01-16 14:20:00,49.99
    1003,user_b,prod_2,2024-01-15 09:00:00,19.99
    1004,user_a,prod_3,2024-01-17 11:45:00,39.99
    1005,user_b,prod_1,2024-01-16 16:30:00,29.99
    1006,user_c,prod_4,2024-01-15 08:00:00,59.99
    1007,user_b,prod_5,2024-01-17 13:00:00,49.99
    1008,user_c,prod_2,2024-01-16 10:15:00,19.99
    1009,user_c,prod_3,2024-01-17 15:30:00,39.99
    """

    # In practice, load from file:
    # df = pd.read_csv('your_transactions.csv')
    from io import StringIO
    df = pd.read_csv(StringIO(raw_data))

    print("Raw data shape:", df.shape)
    print(df.head())

    # Cell 3: Preprocess and Convert IDs
    def convert_to_sasrec_format(df, user_col='user_id', item_col='product_id',
                                  time_col='purchase_time', min_interactions=3):
        """
        Convert transaction data to SASRec format.

        Args:
            df: DataFrame with user, item, and timestamp columns
            user_col: Name of user ID column
            item_col: Name of item ID column
            time_col: Name of timestamp column
            min_interactions: Minimum interactions per user to keep

        Returns:
            dict: user_id -> list of item_ids (chronologically ordered)
        """
        # Make a copy
        data = df.copy()

        # Convert timestamps
        data['timestamp'] = pd.to_datetime(data[time_col])

        # Filter users with minimum interactions
        user_counts = data[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        data = data[data[user_col].isin(valid_users)]
        print(f"Users after filtering: {len(valid_users)}")

        # Create ID mappings (contiguous integers starting from 1)
        user_mapping = {u: i+1 for i, u in enumerate(sorted(data[user_col].unique()))}
        item_mapping = {p: i+1 for i, p in enumerate(sorted(data[item_col].unique()))}

        # Apply mappings
        data['user_int'] = data[user_col].map(user_mapping)
        data['item_int'] = data[item_col].map(item_mapping)

        # Sort by user and timestamp
        data = data.sort_values(['user_int', 'timestamp'])

        # Group into sequences
        user_sequences = {}
        for user_id, group in data.groupby('user_int'):
            user_sequences[user_id] = group['item_int'].tolist()

        # Save mappings for later use
        mappings = {
            'user_to_int': user_mapping,
            'item_to_int': item_mapping,
            'int_to_user': {v: k for k, v in user_mapping.items()},
            'int_to_item': {v: k for k, v in item_mapping.items()},
        }

        return user_sequences, mappings

    # Convert (using min_interactions=2 for this small example)
    sequences, mappings = convert_to_sasrec_format(df, min_interactions=2)

    print(f"\nConverted {len(sequences)} users")
    for uid, seq in sequences.items():
        original_user = mappings['int_to_user'][uid]
        items = [mappings['int_to_item'][i] for i in seq]
        print(f"  User {uid} ({original_user}): {seq} -> {items}")

    # Cell 4: Create Train/Valid/Test Splits
    def create_splits(user_sequences, output_dir):
        """
        Create train/valid/test splits using leave-one-out.

        For each user:
        - Test: last item
        - Valid: second-to-last item
        - Train: all preceding items
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_lines = []
        valid_lines = []
        test_lines = []

        for user_id, items in user_sequences.items():
            if len(items) < 3:
                continue  # Need at least 3 items

            # Split
            train_items = items[:-2]
            valid_item = items[-2]
            test_item = items[-1]

            # Format as strings
            train_line = f"{user_id} " + " ".join(map(str, train_items))
            valid_line = f"{user_id} {valid_item}"
            test_line = f"{user_id} {test_item}"

            train_lines.append(train_line)
            valid_lines.append(valid_line)
            test_lines.append(test_line)

        # Write files
        dataset_name = output_dir.name

        with open(output_dir / f"{dataset_name}_train.txt", 'w') as f:
            f.write('\n'.join(train_lines))

        with open(output_dir / f"{dataset_name}_valid.txt", 'w') as f:
            f.write('\n'.join(valid_lines))

        with open(output_dir / f"{dataset_name}_test.txt", 'w') as f:
            f.write('\n'.join(test_lines))

        print(f"Created splits in {output_dir}:")
        print(f"  Train: {len(train_lines)} users")
        print(f"  Valid: {len(valid_lines)} users")
        print(f"  Test:  {len(test_lines)} users")

    # Create splits
    # create_splits(sequences, 'data/my-ecommerce')

Example 2: Click Stream Data
----------------------------

Convert web click stream data with session handling.

.. code-block:: python
   :caption: clickstream_integration.py

    """
    Custom Dataset Integration: Click Stream Data

    Handle session-based click data with implicit feedback.
    """

    # Cell 1: Session-based Click Data
    def convert_clickstream_data(df, session_timeout_minutes=30):
        """
        Convert click stream data, handling sessions.

        Args:
            df: DataFrame with user_id, page_id, click_time
            session_timeout_minutes: Gap between clicks to start new session
        """
        data = df.copy()
        data['click_time'] = pd.to_datetime(data['click_time'])
        data = data.sort_values(['user_id', 'click_time'])

        # Calculate time gaps
        data['time_diff'] = data.groupby('user_id')['click_time'].diff()

        # Mark new sessions
        timeout = pd.Timedelta(minutes=session_timeout_minutes)
        data['new_session'] = (data['time_diff'] > timeout) | (data['time_diff'].isna())
        data['session_id'] = data.groupby('user_id')['new_session'].cumsum()

        # Create unique session identifier
        data['full_session_id'] = data['user_id'].astype(str) + '_' + data['session_id'].astype(str)

        # Group by session for sequential modeling
        session_sequences = {}
        for session_id, group in data.groupby('full_session_id'):
            pages = group['page_id'].tolist()
            if len(pages) >= 2:  # Need at least 2 pages per session
                session_sequences[session_id] = pages

        return session_sequences

    # Cell 2: Handle Implicit Feedback
    def implicit_to_sequence(interactions, weight_threshold=0):
        """
        Convert implicit feedback (views, clicks) to sequences.

        Args:
            interactions: DataFrame with user_id, item_id, interaction_count, timestamp
            weight_threshold: Minimum interaction count to include
        """
        # Filter by interaction strength
        strong = interactions[interactions['interaction_count'] > weight_threshold]

        # Sort by timestamp for each user
        strong = strong.sort_values(['user_id', 'timestamp'])

        # Group into sequences
        sequences = {}
        for user_id, group in strong.groupby('user_id'):
            sequences[user_id] = group['item_id'].tolist()

        return sequences

Example 3: Create a Reusable Preprocessor
-----------------------------------------

Create a preprocessor class following the framework pattern.

.. code-block:: python
   :caption: custom_preprocessor.py

    """
    Custom Preprocessor Class

    Create a reusable preprocessor that follows the framework pattern.
    """

    from pathlib import Path
    import pandas as pd
    from typing import Dict, List, Optional

    class CustomDatasetPreprocessor:
        """Preprocessor for custom datasets."""

        def __init__(
            self,
            name: str,
            data_dir: str,
            output_dir: str = "data",
            min_interactions: int = 3,
            expected_items: Optional[int] = None
        ):
            self.name = name
            self.data_dir = Path(data_dir)
            self.output_dir = Path(output_dir) / name
            self.min_interactions = min_interactions
            self.expected_items = expected_items

        def load_raw_data(self) -> pd.DataFrame:
            """Override this method to load your specific data format."""
            raise NotImplementedError("Implement load_raw_data() for your dataset")

        def preprocess(self) -> None:
            """Main preprocessing pipeline."""
            print(f"Preprocessing {self.name}...")

            # Step 1: Load raw data
            df = self.load_raw_data()
            print(f"  Loaded {len(df)} interactions")

            # Step 2: Filter users
            df = self._filter_users(df)

            # Step 3: Map IDs
            df, mappings = self._map_ids(df)

            # Step 4: Create sequences
            sequences = self._create_sequences(df)

            # Step 5: Validate
            self._validate(sequences)

            # Step 6: Create splits and save
            self._create_and_save_splits(sequences)

            # Step 7: Save mappings
            self._save_mappings(mappings)

            print(f"  Preprocessing complete!")

        def _filter_users(self, df: pd.DataFrame) -> pd.DataFrame:
            """Filter users with minimum interactions."""
            counts = df['user_id'].value_counts()
            valid = counts[counts >= self.min_interactions].index
            filtered = df[df['user_id'].isin(valid)]
            print(f"  Filtered to {len(valid)} users")
            return filtered

        def _map_ids(self, df: pd.DataFrame):
            """Map IDs to contiguous integers."""
            user_map = {u: i+1 for i, u in enumerate(sorted(df['user_id'].unique()))}
            item_map = {p: i+1 for i, p in enumerate(sorted(df['item_id'].unique()))}

            df['user_int'] = df['user_id'].map(user_map)
            df['item_int'] = df['item_id'].map(item_map)

            mappings = {'user': user_map, 'item': item_map}
            return df, mappings

        def _create_sequences(self, df: pd.DataFrame) -> Dict[int, List[int]]:
            """Group interactions into user sequences."""
            df = df.sort_values(['user_int', 'timestamp'])
            sequences = {}
            for uid, group in df.groupby('user_int'):
                sequences[uid] = group['item_int'].tolist()
            return sequences

        def _validate(self, sequences: Dict[int, List[int]]) -> None:
            """Validate the preprocessed data."""
            all_items = set()
            for seq in sequences.values():
                all_items.update(seq)

            if self.expected_items and len(all_items) != self.expected_items:
                print(f"  WARNING: Expected {self.expected_items} items, got {len(all_items)}")

        def _create_and_save_splits(self, sequences: Dict[int, List[int]]) -> None:
            """Create train/valid/test splits and save."""
            self.output_dir.mkdir(parents=True, exist_ok=True)

            train, valid, test = [], [], []

            for uid, items in sequences.items():
                if len(items) >= 3:
                    train.append(f"{uid} " + " ".join(map(str, items[:-2])))
                    valid.append(f"{uid} {items[-2]}")
                    test.append(f"{uid} {items[-1]}")

            # Save
            for name, lines in [('train', train), ('valid', valid), ('test', test)]:
                path = self.output_dir / f"{self.name}_{name}.txt"
                with open(path, 'w') as f:
                    f.write('\n'.join(lines))

            print(f"  Saved {len(train)} users to {self.output_dir}")

        def _save_mappings(self, mappings: dict) -> None:
            """Save ID mappings for later use."""
            import json
            path = self.output_dir / "id_mappings.json"
            # Convert int keys to strings for JSON
            json_mappings = {
                'user_to_int': mappings['user'],
                'item_to_int': mappings['item'],
            }
            with open(path, 'w') as f:
                json.dump(json_mappings, f, indent=2, default=str)


    # Example Usage:
    class MyEcommercePreprocessor(CustomDatasetPreprocessor):
        """Preprocessor for my e-commerce dataset."""

        def load_raw_data(self) -> pd.DataFrame:
            df = pd.read_csv(self.data_dir / "transactions.csv")
            df = df.rename(columns={
                'customer_id': 'user_id',
                'product_id': 'item_id',
                'purchase_time': 'timestamp'
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df


    # Run preprocessing:
    # preprocessor = MyEcommercePreprocessor(
    #     name='my-ecommerce',
    #     data_dir='raw_data/',
    #     min_interactions=5
    # )
    # preprocessor.preprocess()

Validation Checklist
--------------------

Before using your custom dataset, verify:

.. code-block:: python
   :caption: validation_checklist.py

    def validate_custom_dataset(dataset_dir, dataset_name):
        """Complete validation for custom datasets."""
        from pathlib import Path

        dataset_dir = Path(dataset_dir)
        errors = []
        warnings = []

        # Check 1: Required files exist
        required_files = [
            f"{dataset_name}_train.txt",
            f"{dataset_name}_valid.txt",
            f"{dataset_name}_test.txt"
        ]

        for fname in required_files:
            if not (dataset_dir / fname).exists():
                errors.append(f"Missing required file: {fname}")

        if errors:
            print("ERRORS:")
            for e in errors:
                print(f"  {e}")
            return False

        # Check 2: Load and validate content
        def load_file(path):
            data = {}
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        uid = int(parts[0])
                        items = [int(x) for x in parts[1:]]
                        data[uid] = items
            return data

        train = load_file(dataset_dir / f"{dataset_name}_train.txt")
        valid = load_file(dataset_dir / f"{dataset_name}_valid.txt")
        test = load_file(dataset_dir / f"{dataset_name}_test.txt")

        print(f"Loaded: {len(train)} train, {len(valid)} valid, {len(test)} test")

        # Check 3: User consistency
        train_users = set(train.keys())
        valid_users = set(valid.keys())
        test_users = set(test.keys())

        if not valid_users.issubset(train_users):
            warnings.append("Some valid users not in train")
        if not test_users.issubset(train_users):
            warnings.append("Some test users not in train")

        # Check 4: ID ranges start from 1
        all_items = set()
        for data in [train, valid, test]:
            for items in data.values():
                all_items.update(items)

        if min(all_items) < 1:
            errors.append(f"Item IDs should start from 1, found min: {min(all_items)}")

        # Check 5: Contiguity
        max_item = max(all_items)
        missing = set(range(1, max_item + 1)) - all_items
        if len(missing) > max_item * 0.1:  # More than 10% missing
            warnings.append(f"{len(missing)} item IDs missing (non-contiguous)")

        # Report
        print("\nValidation Results:")
        if errors:
            print("ERRORS:")
            for e in errors:
                print(f"  ❌ {e}")
        if warnings:
            print("WARNINGS:")
            for w in warnings:
                print(f"  ⚠️  {w}")
        if not errors and not warnings:
            print("  ✅ All checks passed!")

        return len(errors) == 0

    # Run validation
    # validate_custom_dataset('data/my-ecommerce', 'my-ecommerce')

Next Steps
----------

- :doc:`quickstart` - Test your dataset with basic loading
- :doc:`data_exploration` - Analyze your custom dataset
- :doc:`training_workflow` - Train a model on your data
