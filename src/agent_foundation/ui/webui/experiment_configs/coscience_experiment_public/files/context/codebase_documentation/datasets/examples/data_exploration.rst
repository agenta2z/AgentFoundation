Data Exploration and Analysis
=============================

This example demonstrates how to explore and analyze recommendation datasets,
visualize distributions, and understand data characteristics before model training.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

Understanding your data is critical for:

1. Choosing appropriate model hyperparameters (sequence length, embedding size)
2. Identifying data quality issues (sparse users, missing timestamps)
3. Understanding user behavior patterns
4. Setting realistic performance expectations

Example 1: Dataset Statistics Analysis
--------------------------------------

Comprehensive statistics for understanding data characteristics.

.. code-block:: python
   :caption: dataset_statistics.py

    """
    Dataset Statistics Analysis

    This notebook analyzes key statistics of recommendation datasets
    to inform model architecture and hyperparameter choices.
    """

    # Cell 1: Imports
    from pathlib import Path
    from collections import Counter
    import numpy as np

    # Cell 2: Load Data
    def load_sasrec_data(filepath):
        """Load SASRec format data."""
        user_sequences = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    item_ids = [int(x) for x in parts[1:]]
                    user_sequences[user_id] = item_ids
        return user_sequences

    # Load all datasets
    datasets = {
        'ml-1m': load_sasrec_data('data/ml-1m/ml-1m_train.txt'),
        'ml-20m': load_sasrec_data('data/ml-20m/ml-20m_train.txt'),
        # 'amzn-books': load_sasrec_data('data/amzn-books/amzn-books_train.txt'),
    }

    # Cell 3: Basic Statistics Function
    def compute_dataset_stats(name, data):
        """Compute comprehensive statistics for a dataset."""
        seq_lengths = [len(seq) for seq in data.values()]
        all_items = set()
        item_counts = Counter()

        for seq in data.values():
            all_items.update(seq)
            item_counts.update(seq)

        stats = {
            'name': name,
            'num_users': len(data),
            'num_items': len(all_items),
            'total_interactions': sum(seq_lengths),
            'avg_seq_length': np.mean(seq_lengths),
            'median_seq_length': np.median(seq_lengths),
            'std_seq_length': np.std(seq_lengths),
            'min_seq_length': min(seq_lengths),
            'max_seq_length': max(seq_lengths),
            'p95_seq_length': np.percentile(seq_lengths, 95),
            'p99_seq_length': np.percentile(seq_lengths, 99),
            'density': sum(seq_lengths) / (len(data) * len(all_items)) * 100,
            'most_popular_items': item_counts.most_common(10),
            'least_popular_items': item_counts.most_common()[-10:],
        }
        return stats

    # Cell 4: Compute Stats for All Datasets
    all_stats = {}
    for name, data in datasets.items():
        stats = compute_dataset_stats(name, data)
        all_stats[name] = stats
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")
        print(f"Users:           {stats['num_users']:,}")
        print(f"Items:           {stats['num_items']:,}")
        print(f"Interactions:    {stats['total_interactions']:,}")
        print(f"Density:         {stats['density']:.4f}%")
        print(f"\nSequence Length Statistics:")
        print(f"  Mean:          {stats['avg_seq_length']:.2f}")
        print(f"  Median:        {stats['median_seq_length']:.2f}")
        print(f"  Std:           {stats['std_seq_length']:.2f}")
        print(f"  Min:           {stats['min_seq_length']}")
        print(f"  Max:           {stats['max_seq_length']}")
        print(f"  95th %ile:     {stats['p95_seq_length']:.2f}")
        print(f"  99th %ile:     {stats['p99_seq_length']:.2f}")

    # Cell 5: Hyperparameter Recommendations
    print("\n" + "="*60)
    print("HYPERPARAMETER RECOMMENDATIONS")
    print("="*60)
    for name, stats in all_stats.items():
        print(f"\n{name}:")
        # Recommended max_seq_len is around 95th percentile
        recommended_seq_len = int(stats['p95_seq_length'])
        print(f"  max_seq_len: {recommended_seq_len} (covers 95% of users)")

        # Embedding dimension based on item count
        if stats['num_items'] < 5000:
            embed_dim = 64
        elif stats['num_items'] < 50000:
            embed_dim = 128
        else:
            embed_dim = 256
        print(f"  embed_dim: {embed_dim} (based on {stats['num_items']} items)")

Example 2: Distribution Visualization
-------------------------------------

Visualize data distributions using matplotlib (can be run in Jupyter).

.. code-block:: python
   :caption: distribution_plots.py

    """
    Distribution Visualization

    Create visualizations of sequence length distributions,
    item popularity, and user activity patterns.
    """

    # Cell 1: Imports
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import Counter

    # Assume data is loaded from previous example
    # data = load_sasrec_data('data/ml-1m/ml-1m_train.txt')

    # Cell 2: Sequence Length Distribution
    def plot_sequence_length_distribution(data, dataset_name):
        """Plot histogram of sequence lengths."""
        seq_lengths = [len(seq) for seq in data.values()]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(seq_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Sequence Length')
        axes[0].set_ylabel('Number of Users')
        axes[0].set_title(f'{dataset_name}: Sequence Length Distribution')
        axes[0].axvline(np.mean(seq_lengths), color='r', linestyle='--',
                        label=f'Mean: {np.mean(seq_lengths):.1f}')
        axes[0].axvline(np.median(seq_lengths), color='g', linestyle='--',
                        label=f'Median: {np.median(seq_lengths):.1f}')
        axes[0].legend()

        # Log-scale histogram (better for long-tail)
        axes[1].hist(seq_lengths, bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Sequence Length')
        axes[1].set_ylabel('Number of Users (log scale)')
        axes[1].set_title(f'{dataset_name}: Sequence Length (Log Scale)')
        axes[1].set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{dataset_name}_seq_length_dist.png', dpi=150)
        plt.show()

    # plot_sequence_length_distribution(data, 'ml-1m')

    # Cell 3: Item Popularity Distribution
    def plot_item_popularity(data, dataset_name, top_n=100):
        """Plot item popularity distribution (power law)."""
        item_counts = Counter()
        for seq in data.values():
            item_counts.update(seq)

        # Sort by popularity
        sorted_counts = sorted(item_counts.values(), reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Top N items
        axes[0].bar(range(top_n), sorted_counts[:top_n], alpha=0.7)
        axes[0].set_xlabel('Item Rank')
        axes[0].set_ylabel('Interaction Count')
        axes[0].set_title(f'{dataset_name}: Top {top_n} Most Popular Items')

        # Full distribution (log-log for power law)
        axes[1].loglog(range(1, len(sorted_counts) + 1), sorted_counts, 'b-', alpha=0.7)
        axes[1].set_xlabel('Item Rank (log)')
        axes[1].set_ylabel('Interaction Count (log)')
        axes[1].set_title(f'{dataset_name}: Item Popularity (Power Law)')

        plt.tight_layout()
        plt.savefig(f'{dataset_name}_item_popularity.png', dpi=150)
        plt.show()

    # plot_item_popularity(data, 'ml-1m')

    # Cell 4: User Activity Over Item Coverage
    def plot_user_item_coverage(data, dataset_name):
        """Analyze how much of item space users cover."""
        all_items = set()
        for seq in data.values():
            all_items.update(seq)

        total_items = len(all_items)
        coverage_per_user = [len(set(seq)) / total_items * 100 for seq in data.values()]

        plt.figure(figsize=(10, 5))
        plt.hist(coverage_per_user, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Item Coverage (%)')
        plt.ylabel('Number of Users')
        plt.title(f'{dataset_name}: Per-User Item Coverage')
        plt.axvline(np.mean(coverage_per_user), color='r', linestyle='--',
                    label=f'Mean: {np.mean(coverage_per_user):.2f}%')
        plt.legend()
        plt.savefig(f'{dataset_name}_user_coverage.png', dpi=150)
        plt.show()

    # plot_user_item_coverage(data, 'ml-1m')

Example 3: Data Quality Checks
------------------------------

Identify potential data quality issues before training.

.. code-block:: python
   :caption: data_quality.py

    """
    Data Quality Checks

    Identify issues that could affect model performance.
    """

    # Cell 1: Quality Check Functions
    def check_data_quality(data, dataset_name):
        """Run comprehensive data quality checks."""
        issues = []

        # Check 1: Users with very short sequences
        short_users = sum(1 for seq in data.values() if len(seq) < 3)
        if short_users > 0:
            pct = short_users / len(data) * 100
            issues.append(f"WARNING: {short_users} users ({pct:.2f}%) have < 3 items")

        # Check 2: Users with single item (can't train)
        single_item_users = sum(1 for seq in data.values() if len(seq) == 1)
        if single_item_users > 0:
            issues.append(f"ERROR: {single_item_users} users have only 1 item")

        # Check 3: Duplicate items in sequences
        dup_users = 0
        for seq in data.values():
            if len(seq) != len(set(seq)):
                dup_users += 1
        if dup_users > 0:
            pct = dup_users / len(data) * 100
            issues.append(f"INFO: {dup_users} users ({pct:.2f}%) have duplicate items")

        # Check 4: Very long sequences (may need truncation)
        long_users = sum(1 for seq in data.values() if len(seq) > 500)
        if long_users > 0:
            pct = long_users / len(data) * 100
            issues.append(f"INFO: {long_users} users ({pct:.2f}%) have > 500 items")

        # Check 5: Cold items (appear only once)
        from collections import Counter
        item_counts = Counter()
        for seq in data.values():
            item_counts.update(seq)
        cold_items = sum(1 for count in item_counts.values() if count == 1)
        total_items = len(item_counts)
        if cold_items > total_items * 0.3:
            pct = cold_items / total_items * 100
            issues.append(f"WARNING: {cold_items} items ({pct:.2f}%) appear only once")

        # Print results
        print(f"\nData Quality Report: {dataset_name}")
        print("=" * 50)
        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  All checks passed!")

        return issues

    # Run checks
    # check_data_quality(data, 'ml-1m')

    # Cell 2: ID Consistency Checks
    def check_id_consistency(train_path, valid_path, test_path):
        """Verify train/valid/test ID consistency."""
        def load_ids(path):
            user_ids = set()
            item_ids = set()
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        user_ids.add(int(parts[0]))
                        item_ids.update(int(x) for x in parts[1:])
            return user_ids, item_ids

        train_users, train_items = load_ids(train_path)
        valid_users, valid_items = load_ids(valid_path)
        test_users, test_items = load_ids(test_path)

        print("ID Consistency Check:")
        print(f"  Train users: {len(train_users)}")
        print(f"  Valid users: {len(valid_users)}")
        print(f"  Test users:  {len(test_users)}")

        # Check all valid/test users are in train
        valid_not_in_train = valid_users - train_users
        test_not_in_train = test_users - train_users
        if valid_not_in_train:
            print(f"  ERROR: {len(valid_not_in_train)} valid users not in train")
        if test_not_in_train:
            print(f"  ERROR: {len(test_not_in_train)} test users not in train")

        # Check for new items in valid/test
        new_valid_items = valid_items - train_items
        new_test_items = test_items - train_items
        if new_valid_items:
            print(f"  WARNING: {len(new_valid_items)} new items in valid set")
        if new_test_items:
            print(f"  WARNING: {len(new_test_items)} new items in test set")

    # check_id_consistency(
    #     'data/ml-1m/ml-1m_train.txt',
    #     'data/ml-1m/ml-1m_valid.txt',
    #     'data/ml-1m/ml-1m_test.txt'
    # )

Complete Notebook: Copy-Paste Ready
-----------------------------------

**Cell 1: Setup**

.. code-block:: python

    from pathlib import Path
    from collections import Counter
    import numpy as np

    def load_sasrec_data(filepath):
        user_sequences = {}
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    item_ids = [int(x) for x in parts[1:]]
                    user_sequences[user_id] = item_ids
        return user_sequences

**Cell 2: Load and Analyze**

.. code-block:: python

    data = load_sasrec_data('data/ml-1m/ml-1m_train.txt')
    seq_lengths = [len(seq) for seq in data.values()]

    print(f"Users: {len(data):,}")
    print(f"Avg Seq Length: {np.mean(seq_lengths):.2f}")
    print(f"95th Percentile: {np.percentile(seq_lengths, 95):.0f}")

**Cell 3: Visualize (Optional)**

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.hist(seq_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('ML-1M: Sequence Length Distribution')
    plt.show()

Key Insights for Model Design
-----------------------------

From the analysis above, you can derive:

+------------------------+------------------------------------------+
| Observation            | Model Decision                           |
+========================+==========================================+
| 95th %ile seq length   | Set ``max_seq_len`` parameter            |
+------------------------+------------------------------------------+
| Power-law item dist    | Use popularity-based negative sampling   |
+------------------------+------------------------------------------+
| High sparsity          | Consider more regularization             |
+------------------------+------------------------------------------+
| Many cold items        | Use item embedding initialization        |
+------------------------+------------------------------------------+
| Long-tail users        | May benefit from data augmentation       |
+------------------------+------------------------------------------+

Next Steps
----------

- :doc:`quickstart` - Basic data loading
- :doc:`custom_integration` - Add your own datasets
- :doc:`training_workflow` - Start training with analyzed data
