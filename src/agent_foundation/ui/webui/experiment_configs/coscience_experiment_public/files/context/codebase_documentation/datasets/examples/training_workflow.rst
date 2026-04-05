Training Workflow: End-to-End Example
======================================

This example demonstrates the complete workflow from data loading to model training,
covering dataset preparation, model configuration, training loop, and evaluation.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The complete training workflow consists of:

1. Data loading and batching
2. Model initialization
3. Training loop with evaluation
4. Checkpoint saving and loading
5. Final evaluation and metrics

Example 1: Complete Training Pipeline
-------------------------------------

End-to-end training using the framework.

.. code-block:: python
   :caption: complete_training.py

    """
    Complete Training Pipeline

    End-to-end example showing data loading, model training,
    and evaluation for sequential recommendation.
    """

    # Cell 1: Imports and Configuration
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from pathlib import Path
    import numpy as np
    from typing import Dict, List, Tuple
    import time

    # Configuration
    CONFIG = {
        'dataset': 'ml-1m',
        'data_dir': 'data/ml-1m',
        'max_seq_len': 50,
        'embed_dim': 64,
        'num_heads': 2,
        'num_layers': 2,
        'dropout': 0.1,
        'batch_size': 256,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'eval_every': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print(f"Training on: {CONFIG['device']}")

    # Cell 2: Dataset Class
    class SequentialRecDataset(Dataset):
        """PyTorch Dataset for sequential recommendation."""

        def __init__(self, data_path: str, max_seq_len: int = 50):
            self.max_seq_len = max_seq_len
            self.sequences = []
            self.targets = []
            self.user_ids = []

            with open(data_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        user_id = int(parts[0])
                        items = [int(x) for x in parts[1:]]

                        # For training: use all prefixes as training samples
                        for i in range(1, len(items)):
                            self.sequences.append(items[:i])
                            self.targets.append(items[i])
                            self.user_ids.append(user_id)

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            seq = self.sequences[idx]
            target = self.targets[idx]

            # Truncate or pad
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
            seq_len = len(seq)
            seq = [0] * (self.max_seq_len - len(seq)) + seq  # Left pad

            return {
                'input_ids': torch.tensor(seq, dtype=torch.long),
                'target_id': torch.tensor(target, dtype=torch.long),
                'seq_len': torch.tensor(seq_len, dtype=torch.long),
            }


    class EvalDataset(Dataset):
        """Dataset for evaluation (single target per user)."""

        def __init__(self, train_path: str, eval_path: str, max_seq_len: int = 50):
            self.max_seq_len = max_seq_len
            self.sequences = []
            self.targets = []
            self.user_ids = []

            # Load training sequences (history)
            train_seqs = {}
            with open(train_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        user_id = int(parts[0])
                        items = [int(x) for x in parts[1:]]
                        train_seqs[user_id] = items

            # Load eval targets
            with open(eval_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        user_id = int(parts[0])
                        target = int(parts[1])
                        if user_id in train_seqs:
                            self.sequences.append(train_seqs[user_id])
                            self.targets.append(target)
                            self.user_ids.append(user_id)

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            seq = self.sequences[idx]
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
            seq_len = len(seq)
            seq = [0] * (self.max_seq_len - len(seq)) + seq

            return {
                'input_ids': torch.tensor(seq, dtype=torch.long),
                'target_id': torch.tensor(self.targets[idx], dtype=torch.long),
                'seq_len': torch.tensor(seq_len, dtype=torch.long),
            }

    # Cell 3: Simple Sequential Model
    class SimpleSeqRec(nn.Module):
        """Simple sequential recommendation model."""

        def __init__(
            self,
            num_items: int,
            embed_dim: int = 64,
            num_heads: int = 2,
            num_layers: int = 2,
            max_seq_len: int = 50,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.num_items = num_items
            self.embed_dim = embed_dim

            # Embeddings
            self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
            self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

            # Transformer layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Output projection
            self.output_layer = nn.Linear(embed_dim, num_items + 1)

            self._init_weights()

        def _init_weights(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """
            Args:
                input_ids: [batch_size, seq_len] - item ID sequence

            Returns:
                logits: [batch_size, num_items+1] - prediction scores
            """
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

            # Embeddings
            item_emb = self.item_embedding(input_ids)  # [B, S, D]
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            pos_emb = self.position_embedding(positions)  # [1, S, D]
            x = item_emb + pos_emb

            # Create attention mask (causal + padding)
            padding_mask = (input_ids == 0)  # [B, S]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()

            # Transformer
            x = self.transformer(
                x,
                mask=causal_mask,
                src_key_padding_mask=padding_mask
            )

            # Take last position output
            last_hidden = x[:, -1, :]  # [B, D]

            # Project to vocabulary
            logits = self.output_layer(last_hidden)  # [B, V]

            return logits

    # Cell 4: Training Functions
    def train_epoch(model, dataloader, optimizer, criterion, device):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['target_id'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches


    def evaluate(model, dataloader, device, k_values=[10, 20]):
        """Evaluate model with ranking metrics."""
        model.eval()
        metrics = {f'HR@{k}': 0.0 for k in k_values}
        metrics.update({f'NDCG@{k}': 0.0 for k in k_values})
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['target_id'].to(device)

                logits = model(input_ids)  # [B, V]

                # Get rankings
                _, indices = torch.topk(logits, max(k_values), dim=1)

                for i in range(targets.size(0)):
                    target = targets[i].item()
                    ranked = indices[i].cpu().numpy()

                    for k in k_values:
                        if target in ranked[:k]:
                            rank = np.where(ranked == target)[0][0] + 1
                            metrics[f'HR@{k}'] += 1
                            metrics[f'NDCG@{k}'] += 1.0 / np.log2(rank + 1)

                    num_samples += 1

        # Average
        for key in metrics:
            metrics[key] /= num_samples

        return metrics

    # Cell 5: Main Training Loop
    def main():
        """Main training function."""
        # Count items
        all_items = set()
        for split in ['train', 'valid', 'test']:
            with open(f"{CONFIG['data_dir']}/{CONFIG['dataset']}_{split}.txt") as f:
                for line in f:
                    parts = line.strip().split()
                    all_items.update(int(x) for x in parts[1:])
        num_items = max(all_items)
        print(f"Number of items: {num_items}")

        # Create datasets
        train_dataset = SequentialRecDataset(
            f"{CONFIG['data_dir']}/{CONFIG['dataset']}_train.txt",
            CONFIG['max_seq_len']
        )
        valid_dataset = EvalDataset(
            f"{CONFIG['data_dir']}/{CONFIG['dataset']}_train.txt",
            f"{CONFIG['data_dir']}/{CONFIG['dataset']}_valid.txt",
            CONFIG['max_seq_len']
        )
        test_dataset = EvalDataset(
            f"{CONFIG['data_dir']}/{CONFIG['dataset']}_train.txt",
            f"{CONFIG['data_dir']}/{CONFIG['dataset']}_test.txt",
            CONFIG['max_seq_len']
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=CONFIG['batch_size'],
            shuffle=True, num_workers=4
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=CONFIG['batch_size'],
            shuffle=False, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=CONFIG['batch_size'],
            shuffle=False, num_workers=4
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Valid samples: {len(valid_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

        # Create model
        model = SimpleSeqRec(
            num_items=num_items,
            embed_dim=CONFIG['embed_dim'],
            num_heads=CONFIG['num_heads'],
            num_layers=CONFIG['num_layers'],
            max_seq_len=CONFIG['max_seq_len'],
            dropout=CONFIG['dropout'],
        ).to(CONFIG['device'])

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_valid_ndcg = 0
        for epoch in range(1, CONFIG['num_epochs'] + 1):
            start_time = time.time()

            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])

            # Evaluate
            if epoch % CONFIG['eval_every'] == 0:
                valid_metrics = evaluate(model, valid_loader, CONFIG['device'])
                elapsed = time.time() - start_time

                print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                      f"Valid HR@10: {valid_metrics['HR@10']:.4f} | "
                      f"Valid NDCG@10: {valid_metrics['NDCG@10']:.4f} | "
                      f"Time: {elapsed:.1f}s")

                # Save best model
                if valid_metrics['NDCG@10'] > best_valid_ndcg:
                    best_valid_ndcg = valid_metrics['NDCG@10']
                    torch.save(model.state_dict(), 'best_model.pt')
                    print(f"  -> New best model saved!")

        # Final evaluation on test set
        model.load_state_dict(torch.load('best_model.pt'))
        test_metrics = evaluate(model, test_loader, CONFIG['device'])

        print("\n" + "="*50)
        print("Final Test Results:")
        print("="*50)
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Run training
    # if __name__ == '__main__':
    #     main()

Example 2: Using Framework Training Script
------------------------------------------

Use the built-in training script with gin configs.

.. code-block:: bash
   :caption: training_commands.sh

    # Train HSTU on MovieLens-1M
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin \
        --master_port=12345

    # Train HSTU on MovieLens-20M
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --gin_config_file=configs/ml-20m/hstu-sampled-softmax-n128-large-final.gin \
        --master_port=12346

    # Train SASRec baseline
    CUDA_VISIBLE_DEVICES=0 python3 main.py \
        --gin_config_file=configs/ml-1m/sasrec-sampled-softmax-n128-final.gin \
        --master_port=12347

Example 3: Custom Training Configuration
----------------------------------------

Modify gin configs for your needs.

.. code-block:: python
   :caption: custom_config.gin

    # Custom training configuration

    # Model architecture
    HSTUConfig.num_layers = 4
    HSTUConfig.embed_dim = 128
    HSTUConfig.num_heads = 4
    HSTUConfig.max_seq_len = 100
    HSTUConfig.dropout = 0.2

    # Training parameters
    TrainerConfig.batch_size = 512
    TrainerConfig.learning_rate = 0.0005
    TrainerConfig.num_epochs = 200
    TrainerConfig.eval_frequency = 5

    # Data
    DataConfig.dataset = 'my-custom-dataset'
    DataConfig.data_dir = 'data/my-custom-dataset'

    # Negative sampling
    NegativeSamplingConfig.num_negatives = 256
    NegativeSamplingConfig.strategy = 'popularity'

Training Tips and Best Practices
--------------------------------

1. **Start with Small Model**
   Begin with smaller embed_dim and fewer layers, then scale up.

2. **Monitor Validation Metrics**
   Always track validation NDCG@10 to detect overfitting early.

3. **Use Learning Rate Schedule**
   Consider warmup and decay for more stable training.

4. **Gradient Clipping**
   Add gradient clipping (max_norm=1.0) to prevent exploding gradients.

5. **Early Stopping**
   Stop training if validation metrics don't improve for 10+ epochs.

.. code-block:: python
   :caption: training_tips.py

    # Learning rate warmup
    def warmup_lr_scheduler(optimizer, warmup_steps, base_lr):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Early stopping
    class EarlyStopping:
        def __init__(self, patience=10):
            self.patience = patience
            self.counter = 0
            self.best_score = None

        def __call__(self, score):
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
            return self.counter >= self.patience

Next Steps
----------

- Review :doc:`../public_datasets` for benchmark datasets
- See :doc:`../preprocessing_pipeline` for data preparation details
- Check :doc:`../custom_datasets` for adding your own data
