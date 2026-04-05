"""
SYNAPSE v2: Temporal Attention Mechanisms

This module provides advanced temporal attention mechanisms that complement
the multi-timescale FLUID layer. These components enable the model to
attend differently to items based on their temporal characteristics.

Key Components:
- TemporalPositionalEncoding: Time-aware positional embeddings
- ContentTemporalAttention: Content-aware temporal attention
- TimescaleAwareAttention: Attention that respects learned timescales
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TemporalPositionalEncoding(nn.Module):
    """Continuous-time positional encoding.

    Unlike discrete positional encodings that use sequence position,
    this encoding uses actual timestamps to create time-aware embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_timescale: float = 1000.0,
        num_timescales: int = 64,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Learnable frequency components
        self.frequencies = nn.Parameter(
            torch.exp(
                torch.linspace(math.log(1.0), math.log(max_timescale), num_timescales)
            )
        )

        # Project to hidden dimension
        self.proj = nn.Linear(num_timescales * 2, hidden_dim)

    def forward(self, timestamps: Tensor) -> Tensor:
        """Compute temporal positional encoding.

        Args:
            timestamps: Absolute timestamps in hours, shape [batch, seq_len]

        Returns:
            Positional embeddings, shape [batch, seq_len, hidden_dim]
        """
        # Expand timestamps for frequency computation
        t = timestamps.unsqueeze(-1)  # [B, N, 1]
        freq = self.frequencies.view(1, 1, -1)  # [1, 1, F]

        # Sinusoidal encoding
        angles = t / freq  # [B, N, F]
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return self.proj(encoding)


class ContentTemporalAttention(nn.Module):
    """Attention mechanism that weights by content-time relevance.

    This attention layer learns to weight past interactions based on
    both their content similarity AND their temporal distance from
    the current time.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Temporal bias projection
        self.temporal_bias = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        time_delta: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute content-temporal attention.

        Args:
            query: Query embeddings, shape [batch, seq_len, hidden_dim]
            key: Key embeddings, shape [batch, seq_len, hidden_dim]
            value: Value embeddings, shape [batch, seq_len, hidden_dim]
            time_delta: Pairwise time differences, shape [batch, seq_len, seq_len]
            mask: Optional attention mask

        Returns:
            Attended values, shape [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = query.shape

        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [B, H, N, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, H, N, N]

        # Add temporal bias
        temporal_bias = self.temporal_bias(time_delta.unsqueeze(-1))  # [B, N, N, H]
        temporal_bias = temporal_bias.permute(0, 3, 1, 2)  # [B, H, N, N]
        attn_scores = attn_scores + temporal_bias

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B, H, N, D]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, N, H, D]
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(attn_output)


class TimescaleAwareAttention(nn.Module):
    """Attention that respects learned timescales.

    This layer uses the timescale predictions from the FLUID layer
    to modulate attention weights. Items with similar timescales
    attend more strongly to each other.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_timescales: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.content_attention = ContentTemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Timescale compatibility projection
        self.timescale_compat = nn.Sequential(
            nn.Linear(num_timescales * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        time_delta: Tensor,
        query_timescales: Tensor,
        key_timescales: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute timescale-aware attention.

        Args:
            query, key, value: Standard attention inputs
            time_delta: Pairwise time differences
            query_timescales: Timescale weights for queries, [B, N, K]
            key_timescales: Timescale weights for keys, [B, N, K]
            mask: Optional attention mask

        Returns:
            Attended values with timescale awareness
        """
        batch_size, seq_len, num_timescales = query_timescales.shape

        # Compute timescale compatibility
        # [B, N, 1, K] and [B, 1, N, K]
        q_ts = query_timescales.unsqueeze(2).expand(-1, -1, seq_len, -1)
        k_ts = key_timescales.unsqueeze(1).expand(-1, seq_len, -1, -1)

        # Concatenate for compatibility computation
        ts_pairs = torch.cat([q_ts, k_ts], dim=-1)  # [B, N, N, 2K]

        # Compute compatibility scores
        compat_scores = self.timescale_compat(ts_pairs).squeeze(-1)  # [B, N, N]

        # Modulate time_delta with compatibility
        modulated_time_delta = time_delta * torch.sigmoid(compat_scores)

        return self.content_attention(
            query,
            key,
            value,
            time_delta=modulated_time_delta,
            mask=mask,
        )


class TemporalTransformerBlock(nn.Module):
    """A transformer block with temporal awareness.

    Combines:
    - Temporal positional encoding
    - Content-temporal attention
    - Feed-forward network
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.temporal_encoding = TemporalPositionalEncoding(hidden_dim)
        self.attention = ContentTemporalAttention(hidden_dim, num_heads, dropout)

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        timestamps: Tensor,
        time_delta: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through temporal transformer block.

        Args:
            x: Input embeddings, shape [batch, seq_len, hidden_dim]
            timestamps: Absolute timestamps, shape [batch, seq_len]
            time_delta: Pairwise time differences, shape [batch, seq_len, seq_len]
            mask: Optional attention mask

        Returns:
            Transformed embeddings, shape [batch, seq_len, hidden_dim]
        """
        # Add temporal positional encoding
        x = x + self.temporal_encoding(timestamps)

        # Self-attention with temporal bias
        attn_out = self.attention(x, x, x, time_delta, mask)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


# Example usage
if __name__ == "__main__":
    batch_size, seq_len, hidden_dim = 4, 10, 256

    # Test temporal positional encoding
    tpe = TemporalPositionalEncoding(hidden_dim)
    timestamps = torch.rand(batch_size, seq_len) * 168  # 0-1 week in hours
    pos_emb = tpe(timestamps)
    print(f"Temporal positional encoding shape: {pos_emb.shape}")

    # Test content-temporal attention
    cta = ContentTemporalAttention(hidden_dim)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    time_delta = torch.rand(batch_size, seq_len, seq_len) * 48
    attn_out = cta(x, x, x, time_delta)
    print(f"Content-temporal attention output shape: {attn_out.shape}")

    # Test full transformer block
    block = TemporalTransformerBlock(hidden_dim)
    out = block(x, timestamps, time_delta)
    print(f"Transformer block output shape: {out.shape}")
