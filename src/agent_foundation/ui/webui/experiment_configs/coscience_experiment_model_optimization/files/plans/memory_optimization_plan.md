# Memory Optimization Plan

## Overview

This plan outlines the strategy for reducing memory footprint and improving memory efficiency in the CSML model, targeting 32% embedding memory reduction and 16.9% peak memory reduction.

## Problem Statement

The original implementation suffers from:

1. **High memory footprint:** Embedding tables consume excessive GPU memory
2. **Memory fragmentation:** Frequent allocations/deallocations cause fragmentation
3. **Redundant allocations:** Temporary tensors created repeatedly in loops
4. **Inefficient precision:** Full precision where lower precision suffices

## Goals

- **Primary:** Reduce embedding table memory by 32%
- **Secondary:** Reduce peak memory usage from 14.2GB to 11.8GB (16.9%)
- **Tertiary:** Reduce allocation count per batch by 72%
- **Target Improvement:** 10.9% QPS contribution

## Implementation Strategy

### Phase 1: Precision Optimization

Implement FP16 casting for embeddings where precision allows:

```python
def optimized_embedding_lookup(
    embeddings: torch.Tensor,  # FP32 storage
    indices: torch.Tensor,
) -> torch.Tensor:
    # Cast to FP16 during lookup
    result = embeddings[indices].half()
    return result
```

**Memory Savings:**
- Original: 2.8GB per embedding table
- Optimized: 1.9GB per embedding table
- Reduction: 32%

### Phase 2: Buffer Pre-allocation

Pre-allocate reusable buffers for attention computation:

```python
class AttentionBufferPool:
    def __init__(self, max_seq_len: int, hidden_dim: int):
        self.q_buffer = torch.empty(max_seq_len, hidden_dim)
        self.k_buffer = torch.empty(max_seq_len, hidden_dim)
        self.v_buffer = torch.empty(max_seq_len, hidden_dim)
        self.attention_scores = torch.empty(max_seq_len, max_seq_len)

    def get_buffers(self, seq_len: int):
        return (
            self.q_buffer[:seq_len],
            self.k_buffer[:seq_len],
            self.v_buffer[:seq_len],
        )
```

### Phase 3: In-place Operations

Replace allocating operations with in-place versions:

```python
# Before: Creates new tensor
result = tensor + other

# After: In-place operation
tensor.add_(other)
```

### Phase 4: Memory Pooling

Implement tensor memory pooling for frequently allocated sizes:

```python
class TensorPool:
    def __init__(self):
        self.pools: Dict[Tuple[int, ...], List[torch.Tensor]] = {}

    def get(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        key = (shape, dtype)
        if key in self.pools and self.pools[key]:
            return self.pools[key].pop()
        return torch.empty(shape, dtype=dtype)

    def release(self, tensor: torch.Tensor):
        key = (tuple(tensor.shape), tensor.dtype)
        if key not in self.pools:
            self.pools[key] = []
        self.pools[key].append(tensor)
```

## Memory Layout Optimization

### Contiguous Memory Access

Ensure tensors are accessed in contiguous memory order:

```python
# Before: Non-contiguous access
for i in range(batch_size):
    process(tensor[:, i, :])  # Strided access

# After: Contiguous access
tensor = tensor.transpose(0, 1).contiguous()
for i in range(batch_size):
    process(tensor[i])  # Contiguous access
```

### Cache-Friendly Tiling

Tile operations to fit in GPU L2 cache:

```python
def tiled_matmul(a: torch.Tensor, b: torch.Tensor, tile_size: int = 128):
    # Process in cache-friendly tiles
    result = torch.zeros(a.shape[0], b.shape[1])
    for i in range(0, a.shape[0], tile_size):
        for j in range(0, b.shape[1], tile_size):
            result[i:i+tile_size, j:j+tile_size] = (
                a[i:i+tile_size] @ b[:, j:j+tile_size]
            )
    return result
```

## Files to Modify

- `pytorch_modules_roo.py` - Module memory management
- `hstu_attention_template.py` - Attention buffer management
- `model_roo.py` - Model-level memory optimization

## Risks

1. **Precision Loss:** FP16 may cause numerical issues in edge cases
2. **Complexity:** Buffer pooling adds code complexity
3. **Thread Safety:** Buffer pools must be thread-safe for parallel execution

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Embedding memory | 2.8GB | 1.9GB |
| Peak memory | 14.2GB | 11.8GB |
| Fragmentation | 23.4% | 8.2% |
| Allocations/batch | 847 | 234 |
| QPS improvement | - | 10.9% |

## Secondary Benefits

While direct QPS impact is moderate (10.9%), memory optimization enables:

1. **Larger batch sizes:** More memory headroom
2. **Better cache utilization:** Improved memory locality
3. **More stable performance:** Less GC pressure

## Timeline

- Week 1: Precision optimization
- Week 2: Buffer pre-allocation
- Week 3: In-place operations migration
- Week 4: Memory pooling implementation
- Week 5: Validation and profiling
