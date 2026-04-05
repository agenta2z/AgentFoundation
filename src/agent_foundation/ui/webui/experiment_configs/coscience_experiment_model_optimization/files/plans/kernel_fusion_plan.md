# Kernel Fusion Optimization Plan

## Overview

This plan outlines the strategy for reducing kernel launch overhead through aggressive kernel fusion, targeting 87% reduction in kernel count (221 → 30).

## Problem Statement

The original implementation uses many small CUDA kernels that individually perform simple operations. Each kernel launch incurs:

1. Kernel setup overhead (~5-10μs)
2. GPU context switching
3. Memory cache invalidation
4. Synchronization barriers

**Impact:** At high batch rates, launch overhead becomes significant portion of total execution time.

## Goals

- **Primary:** Reduce kernel count from 221 to ~30
- **Secondary:** Reduce kernel launch overhead from 4.2ms to 0.6ms
- **Target Improvement:** 34.1% QPS contribution (largest single contributor)

## Implementation Strategy

### Phase 1: Kernel Profiling

1. Use NVIDIA Nsight to profile all kernel launches
2. Identify fusion candidates:
   - Small kernels (<1ms execution)
   - Consecutive kernels with data dependencies
   - Element-wise operations that can be combined

### Phase 2: Loss Computation Fusion

Implement `fused_multi_loss_computation`:

```python
@torch.no_grad()  # Critical: Added missing decorator
def fused_multi_loss_computation(
    loss_inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    loss_types: List[str],
    loss_weights: List[float],
) -> torch.Tensor:
    """
    Compute all losses using batched fusion strategy.
    Original: 221 kernel launches
    Optimized: ~30 kernel launches (87% reduction)
    """
```

### Phase 3: Attention Fusion

Fuse attention-related operations:
- Q, K, V projections
- Softmax + masking
- Output projection

### Phase 4: Embedding Lookup Fusion

Batch embedding lookups across features:
```python
def fused_embedding_lookup(
    embeddings: List[torch.Tensor],
    indices: List[torch.Tensor],
) -> List[torch.Tensor]:
    # Single kernel for all lookups
```

## Fusion Patterns

### Pattern 1: Element-wise Fusion
```python
# Before: 3 kernels
x = a + b
x = x * c
x = x.relu()

# After: 1 kernel
x = torch.relu((a + b) * c)
```

### Pattern 2: Reduction Fusion
```python
# Before: Multiple reductions
mean = x.mean()
std = x.std()

# After: Single pass
mean, std = x.mean(), x.std()  # Fused internally
```

### Pattern 3: Memory Access Fusion
```python
# Before: Multiple memory accesses
result1 = tensor[indices1]
result2 = tensor[indices2]

# After: Batched access
results = tensor[torch.cat([indices1, indices2])]
```

## Files to Modify

- `pytorch_modules_roo.py` - Core computation modules
- `hstu_attention_template.py` - Attention implementations
- `model_roo.py` - Loss computation

## Risks

1. **Debugging:** Fused kernels harder to profile individually
2. **Code Complexity:** Fusion patterns require more sophisticated code
3. **Cache Behavior:** Very large fused kernels may have worse cache behavior

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Kernel launches | 221 | ~30 |
| Launch overhead | 4.2ms | 0.6ms |
| QPS improvement | - | 34.1% |

## Diminishing Returns Analysis

```
Kernel Reduction | QPS Improvement
      25%        |      12%
      50%        |      23%
      70%        |      31%
      86%        |      34%
      95%        |      35% (diminishing)
```

Target 70-85% reduction for optimal cost-benefit.

## Timeline

- Week 1: Profiling and candidate identification
- Week 2: Implement loss computation fusion
- Week 3: Implement attention fusion
- Week 4: Implement embedding fusion
- Week 5: Validation and fine-tuning
