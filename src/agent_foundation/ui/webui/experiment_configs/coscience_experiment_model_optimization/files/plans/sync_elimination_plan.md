# Sync Point Elimination Plan

## Overview

This plan outlines the strategy for eliminating GPU-CPU synchronization points in the CSML model inference path, targeting 83% reduction in sync overhead.

## Problem Statement

The original codebase contains numerous hidden synchronization points through `.item()` calls embedded in conditional logic and loops. Each sync point forces:

1. GPU to complete all queued operations
2. Data transfer from GPU to CPU
3. CPU to wait for transfer completion

**Cost:** 5-50μs per sync depending on GPU load

## Goals

- **Primary:** Reduce sync points from 15-20 per batch to 2-3
- **Secondary:** Reduce sync overhead from 2.3ms to 0.4ms per batch
- **Target Improvement:** 29.5% QPS contribution

## Implementation Strategy

### Phase 1: Audit Sync Points

1. Identify all `.item()` calls in hot paths
2. Categorize by necessity:
   - Essential (required for control flow)
   - Deferrable (can be batched)
   - Eliminable (can use tensor operations)

### Phase 2: NumCandidatesInfo Pattern

Create a dataclass to pre-compute all sync-requiring values once:

```python
@dataclass
class NumCandidatesInfo:
    num_nro_candidates: torch.Tensor
    num_ro_candidates: torch.Tensor
    max_nro_candidates: int  # Single sync here
    max_ro_candidates: int    # Single sync here
    ro_lengths: torch.Tensor
    nro_lengths: torch.Tensor
```

### Phase 3: Refactor Hot Paths

Replace patterns like:
```python
# Before
for i in range(tensor.shape[0].item()):
    if counts[i].item() > 0:
        process(data[i])
```

With:
```python
# After
info = NumCandidatesInfo(...)
for i in range(info.max_candidates):
    # No more syncs in inner loop
```

### Phase 4: Validation

1. Profile sync points before and after
2. Measure QPS improvement
3. Verify correctness with unit tests

## Files to Modify

- `model_roo.py` - Main model forward pass
- `pytorch_modules_roo.py` - Module implementations
- `hstu_attention_template.py` - Attention computation

## Risks

1. **Correctness:** Pre-computed values must match dynamic computation
2. **Memory:** Additional tensors for pre-computed data
3. **Complexity:** Pattern requires consistent application

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Sync points/batch | 15-20 | 2-3 |
| Sync overhead | 2.3ms | 0.4ms |
| QPS improvement | - | 29.5% |

## Timeline

- Week 1: Audit and categorization
- Week 2: Implement NumCandidatesInfo pattern
- Week 3: Refactor hot paths
- Week 4: Validation and optimization
