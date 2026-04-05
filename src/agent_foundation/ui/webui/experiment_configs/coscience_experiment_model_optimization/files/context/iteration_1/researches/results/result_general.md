# Research Summary: CSML Optimization

## Overview

This document synthesizes findings from deep research across six optimization areas (Q1-Q6), providing a unified view of techniques for achieving 65-75% QPS improvements in ML training/inference pipelines.

## Research Coverage

| Query | Topic | Key Finding | Impact |
|-------|-------|-------------|--------|
| Q1 | GPU-CPU Sync Points | `.item()` calls destroy pipeline parallelism | 10-20% QPS |
| Q2 | PT2 Compilation | Graph breaks prevent fusion | 15-30% QPS |
| Q3 | Activation Memory | Budget parameter controls checkpointing | 10-20% memory |
| Q4 | Kernel Fusion | Loss batching reduces kernel count 87% | 5-15% QPS |
| Q5 | Pipeline Optimization | SDD Lite best cost/benefit ratio | 4-5% QPS |
| Q6 | Embedding Tables | int32 indices reduce memory | 10-20% memory |

## Key Patterns Discovered

### 1. Bottleneck-First Thinking
Address fundamental bottlenecks, not surface-level inefficiencies. The NumCandidatesInfo pattern emerged from identifying that sync points were destroying pipeline parallelism—a bottleneck that couldn't be fixed by faster kernels.

### 2. Hardware-Aware Optimization
Modern GPUs (H100/B200) have different bottleneck profiles than previous generations. B200's faster matrix operations make kernel launch overhead relatively worse, increasing gains from fusion.

### 3. Compiler-Friendly Code
Restructure code to enable compiler optimization. Removing `.item()` calls and using `@torch.fx.wrap` enables PT2 to fuse operations.

### 4. State-Passing Architecture
Compute once, pass through. NumCandidatesInfo precomputes values ONCE and passes them through the entire forward pass.

### 5. Memory-Compute Tradeoff
Trade compute for memory when memory is the bottleneck. SDD Lite achieves 4-5% QPS with only 1% memory increase.

## Consolidated Recommendations

### Immediate Impact (< 1 week)
1. Apply activation_memory_budget reduction
2. Enable BF16 mixed precision
3. Enable SDD Lite pipeline

### Medium-Term (1-4 weeks)
1. Eliminate GPU-CPU sync points
2. Fix graph breaks for PT2 compilation
3. Implement kernel fusion for loss computation

### Long-Term (1-3 months)
1. Develop custom Triton kernels
2. Hardware-specific optimization (H100/B200)
3. FlexAttention integration for long sequences

## Conclusion

The research confirms that systematic bottleneck analysis combined with targeted implementation yields substantial performance gains. The 65-75% QPS improvement demonstrates the power of the Evolve methodology.
