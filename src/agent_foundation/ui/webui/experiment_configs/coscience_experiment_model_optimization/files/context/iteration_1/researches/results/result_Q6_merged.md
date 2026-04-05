# Embedding Table Optimization and Kernel-Level Profiling for Large-Scale Recommendation Training

**Executive Summary**: Large-scale recommendation model training with **100M-1B+ embedding entries** faces a fundamental memory-compute tradeoff: embedding tables can consume **50-500GB** of GPU memory, yet only a fraction of entries are accessed per batch. Modern solutions combine aggressive compression, intelligent caching, and fused operators to achieve training within tight memory budgets. Simultaneously, validating kernel-level optimizations requires moving beyond simple end-to-end latency measurements to granular profiling that distinguishes between **launch-bound latency (CPU overhead)** and **compute-bound latency (GPU execution)**, while verifying compiler decisions like SDPA backend selection. For MTML ROO architectures with HSTU and **5% activation memory constraints**, the most impactful optimizations are FBGEMM's fused backward-optimizer (eliminating gradient materialization), software-managed UVM caching (enabling **1-5% GPU memory** utilization), compositional embeddings (achieving **10-1000× compression**), and rigorous kernel profiling to validate that optimizations like `torch.compile` fusion actually reduce kernel counts from **221 → 30 kernels**.

---

## Table of Contents

### Part I: Embedding Table Optimization
1. [Memory Profile of Large-Scale Embeddings](#1-memory-profile-of-large-scale-embeddings)
2. [Compression Techniques During Training](#2-compression-techniques-during-training)
3. [FBGEMM Operators for Memory-Efficient Training](#3-fbgemm-operators-for-memory-efficient-training)
4. [Caching Strategies Beyond GPU Memory Limits](#4-caching-strategies-beyond-gpu-memory-limits)
5. [int32 Optimization for Sparse Tensors](#5-int32-optimization-for-sparse-tensors)
6. [Industry Practices: Hybrid Parallelism and Real-Time Updates](#6-industry-practices-hybrid-parallelism-and-real-time-updates)

### Part II: Kernel-Level Profiling and Benchmarking
7. [Theoretical Foundation for Kernel-Level Visibility](#7-theoretical-foundation-for-kernel-level-visibility)
8. [Counting Kernel Launches and Distribution Analysis](#8-counting-kernel-launches-and-distribution-analysis)
9. [Separating Launch Overhead from Execution Time](#9-separating-launch-overhead-from-execution-time)
10. [SDPA Backend Detection and Verification](#10-sdpa-backend-detection-and-verification)
11. [A/B Benchmarking Infrastructure](#11-ab-benchmarking-infrastructure)
12. [Per-Component Kernel Attribution](#12-per-component-kernel-attribution)
13. [Chrome Trace Export and Visual Analysis](#13-chrome-trace-export-and-visual-analysis)
14. [Integration with NVIDIA Nsight](#14-integration-with-nvidia-nsight)
15. [Complete Production Benchmark Template](#15-complete-production-benchmark-template)

+ [Appendix: Research Questions Quick Answers](#appendix-research-questions-quick-answers)

---

# Part I: Embedding Table Optimization

## 1. Memory Profile of Large-Scale Embeddings

For a single embedding table with **100M entries and 128-dimension** in FP32, the parameter footprint alone reaches **51.2GB**—exceeding most GPU HBM capacities. Training memory follows this formula:

$$\text{Total Memory} = \text{Parameters} + \text{Optimizer States} + \text{Gradients} + \text{Activations}$$

With Adam optimizer in mixed precision, per-parameter costs break down to:
- **6 bytes** for parameters (FP16 + FP32 master)
- **8 bytes** for optimizer states
- **4 bytes** for gradients
- **Total: 18 bytes per parameter**

However, FBGEMM's `SplitTableBatchedEmbeddingBagsCodegen` with fused optimizers eliminates gradient materialization entirely, saving memory equal to parameter size.

### 1.1 Embedding Scale Memory Table

| Embedding Scale | Dimension | FP32 Memory | FP16 Memory |
|-----------------|-----------|-------------|-------------|
| 100M entries | 64 | 25.6 GB | 12.8 GB |
| 100M entries | 128 | 51.2 GB | 25.6 GB |
| 1B entries | 64 | 256 GB | 128 GB |
| 1B entries | 128 | 512 GB | 256 GB |

### 1.2 HSTU Activation Memory Constraints

For HSTU architectures with **512-1024 sequence lengths**, activation memory per attention layer reaches approximately:

$$\text{Activation Memory} = B \times L \times D \times 2\ \text{bytes}$$

A configuration with batch 512, sequence 1024, and dimension 256 consumes ~537MB per layer. Within a **5% activation budget** on an 80GB GPU (~4GB), this constrains depth to 4-6 layers without gradient checkpointing.

### 1.3 KeyedJaggedTensor Memory

The **KeyedJaggedTensor** format efficiently represents variable-length sparse features with flattened values and lengths tensors:

$$\text{KJT\_mem} = \text{values\_count} \times \text{dtype\_size} + \text{lengths\_count} \times \text{offset\_dtype\_size}$$

For training bottleneck analysis, embedding lookups are **memory-bound, not compute-bound**, with all-to-all communication becoming co-dominant at scale.

---

## 2. Compression Techniques During Training

Multiple compression approaches work during training while maintaining model quality.

### 2.1 Mixed-Dimension Embeddings (MDE)

Assign smaller dimensions to rare features, achieving **2-16× compression** with maintained or improved accuracy due to regularization effects on long-tail features. Implementation requires sorting features by popularity and partitioning into blocks with decreasing dimensions.

### 2.2 Quotient-Remainder (QR) Embeddings

Decompose feature ID `i` into quotient `q = i // M` and remainder `r = i % M`, looking up from two smaller tables and combining via element-wise multiplication. This achieves **10-15× compression** for 1B features with negligible accuracy loss.

```python
# QR decomposition: e_i = E_q[i // M] ⊙ E_r[i % M]
# Original: O(N × D) → QR: O(2√N × D)
```

The approach is fully differentiable and requires minimal code changes.

### 2.3 TT-Rec (Tensor Train Decomposition)

Replaces embedding matrices with sequences of smaller tensor cores, achieving **112× compression on Criteo Terabyte with no accuracy loss**. Meta's FBTT-Embedding library provides drop-in PyTorch replacement with LFU caching for frequently-accessed vectors.

### 2.4 ROBE (Random Offset Block Embedding)

Uses a single shared parameter array with block-based universal hashing, achieving **1000× compression** (100GB → 100MB) while meeting MLPerf target AUC of 0.8025. Block-wise access improves cache performance and reduces hash variance compared to standard feature hashing.

### 2.5 Quantization-Aware Training (QAT)

INT4 provides **8× reduction** while the quantization acts as **strong regularization** that can actually improve accuracy by mitigating DLRM overfitting. FBGEMM supports FP16, INT8, INT4, and INT2 weight precisions.

### 2.6 Compression Techniques Comparison

| Technique | Compression | Quality Impact | Training-Compatible |
|-----------|-------------|----------------|-------------------|
| Mixed-Dimension (MDE) | 2-16× | Maintains/improves | ✓ |
| QR Embeddings | 10-15× | Negligible | ✓ |
| TT-Rec | 100-200× | No loss | ✓ |
| ROBE | 1000× | No loss | ✓ |
| QAT INT4 | 8× | Regularization benefit | ✓ |

---

## 3. FBGEMM Operators for Memory-Efficient Training

### 3.1 Core FBGEMM Operator

The core FBGEMM operator `SplitTableBatchedEmbeddingBagsCodegen` provides batched embedding lookups with integrated optimizer updates:

```python
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation

tbe = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=[
        (100_000_000, 128, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA)
    ],
    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
    cache_load_factor=0.2,  # 20% cached in HBM
    prefetch_pipeline=True,
)
```

Key parameters:
- **EmbeddingLocation**: `DEVICE` for HBM, `MANAGED` for UVM, `MANAGED_CACHING` for UVM with HBM cache
- **Fused optimizers**: `EXACT_ROWWISE_ADAGRAD`, `ADAM`, `LAMB`

### 3.2 TorchRec High-Level APIs

TorchRec wraps FBGEMM with high-level APIs:
- **EmbeddingBagCollection**: For pooled embeddings
- **EmbeddingCollection**: For sequence embeddings (critical for HSTU attention)
- **FusedEBC**: Combines table batching, fused optimizer, and UVM caching—achieving **>2× performance** over standard PyTorch EmbeddingBag

For distributed training, **DistributedModelParallel** handles sharding automatically via the **EmbeddingShardingPlanner**.

### 3.3 GPU Kernel Optimizations

- **Table batching**: Reduces kernel launch overhead
- **Coalesced memory reads**: Embedding dimensions aligned to 4
- **Stochastic rounding**: For FP16 stability
- **Fused backward-optimizer**: Eliminates gradient tensor allocation entirely—gradients are applied directly during backward propagation

---

## 4. Caching Strategies Beyond GPU Memory Limits

### 4.1 UVM Caching

For embeddings exceeding GPU memory, **UVM caching** places the full table in unified virtual memory with hot rows cached in HBM. FBGEMM's `MANAGED_CACHING` location with `cache_load_factor=0.2` keeps 20% of rows in fast memory.

Cache policies:
- **LRU**: Adaptive to changing patterns
- **LFU**: Better for stable recommendation access patterns

### 4.2 Software-Managed Caching

**Software-managed caching** outperforms hardware UVM paging. ColossalAI's frequency-aware cache achieves training with **only 1.5-5% of embeddings in GPU memory**—for a 91GB table, only 3.75GB CUDA memory is required. Implementation pre-analyzes dataset for frequency distribution and warms the cache with highest-frequency IDs before training.

### 4.3 ScratchPipe's "Look-Forward" Cache

**ScratchPipe's "look-forward" cache** knows exactly which embeddings will be accessed in upcoming batches, achieving ~100% cache hit rate. This requires only cache size equal to the working set of the current batch. Performance reaches **2.8× average speedup** (up to 4.2×) versus prior GPU embedding systems.

### 4.4 Prefetch Pipelining

**Prefetch pipelining** overlaps cache operations with compute: cache-insert for batch_{i+1} executes in parallel with forward/backward of batch_i. Critical implementation details:
- Preventing immature eviction via `lxu_cache_locking_counter`
- Handling cache invalidation before backward pass for correct gradient writes

### 4.5 Recommended Architecture for 5% Activation Memory Budget

- **3-4%** for embedding cache (software-managed LFU)
- **1%** for prefetch buffer (3-5 batch lookahead)
- Remaining for forward activations with gradient checkpointing
- Use fused optimizers to eliminate gradient memory entirely

---

## 5. int32 Optimization for Sparse Tensors

### 5.1 50% Memory Savings

Switching from int64 to int32 for embedding indices and offsets provides **50% memory reduction** on sparse tensor components. FBGEMM v1.1.0 explicitly introduced int32 support for TBE training, and TorchRec defaults to `lengths_dtype=torch.int32`.

### 5.2 Safety Thresholds

int32 is safe when:
- Embedding table cardinality is under **2.1 billion**
- Cumulative offsets (batch_size × num_features × max_sequence_length) remain under int32 max

For production recommendation systems, feature IDs are typically hashed/modulated to fit within table sizes, making int32 universally applicable.

### 5.3 Implementation Pattern

```python
# Optimized KeyedJaggedTensor construction
sparse_features = KeyedJaggedTensor(
    keys=["user_id", "item_id", "category"],
    values=torch.tensor([...], dtype=torch.int32),    # 50% savings
    lengths=torch.tensor([...], dtype=torch.int32),   # Default
    weights=torch.tensor([...], dtype=torch.float16), # 50% savings on scores
)

# FBGEMM TBE configuration
tbe = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs,
    embedding_table_index_type=torch.int32,
    embedding_table_offset_type=torch.int32,
    weights_precision=SparseType.FP16,
)
```

### 5.4 Additional Micro-Optimizations

- **FP16/BF16 weights** for id_score_list features (50% savings on scores)
- **INT8/INT4 embedding table quantization** (75-87.5% savings)
- Using `optimizer.zero_grad(set_to_none=True)` to free gradient memory immediately

---

## 6. Industry Practices: Hybrid Parallelism and Real-Time Updates

### 6.1 Meta's TorchRec

**Meta's TorchRec** employs hybrid parallelism:
- **Model parallelism** for embedding tables (sharded via table-wise, row-wise, or column-wise strategies)
- **Data parallelism** for dense MLP layers

The EmbeddingShardingPlanner automatically generates optimal sharding plans based on device topology and memory constraints. Production deployments reach **1.25 trillion parameters** with UVM enabling tables larger than GPU memory.

### 6.2 ByteDance's Monolith

**ByteDance's Monolith** introduces collisionless embedding tables via Cuckoo hashing with:
- **Expirable embeddings**: Removed after inactivity
- **Frequency filtering**: Minimum interaction thresholds

Real-time online training achieves sub-minute parameter synchronization through incremental updates of only touched embeddings.

### 6.3 ByteDance's Persia

**ByteDance's Persia** scales to **100 trillion parameters** through hybrid synchronous/asynchronous training:
- Embedding layers update **asynchronously** (memory-intensive, 99.99%+ of parameters)
- Dense networks update **synchronously** (compute-intensive)

This achieves **3.8× higher throughput** versus fully synchronous mode with **7.12× speedup** over baseline systems.

### 6.4 Gradient Compression

**Gradient compression** reduces communication overhead dramatically. Deep Gradient Compression (DGC) demonstrates that **99.9% of gradient exchange is redundant**, achieving 270-600× compression without accuracy loss. TorchRec's Qcomm library enables quantized all-to-all and all-reduce at 4 bits without accuracy degradation.

### 6.5 Recommended Practices for MTML ROO with HSTU

- Use FusedEmbeddingBagCollection with `EXACT_ROWWISE_ADAGRAD` for sparse features
- Enable `prefetch_pipeline=True` with software-managed LFU caching
- Apply QR embeddings or TT-Rec for largest tables (user/item embeddings)
- Use mixed-precision (BF16) for HSTU attention with gradient checkpointing
- Implement int32 indices throughout the sparse feature pipeline
- Consider Monolith-style expirable embeddings for dynamic feature spaces

---

# Part II: Kernel-Level Profiling and Benchmarking

## 7. Theoretical Foundation for Kernel-Level Visibility

### 7.1 The Problem with End-to-End Timing

While macroscopic timing (wall-clock latency) indicates that performance has changed, it fails to explain **why**. To validate specific engineering interventions, such as the efficacy of `torch.compile` (Inductor) or the activation of Flash Attention, one must interrogate the GPU command stream directly.

### 7.2 Eager Execution Overhead

In the eager execution model of PyTorch, the framework acts as an interpreter that dispatches operations one by one to the GPU. This flexibility introduces significant overhead:
1. Python interpreter parses the code
2. PyTorch dispatcher (ATen) resolves the appropriate kernel
3. CUDA driver formats and enqueues the command

### 7.3 The "Black Box" of Compilation and Fusion

`torch.compile` can drastically reduce kernel count by merging operations (e.g., turning a linear+ReLU sequence that used 2 kernels into 1 fused kernel). The challenge is verifying this actually happened.

### 7.4 Key Profiling Requirements

1. **Kernel counting** cannot rely on Python-level operator hooks but must filter low-level profiler events for CUDA device types
2. **Launch overhead** is best isolated through a dual-timing strategy that contrasts asynchronous CPU dispatch time against synchronized GPU execution time
3. **SDPA backend verification** requires signature analysis of the execution trace, as runtime heuristics often silently fallback to slower Math implementations

---

## 8. Counting Kernel Launches and Distribution Analysis

### 8.1 Basic Kernel Counting with torch.profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = ...  # the PyTorch model
inputs = ...  # example input tensor on CUDA

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(inputs)       # forward pass
    output.sum().backward()      # backward pass if needed

# Count GPU kernel events
gpu_kernels = [evt for evt in prof.events() if evt.device_type == 'cuda']
print(f"Total CUDA kernels launched: {len(gpu_kernels)}")
```

### 8.2 Comprehensive Kernel Counting and Categorization

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from collections import defaultdict

def count_and_categorize_kernels(model, inputs):
    """Count total kernel launches and categorize by type."""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("forward_pass"):
            model(inputs)
            torch.cuda.synchronize()

    # Method 1: Count via aggregated statistics
    total_operator_calls = sum(e.count for e in prof.key_averages())

    # Method 2: Count actual CUDA kernels launched
    kernel_types = defaultdict(int)
    total_kernels = 0

    for event in prof.events():
        for kernel in event.kernels:
            total_kernels += 1
            name_lower = kernel.name.lower()
            if "gemm" in name_lower or "matmul" in name_lower:
                kernel_types["GEMM (Matrix Multiply)"] += 1
            elif "flash" in name_lower:
                kernel_types["Flash Attention"] += 1
            elif "conv" in name_lower:
                kernel_types["Convolution"] += 1
            elif "elementwise" in name_lower or "vectorized" in name_lower:
                kernel_types["Elementwise"] += 1
            elif "softmax" in name_lower:
                kernel_types["Softmax"] += 1
            else:
                kernel_types["Other"] += 1

    return {
        'total_kernels': total_kernels,
        'kernel_types': dict(kernel_types),
        'profiler': prof
    }
```

### 8.3 Key Profiler Configuration Options

| Option | Purpose |
|--------|---------|
| `ProfilerActivity.CUDA` | Captures on-device kernels (names like `void at::native::*` or `ampere_sgemm_*`) |
| `ProfilerActivity.CPU` | Shows PyTorch operators (`aten::*`) and launch events (`cudaLaunchKernel`) |
| `with_stack=True` | Adds source location tracking (overhead—use for investigation, disable for timing) |

### 8.4 Validation via Chrome Trace

In Chrome's trace view, each GPU kernel appears as a block on the GPU timeline. The number of these blocks per iteration corresponds to kernel launches. Each fused region will appear as one "CompiledFunction" or Triton kernel launch instead of many smaller ops.

---

## 9. Separating Launch Overhead from Execution Time

### 9.1 The Asynchronous Execution Challenge

CUDA operations are **asynchronous**—the CPU queues work and returns immediately. Without synchronization, naive timing measures only launch overhead (~5-20µs per kernel), not actual GPU compute time.

### 9.2 Visual Identification

In Chrome trace timeline, large gaps between GPU kernel events indicate the GPU is underutilized and waiting on the CPU:

```
Gaps between GPU kernels indicate CPU-side overhead (GPU is idle waiting for kernels to launch).
```

### 9.3 CUDA Events for Precise GPU Timing

```python
def measure_execution_time_accurately(fn, inputs, num_warmup=10, num_iters=100):
    """Measure pure GPU execution time using CUDA events."""
    # Warmup (critical for JIT compilation, cuDNN autotuning)
    for _ in range(num_warmup):
        fn(*inputs)
    torch.cuda.synchronize()

    # Create event pairs for each iteration
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    # Record all events without intermediate synchronization
    for i in range(num_iters):
        start_events[i].record()
        fn(*inputs)
        end_events[i].record()

    # Single synchronize at end—required before elapsed_time()
    torch.cuda.synchronize()

    # Extract GPU execution times (milliseconds)
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return times_ms
```

### 9.4 Measuring Launch Overhead Separately

```python
def measure_launch_overhead_separately(fn, inputs, num_iters=100):
    """Measure launch overhead by comparing async vs sync timing."""
    from time import perf_counter

    # Async timing (launch overhead only)
    launch_times = []
    for _ in range(num_iters):
        start = perf_counter()
        fn(*inputs)  # Returns immediately
        launch_times.append(perf_counter() - start)

    # Sync timing (launch + execution)
    total_times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        start = perf_counter()
        fn(*inputs)
        torch.cuda.synchronize()
        total_times.append(perf_counter() - start)

    import numpy as np
    return {
        'avg_launch_overhead_us': np.mean(launch_times) * 1e6,
        'avg_total_time_us': np.mean(total_times) * 1e6,
        'avg_execution_time_us': (np.mean(total_times) - np.mean(launch_times)) * 1e6
    }
```

### 9.5 Critical Synchronization Rules

- **Always call `torch.cuda.synchronize()` before `elapsed_time()`** or you'll get a RuntimeError
- **Never synchronize inside hot loops**—it forces CPU-GPU serialization and destroys throughput
- **Place synchronization only at measurement boundaries**

### 9.6 Interpreting Results

$$\text{Total elapsed time} = \text{GPU execution time} + \text{CPU launch overhead}$$

For example: "Total step time = 12 ms, GPU kernels execution = 10 ms, launch overhead ≈ 2 ms." A high overhead suggests the model is CPU-bound in launching kernels—techniques like **CUDA Graphs** can help reduce overhead.

**Important Pitfall**: Without `torch.cuda.synchronize()`, we may only capture the cost of launching kernels, not their execution. In one profiling exercise, an optimized model showed ~5.6ms instead of the true ~10ms until proper synchronization was added.

---

## 10. SDPA Backend Detection and Verification

### 10.1 The SDPA Backend Selection Problem

PyTorch's `scaled_dot_product_attention` automatically selects between:
- **Flash Attention**: Fused, GPU-optimized kernel (SM80+ required)
- **Memory-Efficient Attention**: xFormers-based streaming attention
- **cuDNN**: cuDNN 9.0+ backend
- **Math (Fallback)**: Standard matmul + softmax ops

The challenge is that **fallback to Math can happen silently**, causing massive performance degradation.

### 10.2 Querying Backend Availability

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def detect_sdpa_backend_used(query, key, value):
    """Detect which SDPA backend is actually selected at runtime."""

    # Query global enable states
    print("=== SDPA Backend Configuration ===")
    print(f"Flash Attention available: {torch.backends.cuda.is_flash_attention_available()}")
    print(f"Flash enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"Memory-efficient enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"cuDNN enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")
    print(f"Math enabled: {torch.backends.cuda.math_sdp_enabled()}")

    # Check which backends CAN be used for these specific inputs
    from torch.backends.cuda import SDPAParams, can_use_flash_attention, \
        can_use_efficient_attention, can_use_cudnn_attention

    params = SDPAParams(query, key, value, None, 0.0, is_causal=False)
    print(f"\nFor current inputs (shape {query.shape}, dtype {query.dtype}):")
    print(f"  Can use Flash: {can_use_flash_attention(params, debug=True)}")
    print(f"  Can use Efficient: {can_use_efficient_attention(params, debug=True)}")
    print(f"  Can use cuDNN: {can_use_cudnn_attention(params, debug=True)}")
```

### 10.3 Profile-Based Backend Detection

```python
def profile_sdpa_and_identify_backend(query, key, value):
    """Profile SDPA and identify backend from kernel names."""
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        out = F.scaled_dot_product_attention(query, key, value)
        torch.cuda.synchronize()

    # Identify backend from kernel/operator names
    events = prof.key_averages()
    backend_detected = "Unknown"

    for e in events:
        key_lower = e.key.lower()
        if "flash" in key_lower or "_flash_attention" in key_lower:
            backend_detected = "Flash Attention"
            break
        elif "efficient" in key_lower or "_efficient_attention" in key_lower:
            backend_detected = "Memory-Efficient"
            break
        elif "cudnn" in key_lower and "sdpa" in key_lower:
            backend_detected = "cuDNN"
            break

    # Fallback detection: Math backend uses bmm + softmax
    if backend_detected == "Unknown":
        has_bmm = any("bmm" in e.key.lower() for e in events)
        has_softmax = any("softmax" in e.key.lower() for e in events)
        if has_bmm and has_softmax:
            backend_detected = "Math (fallback)"

    return backend_detected, prof
```

### 10.4 Kernel Name Patterns for Backend Identification

| Kernel Pattern | Backend |
|----------------|---------|
| `pytorch_flash::flash_fwd_kernel`, `_flash_attention` | Flash Attention |
| `fmha_*`, `efficient_attention_*`, `_efficient_attention` | Memory-Efficient |
| `cudnn` + `sdpa` | cuDNN |
| `aten::bmm`, `aten::softmax`, `aten::div` (scaling) | Math (fallback) |

### 10.5 Forcing Specific Backends

```python
from torch.backends.cuda import sdp_kernel, SDPBackend

# Force Flash Attention only
with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=True)

# Using newer API (PyTorch 2.3+)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value)
```

### 10.6 Backend Requirements Summary

| Requirement | Flash Attention | Memory-Efficient | Math |
|-------------|-----------------|------------------|------|
| **Dtype** | fp16/bf16 only | fp16/bf16/fp32 | All |
| **GPU** | SM80+ (Ampere/Hopper) | SM50+ | Any |
| **Head dim** | Multiple of 8, ≤256 | Div by 8 (fp16) or 4 (fp32) | No restriction |
| **Custom mask** | ❌ Only `is_causal=True` | ✅ Additive masks | ✅ All |

### 10.7 Optimal Backend Priority by Hardware

```python
def get_optimal_backend_priority(device_capability):
    """Get SDPA backend priorities based on device capability"""
    if device_capability.major >= 8:  # Ampere+
        return [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.MEMORY_EFFICIENT,
            SDPBackend.MATH
        ]
    else:
        return [
            SDPBackend.MEMORY_EFFICIENT,
            SDPBackend.MATH
        ]
```

---

## 11. A/B Benchmarking Infrastructure

### 11.1 Best Practices for Fair Comparison

| Practice | Implementation |
|----------|----------------|
| **Isolate Runs** | Run each configuration separately to avoid interference |
| **Warm Up First** | 10-20 warmup iterations before timing (JIT, cuDNN autotuning) |
| **Synchronized Timing** | Always synchronize before/after timed region |
| **Repeat and Average** | 30+ iterations for statistical significance |
| **Lock GPU Clocks** | Use `nvidia-smi --lock-gpu-clocks` for consistency |
| **Coefficient of Variation** | CV should be below **5%** for reliable benchmarks |

### 11.2 Production A/B Benchmark Class

```python
import torch
import numpy as np
from scipy import stats
import subprocess
import time
import gc

class KernelOptimizationBenchmark:
    """Production A/B benchmark for validating kernel optimizations."""

    def __init__(self, seed=42):
        self.seed = seed

    def setup_gpu_for_benchmarking(self, lock_clocks=True, target_clock_mhz=1500):
        """Lock GPU clocks for consistent measurements."""
        if lock_clocks:
            subprocess.run(['sudo', 'nvidia-smi', '-pm', '1'], check=False)
            subprocess.run([
                'sudo', 'nvidia-smi',
                f'--lock-gpu-clocks={target_clock_mhz},{target_clock_mhz}'
            ], check=False)

    def clear_all_state(self):
        """Clear caches, memory, and compilation state."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch._dynamo.reset()  # Clear torch.compile cache

    def get_gpu_metrics(self):
        """Query GPU temperature and clock speed."""
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu,clocks.current.graphics',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        temp, clock = result.stdout.strip().split(', ')
        return {'temperature': int(temp), 'clock_mhz': int(clock)}

    def run_ab_comparison(self, fn_baseline, fn_optimized, inputs,
                          num_warmup=20, num_iters=200, cooling_seconds=10):
        """Run A/B benchmark with all best practices."""

        results = {}
        for name, fn in [('baseline', fn_baseline), ('optimized', fn_optimized)]:
            self.clear_all_state()

            # Cooling period to prevent thermal variance
            if cooling_seconds > 0:
                time.sleep(cooling_seconds)
                while self.get_gpu_metrics()['temperature'] > 50:
                    time.sleep(5)

            # Warmup phase (JIT compilation, cuDNN autotuning)
            for _ in range(num_warmup):
                fn(*inputs)
            torch.cuda.synchronize()

            # Reset memory tracking after warmup
            torch.cuda.reset_peak_memory_stats()

            # Timed iterations using CUDA events
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

            for i in range(num_iters):
                start_events[i].record()
                fn(*inputs)
                end_events[i].record()

            torch.cuda.synchronize()
            times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

            results[name] = {
                'times': times,
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'median_ms': np.median(times),
                'p95_ms': np.percentile(times, 95),
                'cv': np.std(times) / np.mean(times),  # Should be < 0.05
                'peak_memory_mb': torch.cuda.max_memory_allocated() / 1e6
            }

        # Statistical significance testing
        t_stat, p_value = stats.ttest_ind(
            results['baseline']['times'],
            results['optimized']['times'],
            equal_var=False  # Welch's t-test
        )

        speedup = results['baseline']['mean_ms'] / results['optimized']['mean_ms']

        return {
            'baseline': results['baseline'],
            'optimized': results['optimized'],
            'speedup': speedup,
            'p_value': p_value,
            'significant_at_95pct': p_value < 0.05
        }

    def validate_kernel_reduction(self, fn_baseline, fn_optimized, inputs):
        """Validate kernel count reduction claims (e.g., '221→30 kernels')."""

        def count_kernels(fn):
            torch._dynamo.reset()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                fn(*inputs)
                torch.cuda.synchronize()

            kernel_count = sum(len(e.kernels) for e in prof.events())
            unique_kernels = set()
            for e in prof.events():
                for k in e.kernels:
                    unique_kernels.add(k.name)
            return kernel_count, len(unique_kernels)

        baseline_total, baseline_unique = count_kernels(fn_baseline)
        optimized_total, optimized_unique = count_kernels(fn_optimized)

        return {
            'baseline_kernels': baseline_total,
            'optimized_kernels': optimized_total,
            'reduction': baseline_total - optimized_total,
            'reduction_pct': (baseline_total - optimized_total) / baseline_total * 100,
            'baseline_unique': baseline_unique,
            'optimized_unique': optimized_unique
        }
```

### 11.3 Example A/B Output Format

```
Config               Time per iter (ms)    GPU kernels    SDPA Backend    Memory used (MB)
Eager (no compile)   15.3 ms               221            Math            800 MB
Compiled             9.8 ms                30             Flash           780 MB
```

---

## 12. Per-Component Kernel Attribution

### 12.1 Using record_function for Component Labeling

```python
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

class ProfiledTransformerBlock(nn.Module):
    """Transformer block with hierarchical profiling annotations."""

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        with record_function("TransformerBlock"):
            # Attention sub-block
            with record_function("Attention"):
                with record_function("LayerNorm_1"):
                    x_norm = self.ln1(x)
                with record_function("MultiheadAttention"):
                    attn_out, _ = self.attn(x_norm, x_norm, x_norm)
                x = x + attn_out

            # MLP sub-block
            with record_function("MLP"):
                with record_function("LayerNorm_2"):
                    x_norm = self.ln2(x)
                with record_function("FeedForward"):
                    mlp_out = self.mlp(x_norm)
                x = x + mlp_out

        return x
```

### 12.2 ComponentProfiledModel Wrapper

```python
class ComponentProfiledModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        with torch.profiler.record_function("embedding"):
            x = self.model.embedding(x)

        # Profile each transformer layer
        for i, layer in enumerate(self.model.layers):
            with torch.profiler.record_function(f"layer_{i}"):
                with torch.profiler.record_function(f"layer_{i}_attention"):
                    x = layer.attention(x)
                with torch.profiler.record_function(f"layer_{i}_mlp"):
                    x = layer.mlp(x)

        return x
```

### 12.3 Extracting Component-Level Statistics

```python
def extract_component_stats(prof, component_names):
    """Extract timing stats for specific named components."""
    averages = prof.key_averages()
    stats = {}

    for name in component_names:
        matching = [e for e in averages if name in e.key]
        if matching:
            event = matching[0]
            stats[name] = {
                'cuda_time_ms': event.cuda_time_total / 1000,
                'self_cuda_time_ms': event.self_cuda_time_total / 1000,
                'cpu_time_ms': event.cpu_time_total / 1000,
                'call_count': event.count,
                'cuda_memory_mb': event.cuda_memory_usage / 1e6 if event.cuda_memory_usage else 0
            }

    return stats

# Usage
component_stats = extract_component_stats(prof, [
    "Attention", "MLP", "LayerNorm_1", "LayerNorm_2", "FeedForward"
])
for name, stats in component_stats.items():
    print(f"{name}: {stats['cuda_time_ms']:.2f}ms CUDA, {stats['call_count']} calls")
```

### 12.4 Interpreting Component Attribution

This fine-grained attribution helps pinpoint which part benefits from optimizations. For instance, if after fusion you still see "MLP" taking the bulk of time, you might focus next on optimizing the MLP block (like GEMM autotuning or fusion there).

---

## 13. Chrome Trace Export and Visual Analysis

### 13.1 Exporting Traces

```python
def export_comprehensive_trace(model, inputs, trace_path="analysis_trace.json"):
    """Export detailed trace for Chrome/Perfetto visualization."""

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        # Schedule for multi-iteration capture
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3),
    ) as prof:
        for step in range(6):  # wait + warmup + active
            with record_function(f"step_{step}"):
                model(inputs)
            torch.cuda.synchronize()
            prof.step()

    # Export standard trace
    prof.export_chrome_trace(trace_path)

    # Export flame graph for CPU analysis
    prof.export_stacks("cpu_stacks.txt", metric="self_cpu_time_total")

    return prof
```

### 13.2 A/B Comparison Trace Export

```python
def export_comparison_traces(baseline_prof, optimized_prof, output_dir):
    """Export Chrome traces for A/B comparison"""

    import os
    os.makedirs(output_dir, exist_ok=True)

    # Export individual traces
    baseline_prof.export_chrome_trace(f"{output_dir}/baseline_trace.json")
    optimized_prof.export_chrome_trace(f"{output_dir}/optimized_trace.json")

    print(f"Traces exported to {output_dir}")
    print(f"View baseline: chrome://tracing → Load {output_dir}/baseline_trace.json")
    print(f"View optimized: chrome://tracing → Load {output_dir}/optimized_trace.json")
```

### 13.3 Interpreting Chrome Traces

Open `chrome://tracing` in Chrome or use Perfetto (https://ui.perfetto.dev). The trace shows:

- **CPU row** (top): PyTorch operators, `record_function` labels, `cudaLaunchKernel` calls
- **CUDA row** (bottom): Actual GPU kernel execution
- **Flow arrows**: Connect CPU launch events to GPU kernels
- **Gaps**: Indicate CPU-GPU synchronization points or idle time

**Keyboard shortcuts**: `w`/`s` to zoom in/out, `a`/`d` to pan left/right, click events for details.

### 13.4 What to Look For

- **Kernel overlap** (or lack thereof)
- **Launch gaps** (indicating CPU-bound behavior)
- **Fused regions** appearing as "CompiledFunction" with fewer GPU events
- **Outlier slow kernels** or unusual synchronization points

---

## 14. Integration with NVIDIA Nsight

### 14.1 NVTX Markers for Nsight Systems

```python
import torch.cuda.nvtx as nvtx

def profile_for_nsight(model, inputs, num_iters=10, warmup_iters=5):
    """Annotate code for NVIDIA Nsight Systems profiling."""

    for i in range(num_iters):
        # Start profiling after warmup
        if i == warmup_iters:
            torch.cuda.cudart().cudaProfilerStart()

        profiling = (i >= warmup_iters)

        if profiling:
            nvtx.range_push(f"iteration_{i}")

        # Annotate model phases
        if profiling:
            nvtx.range_push("forward")
        output = model(inputs)
        if profiling:
            nvtx.range_pop()

        if profiling:
            nvtx.range_push("backward")
        loss = output.sum()
        loss.backward()
        if profiling:
            nvtx.range_pop()

        if profiling:
            nvtx.range_pop()  # End iteration

    torch.cuda.cudart().cudaProfilerStop()
```

### 14.2 Nsight Systems CLI

```bash
nsys profile -w true -t cuda,nvtx,osrt -s none \
    --capture-range=cudaProfilerApi \
    -o profile_report python benchmark.py
```

### 14.3 Nsight Compute for Specific Kernel Analysis

```bash
ncu --nvtx --nvtx-include "forward/" \
    -k regex:ampere_sgemm \
    --set full -o kernel_analysis python benchmark.py
```

This provides detailed kernel analysis including occupancy, memory bandwidth, and Tensor Core utilization.

---

## 15. Complete Production Benchmark Template

```python
#!/usr/bin/env python3
"""Production benchmark with kernel-level profiling for GPU optimization validation."""

import torch
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
from scipy import stats
import time
import gc
import json

class EnhancedBenchmark:
    """Complete benchmark suite for kernel optimization validation."""

    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device

    def profile_kernel_distribution(self, inputs):
        """Count and categorize all kernel launches."""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            with record_function("inference"):
                self.model(inputs)
            torch.cuda.synchronize()

        kernel_stats = {'gemm': 0, 'flash': 0, 'elementwise': 0, 'other': 0}
        total_kernels = 0

        for event in prof.events():
            for kernel in event.kernels:
                total_kernels += 1
                name = kernel.name.lower()
                if 'gemm' in name or 'matmul' in name:
                    kernel_stats['gemm'] += 1
                elif 'flash' in name:
                    kernel_stats['flash'] += 1
                elif 'elementwise' in name or 'vectorized' in name:
                    kernel_stats['elementwise'] += 1
                else:
                    kernel_stats['other'] += 1

        return {'total': total_kernels, 'by_type': kernel_stats, 'profiler': prof}

    def verify_sdpa_backend(self, batch_size=2, seq_len=512, heads=8, dim=64):
        """Verify which SDPA backend is being used."""
        q = torch.randn(batch_size, heads, seq_len, dim,
                       dtype=torch.float16, device=self.device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

        for e in prof.key_averages():
            if 'flash' in e.key.lower():
                return 'Flash Attention'
            elif 'efficient' in e.key.lower():
                return 'Memory-Efficient'
            elif 'cudnn' in e.key.lower():
                return 'cuDNN'
        return 'Math (fallback)'

    def benchmark_with_profiling(self, inputs, num_warmup=20, num_iters=100):
        """Full benchmark with timing, memory, and kernel profiling."""

        # Warmup
        for _ in range(num_warmup):
            self.model(inputs)
        torch.cuda.synchronize()

        # Clear state
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        # Timing with CUDA events
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            start_events[i].record()
            self.model(inputs)
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

        # Kernel profiling (single iteration)
        kernel_info = self.profile_kernel_distribution(inputs)

        return {
            'timing': {
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'median_ms': np.median(times),
                'p95_ms': np.percentile(times, 95),
                'cv': np.std(times) / np.mean(times)
            },
            'memory': {
                'peak_mb': torch.cuda.max_memory_allocated() / 1e6,
                'allocated_mb': torch.cuda.memory_allocated() / 1e6
            },
            'kernels': {
                'total': kernel_info['total'],
                'by_type': kernel_info['by_type']
            }
        }

    def export_analysis(self, inputs, output_dir='.'):
        """Export Chrome trace and summary for detailed analysis."""
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with record_function("model_analysis"):
                self.model(inputs)
            torch.cuda.synchronize()

        # Export trace
        trace_path = f"{output_dir}/kernel_trace.json"
        prof.export_chrome_trace(trace_path)

        # Export summary
        summary = {
            'top_cuda_ops': [],
            'sdpa_backend': self.verify_sdpa_backend()
        }

        for e in prof.key_averages()[:10]:
            summary['top_cuda_ops'].append({
                'name': e.key,
                'cuda_time_ms': e.cuda_time_total / 1000,
                'count': e.count
            })

        with open(f"{output_dir}/profile_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        return trace_path, summary


# Example usage
if __name__ == "__main__":
    from torchvision.models import resnet50

    model = resnet50()
    inputs = torch.randn(16, 3, 224, 224, device='cuda')

    bench = EnhancedBenchmark(model)

    # Full benchmark
    results = bench.benchmark_with_profiling(inputs)
    print(f"Mean latency: {results['timing']['mean_ms']:.2f}ms")
    print(f"Total kernels: {results['kernels']['total']}")
    print(f"Peak memory: {results['memory']['peak_mb']:.1f}MB")

    # Export for analysis
    trace_path, summary = bench.export_analysis(inputs)
    print(f"Trace exported to: {trace_path}")
```

### 15.1 Key API Reference

| API | Purpose | Key Parameters |
|-----|---------|----------------|
| `torch.profiler.profile()` | Capture kernel-level traces | `activities`, `record_shapes`, `with_stack`, `profile_memory` |
| `ProfilerActivity.CUDA` | Enable GPU kernel visibility | N/A (enum value) |
| `prof.key_averages()` | Aggregate statistics | `group_by_input_shape`, `group_by_stack_n` |
| `event.kernels` | Access kernel list per event | Returns `List[Kernel]` namedtuples |
| `torch.cuda.Event(enable_timing=True)` | Create timing event | `enable_timing` must be True |
| `event.elapsed_time(end_event)` | Get GPU time in ms | Requires prior `synchronize()` |
| `torch.cuda.synchronize()` | Wait for all GPU work | Required before `elapsed_time()` |
| `record_function(name)` | Label code regions | String name appears in traces |
| `sdpa_kernel(backend)` | Force SDPA backend | `SDPBackend.FLASH_ATTENTION`, etc. |
| `torch.backends.cuda.flash_sdp_enabled()` | Query backend state | Returns bool |
| `prof.export_chrome_trace(path)` | Export for visualization | Path to `.json` or `.json.gz` |

### 15.2 Performance Overhead Considerations

| Feature | Overhead | Recommendation |
|---------|----------|----------------|
| Kernel counting | ~1-2% | Use for all benchmarks |
| Component attribution | ~2-3% | Strategic placement |
| SDPA detection | <1% | Cached backend detection |
| Memory profiling | ~3-5% | Disable in production |
| **Total with all features** | ~5-10% | Development phase only |

### 15.3 Deployment Recommendations

**Development Phase**:
- Enable all profiling features for comprehensive analysis
- Use frequent Chrome trace exports for visual debugging
- Run statistical A/B tests with 30+ iterations

**Production Phase**:
- Disable memory profiling for minimal overhead
- Use sampling-based profiling (1 in 100 runs)
- Focus on kernel counting and component attribution

**CI Integration**:
- Implement automated A/B regression detection
- Set performance thresholds with statistical significance
- Generate automated trace comparison reports

---

## Appendix: Research Questions Quick Answers

### Part I: Embedding Optimization

**RQ1: What is the memory footprint of embedding tables at scale?**

For 100M entries × 128 dimensions in FP32: **51.2GB**. With Adam optimizer in mixed precision: 18 bytes/parameter total. FBGEMM fused optimizers eliminate gradient materialization, saving ~33% of this.

**RQ2: What compression techniques work during training?**

| Technique | Compression | Quality Impact |
|-----------|-------------|----------------|
| QR Embeddings | 10-15× | Negligible |
| TT-Rec | 100-200× | No loss |
| ROBE | 1000× | Meets MLPerf AUC |
| QAT INT4 | 8× | Regularization benefit |

**RQ3: How to fit large embeddings in limited GPU memory?**

Use software-managed UVM caching with **1.5-5% of embeddings in GPU memory**. Enable `prefetch_pipeline=True` for 2.8× speedup. Combine with fused backward-optimizer to eliminate gradient memory.

**RQ4: What immediate memory savings are available?**

Switch to int32 indices/offsets for **50% reduction** on sparse tensors. Safe when cardinality < 2.1B and cumulative offsets fit in int32.

### Part II: Kernel Profiling

**RQ5: How to count GPU kernel launches?**

Use `torch.profiler.profile()` with `ProfilerActivity.CUDA`. Filter events by `evt.device_type == 'cuda'`. For fused models, expect reductions from 221 → 30 kernels.

**RQ6: How to separate launch overhead from execution time?**

Use CUDA events for pure GPU time (`start.elapsed_time(end)`). Compare against wall-clock timing with synchronization. The difference is launch overhead.

**RQ7: How to verify which SDPA backend is being used?**

Profile and check for kernel names containing "flash" (Flash Attention), "bmm" + "softmax" (Math fallback), or "efficient" (Memory-Efficient). Use `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` to force and verify.

**RQ8: How to structure A/B benchmarks?**

1. Warmup 10-20 iterations (JIT, cuDNN autotuning)
2. Run 30+ timed iterations with CUDA events
3. Report mean, std, CV (should be <5%)
4. Use Welch's t-test for statistical significance (p < 0.05)
5. Lock GPU clocks for reproducibility

**RQ9: How to attribute kernel time to model components?**

Use `torch.profiler.record_function("component_name")` to label code regions. Extract stats via `prof.key_averages()`. Export Chrome traces for visual analysis.

---

## References

1. FBGEMM GPU Documentation: https://github.com/pytorch/FBGEMM
2. TorchRec Documentation: https://pytorch.org/torchrec/
3. PyTorch Profiler Documentation: https://pytorch.org/docs/stable/profiler.html
4. TT-Rec: Tensor Train Embeddings for Recommendation: https://arxiv.org/abs/2101.11714
5. ROBE: Random Offset Block Embedding: MLPerf Submission
6. ByteDance Monolith: Real-time Recommendation System
7. ByteDance Persia: 100 Trillion Parameter Training
8. Deep Gradient Compression: https://arxiv.org/abs/1712.01887
9. ColossalAI Embedding Cache Documentation
10. ScratchPipe: Look-Forward Caching
11. PyTorch torch.compile Performance Guide: https://pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
12. Accelerated PyTorch 2 Transformers: https://pytorch.org/blog/accelerated-pytorch-2/
13. NVIDIA Nsight Systems User Guide
14. torch.utils.benchmark Documentation: https://docs.pytorch.org/docs/stable/benchmark_utils.html
15. SDPA Tutorial: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
