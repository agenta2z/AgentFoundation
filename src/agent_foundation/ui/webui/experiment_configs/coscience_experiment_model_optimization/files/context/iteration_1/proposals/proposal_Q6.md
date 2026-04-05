# Proposal Q6: Embedding Table Optimization and Kernel-Level Profiling for MTML ROO Training

## Executive Summary

This proposal presents a **comprehensive, evidence-based optimization strategy** for MTML ROO training with HSTU, synthesizing findings from extensive research on embedding table optimization and kernel-level profiling. The recommendations are structured around **two critical dimensions**:

1. **Memory Optimization** (Part I): Techniques to reduce embedding table footprint from 50-500GB to fit within GPU memory constraints
2. **Performance Validation** (Part II): Kernel-level profiling infrastructure to **verify** optimizations actually deliver expected benefits

### Key Insight: The Verification Gap

> **⚠️ Critical Observation:** Many optimization proposals fail in practice because they lack validation infrastructure. A claim like "torch.compile reduces kernels from 221→30" is meaningless without profiling that proves it. **This proposal mandates profiling as a prerequisite for any optimization deployment.**

### Research Sources

| Source | Status | Key Contributions |
|--------|--------|-------------------|
| **Merged Q6 Research** | ✅ Fully incorporated | Embedding compression, FBGEMM details, profiling methodology |
| **Codebase Verification** | ✅ Applied | Actual config values, not assumed defaults |

### Already Implemented (No Action Required)

| Optimization | Evidence | Impact |
|--------------|----------|--------|
| int32 id_score_list | Commit `1ce9d80b` | 50% indices savings |
| bf16 on task_arch | Commit `1f2108d0` | Dense layer precision |
| activation_memory_budget=0.05 | V2 config | Enables recomputation |
| cache_load_factor=0.1 | Verified in prod configs | UVM caching |
| CacheAlgorithm.LFU | Verified in prod configs | Optimal for recommendations |

---

## Prioritization Matrix Overview

### Risk-Adjusted Priority Scoring

Each optimization is scored using: `Score = (Impact × Success_Probability) / (Effort × Risk_Factor)`

| Tier | Proposals | Combined Score | Recommended Action |
|------|-----------|----------------|-------------------|
| **Tier 1: Low-Hanging Fruits** 🍎 | 4 items | High | Implement Week 1 |
| **Tier 2: Medium Priority** | 5 items | Medium | Plan for Week 2-3 |
| **Tier 3: Research/Future** | 5 items | Variable | Investigate as needed |
| **NEW: Tier 0: Validation Infrastructure** | 1 item | Critical | Implement FIRST |

### Critical Change from Original Proposal

> **🚨 NEW MANDATORY STEP:** Before implementing ANY optimization, deploy the **Kernel-Level Profiling Infrastructure** (Tier 0). This ensures every claimed benefit is measurable and verifiable.

---

## Part I: Embedding Table Memory Optimization

### 1. Memory Profile Analysis

For a single embedding table with **100M entries and 128-dimension** in FP32:

| Scale | Dimension | FP32 Memory | FP16 Memory | With Adam Optimizer |
|-------|-----------|-------------|-------------|---------------------|
| 100M | 64 | 25.6 GB | 12.8 GB | 46.1 GB (18 bytes/param) |
| 100M | 128 | **51.2 GB** | 25.6 GB | 92.2 GB |
| 1B | 128 | 512 GB | 256 GB | 921 GB |

**Training Memory Formula:**
```
Total = Parameters + Optimizer_States + Gradients + Activations
      = W + 2W (Adam m,v) + W (gradients) + Activations
      = 4W + Activations (standard)
      = 2W + Activations (with FBGEMM fused optimizer - eliminates gradients!)
```

**Key Insight from Research:**
> "FBGEMM's `SplitTableBatchedEmbeddingBagsCodegen` with fused optimizers **eliminates gradient materialization entirely**, saving memory equal to parameter size. For 100M × 128D embeddings, this saves **51.2GB** of gradient storage."

### 2. HSTU Activation Memory Constraints

For HSTU with **512-1024 sequence lengths**:

```
Activation per layer ≈ B × L × D × 2 bytes
Example: batch=512, seq=1024, dim=256 → ~537MB per layer
```

With **5% activation budget** on 80GB GPU (~4GB), depth is constrained to **4-6 layers** without gradient checkpointing.

### 3. Access Pattern Statistics (Zipfian Distribution)

> "Approximately **20% of embedding vectors account for 80% of accesses**. This skew allows aggressive optimization of the tail without significantly impacting latency, provided the head is served from the fastest memory tier."

This justifies:
- Mixed-Dimension Embeddings (smaller dims for tail)
- Frequency-aware caching (LFU over LRU)
- INT8/INT4 quantization for cold embeddings

### 3.1 KeyedJaggedTensor Memory Formula

The **KeyedJaggedTensor** format efficiently represents variable-length sparse features with flattened values and lengths tensors:

$$\text{KJT\_mem} = \text{values\_count} \times \text{dtype\_size} + \text{lengths\_count} \times \text{offset\_dtype\_size}$$

**Example calculation:**
- 10M values × 4 bytes (int32) = 40MB
- 1M lengths × 4 bytes (int32) = 4MB
- **Total: 44MB**

With int32 optimization (already implemented):
- Original int64: 10M × 8 + 1M × 8 = 88MB
- Optimized int32: 10M × 4 + 1M × 4 = 44MB (**50% savings**)

For training bottleneck analysis, embedding lookups are **memory-bound, not compute-bound**, with all-to-all communication becoming co-dominant at scale.

### 4. Industry Practices at Scale

| Company | System | Scale | Key Innovation |
|---------|--------|-------|----------------|
| **Meta TorchRec** | Production | 1.25T params | EmbeddingShardingPlanner, hybrid model/data parallelism |
| **ByteDance Monolith** | Collisionless | Real-time | Expirable embeddings, Cuckoo hashing, frequency filtering |
| **ByteDance Persia** | 100T params | 3.8× throughput | Hybrid sync/async (embedding async, dense sync) |
| **Deep Gradient Compression** | Communication | 270-600× | 99.9% gradient redundancy elimination |

**From Research:**
> "ByteDance's Persia scales to **100 trillion parameters** through hybrid synchronous/asynchronous training: embedding layers update **asynchronously** (99.99%+ of parameters), while dense networks update **synchronously**. This achieves **3.8× higher throughput** versus fully synchronous mode."

---

## Part II: Kernel-Level Profiling Infrastructure (NEW)

### Critical Background: Why Profiling is Mandatory

> **The Problem with End-to-End Timing:** "While macroscopic timing (wall-clock latency) indicates that performance has changed, it fails to explain **why**. To validate specific engineering interventions, such as the efficacy of `torch.compile` or Flash Attention activation, one must interrogate the GPU command stream directly."

### Key Profiling Requirements

1. **Kernel counting** cannot rely on Python-level hooks—must filter for CUDA device types
2. **Launch overhead isolation** requires dual-timing strategy (async CPU vs. sync GPU)
3. **SDPA backend verification** requires signature analysis of execution trace (silent fallbacks to Math are common!)

### Profiler Performance Overhead Considerations

| Feature | Overhead | Recommendation |
|---------|----------|----------------|
| Kernel counting | ~1-2% | Use for all benchmarks |
| Component attribution | ~2-3% | Strategic placement |
| SDPA detection | <1% | Cached backend detection |
| Memory profiling | ~3-5% | Disable in production |
| **Total with all features** | ~5-10% | Development phase only |

### Deployment Phase Recommendations

**Development Phase:**
- Enable all profiling features for comprehensive analysis
- Use frequent Chrome trace exports for visual debugging
- Run statistical A/B tests with 30+ iterations

**Production Phase:**
- Disable memory profiling for minimal overhead
- Use sampling-based profiling (1 in 100 runs)
- Focus on kernel counting and component attribution

**CI Integration:**
- Implement automated A/B regression detection
- Set performance thresholds with statistical significance
- Generate automated trace comparison reports

### Production Benchmark Template

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity
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
        """Lock GPU clocks for consistent measurements.

        Critical for reproducible benchmarks - prevents thermal throttling variance.
        """
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
                          num_warmup=20, num_iters=200, cooling_seconds=10,
                          target_temperature=50):
        """Run A/B benchmark with all best practices.

        Args:
            fn_baseline: Baseline function to benchmark
            fn_optimized: Optimized function to benchmark
            inputs: Tuple of inputs to pass to functions
            num_warmup: Number of warmup iterations
            num_iters: Number of timed iterations
            cooling_seconds: Initial cooling period in seconds
            target_temperature: Wait until GPU temp drops to this (Celsius)
        """
        results = {}

        for name, fn in [('baseline', fn_baseline), ('optimized', fn_optimized)]:
            self.clear_all_state()

            # Cooling period to prevent thermal variance
            if cooling_seconds > 0:
                time.sleep(cooling_seconds)
                # Temperature-based cooling wait (CRITICAL for reproducibility)
                while self.get_gpu_metrics()['temperature'] > target_temperature:
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
                'times': times,  # Store raw times for t-test
                'mean_ms': np.mean(times),
                'std_ms': np.std(times),
                'median_ms': np.median(times),
                'p95_ms': np.percentile(times, 95),
                'cv': np.std(times) / np.mean(times),  # Should be < 0.05
                'peak_memory_mb': torch.cuda.max_memory_allocated() / 1e6
            }

        # Statistical significance testing (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(
            results['baseline']['times'],
            results['optimized']['times'],
            equal_var=False  # Welch's t-test for unequal variances
        )

        speedup = results['baseline']['mean_ms'] / results['optimized']['mean_ms']

        return {
            'baseline': results['baseline'],
            'optimized': results['optimized'],
            'speedup': speedup,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_at_95pct': p_value < 0.05,
            'significant_at_99pct': p_value < 0.01
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
            'reduction_pct': (baseline_total - optimized_total) / baseline_total * 100 if baseline_total > 0 else 0,
            'baseline_unique': baseline_unique,
            'optimized_unique': optimized_unique
        }

    def detect_sdpa_backend(self, query, key, value):
        """Detect which SDPA backend is actually used at runtime.

        Uses two methods:
        1. Pre-flight check: Query which backends CAN be used (SDPAParams)
        2. Profile-based detection: Verify which backend WAS actually used
        """
        import torch.nn.functional as F

        # Method 1: Pre-flight check using SDPAParams (before execution)
        print("=== SDPA Backend Configuration ===")
        print(f"Flash Attention available: {torch.backends.cuda.is_flash_attention_available()}")
        print(f"Flash enabled: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"Memory-efficient enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"cuDNN enabled: {torch.backends.cuda.cudnn_sdp_enabled()}")
        print(f"Math enabled: {torch.backends.cuda.math_sdp_enabled()}")

        try:
            from torch.backends.cuda import SDPAParams, can_use_flash_attention, \
                can_use_efficient_attention, can_use_cudnn_attention

            params = SDPAParams(query, key, value, None, 0.0, is_causal=False)
            print(f"\nFor current inputs (shape {query.shape}, dtype {query.dtype}):")
            print(f"  Can use Flash: {can_use_flash_attention(params, debug=True)}")
            print(f"  Can use Efficient: {can_use_efficient_attention(params, debug=True)}")
            print(f"  Can use cuDNN: {can_use_cudnn_attention(params, debug=True)}")
        except ImportError:
            print("SDPAParams not available in this PyTorch version, using profile-based detection only")

        # Method 2: Profile-based detection (verify actual execution)
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            F.scaled_dot_product_attention(query, key, value)
            torch.cuda.synchronize()

        backend = "Unknown"
        for e in prof.key_averages():
            key_lower = e.key.lower()
            if "flash" in key_lower or "_flash_attention" in key_lower:
                backend = "Flash Attention"
                break
            elif "efficient" in key_lower or "_efficient_attention" in key_lower:
                backend = "Memory-Efficient"
                break
            elif "cudnn" in key_lower and "sdpa" in key_lower:
                backend = "cuDNN"
                break

        # Fallback detection: Math backend uses bmm + softmax
        if backend == "Unknown":
            has_bmm = any("bmm" in e.key.lower() for e in prof.key_averages())
            has_softmax = any("softmax" in e.key.lower() for e in prof.key_averages())
            if has_bmm and has_softmax:
                backend = "Math (fallback) ⚠️"

        return backend

    def measure_launch_overhead_separately(self, fn, inputs, num_iters=100):
        """Measure launch overhead by comparing async vs sync timing.

        This helps identify if the model is CPU-bound (high launch overhead)
        vs GPU-bound (most time in execution).

        Returns:
            dict with avg_launch_overhead_us, avg_total_time_us, avg_execution_time_us
        """
        from time import perf_counter

        # Async timing (launch overhead only - returns immediately)
        launch_times = []
        for _ in range(num_iters):
            start = perf_counter()
            fn(*inputs)  # Returns immediately (async)
            launch_times.append(perf_counter() - start)

        # Sync timing (launch + execution)
        total_times = []
        for _ in range(num_iters):
            torch.cuda.synchronize()
            start = perf_counter()
            fn(*inputs)
            torch.cuda.synchronize()
            total_times.append(perf_counter() - start)

        return {
            'avg_launch_overhead_us': np.mean(launch_times) * 1e6,
            'avg_total_time_us': np.mean(total_times) * 1e6,
            'avg_execution_time_us': (np.mean(total_times) - np.mean(launch_times)) * 1e6
        }
```

### Per-Component Kernel Attribution

To identify which model components consume the most GPU time, use hierarchical `record_function` annotations:

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


class ComponentProfiledModel(torch.nn.Module):
    """Generic wrapper for per-component profiling of any model."""

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


# Usage example
component_stats = extract_component_stats(prof, [
    "Attention", "MLP", "LayerNorm_1", "LayerNorm_2", "FeedForward"
])
for name, stats in component_stats.items():
    print(f"{name}: {stats['cuda_time_ms']:.2f}ms CUDA, {stats['call_count']} calls")
```

**Interpreting Component Attribution:**
> This fine-grained attribution helps pinpoint which part benefits from optimizations. For instance, if after fusion you still see "MLP" taking the bulk of time, you might focus next on optimizing the MLP block (like GEMM autotuning or fusion there).

### Chrome Trace Export and Visual Analysis

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


def export_comparison_traces(baseline_prof, optimized_prof, output_dir):
    """Export Chrome traces for A/B comparison."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Export individual traces
    baseline_prof.export_chrome_trace(f"{output_dir}/baseline_trace.json")
    optimized_prof.export_chrome_trace(f"{output_dir}/optimized_trace.json")

    print(f"Traces exported to {output_dir}")
    print(f"View baseline: chrome://tracing → Load {output_dir}/baseline_trace.json")
    print(f"View optimized: chrome://tracing → Load {output_dir}/optimized_trace.json")
```

**Interpreting Chrome Traces:**

Open `chrome://tracing` in Chrome or use Perfetto (https://ui.perfetto.dev). The trace shows:

- **CPU row** (top): PyTorch operators, `record_function` labels, `cudaLaunchKernel` calls
- **CUDA row** (bottom): Actual GPU kernel execution
- **Flow arrows**: Connect CPU launch events to GPU kernels
- **Gaps**: Indicate CPU-GPU synchronization points or idle time

**Keyboard shortcuts:** `w`/`s` to zoom in/out, `a`/`d` to pan left/right, click events for details.

**What to Look For:**
- **Kernel overlap** (or lack thereof)
- **Launch gaps** (indicating CPU-bound behavior)
- **Fused regions** appearing as "CompiledFunction" with fewer GPU events
- **Outlier slow kernels** or unusual synchronization points

### NVIDIA Nsight Integration

For deeper kernel-level analysis beyond PyTorch profiler, use NVIDIA Nsight Systems/Compute:

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

**Nsight Systems CLI:**
```bash
nsys profile -w true -t cuda,nvtx,osrt -s none \
    --capture-range=cudaProfilerApi \
    -o profile_report python benchmark.py
```

**Nsight Compute for Specific Kernel Analysis:**
```bash
ncu --nvtx --nvtx-include "forward/" \
    -k regex:ampere_sgemm \
    --set full -o kernel_analysis python benchmark.py
```

This provides detailed kernel analysis including occupancy, memory bandwidth, and Tensor Core utilization.

### SDPA Backend Detection Warning

> **⚠️ Silent Fallback Problem:** "PyTorch's `scaled_dot_product_attention` automatically selects between Flash Attention, Memory-Efficient, cuDNN, and Math backends. The challenge is that **fallback to Math can happen silently**, causing massive performance degradation."

| Backend | Kernel Signature | Requirements |
|---------|-----------------|--------------|
| Flash Attention | `pytorch_flash::flash_fwd_kernel` | SM80+, fp16/bf16, head_dim≤256 |
| Memory-Efficient | `fmha_*`, `efficient_attention_*` | SM50+, fp16/bf16/fp32 |
| cuDNN | `cudnn` + `sdpa` | cuDNN 9.0+ |
| **Math (fallback)** | `aten::bmm`, `aten::softmax` | ⚠️ Any (but slow!) |

---

## Tier 0: Validation Infrastructure (IMPLEMENT FIRST)

### Proposal V0: Deploy Kernel Profiling Infrastructure

| Criteria | Value | Rationale |
|----------|-------|-----------|
| **Easiness** | 4/5 | Copy production template, customize for model |
| **Complexity** | 2/5 | Standard profiling APIs |
| **Risk Level** | Very Low | Read-only, non-invasive |
| **Dependencies** | None | |
| **Success Estimation** | 99% | Infrastructure, not optimization |
| **Implementation Effort** | 1-2 days | |
| **🔑 Priority** | **CRITICAL** | All other proposals depend on this |

**Deliverables:**
1. `benchmark_utils.py` module with `KernelOptimizationBenchmark` class
2. Integration with training pipeline for periodic profiling
3. Dashboard/logging for kernel counts, memory, and SDPA backend
4. A/B comparison scripts for before/after optimization validation

**Validation Protocol for ALL Subsequent Proposals:**
```
Before Implementation:
  1. Profile baseline (kernel count, memory, latency)
  2. Document SDPA backend in use
  3. Export Chrome trace for reference

After Implementation:
  1. Profile optimized version
  2. Run A/B comparison with 200+ iterations
  3. Verify p_value < 0.05 for claimed improvement
  4. Document actual vs. expected gains
  5. If gains < 50% of expected, investigate before deploying
```

---

## Tier 1: Low-Hanging Fruits 🍎 (Week 1)

### Summary Table

| # | Proposal | Easiness | Complexity | Risk | Success | Impact | Effort |
|---|----------|----------|------------|------|---------|--------|--------|
| 🍎1 | FP16/BF16 weights for id_score_list | **5/5** | **1/5** | Low | 95% | 50% score memory | 0.5 day |
| 🍎2 | Enable `prefetch_pipeline=True` | **4/5** | **2/5** | Low | 85% | Cache overlap | 1 day |
| 🍎3 | Verify fused optimizer enabled | **4/5** | **2/5** | Low | 90% | Eliminate grad storage | 1 day |
| 🍎4 | Embedding dimension alignment | **5/5** | **1/5** | Low | 95% | Optimal FBGEMM perf | 0.5 day |

**Total Tier 1 Effort: ~3 days**

---

### 🍎 LHF-1: FP16/BF16 Weights for id_score_list Features

#### Technical Analysis

The `id_score_list` features contain weight/score values representing feature importance. Currently stored in FP32.

**Why FP16/BF16 is safe:**
1. Scores are typically normalized values (0-1 range or softmax outputs)
2. FP16 has sufficient precision (±65,504 range, 3-4 decimal precision)
3. Already using bf16 on task_arch (commit `1f2108d0`)
4. Score values don't require FP32 precision for gradient computation

**Memory savings:**
- If 10M scores per batch: FP32 = 40MB → FP16 = 20MB (**50% savings**)

#### Implementation

```python
# Current (implicit FP32):
sparse_features = KeyedJaggedTensor(
    keys=["feature_with_scores"],
    values=torch.tensor([...], dtype=torch.int32),     # Already optimized
    lengths=torch.tensor([...], dtype=torch.int32),    # Already optimized
    weights=torch.tensor([...], dtype=torch.float32),  # ← CURRENT
)

# Optimized:
sparse_features = KeyedJaggedTensor(
    keys=["feature_with_scores"],
    values=torch.tensor([...], dtype=torch.int32),
    lengths=torch.tensor([...], dtype=torch.int32),
    weights=torch.tensor([...], dtype=torch.bfloat16),  # ← 50% savings
)
```

**Target Files:** `pytorch_modules_roo.py`, data preprocessing pipeline

#### Validation Protocol

```python
# MANDATORY: Profile before/after
benchmark = KernelOptimizationBenchmark()

# Before
baseline_mem = torch.cuda.max_memory_allocated()

# After
optimized_mem = torch.cuda.max_memory_allocated()

# Verify
assert optimized_mem < baseline_mem * 0.95, "Expected >5% memory reduction"

# Quality check
with torch.no_grad():
    out_fp32 = model(fp32_kjt)
    out_fp16 = model(fp16_kjt)
    rel_error = torch.abs(out_fp32 - out_fp16) / (torch.abs(out_fp32) + 1e-8)
    assert rel_error.max() < 0.01, f"Relative error too high: {rel_error.max()}"
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **5/5** - dtype change only |
| Complexity | **1/5** - No logic changes |
| Risk Level | **Low** - Precision sufficient |
| Success Estimation | **95%** |
| Expected Impact | 50% memory on score tensors |
| 🍎 Low-Hanging Fruit? | **YES** |

---

### 🍎 LHF-2: Enable `prefetch_pipeline=True` in TBE

#### Technical Analysis

FBGEMM's TBE supports prefetch pipelining that overlaps cache operations with compute:

```
Without prefetch_pipeline:
Batch N:   [Cache Miss] → [Fetch UVM] → [Forward] → [Backward]
Batch N+1:                                          [Cache Miss] → ...

With prefetch_pipeline=True:
Batch N:   [Cache Miss] → [Fetch UVM] → [Forward] → [Backward]
Batch N+1:               [Prefetch ←────────────────] [Forward] → ...
                          ↑ Overlapped with batch N compute
```

**From Research:**
> "Prefetch pipelining overlaps cache operations with compute: cache-insert for batch_{i+1} executes in parallel with forward/backward of batch_i."

**Current State (VERIFIED):**
```python
# File: configs.py:513
prefetch_pipeline: bool = False  # ← Currently OFF
```

#### Implementation

```python
# In TBE/TorchRec configuration
tbe = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=embedding_specs,
    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
    prefetch_pipeline=True,  # ← ENABLE THIS
    # ... other params
)
```

#### Validation Protocol

```python
# Profile cache operations overlap
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    for batch in dataloader:
        model(batch)
        break

# Look for "fbgemm::prefetch" and "fbgemm::cache_insert" timing overlap
# Export Chrome trace and verify overlap visually
prof.export_chrome_trace("prefetch_validation.json")
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **4/5** - Config parameter |
| Complexity | **2/5** - Verify pipeline compatibility |
| Risk Level | **Low** - Standard FBGEMM feature |
| Success Estimation | **85%** |
| Expected Impact | Better cache overlap, reduced latency |
| 🍎 Low-Hanging Fruit? | **YES** |

---

### 🍎 LHF-3: Verify Fused Optimizer is Enabled

#### Technical Analysis

FBGEMM's fused backward-optimizer pattern is **critical** for memory efficiency:

**Standard backward-then-optimizer:**
```
Backward:  Compute gradients → Store all gradients
Memory:    [grad_emb_1][grad_emb_2]...[grad_emb_N]  ← ALL stored!
Optimizer: Read gradients → Update weights → Free
```

**Fused backward-optimizer (FBGEMM):**
```
Backward:  Compute grad_i → IMMEDIATELY apply optimizer → Free
Memory:    Never stores full embedding gradients!
```

**From Research:**
> "FBGEMM's fused backward-optimizer **eliminates gradient materialization entirely**—gradients are applied directly during backward propagation."

**Memory savings:** Equal to entire embedding table size!
- 100M × 128D × 4 bytes = **51.2GB** of gradients never allocated

#### Audit Script

```python
def audit_fused_optimizer(model):
    """Check if model uses fused optimizers for embeddings."""
    issues = []

    for name, module in model.named_modules():
        # Check for non-fused EmbeddingBag
        if isinstance(module, nn.EmbeddingBag):
            issues.append(f"Non-fused EmbeddingBag found: {name}")

        # Verify TBE configuration
        if hasattr(module, '_embedding_bags'):
            for tbe_name, tbe in module._embedding_bags.items():
                if hasattr(tbe, 'optimizer'):
                    if tbe.optimizer == OptimType.SGD:
                        issues.append(f"Non-fused SGD in {tbe_name}")

    if issues:
        print("⚠️ Non-fused optimizers detected:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All embeddings use fused optimizers")
        return True
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **4/5** - Audit and verify |
| Complexity | **2/5** - May need migration |
| Risk Level | **Low** - Numerically equivalent |
| Success Estimation | **90%** |
| Expected Impact | Eliminate gradient storage |
| 🍎 Low-Hanging Fruit? | **YES** |

---

### 🍎 LHF-4: Embedding Table Dimension Alignment Check

#### Technical Analysis

**From Research:**
> "GPU kernel optimizations include table batching, **coalesced memory reads (embedding dimensions aligned to 4)**, and stochastic rounding for FP16 stability."

**Why alignment matters:**
- GPU memory is accessed in 128-byte cache lines
- Unaligned accesses cause multiple cache line fetches
- FBGEMM kernels assume `embedding_dim % 4 == 0`
- Unaligned dimensions may fall back to slower code paths

#### Verification Script

```python
def verify_embedding_alignment(model_config, alignment=4):
    """Verify all embedding dimensions are aligned."""
    misaligned = []

    for table_name, config in model_config.embedding_tables.items():
        if config.embedding_dim % alignment != 0:
            misaligned.append(
                f"{table_name}: dim={config.embedding_dim} (not aligned to {alignment})"
            )

    if misaligned:
        print("❌ Misaligned embedding dimensions found:")
        for m in misaligned:
            print(f"  - {m}")
        print(f"\nFix: Round up to nearest multiple of {alignment}")
    else:
        print(f"✓ All embedding dimensions aligned to {alignment}")

    return len(misaligned) == 0

def align_embedding_dim(dim: int, alignment: int = 4) -> int:
    """Round up dimension to nearest multiple of alignment."""
    return ((dim + alignment - 1) // alignment) * alignment
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **5/5** - Verification only |
| Complexity | **1/5** - Config check |
| Risk Level | **Low** - No changes if aligned |
| Success Estimation | **95%** |
| Expected Impact | Ensure optimal FBGEMM perf |
| 🍎 Low-Hanging Fruit? | **YES** |

---

## Tier 2: Medium Priority (Week 2-3)

### Summary Table

| # | Proposal | Easiness | Complexity | Risk | Success | Impact | Effort |
|---|----------|----------|------------|------|---------|--------|--------|
| 5 | **Test cache_load_factor=0.2** | **4/5** | **2/5** | Low | 70% | Potential latency reduction | 1-2 days |
| 6 | Mixed-Dimension Embeddings (MDE) | 2/5 | 3/5 | Medium | 70% | 2-16× compression | 5-7 days |
| 7 | Quotient-Remainder (QR) Embeddings | 2/5 | 3/5 | Medium | 65% | 10-15× compression | 7-10 days |
| 8 | INT8 Embedding Quantization (QAT) | 3/5 | 3/5 | Medium | 70% | 4× reduction | 5-7 days |
| 9 | Software-Managed LFU Cache Warming | 2/5 | 4/5 | Low | 75% | Train with 1.5-5% GPU | 7-10 days |

**Total Tier 2 Effort: ~25-35 days**

---

### Proposal 5: Test cache_load_factor=0.2 (NEW)

#### Technical Analysis

**Current State (VERIFIED):**
```python
# File: flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py:1119
cache_load_factor=0.1  # Currently 10% of rows cached in HBM
```

**From Research:**
> "FBGEMM's `MANAGED_CACHING` location with `cache_load_factor=0.2` keeps 20% of rows in fast memory."

**Clarification:** The original proposal removed the cache_load_factor tuning item because codebase verification showed the value is already 0.1 (not the assumed 0.2 default). However, this raises a **valid optimization question**: should we test 0.2 as a potential improvement?

**Trade-off Analysis:**

| cache_load_factor | HBM Usage | Expected Benefit | Risk |
|-------------------|-----------|------------------|------|
| 0.1 (current) | Lower | Lower latency variance | More UVM fetches |
| 0.2 (proposed test) | Higher | Fewer cache misses | Less HBM for activations |

**Recommendation:** Add this as a Tier 2 experiment to determine if 0.2 provides latency benefits without impacting activation memory budget.

#### Implementation

```python
# A/B test configuration
# Baseline: cache_load_factor=0.1
# Experiment: cache_load_factor=0.2

tbe_experiment = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=embedding_specs,
    cache_load_factor=0.2,  # ← Test this
    # ... other params unchanged
)
```

#### Validation Protocol

```python
benchmark = KernelOptimizationBenchmark()

# Test both configurations
results_01 = benchmark.run_ab_comparison(
    fn_baseline=lambda: model_01(batch),  # cache_load_factor=0.1
    fn_optimized=lambda: model_02(batch), # cache_load_factor=0.2
    inputs=(),
    num_iters=200
)

# Decision criteria
if results_01['p_value'] < 0.05 and results_01['speedup'] > 1.05:
    print("✓ cache_load_factor=0.2 provides statistically significant improvement")
else:
    print("✗ Keep cache_load_factor=0.1")
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **4/5** - Config parameter change |
| Complexity | **2/5** - Requires A/B test infrastructure |
| Risk Level | **Low** - Easily reversible |
| Success Estimation | **70%** |
| Expected Impact | Potential latency reduction |
| Implementation Effort | **1-2 days** |

---

### Proposal 6: Mixed-Dimension Embeddings (MDE)

#### Technical Analysis

**Key Insight:** Not all features need the same embedding dimension.

**From Research:**
> "MDE assigns smaller dimensions to rare features, achieving **2-16× compression** with maintained or improved accuracy due to regularization effects on long-tail features."

**Why this works:**
1. Rare features have fewer training samples → smaller dims prevent overfitting
2. Popular features need larger dims to capture complex patterns
3. Dimension allocation based on frequency is information-theoretically motivated

**Block-Based Implementation (Critical):**
> "Features are grouped into 'blocks' of dimensions (e.g., {16, 32, 64, 128}). All features in the 16-dim bucket are stored in one physical table. This restores regularity, allowing efficient batched kernels."

**Memory savings example:**
```
Original: 100M × 128D × 4 bytes = 51.2GB

Mixed-dimension:
  - 1M high-freq × 128D × 4  = 0.51GB
  - 9M med-freq × 64D × 4   = 2.3GB
  - 40M low-freq × 32D × 4  = 5.1GB
  - 50M tail × 16D × 4      = 3.2GB
  Total: 11.1GB → 4.6× compression
```

#### Implementation

```python
def create_mixed_dimension_configs(
    feature_frequencies: Dict[str, int],
    percentile_thresholds: List[float] = [0.01, 0.10, 0.50],
    dimension_tiers: List[int] = [128, 64, 32, 16],
) -> List[EmbeddingBagConfig]:
    """Create embedding configs with dimensions based on frequency."""

    sorted_features = sorted(
        feature_frequencies.items(),
        key=lambda x: x[1],
        reverse=True
    )
    total_features = len(sorted_features)
    configs = []

    for i, (feature_name, freq) in enumerate(sorted_features):
        percentile = i / total_features

        if percentile < percentile_thresholds[0]:
            dim = dimension_tiers[0]  # High-frequency
        elif percentile < percentile_thresholds[1]:
            dim = dimension_tiers[1]  # Medium
        elif percentile < percentile_thresholds[2]:
            dim = dimension_tiers[2]  # Low
        else:
            dim = dimension_tiers[3]  # Long-tail

        configs.append(EmbeddingBagConfig(
            name=feature_name,
            embedding_dim=dim,
            num_embeddings=...,
            feature_names=[feature_name],
        ))

    return configs
```

#### Critical Thinking: Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Quality degradation on tail features | Medium | A/B test with long-tail metrics |
| Inference latency increase (gathering) | Low | Block-based implementation |
| Integration complexity with TorchRec | Medium | Use EmbeddingBagConfig natively |

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **2/5** - Frequency analysis needed |
| Complexity | **3/5** - Architecture change |
| Risk Level | **Medium** - Quality validation required |
| Success Estimation | **70%** |
| Expected Impact | 2-16× compression |
| Implementation Effort | **5-7 days** |

---

### Proposal 7: Quotient-Remainder (QR) Embeddings

#### Technical Analysis

**Decomposition:**
```
Original:  e_i = E[i]           where E ∈ R^{N × D}
QR:        e_i = E_q[i // M] ⊙ E_r[i % M]

Memory:
  Original: O(N × D)
  QR:       O(2√N × D)  when M = √N

Example (1B features, D=128):
  Original: 512GB
  QR (M=32K): 32.7MB → ~15,600× compression
```

**Why this works:**
- Element-wise multiplication preserves interaction capacity
- Each unique (quotient, remainder) pair produces unique embedding
- Fully differentiable, standard backprop applies
- No collision handling needed (unlike feature hashing)

#### Implementation

```python
class QREmbedding(nn.Module):
    """Quotient-Remainder Embedding for extreme compression."""

    def __init__(self, num_embeddings: int, embedding_dim: int, num_buckets: int = None):
        super().__init__()

        if num_buckets is None:
            num_buckets = int(math.ceil(math.sqrt(num_embeddings)))

        self.num_buckets = num_buckets
        self.num_quotient_buckets = (num_embeddings + num_buckets - 1) // num_buckets

        self.quotient_table = nn.Embedding(self.num_quotient_buckets, embedding_dim)
        self.remainder_table = nn.Embedding(num_buckets, embedding_dim)

        # Initialize to maintain output variance
        nn.init.normal_(self.quotient_table.weight, mean=0, std=1.0)
        nn.init.normal_(self.remainder_table.weight, mean=0, std=1.0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        q = indices // self.num_buckets
        r = indices % self.num_buckets
        return self.quotient_table(q) * self.remainder_table(r)
```

#### Critical Thinking: When to Apply

| Table Size | Recommendation | Rationale |
|------------|----------------|-----------|
| < 10M | **Do not use** | Overhead exceeds benefit |
| 10M - 100M | Consider | Moderate benefit |
| > 100M | **Strongly recommended** | Significant compression |
| > 1B | **Essential** | Cannot fit without compression |

---

### Proposal 8: INT8 Embedding Quantization (QAT)

#### Technical Analysis

**From Research:**
> "QAT with INT8 provides **4× reduction** while quantization acts as **strong regularization** that can actually improve accuracy by mitigating DLRM overfitting."

**Row-wise Quantization (Critical):**
> "The dynamic range of embeddings varies between rows. **Row-wise quantization** stores a separate scale factor per row. Memory overhead is negligible (one float per D bytes), but significantly preserves accuracy. **FBGEMM TBE operators specifically support row-wise quantization.**"

#### FBGEMM Native Support

```python
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import SparseType

# INT8 configuration
tbe_int8 = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=embedding_specs,
    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
    weights_precision=SparseType.INT8,  # 4× memory reduction
)

# INT4 for more aggressive compression
tbe_int4 = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=embedding_specs,
    weights_precision=SparseType.INT4,  # 8× memory reduction
)
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **3/5** - Config change + calibration |
| Complexity | **3/5** - Calibration setup |
| Risk Level | **Medium** - Accuracy depends on calibration |
| Success Estimation | **70%** |
| Expected Impact | 4× (INT8) or 8× (INT4) reduction |
| Implementation Effort | **5-7 days** |

---

### Proposal 9: Software-Managed LFU Cache Warming

#### Technical Analysis

**From Research:**
> "ColossalAI's frequency-aware cache achieves training with **only 1.5-5% of embeddings in GPU memory**—for a 91GB table, only 3.75GB CUDA memory is required."

**Current State:** Codebase already uses `CacheAlgorithm.LFU` ✅

**Opportunity:** Pre-warm cache with highest-frequency IDs before training starts.

#### Implementation

```python
def warm_cache(
    tbe: SplitTableBatchedEmbeddingBagsCodegen,
    frequencies: Dict[str, Counter],
    top_k_percent: float = 5.0,
):
    """Warm TBE cache with highest-frequency feature IDs."""
    for feature_key, freq_counter in frequencies.items():
        total_ids = len(freq_counter)
        top_k = int(total_ids * top_k_percent / 100)

        top_ids = [id_ for id_, count in freq_counter.most_common(top_k)]
        top_ids_tensor = torch.tensor(top_ids, dtype=torch.int64, device='cuda')

        tbe.prefetch(top_ids_tensor)

    print(f"Cache warmed with top {top_k_percent}% frequent IDs")
```

---

### Proposal 10: ScratchPipe "Look-Forward" Cache

#### Technical Analysis

**From Research:**
> "ScratchPipe's 'look-forward' cache knows exactly which embeddings will be accessed in upcoming batches, achieving ~100% cache hit rate. This requires only cache size equal to the working set of the current batch. Performance reaches **2.8× average speedup** (up to 4.2×) versus prior GPU embedding systems."

**Key Difference from Standard Caching:**

| Approach | Cache Policy | Hit Rate | Memory Overhead |
|----------|-------------|----------|------------------|
| LRU/LFU | Reactive | 80-95% | Fixed cache size |
| **Look-Forward** | Proactive | **~100%** | Working set only |

**How it works:**
1. DataLoader prefetches batch_{i+1}, batch_{i+2}, ... batch_{i+k}
2. Before processing batch_i, system knows exact embedding IDs needed for next k batches
3. Embeddings are pre-loaded into cache with near-perfect hit rate
4. No wasted cache capacity on unused embeddings

#### Implementation Considerations

```python
class LookForwardCache:
    """Prefetch-based caching that knows future accesses."""

    def __init__(self, tbe, lookahead_batches=3):
        self.tbe = tbe
        self.lookahead_batches = lookahead_batches
        self.prefetch_queue = deque(maxlen=lookahead_batches)

    def register_upcoming_batches(self, upcoming_batches: List[KeyedJaggedTensor]):
        """Register embedding IDs from upcoming batches for prefetching."""
        for batch in upcoming_batches:
            unique_ids = torch.unique(batch.values())
            self.prefetch_queue.append(unique_ids)

    def prefetch_next(self):
        """Prefetch embeddings for next batch in queue."""
        if self.prefetch_queue:
            ids_to_prefetch = self.prefetch_queue.popleft()
            self.tbe.prefetch(ids_to_prefetch)
```

#### Assessment

| Criteria | Value |
|----------|-------|
| Easiness | **2/5** - Requires DataLoader integration |
| Complexity | **4/5** - Prefetch coordination |
| Risk Level | **Medium** - Need to ensure correct prefetch timing |
| Success Estimation | **65%** |
| Expected Impact | **2.8× average speedup** (up to 4.2×) |
| Implementation Effort | **7-10 days** |

---

## Tier 3: Future/Research (Week 4+)

### Summary Table

| # | Proposal | Easiness | Complexity | Risk | Success | Impact | Effort |
|---|----------|----------|------------|------|---------|--------|--------|
| 10 | ScratchPipe Look-Forward Cache | 2/5 | 4/5 | Medium | 65% | 2.8× speedup | 7-10 days |
| 11 | TT-Rec (Tensor Train) | 2/5 | 4/5 | Medium | 55% | 100-200× | 10+ days |
| 12 | Expirable Embeddings (Monolith) | 1/5 | 5/5 | High | 40% | Dynamic sizing | 15+ days |
| 13 | ROBE | 1/5 | 5/5 | High | 35% | 1000× | 20+ days |
| 14 | CCE (Clustered Compositional) | 2/5 | 4/5 | Medium | 50% | 10-50× | 10-15 days |
| 15 | SSD Offloading | 1/5 | 5/5 | High | 40% | Multi-TB models | 20+ days |

---

### Proposal 11: TT-Rec (Tensor Train Decomposition)

**From Research:**
> "TT-Rec achieves **112× compression on Criteo Terabyte with no accuracy loss**. Meta's FBTT-Embedding library provides drop-in PyTorch replacement."

```python
# FBTT-Embedding usage
from fbtt import TTEmbeddingBag

tt_emb = TTEmbeddingBag(
    num_embeddings=1_000_000_000,
    embedding_dim=128,
    tt_ranks=[64, 64, 64],
    tt_shapes=[[10000, 10000, 10000], [4, 4, 8]],
)
```

---

### Proposal 14: Clustered Compositional Embeddings (CCE)

**From Research:**
> "CCE combines hashing with quantization. CCE **learns a codebook of 'centroids'** and a sparse mapping. Instead of a random hash, the mapping is learned to cluster similar users together. This approaches compression ratio of product quantization while maintaining training efficiency."

**Advantage over QR:** Learned mapping preserves semantic relationships better than random decomposition.

---

### Proposal 15: SSD Offloading (FlashNeuron/Legend)

**From Research:**
> "For multi-terabyte models, even CPU DRAM is insufficient. The next tier is NVMe SSDs. **Legend** employs prefetch-friendly embedding loading and GPU-SSD direct access driver."

**When to consider:**
- Embedding tables exceed 1TB
- CPU DRAM insufficient
- Willing to accept ~10-100μs latency for cold embeddings

---

## Decision Flowchart: Technique Selection

```
                         Start
                           │
                           ▼
                 ┌─────────────────┐
                 │ Deploy Profiling│
                 │ Infrastructure  │
                 └────────┬────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │ Profile Current State │
              │ (kernels, memory,     │
              │  SDPA backend)        │
              └───────────┬───────────┘
                          │
                          ▼
            ┌────────────────────────────┐
            │ Is memory the bottleneck?  │
            └─────────┬────────┬─────────┘
                      │        │
                     Yes       No
                      │        │
                      ▼        ▼
         ┌────────────────┐  ┌──────────────────┐
         │ Table size?    │  │ Focus on compute │
         └──┬──────┬──────┘  │ (torch.compile,  │
            │      │         │  Flash Attention) │
          <10M   >10M        └──────────────────┘
            │      │
            ▼      ▼
      ┌─────────┐ ┌─────────────────────────┐
      │ FP16    │ │ Table size > 100M?      │
      │ weights │ └────────┬──────┬─────────┘
      │ only    │          │      │
      └─────────┘        Yes      No
                          │      │
                          ▼      ▼
                    ┌───────┐  ┌────────┐
                    │ QR or │  │ MDE or │
                    │ TT-Rec│  │ INT8   │
                    └───────┘  └────────┘
```

---

## Implementation Roadmap

### Recommended Execution Order

```
Week 0 (MANDATORY):
  Day 1-2: Deploy Kernel Profiling Infrastructure (Tier 0)
           - Create benchmark_utils.py
           - Integrate with training pipeline
           - Profile baseline state
           - Document SDPA backend, kernel counts, memory

Week 1 (Low-Hanging Fruits):
  Day 1 AM: 🍎 LHF-1: FP16 weights for scores        [0.5 day]
            └─ Validate with profiling
  Day 1 PM: 🍎 LHF-4: Embedding alignment check      [0.5 day]
            └─ Verify all dims % 4 == 0
  Day 2-3:  🍎 LHF-2: prefetch_pipeline=True         [1 day]
            └─ Profile cache overlap improvement
  Day 3:    🍎 LHF-3: Verify fused optimizer         [1 day]
            └─ Audit all embedding modules

  End of Week 1: A/B comparison report with statistical significance

Week 2-3 (Medium Priority - choose based on bottleneck):
  Option A (Memory-constrained):
    - Proposal 5: Test cache_load_factor=0.2
    - Proposal 8: INT8 QAT
    - Proposal 6: Mixed-Dimension Embeddings

  Option B (Large tables, >100M entries):
    - Proposal 7: QR Embeddings
    - Proposal 9: LFU Cache Warming

Week 4+ (Future - as needed):
  - Proposal 10: TT-Rec (if need 100×+ compression)
  - Proposal 13: CCE (if quality-sensitive compression needed)
  - Proposal 14: SSD Offloading (if tables exceed 1TB)
```

### Validation Checkpoints

**After EVERY optimization:**

```python
# 1. Memory Validation
torch.cuda.reset_peak_memory_stats()
# Run training epoch
peak_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak Memory: {peak_mem:.2f} GB")

# 2. Throughput Validation
import time
start = time.time()
for i, batch in enumerate(dataloader):
    if i >= 100: break
    # Training step
qps = 100 * batch_size / (time.time() - start)
print(f"QPS: {qps:.2f}")

# 3. Kernel Count Validation
results = benchmark.validate_kernel_reduction(baseline_fn, optimized_fn, inputs)
print(f"Kernels: {results['baseline_kernels']} → {results['optimized_kernels']}")
print(f"Reduction: {results['reduction_pct']:.1f}%")

# 4. Statistical Significance
ab_results = benchmark.run_ab_comparison(baseline_fn, optimized_fn, inputs)
print(f"Speedup: {ab_results['speedup']:.2f}x")
print(f"p-value: {ab_results['p_value']:.4f}")
print(f"Significant (p<0.05): {ab_results['significant_at_95pct']}")

# 5. SDPA Backend Verification
backend = benchmark.detect_sdpa_backend(query, key, value)
if "fallback" in backend.lower():
    print(f"⚠️ WARNING: SDPA using fallback backend: {backend}")
else:
    print(f"✓ SDPA backend: {backend}")
```

---

## Memory Impact Summary

### Current State vs. Projected Improvements

| Stage | Configuration | Memory | Notes |
|-------|---------------|--------|-------|
| **Current** | int32 indices + bf16 dense | Baseline | Already optimized |
| **After Tier 0** | + Profiling infrastructure | +0% | Validation only |
| **After Tier 1** | + FP16 scores + prefetch + verified fused | -30-50% scores | Low effort |
| **After Tier 2** | + MDE/QR/INT8 | 2-15× compression | Medium effort |
| **After Tier 3** | + TT-Rec/ROBE | 100-1000× | Research effort |

### Memory Budget Breakdown (80GB H100)

```
Current allocation (estimated):
├── Embedding tables:     40GB (50%)
├── Optimizer states:     16GB (20%)  ← Fused optimizer eliminates gradients
├── Activations:           4GB (5%)   ← activation_memory_budget=0.05
├── Dense layers:         10GB (12.5%)
└── Working memory:       10GB (12.5%)

After Tier 1 optimizations:
├── Embedding tables:     35GB (44%)  ← FP16 scores, better caching
├── Optimizer states:     16GB (20%)
├── Activations:           4GB (5%)
├── Dense layers:         10GB (12.5%)
└── Working memory:       15GB (18.5%) ← More headroom
```

---

## Appendix A: FBGEMM API Reference

### SplitTableBatchedEmbeddingBagsCodegen Key Parameters

```python
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    SparseType,
    CacheAlgorithm,
)

tbe = SplitTableBatchedEmbeddingBagsCodegen(
    # Required
    embedding_specs=[
        (num_embeddings, embedding_dim, location, compute_device),
    ],

    # Optimizer (use fused!)
    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,  # or ADAM, LAMB
    learning_rate=0.01,
    eps=1e-8,

    # Caching
    cache_load_factor=0.1,            # Fraction of rows in HBM cache
    cache_algorithm=CacheAlgorithm.LFU,  # LRU or LFU

    # Precision
    weights_precision=SparseType.FP32,  # FP32, FP16, INT8, INT4, INT2

    # Pipeline
    prefetch_pipeline=True,           # Enable prefetch pipelining

    # Index types
    embedding_table_index_type=torch.int32,
    embedding_table_offset_type=torch.int32,
)
```

### EmbeddingLocation Options

| Location | Memory | Speed | Use Case |
|----------|--------|-------|----------|
| `DEVICE` | GPU HBM | Fastest | Small tables |
| `MANAGED` | UVM (CPU) | Slower | Tables > GPU memory |
| `MANAGED_CACHING` | Hybrid | Configurable | Recommended for large tables |
| `HOST` | CPU only | Slowest | Extreme cases |

### SparseType Precision Options

| Type | Bytes | Compression | Notes |
|------|-------|-------------|-------|
| `FP32` | 4 | 1× | Default |
| `FP16` | 2 | 2× | Good for training |
| `BF16` | 2 | 2× | Better dynamic range |
| `INT8` | 1 | 4× | Row-wise quantization |
| `INT4` | 0.5 | 8× | Aggressive compression |
| `INT2` | 0.25 | 16× | Research only |

---

## Appendix B: Profiling API Quick Reference

| API | Purpose | Key Parameters |
|-----|---------|----------------|
| `torch.profiler.profile()` | Capture kernel traces | `activities`, `record_shapes` |
| `ProfilerActivity.CUDA` | GPU kernel visibility | N/A |
| `prof.key_averages()` | Aggregate statistics | `group_by_input_shape` |
| `torch.cuda.Event(enable_timing=True)` | Timing events | Must be True |
| `event.elapsed_time(end_event)` | GPU time (ms) | Requires `synchronize()` |
| `torch.cuda.synchronize()` | Wait for GPU | Required before timing |
| `record_function(name)` | Label code regions | Appears in traces |
| `prof.export_chrome_trace(path)` | Visual analysis | `.json` output |
| `prof.export_stacks(path)` | Flame graph export | CPU analysis |
| `nvtx.range_push(name)` | NVTX marker start | For Nsight |
| `nvtx.range_pop()` | NVTX marker end | For Nsight |

---

## Appendix C: Critical Review Corrections Applied

| Issue | Severity | Correction |
|-------|----------|------------|
| t-test missing `times` array | **BUG** | Fixed: Added `'times': times` to results dict |
| Missing GPU clock locking | HIGH | Added `setup_gpu_for_benchmarking()` method |
| Missing temperature-based cooling | HIGH | Added `target_temperature` parameter with wait loop |
| Missing Per-Component Attribution | MEDIUM | Added Section with `ProfiledTransformerBlock`, `ComponentProfiledModel`, `extract_component_stats()` |
| Missing Chrome Trace details | MEDIUM | Added schedule, flame graph export, keyboard shortcuts, interpretation guide |
| Missing NVIDIA Nsight integration | MEDIUM | Added full Section 14 with NVTX markers and CLI commands |
| Missing Industry Practices | MEDIUM | Added Section 4 with Meta/ByteDance benchmarks |
| Missing Profiler Overhead table | MEDIUM | Added overhead guidance table |
| Missing Deployment Recommendations | MEDIUM | Added Dev/Prod/CI phase recommendations |
| cache_load_factor clarification | LOW | Added Proposal 5 to test 0.2 as potential improvement |
| Missing ScratchPipe look-forward cache | MINOR | Added as Proposal 10 in Tier 3 |
| Missing SDPAParams pre-flight check | MINOR | Added to `detect_sdpa_backend()` method |
| Missing `measure_launch_overhead_separately()` | MINOR | Added to benchmark class |
| Missing KeyedJaggedTensor memory formula | MINOR | Added Section 3.1 with formula and example |

### Verified File Paths

| Config | File | Line |
|--------|------|------|
| cache_load_factor=0.1 | `flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py` | 1119 |
| cache_load_factor=0.1 | `flattened_main_feed_mtml_model_keeper_may_cargo_roo_hstu_datafm.py` | 1301 |
| prefetch_pipeline=False | `configs.py` | 513 |
| weights dtype (FP32) | `model_family.py` | 1600 |
| CacheAlgorithm.LFU | `*_prod_*.py` | 1114-1124 |

---

## References

### Research Documents
- `result_Q6_merged.md`: Comprehensive embedding optimization and kernel profiling research

### Key Papers
- Mixed-Dimension Embeddings: Facebook 2020
- TT-Rec: Meta 2021, FBTT-Embedding library
- ROBE: Random Offset Block Embedding for RecSys
- Monolith: ByteDance 2022
- Persia: ByteDance 2022, 100 Trillion Parameters
- FlashNeuron: SSD-enabled large-batch training (USENIX FAST'21)
- Deep Gradient Compression: arXiv:1712.01887

### Official Documentation
- FBGEMM GPU: https://github.com/pytorch/FBGEMM
- TorchRec: https://pytorch.org/torchrec/
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html
- SDPA Tutorial: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
- NVIDIA Nsight Systems: https://developer.nvidia.com/nsight-systems

---

*Document generated: 2026-01-30*
*Document refactored: 2026-01-30 (comprehensive revision with kernel profiling integration)*
*Document updated: 2026-01-30 (feedback corrections: t-test bug fix, GPU clock locking, temperature cooling, Nsight integration, Chrome trace details, profiler overhead, industry practices)*
*Document updated: 2026-01-30 (minor additions: ScratchPipe cache, SDPAParams pre-flight, measure_launch_overhead_separately, KJT memory formula)*
*Research source: result_Q6_merged.md (Embedding Table Optimization and Kernel-Level Profiling)*
*Focus: Memory optimization + validation infrastructure for MTML ROO training*
