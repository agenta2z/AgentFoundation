# PyTorch kernel-level profiling for GPU optimization validation

Kernel-level profiling in PyTorch requires combining **torch.profiler** for operator-level analysis, **CUDA events** for precise timing, and **NVTX annotations** for deeper integration with NVIDIA tools. This guide provides production-ready patterns for validating GPU optimizations, counting kernel launches, detecting SDPA backends, and structuring A/B benchmarks—everything needed to enhance a benchmark script with kernel-level analysis capabilities.

## Counting kernel launches and analyzing kernel distribution

The `torch.profiler.profile()` context manager with `ProfilerActivity.CUDA` captures kernel-level information. Each `FunctionEvent` contains a `kernels` attribute—a list of `Kernel` namedtuples representing device kernels launched by that operation.

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

**Key profiler configuration options** determine kernel visibility: `ProfilerActivity.CUDA` captures on-device kernels (names like `void at::native::*` or `ampere_sgemm_*`), while `ProfilerActivity.CPU` shows PyTorch operators (`aten::*`) and launch events (`cudaLaunchKernel`). The `with_stack=True` option adds source location tracking but incurs overhead—use for investigation, disable for timing benchmarks.

## Separating kernel launch overhead from execution time

CUDA operations are **asynchronous**—the CPU queues work and returns immediately. Without synchronization, naive timing measures only launch overhead (~5-20µs per kernel), not actual GPU compute time. CUDA events measure time **on the GPU stream**, naturally hiding launch latency.

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

**Critical synchronization rules**: always call `torch.cuda.synchronize()` before `elapsed_time()` or you'll get a RuntimeError. Never synchronize inside hot loops—it forces CPU-GPU serialization and destroys throughput. Place synchronization only at measurement boundaries.

## Detecting and profiling SDPA backend selection

PyTorch's `scaled_dot_product_attention` automatically selects between **Flash Attention**, **Memory-Efficient Attention**, **cuDNN**, and **Math fallback** based on inputs, hardware, and configuration. The new `torch.nn.attention.sdpa_kernel()` context manager controls backend selection.

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import profile, ProfilerActivity

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

def force_and_verify_sdpa_backend(query, key, value, backend: SDPBackend):
    """Force a specific backend and verify it's being used."""
    with sdpa_kernel(backend):
        detected, prof = profile_sdpa_and_identify_backend(query, key, value)
        print(f"Forced {backend.name}, detected: {detected}")
        return detected == backend.name.replace("_", " ").title() or \
               (backend == SDPBackend.MATH and "Math" in detected)
```

**Backend requirements**: Flash Attention requires SM80+ GPUs (Ampere, Ada, Hopper) and FP16/BF16 dtypes with head dimension ≤128. Memory-Efficient works on SM50+ with head dimensions divisible by 4 (FP32) or 8 (FP16). The cuDNN backend requires `TORCH_CUDNN_SDPA_ENABLED=1` environment variable and cuDNN 9.0+.

## A/B benchmarking structure for kernel optimizations

Production-quality A/B benchmarks require careful attention to warmup, statistical rigor, thermal throttling, and GPU state management. The coefficient of variation (CV) should be below **5%** for reliable benchmarks.

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

**Warmup requirements**: Run **10-20 iterations** minimum before timing. For `torch.compile`, the first call triggers compilation which can take seconds—warmup must complete this. Set `torch.backends.cudnn.benchmark = False` during benchmarks to prevent algorithm selection variance.

## Per-component kernel attribution with record_function

The `record_function` context manager creates labeled regions that appear in profiler output and Chrome traces, enabling attribution of kernel time to specific model components.

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

def profile_model_components(model, inputs, export_trace=True):
    """Profile model with per-component attribution."""

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with record_function("model_forward"):
            output = model(inputs)
        torch.cuda.synchronize()

    # Print hierarchical timing breakdown
    print("\n=== Per-Component Timing ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Group by stack to see source locations
    print("\n=== By Source Location ===")
    print(prof.key_averages(group_by_stack_n=3).table(
        sort_by="self_cuda_time_total", row_limit=10))

    if export_trace:
        prof.export_chrome_trace("component_trace.json")
        print("\nExported trace to component_trace.json")
        print("View at: chrome://tracing or https://ui.perfetto.dev")

    return prof
```

**Extracting component-level statistics programmatically**:

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

## Chrome trace export and analysis workflow

The Chrome trace format provides visual timeline analysis of kernel execution, CPU-GPU overlap, and memory operations.

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

**Interpreting Chrome traces**: Open `chrome://tracing` in Chrome or use Perfetto (https://ui.perfetto.dev). The trace shows:
- **CPU row** (top): PyTorch operators, `record_function` labels, `cudaLaunchKernel` calls
- **CUDA row** (bottom): Actual GPU kernel execution
- **Flow arrows**: Connect CPU launch events to GPU kernels
- **Gaps**: Indicate CPU-GPU synchronization points or idle time

**Keyboard shortcuts**: `w`/`s` to zoom in/out, `a`/`d` to pan left/right, click events for details.

## Integrating with NVIDIA Nsight for deep kernel analysis

For detailed kernel analysis (occupancy, memory bandwidth, Tensor Core utilization), integrate with NVIDIA Nsight Systems and Nsight Compute using NVTX markers.

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

**Nsight Systems CLI**:
```bash
nsys profile -w true -t cuda,nvtx,osrt -s none \
    --capture-range=cudaProfilerApi \
    -o profile_report python benchmark.py
```

**Nsight Compute** (for specific kernel analysis):
```bash
ncu --nvtx --nvtx-include "forward/" \
    -k regex:ampere_sgemm \
    --set full -o kernel_analysis python benchmark.py
```

## Complete production benchmark template

This template combines all techniques into a comprehensive benchmark script structure:

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

## Key API reference for kernel profiling

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

The combination of these techniques enables comprehensive validation of GPU optimizations—from counting kernel launches and measuring execution time to verifying SDPA backend selection and attributing time to specific model components. The Chrome trace export provides visual confirmation of kernel fusion results, while statistical methods ensure benchmark results are meaningful and reproducible.
