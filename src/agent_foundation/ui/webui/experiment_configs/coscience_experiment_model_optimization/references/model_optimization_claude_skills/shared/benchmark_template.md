# Benchmark Coding Template

Reusable template and guidelines for creating PyTorch benchmark scripts.

## Benchmark Script Template

Use this template as a starting point for any benchmark script:

```python
#!/usr/bin/env python3
# pyre-strict
"""
Benchmark script for <ModuleName> forward and backward passes.

Usage:
    # Standard run (profiler runs by default, generates trace.json):
    buck run @mode/opt //path/to/tests:benchmark_module

    # Skip profiler for quick debugging:
    buck run @mode/opt //path/to/tests:benchmark_module -- --skip-profile

    # Custom parameters:
    buck run @mode/opt //path/to/tests:benchmark_module -- \
        --batch-size 8 --max-seq-len 256 --num-iterations 50
"""

import argparse
import logging
import time
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.profiler import profile, record_function

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def create_benchmark_inputs(
    batch_size: int,
    max_seq_len: int,
    embedding_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Create inputs for benchmarking the target module.
    Returns a dict with all required input tensors.

    CUSTOMIZE THIS FUNCTION:
    - Match the exact input signature of the target module's forward()
    - Use correct tensor shapes from Phase 1 code analysis
    - Handle special formats (jagged tensors, offsets, etc.)
    """
    # Example: Create mock inputs matching the module's forward() signature
    x = torch.randn(
        (batch_size, max_seq_len, embedding_dim),
        device=device,
        dtype=dtype,
    ).requires_grad_(True)

    return {
        "input_tensor": x,
        # Add other required inputs based on code analysis
    }


def create_model(
    # Add configuration parameters as needed
) -> nn.Module:
    """
    Create and return the model/module to benchmark.

    CUSTOMIZE THIS FUNCTION:
    - Import the target module
    - Use correct constructor arguments
    - Match production configuration
    """
    # Example:
    # from path.to.module import TargetModule
    # return TargetModule(config=...)
    raise NotImplementedError("Customize this function for your target module")


def benchmark_forward_backward(
    model: nn.Module,
    inputs: Dict[str, Any],
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> Dict[str, float]:
    """
    Benchmark forward and backward passes separately.
    Returns timing statistics for both.
    """
    model.train()

    # Warmup - critical for accurate GPU timing
    logger.info(f"Running {num_warmup} warmup iterations...")
    for _ in range(num_warmup):
        # Clone inputs to avoid in-place modification issues
        x = inputs["input_tensor"].detach().clone().requires_grad_(True)
        outputs = model(x)
        loss = outputs.sum()
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark forward pass
    logger.info(f"Running {num_iterations} forward iterations...")
    forward_times = []
    for _ in range(num_iterations):
        x = inputs["input_tensor"].detach().clone().requires_grad_(True)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with record_function("forward"):
            outputs = model(x)

        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)

    # Benchmark backward pass
    logger.info(f"Running {num_iterations} backward iterations...")
    backward_times = []
    for _ in range(num_iterations):
        x = inputs["input_tensor"].detach().clone().requires_grad_(True)
        outputs = model(x)
        loss = outputs.sum()

        torch.cuda.synchronize()
        start = time.perf_counter()

        with record_function("backward"):
            loss.backward()

        torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)

    # Compute statistics
    forward_times_t = torch.tensor(forward_times)
    backward_times_t = torch.tensor(backward_times)

    return {
        "forward_mean_ms": forward_times_t.mean().item() * 1000,
        "forward_std_ms": forward_times_t.std().item() * 1000,
        "forward_min_ms": forward_times_t.min().item() * 1000,
        "forward_max_ms": forward_times_t.max().item() * 1000,
        "backward_mean_ms": backward_times_t.mean().item() * 1000,
        "backward_std_ms": backward_times_t.std().item() * 1000,
        "backward_min_ms": backward_times_t.min().item() * 1000,
        "backward_max_ms": backward_times_t.max().item() * 1000,
    }


def run_profiler(
    model: nn.Module,
    inputs: Dict[str, Any],
    output_dir: str = "./profiler_output",
) -> str:
    """
    Run torch.profiler to get detailed performance breakdown.
    Exports trace JSON and prints summary tables.

    MANDATORY: This function must always be called to generate trace.json
    for use in the optimization phase. The trace file enables:
    - Identifying top time-consuming operations
    - Mapping expensive ops to source code lines
    - Understanding the call hierarchy
    - Guiding optimization decisions

    Args:
        output_dir: Directory for trace.json. Defaults to "./profiler_output".

    Returns:
        Path to the exported trace JSON file.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    model.train()

    logger.info("Running profiler (3 iterations)...")

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(3):
            x = inputs["input_tensor"].detach().clone().requires_grad_(True)

            with record_function("forward_pass"):
                outputs = model(x)

            with record_function("backward_pass"):
                loss = outputs.sum()
                loss.backward()

            torch.cuda.synchronize()

    # Print summary tables
    print("\n" + "=" * 80)
    print("PROFILER SUMMARY - CUDA Time")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    print("\n" + "=" * 80)
    print("PROFILER SUMMARY - CPU Time")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    # Export trace JSON for analysis - MANDATORY for optimization phase
    trace_path = f"{output_dir}/trace.json"
    prof.export_chrome_trace(trace_path)
    logger.info(f"Trace JSON exported to: {trace_path}")
    print(f"\n*** TRACE FILE: {trace_path} ***")
    print("Use this trace file in Phase 3 (optimization) for profiler-guided changes.")

    # Print memory summary
    print("\n" + "=" * 80)
    print("MEMORY SUMMARY")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    return trace_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark <ModuleName>")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num-warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--num-iterations", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--skip-profile", action="store_true", help="Skip profiler (NOT recommended - trace.json is needed for optimization)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="./profiler_output", help="Profiler output directory")
    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Print CUDA diagnostics
    if args.device == "cuda":
        logger.info("CUDA diagnostics:")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        logger.info(f"  Device name: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"  Compute capability: {props.major}.{props.minor}")

    device = torch.device(args.device)
    dtype = torch.bfloat16

    # Log configuration
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max sequence length: {args.max_seq_len}")

    # Create model
    logger.info("Creating model...")
    model = create_model()  # Customize create_model() for your target
    model = model.to(device=device, dtype=dtype)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create inputs
    logger.info("Creating inputs...")
    inputs = create_benchmark_inputs(
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        embedding_dim=args.embedding_dim,
        dtype=dtype,
        device=device,
    )

    # Run benchmark
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING BENCHMARK")
    logger.info("=" * 80)

    timing_stats = benchmark_forward_backward(
        model=model,
        inputs=inputs,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
    )

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Forward pass:  {timing_stats['forward_mean_ms']:.3f} ± {timing_stats['forward_std_ms']:.3f} ms")
    print(f"  min: {timing_stats['forward_min_ms']:.3f} ms, max: {timing_stats['forward_max_ms']:.3f} ms")
    print(f"Backward pass: {timing_stats['backward_mean_ms']:.3f} ± {timing_stats['backward_std_ms']:.3f} ms")
    print(f"  min: {timing_stats['backward_min_ms']:.3f} ms, max: {timing_stats['backward_max_ms']:.3f} ms")
    print(f"Total:         {timing_stats['forward_mean_ms'] + timing_stats['backward_mean_ms']:.3f} ms")
    print("=" * 80)

    # Run profiler to generate trace.json (MANDATORY for optimization phase)
    # Use --skip-profile only for quick iteration during debugging
    if not args.skip_profile:
        trace_path = run_profiler(model=model, inputs=inputs, output_dir=args.output_dir)
        print(f"\n*** Trace file for optimization: {trace_path} ***")
    else:
        logger.warning("Profiler skipped. trace.json will NOT be generated.")
        logger.warning("Re-run without --skip-profile before starting optimization phase.")


if __name__ == "__main__":
    main()
```

---

## BUCK Configuration

### Standard Python Binary Target

```python
python_binary(
    name = "benchmark_module",
    srcs = ["benchmark_module.py"],
    keep_gpu_sections = True,  # REQUIRED for GPU benchmarks
    main_function = "path.to.tests.benchmark_module.main",
    deps = [
        "//caffe2:torch",
        # Add dependencies to the target module
        "//path/to:module",
    ],
)
```

### Important BUCK Notes

| Setting | Requirement | Why |
|---------|-------------|-----|
| `keep_gpu_sections = True` | **Mandatory** | Preserves CUDA kernels; without it GPU sections may be stripped |
| `@mode/opt` | **Required for benchmarking** | Debug mode adds overhead and disables optimizations |
| `gpu_python_binary` | Consider for complex GPU deps | Alternative to `python_binary` for heavy GPU usage |

---

## Key Best Practices

### 1. Input Size Limits for Fast Benchmarking

**CRITICAL**: Use small default input sizes to enable fast benchmark iteration. Production-scale inputs make benchmarking slow and impractical.

| Parameter | Recommended Default | Maximum for Quick Benchmarks | Notes |
|-----------|-------------------|------------------------------|-------|
| `batch_size` | 4 | 8 | Larger batches increase memory and compute time proportionally |
| `max_seq_len` | 256 | 512 | Attention is O(n²); doubling length quadruples time |
| `num_targets` | 10 | 50 | Keep small relative to sequence length |
| `embedding_dim` | 256 | 256 | Match production but don't increase for benchmarks |
| `num_layers` | 4 | 6 | Each layer adds linear overhead |

**Why This Matters:**
- ✅ **Good**: `batch_size=4, max_seq_len=256, num_targets=10` → ~5-20ms per iteration
- ❌ **Bad**: `batch_size=32, max_seq_len=3072, num_targets=1024` → 700+ ms per iteration

**Scaling Guidelines:**
- For attention-based models, compute scales O(batch × seq_len²)
- Keep total benchmark iteration time under 50ms for rapid experimentation
- Use `--batch-size`, `--max-seq-len` CLI flags to test at production scale when needed

**Example Default Args (Good):**
```python
parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
parser.add_argument("--max-seq-len", type=int, default=256, help="Max sequence length")
parser.add_argument("--num-targets", type=int, default=10, help="Number of targets")
```

**Example Default Args (Bad - Avoid):**
```python
# DON'T use production-scale defaults - makes iteration slow
parser.add_argument("--batch-size", default=32, ...)  # Too large
parser.add_argument("--num-targets", default=1024, ...)  # Too large
parser.add_argument("--base-seq-len", default=2048, ...)  # Too large
```

### 2. Warmup Iterations
GPU kernels need warmup for accurate timing. Always run 5+ warmup iterations before measuring.

### 3. Clone Inputs Each Iteration
Use `detach().clone().requires_grad_(True)` to break computation graphs between iterations:
```python
x = inputs["input_tensor"].detach().clone().requires_grad_(True)
```

### 4. CUDA Synchronization
Required before/after timing for accurate GPU measurements:
```python
torch.cuda.synchronize()
start = time.perf_counter()
# ... operation ...
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

### 5. Record Functions
Use `record_function()` to label operations for profiler visibility:
```python
with record_function("forward"):
    outputs = model(x)
```

### 6. Compute Statistics
Report mean, std, min, max for reliable comparisons - single measurements are not reliable.

### 7. Log CUDA Diagnostics
Device name and compute capability help with reproducibility.

### 8. Export Trace JSON
The trace file enables detailed analysis in chrome://tracing or other tools.

---

## Debugging Guidelines

**CRITICAL RULES**:
1. **NEVER modify the source code being tested**
2. **ONLY modify the benchmark tool code**
3. **Focus on mocking input logic**

### Common Issues and Fixes

| Issue | Fix Location | Solution |
|-------|--------------|----------|
| Shape mismatch | `create_benchmark_inputs()` | Adjust tensor dimensions |
| Wrong dtype | `create_benchmark_inputs()` | Change `dtype` parameter |
| Missing inputs | `create_benchmark_inputs()` | Add required fields to return dict |
| Device mismatch | `create_benchmark_inputs()` | Ensure all tensors on same device |
| Constructor error | `create_model()` | Fix constructor arguments |
| Import error | BUCK deps | Add missing dependencies |
| Jagged tensor | `create_benchmark_inputs()` | Create proper jagged/nested format |

### Debugging Workflow

```
Error occurs
    │
    ▼
Identify error type
    │
    ├─► Shape error ──► Fix create_benchmark_inputs()
    ├─► Type error ───► Fix dtype in create_benchmark_inputs()
    ├─► Import error ─► Fix BUCK dependencies
    ├─► Init error ───► Fix create_model()
    │
    ▼
Retry (max 2-3 attempts per error type)
    │
    ▼
If still failing ──► Ask user for help
```

### Max Attempts Before Asking User

| Error Type | Max Self-Fix Attempts |
|------------|----------------------|
| Import errors | 2 |
| Shape mismatches | 2 |
| BUCK build failures | 2 |
| CUDA errors | 1 |
| Model initialization | 1 |

---

## File Naming and Location

| Item | Convention | Example |
|------|------------|---------|
| Directory | `tests/` under source | `/path/to/module/tests/` |
| Filename | `benchmark_<module>.py` | `benchmark_attention.py` |
| BUCK target | Same as filename | `benchmark_attention` |

---

## Run Commands

```bash
# Basic run (always use @mode/opt) - includes profiler by default
# Generates trace.json for use in optimization phase
CUDA_VISIBLE_DEVICES=[gpu] buck run @mode/opt //path/to/tests:benchmark_module

# Skip profiler for quick debugging iterations (NOT recommended for final baseline)
CUDA_VISIBLE_DEVICES=[gpu] buck run @mode/opt //path/to/tests:benchmark_module -- --skip-profile

# Custom parameters
CUDA_VISIBLE_DEVICES=[gpu] buck run @mode/opt //path/to/tests:benchmark_module -- \
    --batch-size 8 --max-seq-len 256 --num-iterations 50

# Custom output directory for trace.json
CUDA_VISIBLE_DEVICES=[gpu] buck run @mode/opt //path/to/tests:benchmark_module -- \
    --output-dir /path/to/traces
```

**Important**: The profiler runs by default to generate `trace.json`. This trace file is **mandatory** for the optimization phase (Phase 3). Only use `--skip-profile` during debugging when you need faster iteration.

---

## Profiling Best Practices

### CRITICAL: Profiler Iteration Limits

**The PyTorch profiler can cause CPU max usage and process hang if misused.** Follow these rules strictly:

| Rule | Requirement | Why |
|------|-------------|-----|
| **Max profiled iterations** | **2-3 iterations only** | Profiler stores detailed timing, stack traces, and shape info for every operation. More iterations = exponential memory growth |
| **Separate from benchmark** | Run profiler in dedicated function | Never wrap full benchmark loop (warmup + benchmark iters) in profiler context |
| **Clone inputs each iteration** | `detach().clone().requires_grad_(True)` | Prevents computation graph accumulation across iterations |

### What Goes Wrong Without These Rules

**Bad Pattern (causes CPU hang):**
```python
# DON'T DO THIS - wraps 30 iterations in profiler
with torch_profile(...) as prof:
    run_benchmark(warmup=5, benchmark=25)  # 30 iterations profiled!
```

**Problems:**
1. **Memory explosion**: Profiler data accumulates for 30 full forward+backward passes
2. **CPU-intensive post-processing**: `prof.key_averages()` must aggregate thousands of recorded operations
3. **Graph accumulation**: Without input cloning, computation graphs grow across iterations
4. **Result**: Process appears stuck at 100% CPU, may take 10+ minutes or OOM

### Correct Profiler Implementation

**Good Pattern:**
```python
def run_profiler(model, inputs, num_iters=3):
    """Run profiler with only 2-3 iterations."""
    with torch_profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(num_iters):  # Only 2-3 iterations!
            # Clone inputs to break computation graph
            x = inputs["input"].detach().clone().requires_grad_(True)
            outputs = model(x)
            loss = outputs.sum()
            loss.backward()
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    # Run benchmark first WITHOUT profiling
    timing_stats = benchmark_forward_backward(model, inputs, ...)

    # Print results
    print(f"Forward: {timing_stats['forward_mean_ms']:.3f} ms")

    # Run profiler by default to generate trace.json (skip with --skip-profile)
    if not args.skip_profile:
        run_profiler(model, inputs, num_iters=3)
```

### Input Cloning Helper

For complex input dictionaries, use a helper function:
```python
def clone_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Clone inputs with fresh tensors to break computation graph."""
    cloned = dict(inputs)

    # Clone main input tensor
    cloned["input_tensor"] = inputs["input_tensor"].detach().clone().requires_grad_(True)

    # Clone nested payloads if present
    if "payloads" in cloned:
        cloned["payloads"] = dict(inputs["payloads"])
        for key in ["embeddings", "nro_embeddings"]:
            if key in cloned["payloads"]:
                cloned["payloads"][key] = (
                    inputs["payloads"][key].detach().clone().requires_grad_(True)
                )

    return cloned
```

### Profiler Checklist

Before implementing profiler support in a benchmark:

- [ ] Profiler runs in separate function from benchmark loop
- [ ] Profiler limited to 2-3 iterations max
- [ ] Inputs are cloned each profiled iteration with `detach().clone()`
- [ ] `torch.cuda.synchronize()` called after each profiled iteration
- [ ] Benchmark runs first, profiler runs after (not instead of) benchmark
