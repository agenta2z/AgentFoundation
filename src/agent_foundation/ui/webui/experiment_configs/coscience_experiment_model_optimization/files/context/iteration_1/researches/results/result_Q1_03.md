# Optimizing PyTorch tensor indexing for large-scale recommendation models

Native slice indexing eliminates kernel launches entirely—`torch.arange() + index_select()` patterns launch **2+ CUDA kernels** while `x[:, start::step, :]` launches **zero**, operating purely on metadata. For a 5.1B parameter transformer with 65+ prediction tasks, this kernel reduction directly translates to reduced latency. However, the backward pass introduces complexity: strided tensors from slicing can cause **8x worse memory bandwidth** utilization, making `.contiguous()` placement critical for training performance.

## Kernel launch reduction is dramatic but context-dependent

The fundamental difference between these patterns lies in their implementation:

| Operation | Kernel Launches | Memory Allocation |
|-----------|-----------------|-------------------|
| `x[:, start::step, :]` (native slice) | **0** | None (view only) |
| `torch.index_select(x, dim, idx)` | **1** (gather kernel) | New output tensor |
| `torch.arange() + index_select()` | **2+** (arange + gather) | Index tensor + output |

Native slicing with steps creates a **view** by modifying only stride metadata and storage offset—no data movement occurs. PyTorch's official documentation confirms: "basic indexing returns views, while advanced indexing returns a copy." When you write `x[:, 0::2, :]`, PyTorch simply doubles the stride value for dimension 1 and adjusts the storage offset.

The `torch.arange() + index_select()` pattern, by contrast, first launches a kernel to generate the index tensor (`arange`), then launches the `indexSelectLargeIndex` or `indexSelectSmallIndex` kernel from `aten/src/ATen/native/cuda/Indexing.cu`. Each CUDA kernel launch incurs **5-15 microseconds** of overhead—negligible for large operations but significant when called repeatedly across 65+ prediction tasks.

## Inductor's combo_kernels doesn't automatically fuse slice operations

The `torch._inductor.config.combo_kernels = True` setting enables **horizontal fusion** of independent operations with no data dependencies into single kernels. However, this primarily benefits reduction operations (sum, max, min) and pointwise operations on different tensors—not slice-to-gather transformations.

Critically, inductor treats slices and index_select fundamentally differently in its IR:
- **Slices**: Lowered to `ReinterpretView` nodes that share underlying storage—essentially free operations that can be "inlined" into downstream computations
- **index_select/gather**: Creates explicit buffers with computed indices, establishing memory dependencies that may block fusion

With `allow_buffer_reuse = False` in your configuration, intermediate tensors persist longer, which can increase memory pressure but ensures deterministic behavior for debugging. The key insight is that combo_kernels won't convert your `index_select` patterns to slices—you must make that change at the source level.

Fusion decisions in inductor's scheduler evaluate candidates based on shared memory dependencies, matching loop dimensions, and memory savings thresholds. Operations with different iteration ranges cannot fuse, and reduction operations with empty ranges after full reduction prevent fusion entirely.

## Memory savings from eliminating intermediate index tensors

The `torch.arange()` call creates a 1D tensor of indices that lives until garbage collected. For your pattern selecting every Nth element across multiple dimensions:

```python
# Old pattern: allocates intermediate tensor
indices = torch.arange(0, seq_len, step, device='cuda')  # Allocates seq_len/step * 8 bytes
result = x.index_select(dim=1, index=indices)  # Allocates full output tensor

# New pattern: zero intermediate allocation
result = x[:, ::step, :]  # View only—no allocation until .contiguous()
```

For a tensor of shape `(batch=64, seq=2048, hidden=4096)` with `step=2`, the index tensor consumes **8KB** (1024 int64 values). While small individually, across 65+ tasks this accumulates. More significantly, `index_select` allocates the **full output tensor** regardless of whether downstream operations need contiguous data—the native slice defers this allocation until explicitly requested.

## When to add .contiguous() after slicing

The decision matrix depends on what operations follow the slice and whether you're training or doing inference:

**Always add `.contiguous()` when:**
- Passing to **cuDNN operations** (convolutions, RNNs, batch normalization)
- Before **`.view()` calls** (required—will error otherwise)
- Feeding into **custom Triton/CUDA kernels** expecting contiguous memory
- The tensor enters **attention mechanisms** with matmul-heavy computation
- Tensor will be used in **≥3 subsequent operations**

**Skip `.contiguous()` when:**
- Tensor is used once in a simple pointwise operation
- Next operation copies anyway (`.clone()`, serialization)
- Working with small tensors under ~100KB
- During **inference-only** with simple feed-forward paths

For backward pass optimization specifically, the heuristic is: if the sliced tensor participates in operations where gradients flow through matrix multiplications or reductions, make it contiguous. NVIDIA benchmarks show strided memory access achieves only **12.5% of contiguous bandwidth** (32 vs 4 memory sectors per request).

```python
# Recommended pattern for training
def efficient_slice(x, step):
    sliced = x[:, ::step, :]
    # Make contiguous if tensor will be used in compute-heavy backward
    if x.requires_grad and x.numel() > 100_000:
        sliced = sliced.contiguous()
    return sliced
```

## Forward pass cost versus backward pass speedup tradeoffs

The `.contiguous()` call copies data, adding forward pass latency proportional to tensor size. The payoff comes from **cumulative backward efficiency**:

| Tensor Size | Forward Cost | Backward Benefit | Recommendation |
|-------------|--------------|------------------|----------------|
| <100KB | Overhead dominates | Negligible | Skip `.contiguous()` |
| 100KB–10MB | Moderate | Measurable | Profile if in hot path |
| 10MB–100MB | Amortized quickly | Significant (**~2x matmul speedup**) | Use `.contiguous()` |
| >100MB | One-time cost | Critical for bandwidth | Always use |

Benchmark data from PyTorch issues shows non-contiguous matmul takes **~0.23s vs ~0.12s** for contiguous—nearly 2x slower. For your 5.1B parameter model, backward passes through transformer attention layers will dominate training time, making the forward-pass copy cost negligible by comparison.

The critical exception: **avoid regular indexing with duplicate indices** in backward passes. GitHub issue #41162 documents **20x slowdowns** when gradients accumulate at repeated index positions. If your recommendation model's task heads share embeddings or features, use `torch.index_select` or `torch.gather` for those specific operations even when native slicing works elsewhere.

## Comprehensive benchmark methodology with torch.profiler

The following profiling setup measures kernel launches, memory allocation, and backward pass behavior:

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

def benchmark_indexing_patterns(batch_size=64, seq_len=2048, hidden=4096, step=2):
    """Compare kernel counts and timing for indexing patterns."""
    device = 'cuda'
    x = torch.randn(batch_size, seq_len, hidden, device=device, requires_grad=True)

    # Pattern 1: torch.arange + index_select
    def pattern_arange_select(tensor):
        indices = torch.arange(0, tensor.size(1), step, device=device)
        return torch.index_select(tensor, dim=1, index=indices)

    # Pattern 2: Native slicing
    def pattern_native_slice(tensor):
        return tensor[:, ::step, :]

    # Pattern 3: Native slicing + contiguous
    def pattern_slice_contiguous(tensor):
        return tensor[:, ::step, :].contiguous()

    patterns = [
        ("arange+index_select", pattern_arange_select),
        ("native_slice", pattern_native_slice),
        ("slice+contiguous", pattern_slice_contiguous),
    ]

    for name, pattern_fn in patterns:
        # Warmup (critical for accurate CUDA timing)
        for _ in range(5):
            y = pattern_fn(x)
            (y.sum()).backward()
            x.grad = None
        torch.cuda.synchronize()

        # Profile forward + backward
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with record_function("forward"):
                y = pattern_fn(x)
            with record_function("backward"):
                y.sum().backward()
            torch.cuda.synchronize()

        print(f"\n{'='*60}")
        print(f"Pattern: {name}")
        print(f"{'='*60}")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=15
        ))

        # Count CUDA kernel launches
        cuda_events = [e for e in prof.key_averages() if e.cuda_time > 0]
        total_calls = sum(e.count for e in cuda_events)
        print(f"\nTotal CUDA kernel calls: {total_calls}")

        # Export for detailed analysis
        prof.export_chrome_trace(f"trace_{name}.json")
        x.grad = None

# Run with environment variable for meaningful kernel names
# TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 python benchmark.py
```

### Profiling backward pass kernel behavior specifically

To isolate backward pass performance, use gradient hooks that inject profiler labels:

```python
def profile_backward_kernels(model, sample_input):
    """Profile backward pass with per-layer kernel attribution."""

    def make_backward_hook(name):
        def hook(module, grad_input, grad_output):
            with record_function(f"backward_{name}"):
                pass  # Hook runs during backward
        return hook

    # Register hooks on all modules
    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_full_backward_hook(make_backward_hook(name)))

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        schedule=torch.profiler.schedule(
            skip_first=2,  # Skip compilation steps
            wait=1,
            warmup=2,
            active=3,
        ),
    ) as prof:
        for step in range(8):
            output = model(sample_input)
            loss = output.sum()
            loss.backward()
            prof.step()

    # Cleanup hooks
    for hook in hooks:
        hook.remove()

    return prof
```

### Interpreting profiler output for kernel fusion verification

When analyzing traces, look for these indicators:

```python
def analyze_fusion(prof):
    """Check if operations were fused by inductor."""
    events = prof.key_averages()

    # Fused kernels have descriptive names with torch.compile
    fused_kernels = [e for e in events if 'triton_poi_fused' in e.key or 'fused' in e.key.lower()]

    # Count separate arange and index_select calls
    arange_calls = sum(e.count for e in events if 'arange' in e.key.lower())
    index_select_calls = sum(e.count for e in events if 'index_select' in e.key.lower())

    print(f"Fused kernel calls: {len(fused_kernels)}")
    print(f"Separate arange calls: {arange_calls}")
    print(f"Separate index_select calls: {index_select_calls}")

    # Memory analysis
    peak_memory = max(e.cuda_memory_usage for e in events if e.cuda_memory_usage)
    print(f"Peak CUDA memory delta: {peak_memory / 1e6:.2f} MB")
```

### Complete benchmark script with torch.compile comparison

```python
import torch
import torch._inductor.config as inductor_config
from torch.profiler import profile, ProfilerActivity, record_function

# Your inductor settings
inductor_config.combo_kernels = True
inductor_config.allow_buffer_reuse = False

class IndexingBenchmark:
    def __init__(self, batch=64, seq=2048, hidden=4096, step=2):
        self.device = 'cuda'
        self.x = torch.randn(batch, seq, hidden, device=self.device, requires_grad=True)
        self.step = step

    def method_arange_select(self, x):
        indices = torch.arange(0, x.size(1), self.step, device=self.device)
        return x.index_select(dim=1, index=indices)

    def method_native_slice(self, x):
        return x[:, ::self.step, :].contiguous()

    def benchmark_method(self, method, name, use_compile=False):
        if use_compile:
            method = torch.compile(method, mode="reduce-overhead")

        # Extended warmup for torch.compile
        warmup_iters = 10 if use_compile else 3
        for _ in range(warmup_iters):
            y = method(self.x)
            y.sum().backward()
            self.x.grad = None
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
        ) as prof:
            for _ in range(5):  # Average over multiple iterations
                with record_function(f"{name}_forward"):
                    y = method(self.x)
                with record_function(f"{name}_backward"):
                    y.sum().backward()
                self.x.grad = None
            torch.cuda.synchronize()

        # Extract metrics
        cuda_time = sum(e.cuda_time_total for e in prof.key_averages())
        kernel_count = sum(e.count for e in prof.key_averages() if e.cuda_time > 0)

        return {
            "name": name,
            "compiled": use_compile,
            "cuda_time_us": cuda_time,
            "kernel_launches": kernel_count,
            "profiler": prof
        }

# Run benchmarks
bench = IndexingBenchmark()
results = [
    bench.benchmark_method(bench.method_arange_select, "arange_select", False),
    bench.benchmark_method(bench.method_arange_select, "arange_select", True),
    bench.benchmark_method(bench.method_native_slice, "native_slice", False),
    bench.benchmark_method(bench.method_native_slice, "native_slice", True),
]

print("\n" + "="*70)
print(f"{'Method':<25} {'Compiled':<10} {'CUDA Time (us)':<15} {'Kernels':<10}")
print("="*70)
for r in results:
    print(f"{r['name']:<25} {str(r['compiled']):<10} {r['cuda_time_us']:<15.0f} {r['kernel_launches']:<10}")
```

## Key recommendations for your 5.1B parameter model

Given your specific context—transformer-based architecture with 65+ prediction tasks—these optimizations provide the highest impact:

1. **Replace all `torch.arange() + index_select()` patterns** with native slicing where indices follow regular step patterns. Expected reduction: 2 kernel launches → 0 per operation, multiplied across all tasks.

2. **Add `.contiguous()` strategically**: Place it after slicing operations that feed into attention layers or linear projections, but skip it for tensors only used in elementwise operations.

3. **Enable `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1`** when profiling to get meaningful kernel names in Chrome traces for fusion verification.

4. **Watch for duplicate indices**: If any prediction tasks share feature selections with overlapping indices, keep `index_select` for those specific cases to avoid the 20x backward slowdown.

5. **Profile before and after** using the methodology above, exporting Chrome traces to `chrome://tracing` or Perfetto for visual kernel fusion verification. Look for reduced kernel density in the timeline view as confirmation of successful optimization.
