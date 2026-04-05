# CUDA kernel fusion and launch optimization for H100 training

Kernel launch overhead of **5-10μs per operation** compounds dramatically in modern PyTorch training, where transformer layers execute 50+ operations per forward pass. On H100 GPUs, this creates a critical optimization opportunity: reducing kernel counts from hundreds to dozens can deliver **5-15% QPS improvements** in production training. The most effective techniques combine `torch.compile` with TorchInductor fusion (achieving **1.41x training speedup** on average), CUDA graphs for eliminating launch overhead entirely, and custom Triton kernels leveraging H100's Tensor Memory Accelerator for **59% memory throughput improvement**. DeepSpeed's production benchmarks demonstrate fused attention achieving **2.9x speedup** over unfused implementations, while LinkedIn's Liger Kernel reports **20% training throughput gains** with Triton-based fusion.

## Kernel launch overhead creates a hidden training bottleneck

NVIDIA benchmarks on V100 demonstrate the compounding effect clearly: a simple elementwise kernel executing in **2.9μs** incurs an effective per-kernel time of **9.6μs** when synchronization exposes launch overhead—a **3.3x penalty**. With CUDA graphs, this drops to 3.4μs, representing only 17% overhead versus 230% without graphs. The overhead stems from three components: GPU launch latency (~1-2μs for the GPU to begin execution), CPU wrapper overhead (~3-10μs for the CUDA API call), and implicit synchronizations triggered by operations like `tensor.item()` or conditional logic.

Modern training loops exhibit classic launch-bound symptoms that profiling tools readily identify. In Nsight Systems traces, look for gaps between consecutive kernel executions in the GPU timeline, situations where CPU API time dominates GPU execution time, and kernel durations under 10μs paired with high total step time. The diagnostic formula is straightforward: if `(total_step_time - sum(kernel_times)) / total_step_time > 0.3`, the workload is launch-bound. Meta's MAIProf analysis of production models found GPU idle exceeding **50% of training time** in unoptimized configurations, with SM utilization at just 9.1%.

H100 introduces architectural improvements that indirectly reduce launch impact. The **Tensor Memory Accelerator (TMA)** enables async data transfers with minimal CPU involvement, while Thread Block Clusters provide hardware-accelerated barriers across SMs. PCIe Gen5 doubles host-device communication bandwidth to 128 GB/s. However, the fundamental ~5-10μs baseline launch latency remains similar to A100—the key difference is H100's ability to better hide this latency through overlap and its faster kernels that shift the relative overhead balance.

## TorchInductor fusion delivers automatic kernel reduction

TorchInductor, PyTorch's default compiler backend for `torch.compile`, achieves **2.27x inference and 1.41x training geometric mean speedup** across 180+ real-world models on A100. The fusion pipeline works through several stages: TorchDynamo captures an FX graph via bytecode transformation, AOTAutograd generates forward and backward graphs, then 191 decomposition passes break complex operations into fuseable primitives before the scheduler identifies fusion opportunities.

The most impactful fusion patterns target elementwise operation sequences—chains like `mul → add → relu → sigmoid` that would otherwise launch four separate kernels collapse into one. Reduction patterns similarly benefit, with combinations like `(x * y).sum()` fusing into single kernels that avoid intermediate memory writes. For transformers, the critical fusion is scaled dot-product attention: PyTorch 2.2+ automatically dispatches to FlashAttention-2, which fuses the entire Q×K^T → softmax → dropout → ×V sequence while keeping intermediate attention scores in SRAM rather than writing to HBM, reducing memory complexity from O(N²) to O(N).

Key configuration options maximize fusion aggressiveness:

```python
import torch._inductor.config as config
config.max_autotune = True           # Profile multiple kernel configurations
config.epilogue_fusion = True        # Fuse pointwise ops into matmul templates
config.aggressive_fusion = True      # Fuse even without shared memory access
config.triton.cudagraphs = True      # Combine with CUDA graphs for launch reduction

model = torch.compile(model, mode="max-autotune", fullgraph=True)
```

The `fullgraph=True` parameter forces an error if graph breaks occur, preventing silent fallback to eager mode. Mode presets simplify configuration: `"default"` provides standard fusion, `"reduce-overhead"` enables CUDA graphs for launch elimination, and `"max-autotune"` performs exhaustive kernel search at the cost of longer compilation.

Fusion limitations arise with data-dependent control flow (which creates graph breaks), external library calls to cuBLAS/cuDNN (which remain as opaque nodes), and dependency conflicts where producer-consumer access patterns misalign. When these limitations block automatic fusion, manual intervention through custom Triton kernels becomes necessary.

## Loss computation consolidation reduces kernel counts dramatically

Multi-task training scenarios often spawn excessive kernel launches from separate loss computations. The target reduction from **221 to ~30 kernels** is achievable through systematic batching of similar loss types. The core technique uses `reduction='none'` to compute per-element losses, then applies custom weighting and reduction in vectorized operations.

For cross-entropy losses across multiple classification heads with the same number of classes, concatenating predictions and targets enables single-kernel computation:

```python
@torch.compile(mode="max-autotune", fullgraph=True)
def batched_cross_entropy(preds_list, targets_list, weights=None):
    # Batch all predictions: [total_samples, num_classes]
    preds_batched = torch.cat(preds_list, dim=0)
    targets_batched = torch.cat(targets_list, dim=0)

    # Single kernel for all losses
    loss_per_sample = F.cross_entropy(preds_batched, targets_batched, reduction='none')

    # Apply per-task weighting if needed
    if weights is not None:
        # Compute weighted average per original batch segment
        idx = 0
        weighted_sum = 0.0
        for i, pred in enumerate(preds_list):
            n = len(pred)
            weighted_sum += weights[i] * loss_per_sample[idx:idx+n].mean()
            idx += n
        return weighted_sum
    return loss_per_sample.mean()
```

Dynamic loss balancing adds another optimization layer. The uncertainty-based weighting from Kendall et al. learns task-specific precision parameters: `total_loss = sum(exp(-log_var[i]) * loss[i] + log_var[i])`. This not only balances gradients across tasks but expresses elegantly in fused form. The simpler "RLW" approach normalizes each loss by its detached value—`sum(loss / loss.detach())`—achieving automatic magnitude balancing without learned parameters.

Gradient accumulation introduces complications that require careful handling. When using mixed precision, a **single GradScaler must serve all losses**—using separate scalers corrupts the gradient accumulation math. The pattern requires scaling the combined loss once before backward, then calling `scaler.step()` and `scaler.update()` only on accumulation boundaries.

## Triton kernels unlock H100-specific performance

Triton's block-based programming model enables near-expert-level GPU performance with significantly reduced development time. Where CUDA requires 100-500+ lines for a GEMM kernel, Triton achieves equivalent functionality in 25-50 lines by abstracting away thread/warp management, memory coalescing, and Tensor Core dispatch. The performance gap narrows considerably: Triton achieves **80-95% of hand-tuned CUDA** for most workloads, with some specialized kernels exceeding cuBLAS due to better problem-specific optimization.

H100's Tensor Memory Accelerator represents the most impactful Hopper-specific optimization available in Triton. TMA provides a dedicated hardware unit for async bulk data transfer between global and shared memory, bypassing the register file entirely. PyTorch benchmarks demonstrate TMA-enabled Triton GEMM achieving **1.45 TB/s memory throughput** versus 910 GB/s without TMA—a 59% improvement. The API remains experimental but is accessible:

```python
@triton.jit
def kernel_with_tma(a_desc_ptr, block_m, block_k, ...):
    offs_am = tl.program_id(0) * block_m
    offs_k = 0
    # TMA descriptor handles address calculation and transfer
    a = tl._experimental_descriptor_load(
        a_desc_ptr, [offs_am, offs_k], [block_m, block_k], tl.float8e4nv
    )
```

Warp specialization divides warps into producer (data movement) and consumer (compute) roles, automatically enabled via autotune parameters: `num_consumer_groups=2` and `num_buffers_warp_spec=3`. This delivers **10-15% speedup** on FlashAttention and FP8 GEMM kernels by overlapping data loading with computation. The Tawa automatic warp specialization framework achieves **1.21x speedup** over baseline Triton by automatically applying this technique.

Integration with PyTorch training uses `torch.library.triton_op` for seamless autograd support:

```python
from torch.library import triton_op, wrap_triton

@triton_op("mylib::fused_gelu", mutates_args={})
def fused_gelu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    wrap_triton(gelu_kernel)[(triton.cdiv(n, 1024),)](x, out, n, BLOCK_SIZE=1024)
    return out

@fused_gelu.register_autograd
def backward(ctx, grad):
    x, = ctx.saved_tensors
    return grad * gelu_backward_impl(x)  # Uses another triton_op
```

## CUDA graphs eliminate launch overhead for static workloads

CUDA graphs record sequences of GPU operations during capture and replay them with a single `cudaGraphLaunch` call, eliminating Python/C++/driver overhead entirely. MLPerf training v1.0 demonstrated **~1.7x speedup** for Mask R-CNN and ~1.12x for BERT at scale using this technique. The approach works best for small batch sizes where launch overhead dominates and for static workloads where operation sequences don't change.

The capture process requires careful warmup to avoid capturing library initialization code:

```python
# Warmup on side stream (critical for cuBLAS/cuDNN initialization)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):  # Minimum 3 iterations; 11 for DDP
        optimizer.zero_grad(set_to_none=True)  # set_to_none is cleaner for graphs
        with torch.autocast('cuda', torch.float16):
            y = model(static_input)
            loss = loss_fn(y, static_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)
    static_loss = loss_fn(static_output, static_target)
    scaler.scale(static_loss).backward()
```

Critical constraints apply during capture: no dynamic tensor shapes (batch size and sequence length must be fixed), no CPU-GPU synchronization points, no data-dependent control flow, and no memory allocation via `cudaMalloc`. For training loops with variable-length sequences, the workaround involves pre-capturing multiple graphs for different shape buckets or padding to maximum length with attention masks.

`torch.compile(mode="reduce-overhead")` provides a higher-level alternative that manages CUDA graphs automatically, including re-recording for new shapes through CUDA Graph Trees. This approach handles many edge cases that manual graph management cannot, including better compatibility with gradient checkpointing (which fails under direct stream capture due to RNG state issues).

Memory implications are significant: NVIDIA allocates **~64 KB per kernel** in graph overhead (reduced in CUDA 12.4+). For a model with 100 kernel launches across 10 different batch sizes, this totals ~64 MB of graph overhead. Pool sharing via `torch.cuda.graph_pool_handle()` allows multiple graphs to share memory, and `torch.compile` handles this automatically.

## Production impact validates the optimization investment

DeepSpeed's deep fusion benchmarks quantify the kernel-level gains: Input LayerNorm + QKV GEMM + bias adds achieves **1.5x speedup** over unfused cuBLAS, attention computation with implicit matrix transformation reaches **2.9x**, and the intermediate FF + LayerNorm + bias + residual + GELU fusion hits **3.0x**. Their MoE kernel optimization delivers **over 6x reduction** in MoE-related latency, critical for mixture-of-experts training at scale.

LinkedIn's Liger Kernel project, built entirely in Triton, reports **20% multi-GPU training throughput improvement** with **60% memory reduction** from fused operations including RMSNorm, RoPE, SwiGLU, and FusedLinearCrossEntropy. The FusedLinearCrossEntropy kernel is particularly impactful, avoiding materialization of the full logit tensor for vocabulary projection.

Measurement methodology matters for validating improvements. Use CUDA events for accurate timing (not Python `time.time()`), warm up for 10+ iterations before measurement, run 50+ iterations taking the median, and profile with Nsight Systems to visualize the kernel timeline before and after optimization. Key metrics to track include kernel count per training step, achieved SM occupancy (target >50%), memory bandwidth utilization (H100 theoretical: 3.35 TB/s HBM3), and end-to-end throughput in samples/second.

The expected impact for a well-optimized training loop targeting 221→30 kernel reduction: **5-15% QPS improvement** from launch overhead elimination alone, with additional gains from improved memory bandwidth utilization and better Tensor Core occupancy. Complex models with many fuseable elementwise operations see the largest benefits, while already-optimized GEMM-dominated workloads see smaller relative gains.

## Conclusion: A layered optimization strategy maximizes training efficiency

The most effective approach layers three complementary techniques. First, apply `torch.compile(mode="max-autotune")` as the baseline—this provides automatic fusion, CUDA graph integration, and shape specialization with minimal code changes, delivering the **1.41x training speedup** Meta reports across diverse models. Second, for multi-task losses and domain-specific patterns where compiler fusion falls short, implement batched loss computations using the concatenate-compute-split pattern to consolidate kernel launches. Third, write custom Triton kernels only for truly novel operations or when profiling reveals specific bottlenecks, leveraging TMA and warp specialization for H100-specific gains.

Profile before optimizing using Nsight Systems to identify whether workloads are launch-bound (gaps between kernels, GPU utilization <70%) or compute-bound (continuous kernel execution, high SM occupancy). Meta's production analysis found GPU idle time exceeding 50%—in such cases, the optimizations described here deliver transformative rather than incremental improvement. The combination of automatic compiler optimization, careful loss batching, and targeted custom kernels represents the production-ready path to maximizing H100 training throughput.
