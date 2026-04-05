Optimizing High-Parameter Recommendation Architectures: A Micro-Architectural Analysis of Tensor Indexing, Layout Contiguity, and Compiler Fusion in PyTorch 2.0
1. Introduction: The Scalability Challenge in Deep Recommender Systems
The deployment of large-scale recommendation systems, particularly those scaling to 5.1 billion parameters with over 65 prediction tasks, represents one of the most demanding frontiers in modern high-performance computing (HPC). Unlike dense Large Language Models (LLMs) or Computer Vision architectures, which are often characterized by uniform, compute-bound General Matrix Multiplications (GEMMs), recommendation systems operate in a regime of extreme heterogeneity. They combine massive, memory-bound embedding lookups with sparse feature interactions and deep, compute-intensive transformer stacks. In this hybrid landscape, system performance is frequently dictated not by the raw floating-point capability of the GPU, but by the efficiency of data movement and the latency of kernel orchestration.
This report addresses a critical optimization vector identified within a production-grade 5.1B parameter recommendation model: the transition from indirect tensor indexing mechanisms—specifically the torch.arange() + torch.index_select() pattern—to native, stride-based slicing techniques (x[:, start::step, :]). While seemingly a minor syntactical refactor, this change touches upon fundamental aspects of GPU micro-architecture, memory coalescing, autograd mechanics, and the graph-lowering strategies of the PyTorch 2.0 Inductor compiler.
The primary bottleneck identified is the "kernel launch latency" phenomenon. In eager execution modes, or even within suboptimal compilation paths, the arange plus index_select pattern necessitates multiple kernel launches: one to generate the index sequence and another to perform the gather operation. For a model with dozens of prediction heads and deep transformer layers, the accumulation of these micro-kernels can lead to a scenario where the GPU's Command Processor (CP) becomes the bottleneck, leaving the Streaming Multiprocessors (SMs) underutilized. Transitioning to native slicing offers the theoretical promise of "zero-kernel" indexing via view mechanics, yet it introduces new complexities regarding memory contiguity and the efficiency of the backward pass.2
Furthermore, the optimization landscape is complicated by the configuration of the PyTorch 2.0 Inductor backend. The presence of torch._inductor.config.combo_kernels = True suggests an intent to leverage horizontal fusion, while the constraint torch._inductor.config.allow_buffer_reuse = False imposes significant penalties on intermediate memory allocations.4 This report provides an exhaustive analysis of these interactions, quantifying the theoretical and empirical tradeoffs between forward-pass overhead (explicit contiguous copies) and backward-pass throughput (optimized gradient accumulation).
1.1 The Operational Context: 5.1B Parameters and 65+ Tasks
At the scale of 5.1 billion parameters, the model likely partitions its capacity between massive embedding tables (potentially sharded across GPUs) and a dense "backbone" of transformer layers. The "65+ prediction tasks" implies a Multi-Task Learning (MTL) architecture, likely culminating in a "Multi-Gate Mixture-of-Experts" (MMoE) or a massive multi-head output layer.
This architecture creates a specific stress profile on the hardware:
Memory Bandwidth Saturation: The embedding lookups are sparse and bandwidth-intensive.
Kernel Fragmentation: The 65+ tasks often result in 65+ small, independent linear projections and loss calculations at the end of the forward pass.
Data Movement Overhead: The transformer backbone requires dense, contiguous data for maximum cuBLAS efficiency. Any indexing operation that disrupts this contiguity acts as a barrier, forcing data realignment or strided access penalties.
The proposed optimization—switching to strided slicing—targets the interface between these components, specifically where sequence data is downsampled or reshaped before entering the transformer stacks or prediction heads.
2. The Physics of Tensor Indexing on GPUs
To rigorously evaluate the proposed optimization, one must first deconstruct the physical execution of indexing operations on NVIDIA GPU architectures (Volta, Ampere, Hopper). The distinction between "indirect indexing" and "strided slicing" is not merely semantic; it represents two fundamentally different instructions streams and memory access patterns.
2.1 Indirect Indexing: The "Gather" Paradigm
The legacy pattern torch.index_select(x, dim, indices) implements a "gather" operation. Mathematically, for an input tensor $A$ and an index tensor $I$, the output $B$ is constructed such that $B[i] = A[I[i]]$.
2.1.1 Micro-Architectural Execution Sequence
When index_select is executed (even within a compiled kernel), the GPU threads must perform the following sequence:
Load Index: Each thread $t$ loads its assigned index value $I[t]$ from Global Memory into a register.
Address Calculation: The thread computes the source address: $Addr_{src} = BasePtr_A + (I[t] \times Stride_A)$.
Global Memory Load (Gather): The thread requests the data at $Addr_{src}$.
2.1.2 The Stochasticity Penalty
The critical flaw in this pattern, even when $I$ happens to contain a linear sequence (generated by arange), is that the GPU hardware and the compiler's static analysis cannot inherently guarantee this linearity without explicit optimization passes.
Memory Divergence: In a general gather, adjacent threads $t$ and $t+1$ might load indices pointing to memory locations far apart in the physical address space (e.g., $I[t]=0$ and $I[t+1]=1000$). This breaks memory coalescing. A GPU warp (32 threads) works most efficiently when it accesses a contiguous 128-byte cache line. If the gather is scattered, the memory controller must service up to 32 separate cache line transactions, drastically reducing effective bandwidth utilization.1
Translation Lookaside Buffer (TLB) Thrashing: Random access patterns across a multi-gigabyte tensor can cause high TLB miss rates, adding significant latency to address translation.
2.1.3 Kernel Launch Overhead
In eager execution, torch.arange() launches a kernel to populate the index tensor, and torch.index_select() launches a second kernel to perform the copy.
Launch Latency: A CUDA kernel launch typically incurs 3-5 microseconds of CPU-side dispatch overhead and GPU-side command processor latency.
Synchronization: The index_select kernel implicitly depends on the completion of the arange kernel. While CUDA streams allow asynchronous execution, the dependency chain ensures sequentiality.
2.2 Native Slicing: The "Strided View" Paradigm
The proposed pattern x[:, start::step, :] utilizes PyTorch's view mechanics. This is a metadata-only operation in the eager runtime.
2.2.1 The Arithmetic of Strides
A view does not copy data. It creates a new TensorImpl structure sharing the same storage pointer but with a modified stride. For a stride of $S_{step}$, the memory address for element $i$ is calculated as:

$$Addr_{i} = BasePtr + (i \times S_{old} \times S_{step})$$
Crucially, this address calculation is deterministic and linear.
2.2.2 Zero-Kernel Execution
In the forward pass (eager mode), slicing launches zero kernels. It is purely a CPU-side pointer arithmetic operation on the PyObject.2 This immediately eliminates the "death by a thousand kernels" problem for the indexing step itself.
2.2.3 Memory Coalescing and Gaps
The physical memory layout of a slice ::2 involves gaps.
Access Pattern: Thread $t$ accesses element $i$, Thread $t+1$ accesses element $i+2$.
Coalescing Efficiency: While not fully contiguous, this pattern is regular. The GPU memory controller can often coalesce these into fewer transactions than a random gather. However, bandwidth is still wasted. If the hardware fetches a 128-byte line but the warp only uses every other 4 bytes, effective bandwidth is halved (50% utilization) compared to a dense load.8
Vectorization Loss: Contiguous tensors allow the compiler to emit LDG.128 (load 128 bits / 4 floats) instructions. A stride-2 tensor forces the use of LDG.32 (load 32 bits / 1 float) or complex masking, reducing the instruction-level parallelism (ILP) and increasing the total number of load instructions issued.9
3. PyTorch 2.0 Inductor: Compilation, Fusion, and Lowering
The transition to PyTorch 2.0 and the Inductor backend shifts the optimization landscape from runtime kernel management to compile-time graph lowering. Inductor captures the PyTorch program into an FX graph and lowers it to Triton kernels.
3.1 The optimize_indexing Pass
Inductor contains a specific optimization pass located in torch/_inductor/optimize_indexing.py designed to mitigate the cost of indirect indexing.10
Functionality: This pass analyzes index_select operations. If it can prove via symbolic shape analysis that the index tensor is derived from a linear function (like arange), it attempts to replace the indirect "gather" with computed arithmetic indexing.
Fragility: This optimization relies on pattern matching. If the arange is created in a complex manner, or if the graph structure obscures the lineage of the index tensor (e.g., passing through multiple views or non-functionalized ops), the compiler may fail to trigger this optimization.
Superiority of Slicing: Using native Python slicing (x[::step]) explicitly encodes the linearity in the Intermediate Representation (IR). The stride is a property of the tensor node, not a computed value in a separate buffer. This guarantees that Inductor will generate arithmetic pointer math (ptr + idx * step) rather than memory-dependent lookups (ptr + load(idx_ptr)), ensuring the most efficient Triton code generation.11
3.2 Horizontal vs. Vertical Fusion
The configuration torch._inductor.config.combo_kernels = True enables horizontal fusion.13
Horizontal Fusion: Combines independent operations (e.g., four separate sum() reductions on four different tensors) into a single kernel launch. This is critical for the "65+ prediction tasks" mentioned in the query, as it prevents the tail of the model from fracturing into dozens of tiny kernels.
Vertical Fusion: Combines producer-consumer chains (e.g., x -> slice -> relu -> y). Inductor performs this by default.
Interaction: combo_kernels does not help fuse the sequential arange + index_select pattern because they are dependent (vertical). However, native slicing (which is just metadata) allows Inductor to easily fuse the slicing logic into subsequent elementwise operations (like ReLU or LayerNorm) via vertical fusion, creating a single kernel that reads with strides and computes.
3.3 The allow_buffer_reuse Constraint
The user notes torch._inductor.config.allow_buffer_reuse = False.4
Implication: Normally, Inductor's memory planner reclaims memory from dead tensors. If tensor $A$ is used to compute $B$ and then never used again, $B$ can overwrite $A$'s memory (if size permits) or $A$'s memory is marked free for $C$.
Impact on index_select: The arange tensor and the output of index_select (before any downstream ops) are distinct allocations. With reuse disabled, these allocations persist longer or require fresh calls to the allocator, increasing VRAM fragmentation and peak usage.
Impact on Slicing: Slicing creates a view. A view does not have its own storage; it aliases the input. Therefore, it does not request a new buffer from the memory pool. This completely bypasses the penalty of allow_buffer_reuse=False for the intermediate step. This is a massive hidden advantage for the slicing approach in this specific constrained environment.5
4. The Contiguity Conundrum: Forward Overhead vs. Backward Throughput
The most nuanced aspect of this optimization is the decision to append .contiguous() after slicing: x[:, ::2, :].contiguous(). This operation forces a physical data copy in the forward pass. Why would we voluntarily introduce a copy (a bandwidth-bound operation) to optimize performance?
4.1 The Cost of Non-Contiguity in Transformers
The recommendation model relies on "transformer-based architectures." The core computational primitive of a transformer is the General Matrix Multiplication (GEMM), executed via Linear layers and MultiheadAttention.
4.1.1 The cuBLAS/Triton GEMM Constraint
High-performance GEMM kernels (e.g., NVIDIA's cuBLAS or Triton's dot ops) are heavily optimized for specific memory layouts, typically Row-Major (contiguous in last dimension) or Column-Major.
Strided Inputs: If a GEMM receives a generic strided tensor (e.g., stride 2 on the sequence dimension), the kernel cannot efficiently load tiles of data into the Tensor Cores.
Implicit Copies: If PyTorch detects a non-contiguous input to a Linear layer that requires contiguity (often for optimal Tensor Core usage), it may implicitly launch an aten::contiguous or aten::copy kernel immediately before the GEMM.
Result: You pay the copy cost anyway, but potentially in a way that is hidden from the graph optimizer or less efficient.
4.2 The Backward Pass Dynamics
The user's hypothesis states: "Autograd operations on contiguous tensors are generally faster." This is empirically true and rooted in the implementation of backward kernels.9
4.2.1 index_select Backward: The Atomic Bottleneck
The backward operation for index_select is index_add_ (or scatter_add).
Gradients: To compute the gradient with respect to the input, the system must scatter the incoming gradients back to the source indices.
Atomic Operations: Since multiple indices could theoretically point to the same source row (even if arange ensures uniqueness, the kernel logic is generic), the GPU must use atomic addition instructions (atomicAdd) to ensure correctness.
Performance: Atomic operations on global memory are significantly slower than register-based accumulation. They serialize access to the same memory addresses and disable cache write-combining optimizations.17
4.2.2 Strided Slice Backward: Strided Reads/Writes
The backward pass of a strided slice involves mapping gradients from the smaller output tensor back to the strided positions in the larger input gradient tensor.
Mechanism: This is a purely deterministic strided copy/accumulate. No atomics are required because the mapping is injective (one-to-one).
Efficiency: While faster than atomics, writing to strided memory (e.g., every 2nd element) causes write bandwidth wastage. A 128-byte cache line is fetched, 64 bytes are modified, and the line is written back. This consumes full bandwidth for half the data throughput.
4.2.3 Contiguous Backward: The Dense Path
By forcing .contiguous() in the forward pass:
Forward: We incur a dense D2D copy cost. Input (strided) -> Copy -> Input_Contiguous.
Transformer: The Transformer consumes Input_Contiguous.
Backward: The Transformer produces Grad_Contiguous.
Copy Backward: The backward of the .contiguous() op is simply a copy (or strided copy) back to the strided layout.
The Winning Factor: The massive computation inside the Transformer (GEMMs, Softmax, LayerNorm) happens on contiguous data.
Cache Locality: Spatial locality is maximized. Prefetchers work perfectly.
Vectorization: All loads/stores use LDG.128/STG.128.
Kernel Occupancy: Since the kernels are not stalled by memory latency (due to optimal coalescing), the SMs remain saturated with compute instructions.
Autograd Overhead: Eager autograd often has fast-paths for contiguous tensors. Non-contiguous tensors trigger more complex strides-checking logic in the C++ dispatcher.
Tradeoff Verdict: In a 5.1B parameter model, the execution time is dominated by the compute-heavy Transformer layers. Optimizing the data layout entering these layers yields performance gains that dwarf the cost of a single memory copy. The copy is a linear $O(N)$ operation; the Transformer is often super-linear or at least has a very large constant factor due to $O(N^2)$ attention or massive hidden dimensions.
5. Benchmarking Methodology and Profiling Strategy
To validate these theoretical assertions, we define a rigorous benchmarking methodology using torch.profiler (Kineto). The goal is to isolate the indexing subsystem while maintaining the stress characteristics of the full model.
5.1 Profiling Tools and Metrics
We leverage the PyTorch Kineto profiler, which provides nanosecond-level visibility into CUDA kernel launches and memory events.19
Key Metrics:
Kernel Launch Count: Absolute number of kernels launched per step.
CUDA Time Total: Total time spent on GPU.
Memory Allocations: Number and size of memory allocation events (to test the buffer_reuse impact).
DRAM Bandwidth: To verify the efficiency of the backward pass (coalescing).
5.2 Benchmark Script
The following script benchmarks three variants: Legacy (IndexSelect), SliceView (Strided), and SliceContiguous (Optimized).

Python


import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

# --- Configuration to match User Environment ---
# Enabling Inductor Fusion
torch._inductor.config.combo_kernels = True
# Disabling Buffer Reuse as per user constraint
torch._inductor.config.allow_buffer_reuse = False

# Simulation Constants for 5.1B Model Scale
BATCH_SIZE = 128
SEQ_LEN = 1024
EMB_DIM = 4096 # High embedding dim typical of large RecSys
STRIDE = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RecommendationIndexer(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        # Simulate a heavy downstream Transformer block
        # This ensures we measure the impact of layout on the consumer
        self.transformer_block = nn.Sequential(
            nn.Linear(EMB_DIM, EMB_DIM * 4),
            nn.GELU(),
            nn.Linear(EMB_DIM * 4, EMB_DIM)
        )

    def forward(self, x):
        # x shape:
        if self.mode == 'legacy':
            # Pattern: arange + index_select
            indices = torch.arange(0, x.shape, STRIDE, device=x.device)
            out = torch.index_select(x, 1, indices)

        elif self.mode == 'slice_view':
            # Pattern: Native Strided Slice
            out = x

        elif self.mode == 'slice_contiguous':
            # Pattern: Slice + Explicit Contiguous Copy
            out = x.contiguous()

        # Pass through heavy compute
        return self.transformer_block(out)

def run_benchmark(mode, input_tensor):
    print(f"--- Benchmarking: {mode} ---")
    model = RecommendationIndexer(mode).to(device)

    # Compile with Inductor
    compiled_model = torch.compile(model)

    # Warmup to trigger compilation and JIT fusion
    for _ in range(5):
        loss = compiled_model(input_tensor).sum()
        loss.backward()

    # Profile
    with profile(
        activities=,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=tensorboard_trace_handler(f"./log/{mode}")
    ) as prof:
        for i in range(10):
            with record_function("training_step"):
                # Forward
                output = compiled_model(input_tensor)
                loss = output.sum()
                # Backward
                loss.backward()

    # Reporting
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == "__main__":
    # Input Tensor (Gradient required for backward pass analysis)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMB_DIM, device=device, requires_grad=True)

    run_benchmark('legacy', x)
    run_benchmark('slice_view', x)
    run_benchmark('slice_contiguous', x)


5.3 Interpreting the Results
Legacy: Look for aten::arange and aten::index_select (or triton_poi_fused_index_select) in the trace. The backward pass will show aten::index_add_ or aten::scatter_add.
SliceView: The specific indexing kernels should disappear. However, check the Linear layer execution. Does it show aten::copy happening implicitly? Or does the mm kernel execution time increase due to non-contiguous loading?
SliceContiguous: You should see an explicit aten::copy or triton_poi_fused_copy kernel. Crucially, verify that the mm (Matrix Multiply) kernels in the backward pass (mm_backward) are faster than in the SliceView case.
6. Addressing the Research Questions
6.1 RQ1: Effect on Kernel Launch Count
Analysis: Replacing torch.arange + index_select with native slicing reduces the kernel count for the indexing operation from 2 to 0 (pure view) or 1 (if contiguous).
Legacy: Launches arange kernel (to generate indices) + index_select kernel (to gather data).
Slicing: No kernel is launched for the slice itself. It is a CPU-side metadata update.
Impact: For a graph with many such operations, this significantly reduces CPU dispatcher pressure.
6.2 RQ2: combo_kernels=True and Automatic Fusion
Analysis: No, combo_kernels does not automatically fuse slice operations.
Horizontal vs. Vertical: combo_kernels handles horizontal fusion (independent parallel ops). Slicing requires vertical fusion (producer-consumer).
Inductor's Role: Inductor does fuse slicing vertically, but this is handled by the core scheduler and the optimize_indexing pass, not the combo_kernels flag. combo_kernels is relevant for the multi-head prediction tasks later in the model but is orthogonal to the slicing optimization.
6.3 RQ3: Memory Allocation Savings
Analysis:
Index Tensor: Eliminating arange saves $Batch \times Seq_{out} \times 8$ bytes.
Intermediate Buffer: index_select allocates a new dense tensor for the output. Slicing allocates a view (0 bytes).
allow_buffer_reuse=False Context: This is the critical win. Because buffer reuse is disabled, the system cannot efficiently recycle the memory used by index_select's output. By using a view, we avoid requesting a buffer from the allocator entirely, bypassing the fragmentation/overhead penalty of the disabled reuse flag.
6.4 RQ4: When to Add .contiguous()?
Analysis: Add .contiguous() immediately before Compute-Bound kernels that are sensitive to memory layout.
Best Practice: Slice -> Elementwise Ops (ReLU/Norm) ->.contiguous() -> Linear/Attention.
Reasoning: Inductor can fuse the stride into elementwise ops for free. It cannot fuse stride into a GEMM without either a copy or a slow strided kernel. By placing contiguous before the GEMM, you ensure the heavy compute runs at peak efficiency.
6.5 RQ5: Tradeoff Analysis (Forward Overhead vs. Backward Speedup)
Analysis:
Forward Cost: A contiguous copy is a streaming D2D operation. On an A100 (1.5TB/s bandwidth), copying a 1GB tensor takes ~0.7ms.
Backward Speedup:
Avoids index_add_ (atomics).
Enables dense dgrad computation in Transformers.
Improves L2 cache hit rates for gradients.
Verdict: For 5.1B parameter models, the backward pass compute dominates the runtime. The speedup in gradient computation (typically 20-30% faster for contiguous vs. strided GEMM backward) vastly outweighs the sub-millisecond cost of the forward copy.
7. Conclusion
The optimization of replacing index_select with strided slicing followed by .contiguous() represents a high-value architectural shift for large-scale recommendation models. It aligns the model's execution pattern with the strengths of the PyTorch 2.0 Inductor compiler (arithmetic indexing over gather) and the GPU hardware (dense vectorization over atomics).
By eliminating the arange kernel and the index_select gather, we reduce kernel launch overhead. By strategically re-introducing contiguity before the transformer stack, we ensure that the massive matrix multiplications—which constitute the vast majority of the FLOPs—execute on optimal memory layouts. Finally, leveraging views bypasses the memory allocation inefficiencies imposed by the allow_buffer_reuse=False constraint, improving the model's memory hygiene.
Recommendation: Implement x[:, ::step, :].contiguous() immediately. Ensure torch._inductor.config allows for vertical fusion in elementwise ops preceding the copy. Monitor the backward pass throughput (Tokens/Sec) as the primary success metric.
Table 1: Comparative Summary of Indexing Approaches
Feature
Indirect Indexing (index_select)
Native Slicing (view)
Optimized (slice + contiguous)
Kernel Launches
2 (arange + gather)
0 (Metadata only)
1 (copy)
Memory Access
Indirect (Gather)
Strided (Regular Gaps)
Contiguous (Dense)
Inductor Fusion
Hard (requires optimize_indexing match)
Easy (Arithmetic math)
Easy (Fuses into copy)
Backward Pass
index_add_ (Atomics, Slow)
Strided Accumulation
Dense mm_backward (Fast)
Memory Alloc
New Buffer + Index Buffer
0 (View)
New Buffer
buffer_reuse=False Penalty
High (Leaks capacity)
None (No alloc)
Moderate (Allocates output)

This structural change transforms the indexing layer from a legacy bottleneck into a streamlined, compiler-friendly operation, unlocking the full potential of the GPU hardware for the 5.1B parameter workload.
Works cited
Why torch.take is tremendously slower than torch.index_select with two reshapes?, accessed January 30, 2026, https://discuss.pytorch.org/t/why-torch-take-is-tremendously-slower-than-torch-index-select-with-two-reshapes/88020
In pytorch, what is the difference between indexing with square brackets and "index_select"?, accessed January 30, 2026, https://stackoverflow.com/questions/69824591/in-pytorch-what-is-the-difference-between-indexing-with-square-brackets-and-in
What does .contiguous() do in PyTorch? - Stack Overflow, accessed January 30, 2026, https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
[FSDP][torch._dynamo.compiled_autograd] Final callbacks can only be installed during backward pass · Issue #121071 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/121071
pytorch/torch/_inductor/config.py at main - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
The performance of `torch.index_select` and regular indexing differs dramatically based on the size of the tensor it is indexing · Issue #116555 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/116555
torch.select — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/generated/torch.select.html
Contigious vs non-contigious tensor - PyTorch Forums, accessed January 30, 2026, https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107
Does performance of different operations depend on contiguity? - PyTorch Forums, accessed January 30, 2026, https://discuss.pytorch.org/t/does-performance-of-different-operations-depend-on-contiguity/17787
eac5e1254883e693b028db66e7fc90b3679acbd0 - platform, accessed January 30, 2026, https://android.googlesource.com/platform/external/pytorch/+/eac5e1254883e693b028db66e7fc90b3679acbd0
Torch Dynamo backend compilation error with dynamic = True · Issue #96469 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/96469
TorchInductor and AOTInductor Provenance Tracking - PyTorch documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/torch.compiler_inductor_provenance.html
[RFC][Inductor] enablement of combo-kernels - experimental horizontal optimization in torchinductor · Issue #170268 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/170268
pytorch/pytorch v2.10.0 on GitHub - NewReleases.io, accessed January 30, 2026, https://newreleases.io/project/github/pytorch/pytorch/release/v2.10.0
Compiled functional collectives fail without graph breaks on CUDA · Issue #108780 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/108780
.venv/lib/python3.11/site-packages/torch/_inductor/config.py · koichi12/llm_tutorial at 70fbf20b4ddc5dc677bf521e5e39c22461fa7256 - Hugging Face, accessed January 30, 2026, https://huggingface.co/koichi12/llm_tutorial/blob/70fbf20b4ddc5dc677bf521e5e39c22461fa7256/.venv/lib/python3.11/site-packages/torch/_inductor/config.py
[PERFORMANCE] Use index_select instead of regular indexing · Issue #3729 · dmlc/dgl, accessed January 30, 2026, https://github.com/dmlc/dgl/issues/3729
Insum: Sparse GPU Kernels Simplified and Optimized with Indirect Einsums - arXiv, accessed January 30, 2026, https://arxiv.org/pdf/2510.17505
17.3. GPU Profiling — Kempner Institute Computing Handbook, accessed January 30, 2026, https://handbook.eng.kempnerinstitute.harvard.edu/s5_ai_scaling_and_engineering/scalability/gpu_profiling.html
Automated trace collection and analysis - PyTorch, accessed January 30, 2026, https://pytorch.org/blog/automated-trace-collection/
