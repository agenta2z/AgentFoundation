Q6: Benchmark Tool & Kernel Profiling Analysis
1. Executive Summary
The optimization of deep learning models in PyTorch has transitioned from a focus on high-level algorithmic adjustments to low-level systems engineering. As models scale in parameter count and complexity, the bottleneck often shifts from pure arithmetic intensity to the overhead of the framework and the efficiency of kernel orchestration. The user’s objective—validating kernel-level optimizations such as operator fusion and backend selection—requires a departure from standard end-to-end latency measurements. While macroscopic timing (wall-clock latency) indicates that performance has changed, it fails to explain why. To validate specific engineering interventions, such as the efficacy of torch.compile (Inductor) or the activation of Flash Attention, one must interrogate the GPU command stream directly.
This report establishes a rigorous methodology for "Enhanced Benchmarking" in PyTorch. It addresses the critical deficit in current tooling: the inability to distinguish between launch-bound latency (CPU overhead) and compute-bound latency (GPU execution), and the lack of visibility into compiler decisions. By synthesizing capabilities from torch.profiler, torch.utils.benchmark, and the underlying Kineto tracing library, this analysis provides a blueprint for a diagnostic tool capable of granular kernel introspection.
Key findings dictate that accurate kernel counting cannot rely on Python-level operator hooks but must filter low-level profiler events for CUDA device types. Launch overhead is best isolated through a dual-timing strategy that contrasts asynchronous CPU dispatch time against synchronized GPU execution time. Furthermore, verifying Scaled Dot Product Attention (SDPA) backends requires signature analysis of the execution trace, as runtime heuristics often silently fallback to slower math implementations due to memory alignment or data type mismatches.
The following sections detail the theoretical basis and practical implementation of these profiling strategies, culminating in a comprehensive specification for an A/B benchmarking architecture designed to support iterative kernel optimization.
2. The Theoretical Imperative for Kernel-Level Visibility
In the eager execution model of PyTorch, the framework acts as an interpreter that dispatches operations one by one to the GPU. This flexibility, while beneficial for debugging, introduces significant overhead. Each Python line corresponds to a sequence of dispatch events: the Python interpreter parses the code, the PyTorch dispatcher (ATen) resolves the appropriate kernel for the tensor's device and dtype, and the CUDA driver formats and enqueues the command for the GPU.
2.1 The "Black Box" of Compilation and Fusion
With the advent of PyTorch 2.0 and torch.compile, the execution model shifts towards graph capture and holistic optimization. The compiler, typically TorchInductor, analyzes the computation graph to identify opportunities for vertical and horizontal fusion.
Vertical Fusion: Combining a producer (e.g., Matrix Multiplication) and a consumer (e.g., ReLU) into a single kernel to keep data in GPU SRAM (L1/Shared Memory) and avoid costly round-trips to HBM (High Bandwidth Memory).
Horizontal Fusion: Combining independent operations that share input data.
The user's specific scenario—validating a claim of "221 kernels reduced to 30"—is a direct test of this fusion logic. A standard timer cannot validate this. If the latency drops by 10%, it could be due to fusion, or it could be due to better clock speeds, or simply less contention on the PCIe bus. Without a kernel counter, the engineer is flying blind. If the compiler failed to fuse the kernels (due to a graph break or dynamic shape issue), the latency might still look acceptable, but the optimization ceiling would remain untouched. Thus, kernel counting is not just a metric; it is a verification of the compilation process itself.1
2.2 The Launch-Bound Regime
Deep learning workloads are often categorized as "Compute Bound" (limited by FLOPs) or "Memory Bound" (limited by VRAM bandwidth). However, a third category is increasingly prevalent: "Launch Bound" (or Latency Bound). This occurs when the time taken by the CPU to dispatch a kernel ($T_{cpu}$) exceeds the time taken by the GPU to execute it ($T_{gpu}$).

$$T_{overhead} = T_{python} + T_{dispatch} + T_{driver}$$
If $T_{overhead} > T_{gpu}$, the GPU experiences starvation. It finishes its work and sits idle, waiting for the CPU to send the next command. This "bubble" in the pipeline is invisible to a synchronous timer, which simply sees the total sum of time. To diagnose this, the benchmark must separate the timeline of the CPU from the timeline of the GPU. This separation is crucial for determining whether to invest in kernel fusion (to reduce the number of launches) or in CUDA Graphs (to reduce the cost per launch).4
2.3 Granularity of Observation
Traditional profiling methods (e.g., cProfile for Python) stop at the CPU boundary. They can tell you that self.layer1(x) took 50ms, but they cannot tell you if that 50ms was spent calculating convolution or waiting for memory allocation. The torch.profiler API bridges this gap by integrating with NVIDIA's CUPTI (CUDA Profiling Tools Interface) via the Kineto library. This allows the profiler to record events on the device timeline, providing the ground truth of hardware execution. This report leverages Kineto's capabilities to answer the user's specific questions regarding kernel counts, backend selection, and component attribution.1
3. Methodology: Programmatic Kernel Counting
The first research question asks: "How to count total kernel launches in a PyTorch forward/backward pass?" This sounds trivial but is complicated by the disconnect between Python operators and CUDA kernels.
3.1 The Discrepancy: Operators vs. Kernels
A single PyTorch operator does not always equal one kernel.
Legacy Ops: An operation like torch.batch_norm might launch separate kernels for mean calculation, variance calculation, and normalization.
Fused Ops: A sequence like x * y + z in Eager mode launches two kernels (Mul, Add). In Compiled mode (Inductor), it launches one Triton kernel.
Utility Kernels: PyTorch transparently launches kernels for memory management, contiguous checks, and zeroing out buffers (aten::zero_).
Counting Python function calls is therefore a poor proxy for GPU activity. The counting mechanism must exist at the profiler level, inspecting the trace events that originate from the device.
3.2 Leveraging torch.profiler for Accurate Counts
The torch.profiler exposes the execution trace through the EventList object. This object contains a flattened list of all recorded events, both CPU and GPU. To count kernels, we must filter this list based on the device_type.
3.2.1 Configuration
The profiler must be initialized with ProfilerActivity.CUDA. Without this, Kineto will not initialize the CUPTI callbacks, and the trace will contain only CPU-side metadata (dispatch records) but no actual kernel launches.1

Python


from torch.profiler import profile, ProfilerActivity

activities =
with profile(activities=activities, record_shapes=False) as prof:
    model(inputs)
    torch.cuda.synchronize()


The call to torch.cuda.synchronize() is vital. Profiling is asynchronous; if the context manager exits before the GPU finishes, the trace data for the trailing kernels will be lost or truncated, leading to an undercount.9
3.2.2 Filtering Logic
The key_averages() method aggregates events by name. However, a robust count should not rely on aggregation alone, as we need the total count, not just the count per unique name. The most accurate method is to iterate over the raw events and sum those that belong to the CUDA device.
In the FunctionEvent object (and its aggregated counterpart FunctionEventAvg), the device_type attribute indicates the origin.
device_type=0: CPU
device_type=1: CUDA (typically, though enum values may vary, relying on DeviceType.CUDA is safer).
The benchmark tool should implement a scanner function:

Python


def count_total_kernels(prof_result):
    total_kernels = 0
    # key_averages() groups by operator name
    averages = prof_result.key_averages()
    
    for event in averages:
        # We only care about CUDA kernels. 
        # Note: In some versions, device_type is on the raw event, 
        # but key_averages aggregates mixed devices if names match.
        # It is safer to inspect the `self_cuda_time_total` > 0 check 
        # or filter raw events.
        
        if event.device_type == torch.device('cuda').type or event.cuda_time_total > 0:
             # Some CPU ops have cuda_time if they are wrappers, 
             # so we must be careful. Ideally, use raw events.
             pass 

    # Better Method: Raw Event Iteration
    raw_kernel_count = 0
    for event in prof_result.events():
        if event.device_type == torch.device('cuda').type:
             # Filter out Memcpy if strictly counting compute kernels
             if "Memcpy" not in event.name and "memset" not in event.name:
                 raw_kernel_count += 1
    return raw_kernel_count


Research Nuance: The snippet 11 from the PyTorch autograd profiler source code reveals that kineto_event.device_type() is the source of truth. It also shows that Memcpy and Memset are treated as device events. Depending on the user's definition of "kernel" (compute vs. all traffic), the benchmark might need to offer a toggle to exclude memory operations from the count. For validating fusion ("221 -> 30"), typically all kernel launches including memory ops are relevant, as fusion often eliminates intermediate memory moves.7
3.3 Validating the "221 → 30" Claim
When torch.compile is active, the kernel names change.
Eager: aten::add, aten::mul, void c10::cuda::kernel...
Inductor: triton_poi_fused_add_mul_..., triton_red_fused_...
The enhanced benchmark must not only count the kernels but also output a distribution of names. A reduction in count accompanied by the appearance of "triton_" prefixed names confirms that Inductor is successfully generating fused kernels. If the count remains high and the names remain aten::..., graph breaks are likely occurring, causing the compiler to fall back to eager execution for parts of the graph.3
4. Deconstructing Latency: Launch Overhead vs. Compute
The second research question—measuring kernel launch overhead separately from kernel execution time—addresses the efficiency of the CPU-GPU communication channel.
4.1 The Physics of Latency
Total latency is a pipelined sum of CPU work and GPU work.
CPU Dispatch Time ($T_{dispatch}$): The time the CPU spends in the Python interpreter, the PyTorch dispatcher, and the CUDA driver (cudaLaunchKernel). During this time, the CPU is blocked.
GPU Execution Time ($T_{compute}$): The time the GPU spends executing the kernel.
Synchronization Overhead: Time spent waiting for the two distinct timelines to align.
In an ideal asynchronous execution, the CPU dispatches Kernel A, returns immediately, and starts processing Kernel B while the GPU executes Kernel A. However, if Kernel A is very fast (e.g., 1μs) and the dispatch takes 10μs, the GPU finishes Kernel A and idles for 9μs. This is the definition of "Launch Bound".5
4.2 Measurement Strategy: The Dual-Timer Approach
To isolate these metrics, the benchmark must employ two different timing scopes within the same script logic.
4.2.1 Measuring CPU Dispatch (Launch Overhead)
We utilize the asynchronous nature of CUDA calls. By measuring the wall-clock time of the Python call without synchronization, we capture how fast the CPU can enqueue work.

Python


# Measure Launch Overhead (CPU Dispatch)
start_cpu = time.perf_counter()
model(inputs) 
# We do NOT synchronize here. 
# The function returns as soon as kernels are enqueued.
end_cpu = time.perf_counter()
cpu_dispatch_latency = end_cpu - start_cpu


Interpretation: This value represents the burden on the CPU. If this value is high (e.g., 20ms for a forward pass), the model architecture involves too many Python-side operations or too many small kernels. Optimizations should focus on torch.compile (to move loop logic to C++) or CUDA Graphs.9
4.2.2 Measuring GPU Execution (True Compute)
To measure the actual GPU work, we cannot use CPU wall clocks because the CPU might finish long before the GPU does. We must use torch.cuda.Event.

Python


# Measure GPU Time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
model(inputs)
end_event.record()

# Now we must wait to read the timer
torch.cuda.synchronize()
gpu_execution_latency = start_event.elapsed_time(end_event) # Returns milliseconds


Interpretation: This is the physical limit of the hardware given the current kernels. If this value is high, the model is Compute Bound or Memory Bandwidth Bound. Optimizations should focus on kernel efficiency (Flash Attention, Triton) or quantization.10
4.3 The "Empty Kernel" Calibration
To provide a baseline for "Launch Overhead," standard practice involves measuring the cost of a no-op.
Technique: Create a dummy tensor operation (e.g., adding 0 to a 1-element tensor) or a raw CUDA kernel that returns immediately.
Measurement: Measure the dispatch time.
Benchmark Value: On modern PyTorch/Drivers, this is typically 3μs - 10μs.
Application: If the model's average per-kernel launch cost (Total Dispatch Time / Kernel Count) is significantly higher than this baseline (e.g., 50μs), it indicates heavy framework overhead (autograd, dynamic shape checking) rather than just driver overhead. The enhanced benchmark should calculate and report this "Overhead per Kernel" metric.5
4.4 Amortization via CUDA Graphs
The report must highlight the role of torch.cuda.CUDAGraphs in eliminating launch overhead. By capturing the graph, the CPU submits a single command (cudaGraphLaunch) instead of N commands.
Benchmark Integration: The tool should offer a flag --enable-cuda-graphs.
Mechanism:
Warmup the model.
Capture the graph: g = torch.cuda.CUDAGraph(); with torch.cuda.graph(g): model(x).
Replay in the timing loop: g.replay().
Contrast: Comparing the cpu_dispatch_latency of the Graph run vs. the Eager run provides the exact magnitude of the launch overhead savings.18
5. Scaled Dot Product Attention (SDPA) Forensics
Research question 3 ("How to profile which SDPA backend is selected?") targets the opacity of the F.scaled_dot_product_attention operator. This operator acts as a runtime dispatcher, selecting between three primary backends:
Flash Attention: (sdp_kernel context: FLASH_ATTENTION) - The fastest, tiling-based implementation.
Memory Efficient Attention: (sdp_kernel context: EFFICIENT_ATTENTION) - Based on xFormers, slightly slower but supports more hardware/dtypes.
Math: (sdp_kernel context: MATH) - The C++ fallback using standard matrix multiplications. Slow and memory hungry.
5.1 The Verification Gap
Users often assume "PyTorch 2.0 = Flash Attention." However, the dispatcher silently falls back to Math if constraints are not met:
Dtype: Flash usually requires FP16 or BF16. FP32 often triggers fallback.
Hardware: Requires Compute Capability 8.0+ (Ampere) for v2.
Alignment: Tensors must be contiguous in memory.
Head Dimension: Earlier versions required head_dim to be a multiple of 8.20
Since sdpa_kernel acts as a constraint enforcer rather than a reporter, simply running the code doesn't tell you what happened unless you force a backend and catch the RuntimeError. For profiling what happened in a benchmark run, we need forensic analysis of the trace.
5.2 Signature Analysis via Profiler
The most reliable non-intrusive method is to inspect the kernel names generated in the profiler trace. The distinct implementations emit unique kernel symbols.
5.2.1 Identifying Flash Attention
Trace events will contain specific substrings.
Modern PyTorch: flash_fwd, flash_bwd.
Low-level CuDNN/Cutlass: cutlass::Kernel, fmha_....
Note: In torch.compile mode, these might appear as impl_abstract("xformers_flash::flash_fwd") or similar wrapped calls.23
5.2.2 Identifying Memory Efficient Attention
Keywords: mem_efficient, efficient_attention_forward.
Origin: Often linked to xformers implementations integrated into ATen.
5.2.3 Identifying Math Fallback
Keywords: The absence of the above.
Pattern: A cluster of bmm, div, softmax, transpose, bmm appearing inside the scope of the attention layer.
Detection Logic: If the profiler sees aten::scaled_dot_product_attention on the CPU side, but the GPU side shows standard GEMM (General Matrix Multiply) kernels instead of specialized attention kernels, the fallback occurred.26
5.3 Benchmarking Logic for Verification
The enhanced benchmark tool should include a verify_sdpa_backend() routine:
Run a single forward pass with torch.profiler.
Extract all kernel names from prof.events().
Apply regex matching against the known signatures.
Report: "SDPA Backend Detected: [Flash | Efficient | Math]".
Additionally, the tool should support A/B testing by forcing backends:

Python


from torch.nn.attention import sdpa_kernel, SDPBackend

# Benchmark Config A: Force Math
with sdpa_kernel(SDPBackend.MATH):
    run_benchmark()

# Benchmark Config B: Force Flash
try:
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        run_benchmark()
except RuntimeError:
    print("Flash Attention not supported configuration.")


This contrast explicitly quantifies the speedup gained from the attention mechanism itself, separate from other model optimizations.27
6. Component-Level Attribution & Hierarchical Profiling
Research question 5 asks how to attribute kernel time to specific model components (Embedding vs. Attention vs. MLP). This is the "Blame Assignment" problem. In synchronous code, this is easy. In asynchronous GPU execution, the CPU moves on to the MLP layer while the GPU is still crunching the Embedding layer. If we simply time the CPU scope, we measure dispatch time, not compute time.
6.1 The Solution: record_function and Correlation
PyTorch's profiler solves this via Correlation IDs. When a user enters a record_function scope on the CPU, PyTorch generates a correlation ID. Any CUDA kernel launched within that scope inherits this ID. When the Kineto trace is reconstructed, the GPU kernel blocks are visually or logically nested under the CPU scope that spawned them, even if they execute much later in time.
6.2 Implementing Hierarchical Scopes
The benchmark must inject these scopes.
6.2.1 Manual Injection
For granular control, the user can wrap code blocks:

Python


def forward(self, x):
    with torch.profiler.record_function("Embedding Block"):
        x = self.embed(x)
    with torch.profiler.record_function("Attention Block"):
        x = self.attn(x)
    return x


6.2.2 Automatic Injection: with_modules=True
For the "Enhanced Benchmark" tool, relying on manual user code changes is brittle. The tool should utilize torch.profiler.profile(..., with_modules=True).
Mechanism: This flag adds hooks to nn.Module.__call__.
Result: Every layer invocation (ResNet, Layer1, Conv2d) automatically gets a record_function scope named after the module class and instance.
Benefit: The trace automatically reflects the model's architecture hierarchy without changing a line of model code.1
6.3 Aggregating Time by Component
To extract the data programmatically (e.g., "Attention took 40% of time"), the benchmark relies on key_averages(group_by_stack_n=...).
However, group_by_stack_n groups by the Python stack trace, which can be verbose. A cleaner programmatic approach is to parse the EventList:
Filter events that are DeviceType.CUDA.
Look at the scope or name of the event. When with_modules=True is on, the name often includes the module hierarchy.
Sum cuda_time_total for all kernels where the scope string contains "Attention".
Caution: cuda_time_total in FunctionEventAvg for a high-level scope (like "ResNet") includes the time of its children. When creating a pie chart, one must be careful not to double-count (e.g., don't sum "ResNet" and "ResNet/Layer1"). The benchmark should target leaf nodes or specific depth levels (e.g., Depth 1: "Encoder", "Decoder").1
7. Recommended Structure for A/B Benchmarking
Research question 4 seeks the "recommended structure." Ad-hoc scripts using time.time() are prone to OS jitter, thermal throttling, and background noise. The professional standard for PyTorch benchmarking is torch.utils.benchmark.
7.1 Architecture of the Comparative Suite
The enhanced benchmark should not be a linear script but a structured application. We recommend utilizing the Timer and Compare classes from torch.utils.benchmark to ensure statistical rigor.
7.1.1 The Timer Class vs. timeit
torch.utils.benchmark.Timer is superior to Python's timeit because:
Thread Control: It defaults to single-threaded execution (num_threads=1). PyTorch by default uses all available cores for intra-op parallelism, which introduces high variance in benchmarks due to thread contention. Single-threaded is more stable for relative comparisons.
Global Handling: It simplifies passing globals (stmt="model(x)", globals={'model': model, 'x': x}).
Adaptive Runs: It implements blocked_autorange, which determines how many iterations are needed to get a statistically significant sample based on the execution duration (e.g., ensure at least 2 seconds of runtime).32
7.2 Benchmark Configuration Management
A robust A/B test requires controlling variables. The benchmark tool should define a BenchmarkCase structure:
Parameter
Description
Label
Top-level name (e.g., "ResNet50")
Sub-label
Variant name (e.g., "Eager" vs "Inductor")
Description
Input details (e.g., "Batch 32, FP16")
Env
Critical for A/B. Tag runs as "Baseline" or "Optimized".

7.3 Statistical Metrics: IQR over Mean
The report emphasizes the use of Interquartile Range (IQR) provided by the Measurement object.
Mean/Median: Good for central tendency.
IQR: Measures spread. A high IQR in kernel benchmarking indicates Jitter—likely due to GPU throttling, garbage collection spikes, or background processes.
Recommendation: The tool should flag any run where IQR > 10% of Median as "Unstable" and recommend re-running with a strictly isolated environment.33
8. Visual Analysis via Chrome Traces
The "Expected Deliverable" includes Chrome trace export. This is the visual counterpart to the programmatic data.
8.1 Generating the JSON
The profiler's export_chrome_trace("trace.json") method dumps the event list into the Google Trace Event Format.
Format: A JSON list of dictionaries. Keys include ph (phase), ts (timestamp), dur (duration), name, pid, tid.
Compression: The benchmark should default to .json.gz as traces can grow to hundreds of megabytes quickly.1
8.2 Reading the Tea Leaves
The report guides the user on what to look for in chrome://tracing (or the newer ui.perfetto.dev):
Gaps (The Confetti): If the GPU timeline shows many small colored blocks with whitespace between them, the model is Launch Bound. The gaps are the CPU dispatch latency.
Overlaps (Streams): If blocks are stacked vertically, concurrent CUDA streams are active (e.g., overlap of communication and computation).
Super-Blocks: Large, solid blocks indicate fused kernels (Inductor/Triton).
Flow Arrows: Selecting a kernel shows an arrow back to the CPU thread. A long diagonal arrow indicates a large latency between "Order Given" and "Order Executed".8
9. Implementation Specification: The Enhanced Benchmark Tool
Based on the analysis, we define the specification for the tool the user should build. This satisfies the "Deliverable" requirement.
9.1 Tool Architecture
The tool torch_bench_pro.py will have three main modes:
Latency Mode: Uses torch.utils.benchmark for A/B testing.
Profile Mode: Uses torch.profiler for counting and tracing.
Overhead Mode: Uses dual-timers for launch analysis.
9.2 Core Class Structure (Pseudo-Code)

Python


class ProfilerSuite:
    def __init__(self, model, input_tensor):
        self.model = model
        self.input = input_tensor
        self.results = {}

    def count_kernels(self):
        """
        Research Question 1: Kernel Counting
        """
        with torch.profiler.profile(activities=) as p:
            self.model(self.input)
            torch.cuda.synchronize()
        
        # Filter for CUDA events
        count = sum(1 for e in p.events() if e.device_type == torch.device('cuda').type)
        return count

    def measure_overhead(self):
        """
        Research Question 2: Launch Overhead
        """
        # Async Dispatch Time
        t0 = time.perf_counter()
        for _ in range(50):
            self.model(self.input)
        t1 = time.perf_counter()
        dispatch_avg = (t1 - t0) / 50

        # Sync Execution Time
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(50):
            self.model(self.input)
        end_evt.record()
        torch.cuda.synchronize()
        exec_avg = start_evt.elapsed_time(end_evt) / 50 # ms

        return {"dispatch_us": dispatch_avg*1e6, "exec_us": exec_avg*1000}

    def verify_backend(self):
        """
        Research Question 3: SDPA Verification
        """
        with torch.profiler.profile(activities=) as p:
            self.model(self.input)
        
        kernel_names = [e.name for e in p.events()]
        if any("flash" in n for n in kernel_names): return "Flash"
        if any("mem_efficient" in n for n in kernel_names): return "Efficient"
        return "Math"

    def run_ab_comparison(self, optimized_model):
        """
        Research Question 4: A/B Structure
        """
        t1 = torch.utils.benchmark.Timer(
            stmt="m(i)", globals={"m": self.model, "i": self.input}, 
            label="Baseline"
        )
        t2 = torch.utils.benchmark.Timer(
            stmt="m(i)", globals={"m": optimized_model, "i": self.input}, 
            label="Optimized"
        )
        return torch.utils.benchmark.Compare([t1.blocked_autorange(), t2.blocked_autorange()])


9.3 Output Artifacts
The tool will generate:
Console Report: Tables showing Kernel Counts (A vs B), Backend usage, and Latency stats (Median/IQR).
Trace File: profile_trace.json.gz for visual inspection.
Component Breakdown: A text summary of % CUDA Time per top-level module (using key_averages(group_by_stack_n=1)).
10. Conclusion
The transition from a 650-line timing script to a kernel-aware profiling suite represents a maturity milestone in deep learning engineering. By adopting the methodologies detailed in this report, the user can move beyond observing that the model is faster, to proving why.
The combination of Kineto-based event filtering for kernel counting, Asynchronous timing for overhead isolation, and Signature analysis for backend verification provides a complete forensic picture of PyTorch execution. This rigor is essential for validating the complex behaviors of modern compilers like TorchInductor and ensuring that sophisticated optimizations like Flash Attention are correctly engaged. The proposed tool architecture not only answers the immediate research questions but establishes a reusable platform for all future optimization campaigns.
Works cited
torch.profiler — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/profiler.html
Measuring Automated Kernel Engineering - METR, accessed January 30, 2026, https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering/
Profiling to understand torch.compile performance — PyTorch 2.9 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html
Large Total Time for cudaLaunchKernel - Profiling Linux Targets, accessed January 30, 2026, https://forums.developer.nvidia.com/t/large-total-time-for-cudalaunchkernel/304875
How to measure overhead of a kernel launch in CUDA - Stack Overflow, accessed January 30, 2026, https://stackoverflow.com/questions/24375432/how-to-measure-overhead-of-a-kernel-launch-in-cuda
Automated trace collection and analysis - PyTorch, accessed January 30, 2026, https://pytorch.org/blog/automated-trace-collection/
pytorch/torch/autograd/profiler_util.py at main - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/blob/main/torch/autograd/profiler_util.py
PyTorch Profiler — PyTorch Tutorials 2.10.0+cu128 documentation, accessed January 30, 2026, https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
How to Accurately Time CUDA Kernels in Pytorch - Speechmatics, accessed January 30, 2026, https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
CUDA semantics — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/notes/cuda.html
pytorch-pytorch/torch/autograd/profiler.py at master · hughperkins/pytorch-pytorch - GitHub, accessed January 30, 2026, https://github.com/hughperkins/pytorch-pytorch/blob/master/torch/autograd/profiler.py
Is there a way to find out how many times of cuda kernels called/launched? - C++, accessed January 30, 2026, https://discuss.pytorch.org/t/is-there-a-way-to-find-out-how-many-times-of-cuda-kernels-called-launched/190255
How to trace kernels in pytorch, accessed January 30, 2026, https://discuss.pytorch.org/t/how-to-trace-kernels-in-pytorch/160340
MultiKernelBench: A Multi-Platform Benchmark for Kernel Generation - arXiv, accessed January 30, 2026, https://arxiv.org/html/2507.17773v2
How can I measure kernel launch overhead using ncu - NVIDIA Developer Forums, accessed January 30, 2026, https://forums.developer.nvidia.com/t/how-can-i-measure-kernel-launch-overhead-using-ncu/250619
CPU execution/dispatch time dominates and slows down small TorchScript GPU models · Issue #72746 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/72746
How to measure time in PyTorch, accessed January 30, 2026, https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
Accelerating PyTorch with CUDA Graphs, accessed January 30, 2026, https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
Constant Time Launch for Straight-Line CUDA Graphs and Other Performance Enhancements | NVIDIA Technical Blog, accessed January 30, 2026, https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/
SDPA memory efficient and flash attention kernels don't work with singleton dimensions · Issue #127523 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/127523
Accelerated PyTorch 2 Transformers, accessed January 30, 2026, https://pytorch.org/blog/accelerated-pytorch-2/
Out of the box acceleration and memory savings of decoder models with PyTorch 2.0, accessed January 30, 2026, https://pytorch.org/blog/out-of-the-box-acceleration/
AXLearn: Modular Large Model Training on Heterogeneous Infrastructure - arXiv, accessed January 30, 2026, https://arxiv.org/html/2507.05411v2
See raw diff - Hugging Face, accessed January 30, 2026, https://huggingface.co/spaces/facebook/MusicGen/commit/5238467a52adc08ddc72ffd7b6ec8ffb74528af2.diff
[Bug]: The new version (v0.5.4) cannot load the gptq model, but the old version (vllm=0.5.3.post1) can do it. · Issue #7240 - GitHub, accessed January 30, 2026, https://github.com/vllm-project/vllm/issues/7240
(Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA) - PyTorch documentation, accessed January 30, 2026, https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
SDPBackend — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.SDPBackend.html
Is it possible for torch SDPA to be slower than manual attention? - PyTorch Forums, accessed January 30, 2026, https://discuss.pytorch.org/t/is-it-possible-for-torch-sdpa-to-be-slower-than-manual-attention/218198
torch.nn.attention.sdpa_kernel - PyTorch documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
pytorch/torch/profiler/profiler.py at main - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/blob/main/torch/profiler/profiler.py
torch.autograd.profiler.profile.key_averages — PyTorch 2.9 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/generated/torch.autograd.profiler.profile.key_averages.html
torch.utils.benchmark — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/benchmark_utils.html
Timer quick start — PyTorch Tutorials 2.10.0+cu128 documentation, accessed January 30, 2026, https://docs.pytorch.org/tutorials/recipes/recipes/timer_quick_start.html
Download Python source code: benchmark.py, accessed January 30, 2026, https://docs.pytorch.org/tutorials/_downloads/72c2f17ac50228049705f9a4d76c7815/benchmark.py
