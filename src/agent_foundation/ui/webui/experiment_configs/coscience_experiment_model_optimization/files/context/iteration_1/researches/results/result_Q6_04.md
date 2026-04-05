Benchmark Tool & Kernel Profiling Analysis

Goal: Enhance the existing PyTorch benchmarking script to validate kernel-level optimizations. We want to measure not just end-to-end latency, but detailed GPU kernel metrics – including total kernel launches, launch overhead vs execution time, which attention kernel (SDPA) backend was used, and per-component timing – while providing A/B comparisons and Chrome trace visualization for deeper analysis.
Counting Total GPU Kernel Launches per Pass

In PyTorch’s eager mode, each high-level operation typically triggers a separate CUDA kernel launch. For example, an elementwise activation after a matrix multiply would normally be a second kernel launch following the GEMM. The new torch.compile fused execution can drastically reduce this count by merging ops (e.g. turning a linear+ReLU sequence that used 2 kernels into 1 fused kernel). To count the total GPU kernels in a forward/backward pass, we can leverage PyTorch Profiler to capture all CUDA kernel events:

    Use torch.profiler: Wrap the model forward (and backward) in torch.profiler.profile with CUDA activities enabled. Every GPU kernel executed will be recorded as an event. For each operator invocation on GPU, the profiler logs a CPU launch call and a corresponding GPU kernel event. Thus, counting the GPU events equates to counting kernel launches.

import torch
from torch.profiler import profile, ProfilerActivity

model = ...  # the PyTorch model
inputs = ...  # example input tensor on CUDA
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(inputs)       # forward pass
    output.sum().backward()      # backward pass if needed
# Count GPU kernel events:
gpu_kernels = [evt for evt in prof.events() if evt.device_type == 'cuda']
print(f"Total CUDA kernels launched: {len(gpu_kernels)}")

This will iterate over all profiler events and filter those executed on the CUDA device. Each such event represents a launched GPU kernel (either an ATen op kernel or a fused kernel). In our optimized vs baseline scenario, we expect to see the compiled model launch significantly fewer kernels (e.g. a reduction from hundreds of kernels down to a few dozen, as claimed by torch.compile fusion) – which this count will verify.

Validation: The profiler’s Chrome trace and textual summary can confirm the count. In Chrome’s trace view (see section on Chrome trace below), each GPU kernel appears as a block on the GPU timeline. The number of these blocks per iteration corresponds to kernel launches. The profiler’s tabular output (prof.key_averages()) can also show how many times each op ran; summing those can cross-check the total. Each fused region will appear as one “CompiledFunction” or Triton kernel launch instead of many smaller ops. By collecting these metrics, we can confirm claims like “221 → 30 kernels” after fusion by directly comparing the counts from the profiler logs.
Measuring Kernel Launch Overhead vs Execution Time

Launching many small kernels can degrade performance due to CPU overhead, where the GPU is often idle waiting for the next launch. We need to separate the launch overhead (CPU-side delay between kernels) from the actual GPU execution time of the kernels:

    Identify Gaps: In the Chrome trace timeline, large gaps between GPU kernel events indicate the GPU is underutilized and waiting on the CPU. Below is a visualization showing noticeable idle gaps (gray areas) between GPU kernels, caused by launch overhead on the CPU side:

Gaps between GPU kernels indicate CPU-side overhead (GPU is idle waiting for kernels to launch).

    Use CUDA Events: For precise measurement, insert CUDA events around the workload. A pair of torch.cuda.Event objects can measure pure GPU execution time without blocking the CPU. By recording an event right before and after the model execution, and then calling start.elapsed_time(end), we get the total time spent on GPU kernels. This method inherently excludes most launch latency because it times actual GPU work between events. For example:

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
output = model(inputs)
output.sum().backward()
end.record()
torch.cuda.synchronize()  # wait for GPU work to finish
gpu_time_ms = start.elapsed_time(end)

This yields the aggregated GPU execution time for the forward/backward pass (in milliseconds). By contrast, a simple wall-clock timing around the same code (with synchronization) would include both GPU execution and any launch overhead. The difference between the wall-clock time (with torch.cuda.synchronize() to ensure all work is done) and the GPU event time approximates the launch overhead. In other words:

    Total elapsed time (CPU perspective, including overhead) = GPU execution time + CPU launch overhead.

Important: Always synchronize before stopping a timer on the host. If we measure time on the CPU without synchronization, we may only capture the cost of launching kernels, not their execution. Calling torch.cuda.synchronize() ensures the CPU waits for all kernels to finish, giving an accurate end-to-end time.

Using the above approach, our enhanced benchmark can report, for example: “Total step time = 12 ms, GPU kernels execution = 10 ms, launch overhead ≈ 2 ms.” This quantifies how much time is lost to dispatching kernels. A high overhead (manifested as large idle gaps on GPU) suggests the model is CPU-bound in launching kernels. In such cases techniques like CUDA Graphs (which batch kernel launches) can help reduce overhead.
Profiling SDPA Backend Selection (FlashAttention vs Math)

Modern PyTorch uses a optimized Scaled Dot Product Attention (SDPA) that can choose among multiple implementations (backends) at runtime:

    Flash Attention – fused, GPU-optimized kernel (requires GPU support, e.g. NVIDIA SM80+ for certain sizes).

    Memory-Efficient Attention – an XFormers-based streaming attention (also GPU optimized but more broadly applicable).

    Math (Standard) Attention – the fallback implementation using regular matrix multiply + softmax ops on any device.

We need to verify which backend was used during a run (especially on GPUs like A100 vs. H100). To do this, we can rely on profiler logs and PyTorch’s APIs:

1. Use Profiler Events: If the model uses PyTorch’s built-in attention (e.g. torch.nn.MultiheadAttention or F.scaled_dot_product_attention in PyTorch 2.x), the chosen backend is reflected in the operator names. In a profiler trace or summary:

    Flash Attention in use: You will see ops named _scaled_dot_product_flash_attention (and underlying kernels like _flash_attention_forward) in the trace. These indicate the fused flash-attention kernel ran.

    Math/backward fallback: If flash was not used, the attention will be computed via standard ops (matmul + softmax, etc.). The profiler would then show separate kernels such as aten::matmul, aten::softmax, etc., rather than a single fused SDPA op. In other words, the absence of the flash-specific op is a clue that the math path executed. The “math” implementation even does an extra scaling of Q and K for numerical stability (launching a couple of aten::div kernels) – seeing those div ops in the profile is another hint that the math backend ran.

For example, a profiler snippet might look like:

aten::_scaled_dot_product_flash_attention         [runs] 1    [GPU time] 2.8ms
    └─ … flash_fwd_kernel (CUDA)                  [runs] 1    [GPU time] 2.8ms

This indicates the FlashAttention path was taken. By contrast, if the profile instead shows a series of matrix multiplies and softmax without any _flash kernel, the math backend was used.

2. Programmatic Check (Optional): PyTorch provides a context manager and flags in torch.backends.cuda to control and query SDPA backends. You can temporarily force each backend to verify behavior:

from torch.backends.cuda import sdp_kernel, SDPBackend
# e.g., force math only
with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    out = F.scaled_dot_product_attention(q, k, v, attn_mask, 0.0, is_causal)

However, simply observing the default behavior with the profiler is usually sufficient. Additionally, functions like torch.backends.cuda.is_flash_attention_available() can tell if flash kernels are built for the current hardware (H100 GPUs support FlashAttention v2 which can handle larger sizes than A100, etc.), and torch.backends.cuda.can_use_flash_attention(params, debug=True) will even log why flash was or wasn’t chosen for given Q,K,V tensors. These tools can be used to double-check why a certain backend was selected (e.g., flash might be unavailable if sequence length or head dimension is out of supported range).

In summary, our benchmark will use the profiler output to identify the SDPA kernel: seeing a flash-attention kernel event confirms the Flash backend, whereas its absence (and presence of standard ops) implies the math path. This can be printed or logged for each run (“Attention backend used: FlashAttention” or “used math fallback”), providing transparency about which GPU kernel was actually employed.
Structuring A/B Benchmarking for Kernel Optimizations

When comparing two scenarios – e.g. baseline vs. optimized (compiled) – it’s crucial to structure the benchmark to get fair, reliable measurements. Here are best practices for A/B testing kernel performance:

    Isolate Runs: If possible, run each configuration separately to avoid interference. For instance, run the baseline model in its own block (or even separate process) and the optimized model in another. This prevents any GPU memory caching or CUDA context effects from giving one an unintended advantage. If running in one script, ensure you reset any global state that might carry over (cache allocator, etc., though PyTorch’s caching usually won’t dramatically skew relative timings).

    Warm Up First: Always perform a few warm-up iterations for each model before timing. The first iteration can include one-time overheads like JIT compilation (for torch.compile), CUDA memory allocator initialization, cuDNN autotuning, etc.. Discard these initial iterations. For example:

    for _ in range(3):
        model(inputs); torch.cuda.synchronize()

    (Do this for both A and B before measuring.)

    Use Synchronized Timing: Time each trial with proper synchronization. As discussed, wrap the forward/backward in torch.cuda.synchronize() calls (or use CUDA events for precision). This ensures each measured iteration truly finished all GPU work. A common pattern is:

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = model(inputs); output.sum().backward()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    By synchronizing before and after, we capture the full latency of the pass. (Without the initial sync, if a previous iteration is still finishing asynchronously, it could skew the timing.)

    Repeat and Average: Run multiple iterations for each mode and report an average or median. GPU timings can vary slightly run to run. For instance, do 10 timed iterations on baseline (after warmup) and 10 on optimized, then compare the means. This helps smooth out noise (like occasional OS interrupts or thermal throttling).

    Alternate Order (if in one process): To mitigate any order-dependent effects (like cache warmup benefiting the second model), you can alternate runs A and B or run B first on some trials. However, if each is warmed and run in isolation, this is less of a concern. The key is consistency: use the same input data, batch size, and environment for both. Also ensure any random seeds are fixed if the model has stochastic behavior, so that each sees an equivalent workload.

    Collect Kernel Stats Too: Besides timing, use the profiler or counters in each run to collect the metrics discussed above (kernel count, etc.). Our benchmark can output something like:
    Config	Time per iter (ms)	GPU kernels	SDPA Backend	Memory used (MB)
    Eager (no compile)	15.3 ms	221	Math	800 MB
    Compiled	9.8 ms	30	Flash	780 MB

    This kind of summary makes the improvements clear and ties together the analysis.

    Consider Overheads in Measurement: Be mindful that enabling the profiler itself adds overhead. For pure timing comparison, do one set of runs with minimal instrumentation (just timers and sync). Use the profiler in separate runs to dig into details. This way the act of profiling doesn’t skew the timing comparison too much. If you do profile both A and B, ensure both are equally affected (the PyTorch profiler will slow both similarly).

    GPU Steady-State: Ensure the GPU is in a comparable performance state for both runs. If one model runs right after a heavy workload, the GPU might be hotter or power-throttled. It’s best to run experiments on an otherwise idle machine and even fix GPU clocks for consistency if needed (NVIDIA’s nvidia-smi -lgc command can lock clock rates). This level of rigor might be overkill for quick checks, but it’s noted for completeness.

By following these steps, the A/B benchmark will yield a fair comparison. For example, you might find the compiled model is ~1.5× faster per iteration and launches ~⅓ the number of kernels. The structure above ensures that conclusion is based on accurate and reproducible measurements, not artifacts of the testing method.

Side note: In one of our profiling exercises, we noticed an unexpectedly low iteration time for the optimized model until we added an explicit torch.cuda.synchronize() at the end of each iteration. Without that, the CPU moved on while the GPU was still processing in the background, giving an illusion of faster time. After synchronization, the true latency was reported (e.g. ~10 ms instead of ~5.6 ms). This highlights why careful synchronization is essential in benchmarking.
Attributing Kernel Time to Model Components (Embedding vs Attention vs MLP)

To understand which parts of the model consume the most time (e.g. embedding layer vs. attention vs. MLP feed-forward), we should break down the profiling by component. There are a couple of approaches:

1. Manual timing or profiler ranges: The simplest method is to instrument the forward pass by inserting timing or profiler labels around each component. PyTorch’s torch.profiler.record_function context manager is very useful here – it lets you mark custom regions in the code which appear as labeled events in the profiler. For example, if the model’s forward method looks like:

def forward(self, x):
    with torch.profiler.record_function("Embed"):
        x = self.embedding(x)
    with torch.profiler.record_function("Attention"):
        x = self.attention_layer(x)
    with torch.profiler.record_function("MLP"):
        x = self.mlp_block(x)
    return x

Each record_function block will group the operations (and kernels) executed inside it under a named label. In the Chrome trace or profiler output, you will then see these high-level labels, and you can aggregate timing by them. This approach was demonstrated in a custom loss function profiling example: the author wrapped each line (log-softmax, weight initialization, NLL computation) with a record_function label, which made it easy to identify which step was the bottleneck. In our case, we’d label "Embed", "Attention", "MLP" as above. The profiler’s key_averages can then report the total time spent in each labeled region.

2. Module-wise timing: If modifying the model code is undesirable, another approach is to time sub-modules externally. For instance, you could fetch the input to each sub-component and run that module separately to measure its time. However, this can be complicated for intertwined models (and might disrupt the GPU pipeline). Using record_function is simpler and non-intrusive (it only adds profiling markers, without changing computation).

3. Analyzer tools: The PyTorch profiler with the TensorBoard plugin or the Chrome trace UI also lets you inspect the call stack. If the model is structured such that embedding, attention, MLP are separate submodules (which is common in Transformer architectures), the profiler can aggregate by the module call. In Chrome trace, you can often expand the CPU thread view to see functions calls (if you ran with with_stack=True). However, this is less straightforward than simply adding named ranges as above.

We recommend instrumenting the code with record_function for clarity. For example:

with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(inputs)  # model.forward has record_function blocks for components
prof.key_averages(group_by_input_shape=False).table(sort_by="self_cuda_time_total")

In the printed table (or in TensorBoard), you will see entries like “Embed”, “Attention”, “MLP” with their inclusive times. This directly tells you, say, 40% of the time is in attention, 50% in MLP, etc. You can further break it down (e.g., within “Attention” you might add labels for QKV projection, SDPA, output projection if needed). This fine-grained attribution is extremely helpful for pinpointing which part benefits from optimizations. For instance, if after fusion you still see “MLP” taking the bulk of time, you might focus next on optimizing the MLP block (like GEMM autotuning or fusion there).

Using this method, we ensure the benchmark not only reports overall speed but also provides a per-component profile. This way, any regressions or improvements in specific sub-components after an optimization can be observed (e.g., “Attention kernels reduced from 8ms to 3ms after enabling FlashAttention, while Embedding remained ~1ms”). It’s a form of performance “budget” breakdown for the model.
Chrome Trace Export for Visual Analysis

Finally, for deeper analysis and verification, the benchmark should export a Chrome trace file. PyTorch profiler can save the timeline of events to a JSON trace that Chrome’s built-in tracer (chrome://tracing) can load. We can do this easily with prof.export_chrome_trace("trace.json") after profiling:

with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                             record_shapes=True) as prof:
    # ... run model as needed, possibly multiple steps with prof.step()
    pass
prof.export_chrome_trace("benchmark_trace.json")

This will produce a benchmark_trace.json file. Opening Chrome and loading this file in the chrome://tracing UI allows us to visually inspect the timeline of CPU and GPU activities. In the trace view you can zoom and scroll to see details:

    CPU events (top) showing Python functions, launch calls (cudaLaunchKernel), etc.

    GPU events (bottom) showing each kernel execution and duration.

    Colored bars or groupings for our record_function labels (e.g. “Embed”, “Attention”) if added, which helps visually segment the model’s timeline.

    Async “flow events” that connect CPU launch events to GPU kernel events – useful to see which CPU task launched which GPU kernel.

Using the trace, we can visually confirm things like kernel overlap (or lack thereof), launch gaps, and the order of operations. For example, if using torch.compile, the trace will show regions labeled “CompiledFunction” and you might observe far fewer GPU events, aligned with the fused kernels. The earlier image we embedded of gaps was taken from such a trace; you could similarly capture a screenshot to include in reports, highlighting how an optimization removed the idle gaps between kernels.

Moreover, the Chrome trace is invaluable for sanity-checking our numeric counts. If our script says “30 kernels”, one can literally count ~30 GPU events in the trace for that iteration. It also helps in identifying any outlier slow kernels or unusual events (e.g., CPU JIT compile steps, synchronization points, etc., will appear on the timeline).

Summary: By implementing the above enhancements, our benchmark script will be able to:

    Count and compare the number of GPU kernel launches in each scenario (eager vs compiled).

    Measure and separate kernel launch overhead from execution time using synchronization and CUDA events.

    Determine which SDPA (scaled dot-product attention) backend was used (FlashAttention vs others) during runtime.

    Provide structured A/B comparisons with proper warmup and synchronized timing to validate performance gains from kernel optimizations.

    Attribute execution time to high-level model components (embedding, attention, MLP) via profiling labels.

    Export a Chrome tracing file for visual analysis, allowing deeper inspection of kernel-level behavior and verification of the above results.

With these tools and methods in place, we can confidently validate the kernel-level optimizations on hardware like A100, H100, and future H200 GPUs. (Notably, newer GPUs may unlock additional fused kernels or backend capabilities – e.g. H100 supports FlashAttention v2 – but the profiling approach remains the same.) The end result is a comprehensive performance profile that goes beyond total latency, illuminating why one version is faster and how exactly the GPU workload has been improved.

Sources:

    PyTorch Profiler documentation and usage examples

    PyTorch 2.x SDPA (scaled dot product attention) backend info

    Best practices for CUDA timing and synchronization

    PyTorch performance tuning blogs (kernel fusion, overhead)

    Case study on profiling model components with record_function

Citations

Accelerating PyTorch Models: Inside torch.compile’s Kernel Optimization | Abhik Sarkar
https://www.abhik.ai/articles/compiling-pytorch-kernel

Accelerating PyTorch Models: Inside torch.compile’s Kernel Optimization | Abhik Sarkar
https://www.abhik.ai/articles/compiling-pytorch-kernel

Profiling to understand torch.compile performance — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html

Profiling to understand torch.compile performance — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html

How to Accurately Time CUDA Kernels in Pytorch
https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch

How to Accurately Time CUDA Kernels in Pytorch
https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch

How to Accurately Time CUDA Kernels in Pytorch
https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch

Profiling to understand torch.compile performance — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html

Profiling to understand torch.compile performance — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html

Out of the box acceleration and memory savings of decoder models with PyTorch 2.0 – PyTorch
https://pytorch.org/blog/out-of-the-box-acceleration/

(Beta) Implementing High-Performance Transformers with Scaled Dot Product Attention (SDPA) — PyTorch Tutorials 2.7.0+cu126 documentation
https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html

Out of the box acceleration and memory savings of decoder models with PyTorch 2.0 – PyTorch
https://pytorch.org/blog/out-of-the-box-acceleration/

torch.backends — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/backends.html

torch.backends — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/backends.html

torch.backends — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/backends.html

Out of the box acceleration and memory savings of decoder models with PyTorch 2.0 – PyTorch
https://pytorch.org/blog/out-of-the-box-acceleration/

How to Accurately Time CUDA Kernels in Pytorch
https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch

How to Accurately Time CUDA Kernels in Pytorch
https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch

PyTorch Model Performance Analysis and Optimization — Part 2 | by Chaim Rand | TDS Archive | Medium
https://medium.com/data-science/pytorch-model-performance-analysis-and-optimization-part-2-3bc241be91

PyTorch Model Performance Analysis and Optimization — Part 2 | by Chaim Rand | TDS Archive | Medium
https://medium.com/data-science/pytorch-model-performance-analysis-and-optimization-part-2-3bc241be91

PyTorch Model Performance Analysis and Optimization — Part 2 | by Chaim Rand | TDS Archive | Medium
https://medium.com/data-science/pytorch-model-performance-analysis-and-optimization-part-2-3bc241be91

PyTorch Model Performance Analysis and Optimization — Part 2 | by Chaim Rand | TDS Archive | Medium
https://medium.com/data-science/pytorch-model-performance-analysis-and-optimization-part-2-3bc241be91

Profiling to understand torch.compile performance — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html

Profiling to understand torch.compile performance — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html

Profiling to understand torch.compile performance — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html

Accelerating PyTorch Models: Inside torch.compile’s Kernel Optimization | Abhik Sarkar
https://www.abhik.ai/articles/compiling-pytorch-kernel
All Sources
abhik
docs.pytorch
speechmatics
pytorch
medium
