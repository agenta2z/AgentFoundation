Performance Optimization in Large-Scale PyTorch
Models
Reducing Kernel Launch Count with Native Slicing
Using native slicing instead of a combination of torch.arange plus torch.index_select can
significantly reduce GPU kernel launches. In the original pattern, torch.arange creates an index tensor
and index_select performs a gather, resulting in two separate GPU kernels (one to generate indices,
one to index) . Replacing this with tensor slicing (e.g. x[:, start::step, :] ) allows the operation to
be handled as a single strided view or copy, effectively cutting the kernel launch count from 2 to 1 in the
forward pass. Fewer kernel launches mean less scheduling overhead and can improve throughput .
Moreover, slicing returns a view on the original tensor without immediately copying data , so it avoids
the extra memory kernel that index_select would use to copy indexed elements into a new tensor.
Fusion of Slice Operations in TorchInductor
PyTorch 2.x’s Inductor compiler can fuse slice operations with other computations when
torch.compile() is used (especially with torch._inductor.config.combo_kernels = True ).
TorchInductor’s loop-level IR treats slicing or indexing as just an offset/stride in memory, which makes it
easy to combine with subsequent operations in one fused kernel . For example, Inductor can compile an
expression like y = x[:, ::2] * 2 into a single GPU kernel, internally handling the strided access
pattern in the loop index formula . With combo_kernels=True enabled, Inductor aggressively groups
operations, so a slice followed by element-wise transforms or reductions will likely run as one fused Triton
kernel rather than multiple kernels. This means that slice views do not inherently trigger separate
kernels – they are merged into larger kernels, reducing launch overhead.
Eliminating Intermediate Index Tensors (Memory Savings)
Removing the intermediate index tensors (from torch.arange or similar) yields notable memory
allocation savings. In the original approach, torch.arange produces a long index tensor (on GPU if not
careful), and index_select then allocates an output tensor for the selected elements. By using slicing,
we avoid allocating a separate index tensor and often avoid an immediate data copy for the output (when
using a view). This saves both GPU memory and memory bandwidth. In particular, avoiding
index_select bypasses its internal index expansion and copy overhead , which can involve extra
host-device transfers or buffer creations. Overall, using a slice view means no new storage is needed for
indices or the sliced output (unless we explicitly copy it), whereas index_select “returns a new tensor
which copies the indexed fields into a new memory location” . This reduction in intermediate allocations
not only saves memory (beneficial in a 5B+ parameter model), but also can improve performance by
reducing GPU memory fragmentation and freeing up bandwidth for actual model computation.
1
2
1
3
4
5
1
1
Adding .contiguous() for Backward Pass Optimization
When a tensor is sliced with a stride (for example, x[:, ::2, :] taking every 2nd element), the result is
a non-contiguous view of the data. In these cases, appending .contiguous() is a recommended
optimization for the backward pass. Making the slice contiguous will create a compact copy in memory ,
which can substantially speed up gradient computations. The backward pass benefits in several ways:
Better cache locality for gradient accumulation – contiguous memory ensures that accumulating
gradients (often a sequential memory write) hits cache lines efficiently.
Coalesced memory access in CUDA kernels – GPUs achieve peak throughput when threads access
memory in uniform, sequential patterns . Contiguous gradients allow warp memory accesses
to be coalesced, whereas a strided, non-contiguous gradient would lead to scattered reads/writes
that underutilize the GPU.
Faster autograd ops on contiguous tensors – Many PyTorch operations (and their backwards) are
optimized for contiguous inputs. Operations on non-contiguous gradients may internally need to
reorder data or use less efficient code paths. By ensuring the tensor is contiguous, we let autograd
use the most efficient implementation, avoiding any hidden overhead of handling strides.
In practice, you should add .contiguous() after slicing whenever the slice result will be used in
further computations or backward passes that are performance-critical. This is especially true if the
slicing involves a step (non-unit stride) or other transformations that produce a discontiguous view. For
example, taking every Nth element or transposing will make a tensor non-contiguous, so calling
.contiguous() before intensive matrix operations or before backward can yield a net performance gain.
If a slice is a simple narrow (continuous range in memory), it might already be contiguous in storage;
otherwise, using .contiguous() guards against slowdowns in later use.
Forward vs. Backward Trade-Off for .contiguous()
There is a trade-off when using .contiguous() : we incur a small forward-pass overhead in order to
gain a larger backward-pass speedup. The forward overhead comes from the explicit copy operation –
calling contiguous() on a non-contiguous slice allocates new memory and copies the slice data into a
dense layout . This is an extra kernel/memory operation in the forward pass, adding some latency.
However, the payoff is that the backward pass (and any further forward ops on that tensor) can run much
faster on contiguous data. In training, backward computations typically involve heavy tensor operations
(e.g. large matrix multiplications or gradient accumulations) that dominate runtime, so optimizing them
often outweighs a one-time copy cost. Non-contiguous memory access in backward can severely bottleneck
GPU utilization – as noted, scattered access patterns cause the GPU to spend cycles waiting on memory
rather than doing math . By paying a small cost upfront to make data contiguous, we ensure the
backward kernels can use optimal memory access patterns and fully leverage GPU throughput. In
summary, .contiguous() often yields a net positive trade-off: the minor extra work in forward is
compensated by a smoother and faster backward pass. The exception would be if the sliced tensor is only
used in lightweight operations or the model is inference-only; in those cases, you might skip
.contiguous() to save the copy. But for large-scale training (like this 5.1B parameter model), the
consensus is that making frequently used sliced outputs contiguous is worth it for the overall training
speed .
6
•
•
7 8
•
6
7 8
8
2
Benchmarking with torch.profiler (Methodology)
To quantitatively verify these optimizations, we can use torch.profiler to benchmark the model before
and after the changes. The methodology is as follows:
Setup Profiling Runs: Wrap the model’s forward + backward pass in a torch.profiler.profile
context for both the original code (using arange + index_select ) and the optimized code
(using slicing and .contiguous() where appropriate). Enable CUDA activity tracking so we
capture GPU kernel events . For example:
with
torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
as prof:
output = model(input_batch)
loss = output.sum()
loss.backward()
Run this for a few iterations to get stable measurements (include a warm-up iteration to account for
compile or caching effects) .
Measure Kernel Launches: Examine the profiler results to count GPU kernel launches in each
scenario. The profiler’s trace or aggregated stats ( prof.key_averages() ) will show each kernel
or operation. Specifically, look for the indexing operations: in the original run you should see a
kernel for torch.arange and one for index_select (or an advanced indexing kernel) per use,
whereas in the optimized run you should see fewer separate kernels for indexing (a contiguous copy
might appear if .contiguous() is used). By comparing the traces, confirm that the slicing
approach launches roughly half the number of kernels for those indexing parts of the model (2→1
kernel reduction per indexing operation).
Analyze Backward Pass Efficiency: Use profiler timestamps or CUDA kernel times to compare
backward pass durations. Pay attention to operations like gradient accumulation or scatter in the
original vs. optimized runs. We expect the backward phase to be faster when using contiguous
slices, which might show up as reduced time in functions like index-add (used in index select
backward) or overall lower backward compute time. The profiler can also report GPU memory
utilization and any extra memcpy operations. Verify that eliminating index tensors saves memory
allocations (the profiler’s memory view or even just monitoring
torch.cuda.memory_allocated() inside the profile scope can show lower peak memory in the
optimized version).
Hardware Considerations: Run these profiles on the target hardware (e.g. an NVIDIA H100 GPU,
and possibly an upcoming H200 if available) to capture any hardware-specific effects. Newer GPUs
might handle certain patterns more efficiently, but the relative improvements from reducing kernel
launches and using contiguous memory should hold universally. Ensure that Tensor Core usage (if
any) isn’t hindered by non-contiguous inputs – the profiler’s device metrics can confirm if you achieve
higher FLOPs utilization after optimization .
1.
9 10
11 12
2.
3.
4.
8
3
By following this profiling approach, you can concretely demonstrate the impact of the changes: fewer GPU
kernels launched in the forward pass, less intermediate memory allocated, and a faster backward pass
due to improved memory access patterns. All these should translate into end-to-end training speedups for
the recommendation model. The profiler data provides the evidence to quantify these benefits and ensure
no regressions were introduced in the forward pass.
illegalaccessexception - In pytorch, what is the difference between indexing with square brackets and
"index_select"? - Stack Overflow
https://stackoverflow.com/questions/69824591/in-pytorch-what-is-the-difference-between-indexing-with-square-brackets-and-in
Efficient PyTorch Programming Guide
https://www.allpcb.com/allelectrohub/efficient-pytorch-programming-guide
TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes - compiler -
PyTorch Developer Mailing List
https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
PyTorch Memory Management Strategies
https://apxml.com/courses/advanced-pytorch/chapter-1-pytorch-internals-autograd/memory-management
Flash Sparse Attention: An Alternative Efficient Implementation of Native Sparse Attention Kernel
https://arxiv.org/html/2508.18224v1
Profiling to understand torch.compile performance — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html
1
2 5
3 4
6
7 8
9 10 11 12
4
