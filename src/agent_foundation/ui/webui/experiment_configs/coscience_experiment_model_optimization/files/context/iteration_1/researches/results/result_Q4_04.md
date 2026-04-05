Optimizing Memory Bandwidth & Kernel Efficiency in PyTorch
Broadcasting vs. Explicit .expand() (TorchInductor Behavior)

TorchInductor’s handling of broadcasting vs. expand: PyTorch’s new compiler backend, TorchInductor, is designed to optimize away differences between explicit expansions and implicit broadcasting. In fact, Inductor actively eliminates view operations like .expand and removes broadcasting overhead by adjusting tensor shapes in its IR. This means that whether you write code using an explicit tensor.expand(...) or rely on PyTorch’s automatic broadcasting, Inductor will treat them similarly under the hood. The expand becomes just a stride-based view with no data copy, and Inductor’s codegen will generate loops that handle the broadcasting without extra memory traffic. In essence, Inductor does not require manual .expand() to optimize kernels – it will fuse and handle broadcasts efficiently on its own. (Early versions had minor broadcasting bugs, but those have since been resolved.) For developers, this implies that using PyTorch’s natural broadcasting (Pattern B) is typically preferred; Inductor will compile it into efficient code that is on par with or better than a manual expand. Overall, there should be little to no performance difference between explicit expand vs. implicit broadcast when using torch.compile (Inductor), since Inductor normalizes both patterns into the same optimized operations.
When Does .expand() Allocate Memory vs. Remain a View?

By design, tensor.expand(new_shape) does not allocate new memory – it returns a view that virtually tilts or repeats the tensor along the expanded dimensions. The expanded tensor shares the same underlying data, using stride 0 in the expanded dimension to refer to the original data. In normal use, this means .expand() is cheap: you can expand a [1 × N] tensor to [batch_size × N] without copying data. For example, broadcasting a 20×1 and 1×20 tensor to perform an elementwise multiply showed no increase in memory usage compared to using .expand beforehand. Both approaches only allocate memory for the result, not for any expanded intermediate.

However, there are cases where an expanded view might trigger a real allocation down the line. The key point is that an expanded tensor often has non-contiguous strides, and certain operations may require contiguity or a full copy:

    If you call .contiguous() or .clone() on an expanded tensor, it will allocate a new tensor with the expanded data (since making it truly contiguous requires copying the repeated values). In other words, if an operation explicitly needs a contiguous memory layout, the expanded view must be materialized into real memory.

    Using .view() on a non-contiguous expanded tensor can error or force a copy. For example, attempting to .view an expanded tensor into a new shape is not possible without it being contiguous. You’d need to use .reshape(), which may allocate memory to maintain the data layout.

    In-place writes to an expanded tensor are restricted. You generally should not modify an expanded view in-place, because a single memory element corresponds to multiple indices of the expanded tensor. PyTorch will either prevent such operations or they will write to the base tensor (affecting all “expanded” positions simultaneously). For safety, if you need to modify expanded data independently, you’d have to allocate a full copy.

In summary, .expand() itself remains a view and avoids allocation in all typical read-only scenarios. It only causes an allocation if you subsequently perform an operation that inherently requires a distinct physical memory layout (like making it contiguous or altering values in separate expanded positions). Under normal use (e.g., doing math with the expanded tensor), no additional memory is consumed beyond the original tensor and the final result.
Broadcasting and Memory Bandwidth in Attention Computation

Memory-bandwidth bottlenecks in attention: Modern attention mechanisms (e.g. Transformer self-attention) are often limited by memory transfers rather than compute. In fact, profiling shows that in a naive attention implementation, most of the time is spent on memory-intensive steps like applying masks, softmax, and dropout, not on the large matrix multiplications themselves. The GPU’s compute units are extremely fast at matrix math (like Q·Kᵀ), but they often sit idle waiting for data because the huge matrices (queries, keys, values, and intermediate scores) have to be read/written from GPU memory (HBM) repeatedly. In other words, attention is typically memory-bound, meaning the throughput of memory accesses (GB/s) is the limiting factor, more so than FLOPs.

How broadcasting helps: Broadcasting can alleviate some memory pressure by avoiding unnecessary data replication and transfers. When computing attention scores and applying masks or biases, using broadcasting means we do not create large intermediate tensors in memory for the expanded mask or bias. For example, if you have an attention mask of shape [Batch, 1, 1, SeqLen] that needs to be added to the scores of shape [Batch, Heads, Qlen, SeqLen], you can rely on broadcasting instead of explicitly expanding the mask. This way, the addition kernel will fetch each mask value as needed (likely leveraging cache) rather than reading a huge pre-expanded mask from global memory. The GPU threads will repeatedly use the same small set of mask values across many positions, which is cache-friendly. In contrast, if you explicitly expanded the mask to [Batch, Heads, Qlen, SeqLen] in memory (a massive tensor), adding it would require streaming that entire expanded mask through memory. That would be an enormous increase in memory traffic, given that Heads*Qlen copies of the mask would be read. Broadcasting avoids that overhead – the expansion is done conceptually in registers/on-chip, not by actual data duplication in HBM. This reduction in memory access is crucial because, as noted, attention’s performance is often constrained by memory bandwidth.

In essence, broadcasting keeps the attention computation more IO-efficient. The GPU only stores and moves the minimal data necessary (e.g. the small mask tensor and the large score tensor) and doesn’t shuffle around extra copies. Every time we can eliminate a large memory read/write, we help feed the compute units faster. Modern fused attention kernels (like FlashAttention) take this to the extreme by never materializing the full score matrix in memory at all – they compute and apply softmax in tiles to stay in fast on-chip memory. While our context is simpler (broadcast vs. expand), the same principle applies: fewer and smaller memory accesses = better performance in memory-bound operations. Broadcasting ensures attention mask and bias additions incur minimal memory overhead beyond the core operations.
Memory Footprint for Batch Size 256K: Expand vs. Broadcast

With an extremely large batch (256,000 samples), any extra memory allocation is costly. Fortunately, when using the two patterns in question – explicit expand vs. implicit broadcast – the memory footprint is essentially the same, as long as expand is only creating a view. In both Pattern A and Pattern B, no intermediate giant tensor needs to be stored in memory if done correctly:

    Pattern A (explicit expand + op): tensor.expand(batch_size, ...) creates a view, not a real tensor copy. Thus, right before the operation, you still only hold the original tensor’s data in memory. The subsequent elementwise operation (e.g. addition) will produce an output tensor of the broadcasted shape, which is the same output you’d get with implicit broadcasting.

    Pattern B (implicit broadcasting in-kernel): PyTorch will internally handle the broadcast during the operation without ever allocating an expanded tensor. It reads the smaller tensor as needed for each chunk of the output computation. The only additional memory usage is for the output of the operation.

As evidence, a simple test showed that multiplying two tensors with broadcasting versus doing an explicit expand yielded virtually no difference in peak memory usage. This indicates that PyTorch isn’t making a large temporary copy in either case. So for a batch of 256K, if you add, say, a [256000 × D] tensor with a [1 × D] tensor, the memory will primarily be used by the input tensors and the [256000 × D] result – not by an extra expanded copy of the smaller tensor. The .expand view costs almost nothing extra. The implicit broadcast also avoids any extra allocation.

However, it’s important to ensure that .expand remains a view. If one were to inadvertently convert that view into a real tensor (for example, by calling .contiguous() on the expanded result or using an operation that forces a copy), the memory footprint would blow up. For instance, expanding a [1 × 128] tensor to [256000 × 128] logically represents ~32 million elements; if that were materialized, it would consume on the order of 120–125 MB of memory (assuming 32-bit floats). Doing that repeatedly or for many such tensors would quickly exhaust memory. So the bottom line is: Under normal usage, Pattern A and Pattern B have the same low memory footprint. The difference comes only if you accidentally cause real expansion. Stick to pure broadcasting operations (or views that stay as views), and even at batch=256K you won’t pay an extra memory cost.
Hoisting Common Operations Out of Conditionals

Identifying and hoisting common sub-operations: The example given shows a typical scenario where an operation (unsqueeze(1)) was performed in both branches of an if/else. To systematically catch these, you can follow a few strategies:

    Code inspection & refactoring: Manually look at your conditional branches for duplicate computations. If two branches of an if perform the same tensor operation (like reshaping, unsqueezing, or a costly function) on the same input, that’s a candidate to hoist. In the example, both branches called tensor.unsqueeze(1) before passing it to func, so it was safe to do that once before the if. By doing so, we removed the redundancy. As a rule of thumb, factor out any branch-invariant work – do it outside the conditional, store it in a temporary variable, and then inside the if/else just use that result or apply the small differing step (like a dtype conversion). This not only improves efficiency but also simplifies the code.

    Leverage context managers or wrappers for minor differences: In the example, the only difference between branches was whether to cast to bf16. This was elegantly handled by hoisting the unsqueeze and then using with torch.autocast(...): inside which the function is called. Another approach could be to perform the cast inside the function func if possible, or use a conditional operator on a tensor (though for casting, autocast is cleaner). The goal is to avoid duplicating the main operation pipeline across branches. Identify what truly must happen differently in each branch and separate it from what can be shared.

    Automated detection (advanced): While standard Python won’t automatically do this for you in eager mode, tools do exist in more static contexts. For example, PyTorch’s JIT compiler (TorchScript) includes an optimization pass for common subexpression elimination (CSE) in the graph. This means if you script your model and both branches end up containing the same sub-computation, the JIT might merge them. However, in dynamic Python code, the runtime can’t easily “see ahead” to eliminate redundant ops across branches – each branch executes separately if and when its turn comes. In fact, it’s noted that in a purely dynamic graph, CSE isn’t feasible because the framework cannot look across different execution paths in advance. Therefore, it’s on the developer (or a compiler that traces both paths) to handle this. Using torch.compile (Inductor) might also eliminate some redundant view operations or such, but it’s safest not to rely on it for complex Python control flow.

Hoisting best practices: Think of it like algebraic simplification of code. If you have:

if condition:
    y = f(x) + 5
else:
    y = f(x) + 3

you can hoist temp = f(x) and then just do temp+5 or temp+3 in each branch. The same concept applies to tensor ops. Identify f(x) as common, do it once. In large codebases, you can search for identical code snippets in both branches or use diff tools on the branches. Also, keep an eye on your profiler output – if a certain operation appears twice when logically it could be done once, that’s a sign. By systematically hoisting common operations, you reduce duplicate work and potentially improve performance (even if each instance is minor, it adds up across thousands of iterations). This practice also tends to make the code cleaner and less error-prone. In summary, hoist invariant computations out of conditionals whenever possible, just as was done with the unsqueeze example, to eliminate redundant tensor ops and streamline the kernel execution flow.

Sources: The information above is drawn from PyTorch documentation and performance guides. Notably, PyTorch’s own docs emphasize that Tensor.expand is a view operation with no memory cost, and memory profiling confirms no difference between using expand vs. direct broadcasting. The TorchInductor compiler internals are described in an official blog, which highlights elimination of views and broadcast overhead. Discussion of attention performance comes from research blogs that explain attention is typically memory-bandwidth-bound and spends much of its time in elementwise ops and data movement. These sources collectively support the strategies outlined for optimizing memory usage and kernel efficiency in large-batch PyTorch models.
Citations

Inductor notes – Ian’s Blog
https://ianbarber.blog/2024/01/16/inductor-notes/

[inductor] Cannot broadcast, the expanded size of the tensor · Issue #1654 · pytorch/torchdynamo · GitHub
https://github.com/pytorch/torchdynamo/issues/1654

python - Does pytorch broadcast consume less memory than expand? - Stack Overflow
https://stackoverflow.com/questions/65900110/does-pytorch-broadcast-consume-less-memory-than-expand

python - Does pytorch broadcast consume less memory than expand? - Stack Overflow
https://stackoverflow.com/questions/65900110/does-pytorch-broadcast-consume-less-memory-than-expand

torch.view() after torch.expand() complains about non-contiguous tensor · Issue #47146 · pytorch/pytorch · GitHub
https://github.com/pytorch/pytorch/issues/47146

Why FlashAttention?. On GPU Memory Bandwidth, Tiling and… | by Katherine Oluwadarasimi Olowookere | Dec, 2025 | Medium
https://medium.com/@katherineolowookere/why-flashattention-4b0f6cca8653

Why FlashAttention?. On GPU Memory Bandwidth, Tiling and… | by Katherine Oluwadarasimi Olowookere | Dec, 2025 | Medium
https://medium.com/@katherineolowookere/why-flashattention-4b0f6cca8653

Why FlashAttention?. On GPU Memory Bandwidth, Tiling and… | by Katherine Oluwadarasimi Olowookere | Dec, 2025 | Medium
https://medium.com/@katherineolowookere/why-flashattention-4b0f6cca8653

Lowering Phase — Torch-TensorRT v2.11.0.dev0+f60963f documentation
https://docs.pytorch.org/TensorRT/contributors/lowering.html

NNC Dynamic Graph Execution — nnc, a deep learning framework from ccv
https://libnnc.org/tech/nnc-dy/
All Sources
ianbarber
github
stackoverflow
medium
docs.pytorch
libnnc
