Memory Bandwidth Optimization and Kernel Efficiency in High-Scale DLRMs: A Comprehensive Analysis of Tensor Expansion, Broadcasting, and Compiler Graph Capture
1. Executive Technical Synthesis
The optimization of Deep Learning Recommendation Models (DLRMs) operating at extreme batch sizes—specifically the 256,000-sample regime identified in the query—requires a fundamental shift in performance engineering strategy. At this scale, the primary constraint on system throughput transitions from arithmetic compute capability (FLOPs) to the memory subsystem's ability to supply data to the streaming multiprocessors (SMs). This report provides an exhaustive analysis of the interplay between explicit tensor expansion (.expand()) and implicit kernel-level broadcasting, specifically within the context of PyTorch’s execution engine and its Next-Generation compiler stack (torch._inductor).
Our analysis indicates that while explicit expansion (Pattern A) and implicit broadcasting (Pattern B) are semantically equivalent in an idealized execution environment, their practical implications diverge significantly when subjected to the complexities of graph capture, memory allocation heuristics, and legacy operator constraints. The distinction is not merely syntactic; it touches upon the fundamental mechanics of how PyTorch manages tensor views versus physical storage, and how the torch.compile stack lowers these high-level Python abstractions into optimized Triton kernels.
Furthermore, we address the architectural inefficiency introduced by conditional control flow in the forward pass. We provide a rigorous methodology for "hoisting" invariant operations to eliminate graph breaks—a phenomenon that fractures the compilation unit and forces costly exits to the Python interpreter. By fusing element-wise operations like Swish and LayerNorm and leveraging broadcasting semantics effectively, we demonstrate pathways to maximize arithmetic intensity and minimize memory transaction overheads, ensuring that the massive parallelism available on modern GPU architectures (e.g., NVIDIA A100/H100) is fully utilized rather than throttled by redundant global memory traffic.
2. Architectural Context: The Memory Wall in DLRMs
To understand the critical nature of the trade-offs between tensor expansion and broadcasting, one must first deconstruct the operating environment of a large-scale Recommendation Model. Unlike dense Large Language Models (LLMs) or Convolutional Neural Networks (CNNs) where compute intensity is often high due to massive matrix multiplications, DLRMs present a heterogeneous workload that places unique stress on the memory hierarchy.
2.1 The Physics of Batch Size 256K
Processing a batch size of 256,000 samples creates a distinct performance profile dominated by memory bandwidth utilization. In typical deep learning workloads, batch sizes range from 32 to perhaps 4,096. At 256,000, the intermediate activation tensors become enormous.
Consider a standard hidden layer dimension ($H$) of 1,024, typical in recommendation architectures. A single intermediate activation tensor for this batch size, stored in Brain Float 16 (bfloat16, 2 bytes per element), consumes:

$$\text{Memory}_{\text{tensor}} = \text{Batch} \times \text{Hidden} \times \text{Sizeof}(\text{bfloat16})$$

$$\text{Memory}_{\text{tensor}} = 256,000 \times 1,024 \times 2 \text{ bytes} \approx 524,288,000 \text{ bytes} \approx 500 \text{ MB}$$
A single tensor occupies half a gigabyte of GPU memory. In a deep network with hundreds of layers, the cumulative memory pressure is immense. More importantly, any operation that reads this tensor, performs a simple element-wise computation (like an activation function or a residual addition), and writes it back, must transfer 1 GB of data across the memory bus (500 MB read + 500 MB write).
On an NVIDIA A100 GPU with a peak memory bandwidth of approximately 1,555 GB/s (SXM4) or 1,935 GB/s (80GB version) 1, this operation has a theoretical minimum latency.

$$\text{Time}_{\text{min}} = \frac{1 \text{ GB}}{1,555 \text{ GB/s}} \approx 0.64 \text{ ms}$$
This calculation assumes perfect bandwidth utilization. In reality, efficiency losses due to non-contiguous memory access, DRAM page misses, and protocol overheads reduce effective bandwidth. If the model accidentally triggers a memory copy due to inefficient tensor handling (Pattern A), this latency doubles or triples, directly impacting training throughput.
2.2 The Arithmetic Intensity Gap
The central challenge in optimizing these kernels is Arithmetic Intensity ($AI$), defined as the ratio of floating-point operations performed to bytes accessed from the main memory.3

$$AI = \frac{\text{FLOPs}}{\text{Bytes Accessed}}$$
Operations like Matrix Multiplication (GEMM) have high arithmetic intensity because they perform $O(N^3)$ computations on $O(N^2)$ data. As the matrix size grows, $AI$ increases, allowing the workload to become compute-bound, where the GPU's tensor cores are the limiting factor.
Conversely, the operations highlighted in the user query—element-wise additions, Swish activations, and LayerNorms—are inherently memory-bound.
Element-wise Addition: $C = A + B$.
Reads: 2 elements ($A_i, B_i$).
Ops: 1 addition.
Writes: 1 element ($C_i$).
$AI \approx 1/3$ FLOPs/element (or significantly lower in terms of bytes).
At batch size 256K, these low-intensity operations saturate the memory bandwidth long before they tax the compute units. This is why "Memory Bandwidth" is cited as the bottleneck in the background description. The goal of optimization in this regime is not to reduce the number of calculations (FLOPs), but to reduce the number of bytes transferred to and from High Bandwidth Memory (HBM).
2.3 The Hierarchy of Broadcasting Economics
Broadcasting serves as a primary mechanism to artificially inflate Arithmetic Intensity. By engaging in an operation where one operand is large (Batch 256K) and one is small (e.g., a bias vector of size 1024), we change the memory equation.
Without Broadcasting (Materialized):
Read Tensor A (500 MB).
Read Tensor B (500 MB - explicitly expanded).
Write Result (500 MB).
Total Traffic: 1.5 GB.
With Broadcasting (Virtual):
Read Tensor A (500 MB).
Read Vector b (2 KB - stays in L2/L1 cache).
Write Result (500 MB).
Total Traffic: 1.0 GB.
The use of broadcasting immediately reduces the required memory bandwidth by 33%. In the context of the attention mechanism ($QK^T$), broadcasting allows us to apply masks or positional biases without paying the cost of reading a full $Batch \times Seq \times Seq$ matrix from global memory.
Therefore, the decision between Pattern A (Explicit Expand) and Pattern B (Implicit Broadcasting) is not merely stylistic. It is a decision about whether we expose the system to the risk of reverting to the "Without Broadcasting" scenario. If an explicit expand is materialized by a downstream operator, the 33% bandwidth saving is lost, and an additional 500 MB of capacity is consumed.
3. PyTorch Internals: Views, Strides, and Materialization Risks
To answer the research questions regarding when allocation occurs, we must dissect how PyTorch represents tensors in memory.
3.1 The Anatomy of a Tensor View
A PyTorch tensor is composed of two distinct entities:
Storage: A container holding the raw data pointer (e.g., float*) to the physical memory allocation on the GPU.
TensorImpl (Metadata): A lightweight structure containing the shape (sizes), the traversal logic (strides), the data type (dtype), and the offset from the storage base pointer.
Pattern A: expanded = tensor.expand(batch_size, -1, -1) When .expand() is invoked, PyTorch performs a purely metadata operation.5 It creates a new TensorImpl that points to the same Storage. Crucially, it manipulates the stride array.
Stride-0 Semantics: If a dimension has size $N > 1$ but stride $0$, it indicates that advancing the index in that dimension does not advance the pointer in memory. This is the internal representation of a broadcast.
Allocation: No new GPU memory is allocated for the data. The cost is negligible (microseconds of CPU time to allocate the metadata struct).
Pattern B: result = tensor + other_tensor
When implicit broadcasting occurs, the PyTorch dispatcher (or the Inductor compiler) calculates the effective output shape and the necessary strides on the fly.
Ephemeral View: Conceptually, an ephemeral view is created to align the dimensions of other_tensor with tensor, using the same stride-0 logic.
Allocation: Zero additional memory for input expansion.
3.2 The Materialization Trap: When Views Become Copies
The critical danger of Pattern A lies in the "fragility" of views. While .expand() itself is zero-copy, the resulting tensor is non-contiguous. Many PyTorch operations, particularly those that interface with legacy C++ libraries (like LAPACK) or certain optimized CUDA kernels that demand contiguous memory layouts, will implicitly trigger a copy.
Table 1: Operations Triggering Materialization of Expanded Tensors

Operation Type
Examples
Behavior with Expanded View
Consequence at Batch 256K
Contiguity Check
.contiguous()
Allocates new memory; copies data.
Catastrophic. Allocates 500MB, initiates 500MB Read + 500MB Write.
Reshaping
.view() (some cases), .reshape()
.reshape() may call .contiguous() if strides are incompatible.
High Risk. If stride logic fails, a copy is forced.8
Legacy Kernels
Some torch.mm paths, custom C++ extensions
May internally call x.contiguous() before processing.
Hidden Latency. Developer may not see explicit copy in Python code.
In-Place Ops
x += y
Modifying an expanded tensor in-place is illegal/undefined.
Error/Copy. PyTorch will raise an error or force a clone to safeguard data.6
Cross-Device
.to(device)
Transfers data.
Context Dependent. Usually copies the underlying storage. If expanded, might materialize on destination.9

Detailed Analysis of the Trap:
If a developer writes expanded = tensor.expand(...) and then passes expanded to a function that includes a line like if not input.is_contiguous(): input = input.contiguous(), the optimization is instantly negated. At batch size 256K, this silent copy operation saturates the HBM bandwidth. The profiler will show a generic aten::copy_ or aten::contiguous kernel dominating the timeline, often obscuring the root cause.
In contrast, Pattern B (Implicit Broadcasting) handles the expansion logic inside the element-wise kernel iterator. The kernel is generated to handle the stride-0 access natively. Since the "expanded" tensor never exists as a Python object passed around to other functions, it cannot be accidentally materialized by a distinct function call. Pattern B acts as a safeguard against accidental materialization.
3.3 Memory Footprint Differential
Research Question 4: What's the memory footprint difference for batch_size=256K between the two patterns?
Scenario 1: Ideal Execution (No Materialization)
Pattern A: Footprint $\approx 0$. (Only CPU-side metadata overhead).
Pattern B: Footprint $= 0$.
Difference: Negligible.
Scenario 2: Accidental Materialization (The Risk Case)
Pattern A: If the expanded tensor (BF16, $256K \times 1024$) is materialized:
Allocation: $256,000 \times 1,024 \times 2 \text{ bytes} = 500 \text{ MB}$.
Bandwidth Cost: 500 MB Write (allocation) + 500 MB Read (usage).
Pattern B: Remains 0 bytes.
Difference: 500 MB per tensor.
In a complex DLRM forward pass with thousands of operations, even a 1% rate of accidental materialization can lead to Out-Of-Memory (OOM) errors or significant throughput degradation due to memory thrashing.
4. The Compiler Stack: TorchInductor and Triton
The most significant development in PyTorch's evolution is the torch.compile stack, specifically TorchInductor. This compiler fundamentally alters how broadcasting is handled, rendering the manual "Pattern A vs. Pattern B" debate largely obsolete within compiled regions, provided graph breaks are managed.
4.1 Inductor's Handling of Broadcasting vs. Expand
Research Question 1: Does torch._inductor handle broadcasting differently than explicit .expand()?
The Answer: No. TorchInductor aggressively normalizes these patterns during the graph lowering phase.
Mechanism of Normalization:
AOT Autograd: The first stage of compilation captures the PyTorch code into an FX graph. During this tracing, explicit .expand() calls are recorded as aten.expand nodes. Implicit broadcasts in operations like aten.add are recorded as aten.add nodes with shape mismatch.10
Decomposition & Lowering: Inductor lowers these high-level ops to a simpler IR. Crucially, Inductor's scheduler analyzes the data dependency graph. It identifies that aten.expand is a View Operation—it changes metadata but requires no compute.11
Kernel Fusion: Inductor's primary optimization is vertical fusion. It fuses sequences of pointwise operations (like expand -> add -> swish) into a single kernel.
Triton Code Generation: Whether the input was explicitly expanded or implicitly broadcasted, Inductor generates a Triton kernel that utilizes indexing arithmetic.
Generated Triton IR Analysis:
In Triton, broadcasting is implemented via masking and modular arithmetic on pointers, not by data duplication.

Python


# Conceptual Triton Kernel generated by Inductor
@triton.jit
def fused_add_kernel(ptr_x, ptr_bias, ptr_out, n_elements,...):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load large tensor X (contiguous)
    x = tl.load(ptr_x + offsets, mask=mask)

    # Load small bias (broadcasted)
    # The stride for the bias dimension is effectively 0 in the batch dimension
    # Triton handles this by mapping the offsets to the smaller shape
    bias_index = offsets % BIAS_SIZE  # Simplified broadcasting logic
    bias = tl.load(ptr_bias + bias_index, mask=mask)

    # Compute
    output = x + bias
    tl.store(ptr_out + offsets, output, mask=mask)


Because the bias tensor is small (relative to the 256K batch), it resides almost entirely in the GPU's L2 cache (or even L1 cache). The generated machine code (SASS) will issue load instructions that hit the cache >99% of the time. This behavior is identical whether the user wrote bias.expand(...) or just x + bias.
Conclusion: In torch.compile mode, Pattern A and Pattern B result in identical binary code, provided the .expand() does not interact with a graph break.
4.2 Graph Breaks: The Achilles Heel
The compiler's ability to optimize is contingent on capturing a complete graph. A Graph Break occurs when Dynamo (the frontend tracer) encounters Python structures it cannot symbolize, such as data-dependent control flow.12
The Impact of Graph Breaks on expand():
If a graph break occurs after an explicit .expand() but before its consumption, Inductor must "materialize" the state of the system to return control to the Python interpreter.
Scenario:
Python
expanded = tensor.expand(256000, -1)
print(expanded.shape) # GRAPH BREAK caused by side-effect/I/O
result = expanded + other


Consequence: While PyTorch is smart enough to pass expanded as a view back to Python, the break interrupts the fusion opportunity. The compiler creates one kernel for the pre-print graph and one for the post-print graph.
Optimization Loss: While the view itself might survive the break without a copy, the splitting of the fusion group means we lose the kernel efficiency. We incur the overhead of kernel launch latency and potentially extra reads/writes to HBM for intermediate variables that could have been kept in registers.
5. Micro-Optimization of Key Kernels: Attention and SwishLayerNorm
The user query highlights two specific areas: Attention mechanisms and SwishLayerNorm. These are the critical paths for memory bandwidth.
5.1 Memory Bandwidth in Attention Score Computation
Research Question 3: How does broadcasting affect memory bandwidth in attention score computation?
Attention involves the computation $S = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)$, where $M$ is a mask or bias.
Dimensions: Batch $B$, Heads $H$, Sequence $L$.
Score Matrix Size: $B \times H \times L \times L$.
Bias Mask $M$: Often $1 \times 1 \times L \times L$ (e.g., causal mask) or $1 \times H \times 1 \times 1$ (head bias).
Without Broadcasting (Materialized Mask):
If $M$ is materialized to match the score matrix size (Pattern A + Accidental Copy):
Size of $M$: $256,000 \times H \times L \times L \times 2$ bytes.
For $L=128, H=8$: $256,000 \times 8 \times 128^2 \times 2 \approx 67 \text{ GB}$.
Result: Immediate OOM. It is physically impossible to materialize the mask at this batch size on current hardware.
With Broadcasting (Implicit/Virtual):
The GPU reads the Score Matrix ($S$) from HBM.
The GPU reads the Mask ($M$) from L2 Cache (since $M$ is small and reused across the batch).
Bandwidth Savings: The "Read" bandwidth for the add operation is dominated solely by $S$. The cost of reading $M$ is effectively zero in terms of HBM bandwidth.
Optimization Insight: The bottleneck is the read/write of $S$. Fusion is critical here. If we compute $QK^T$, write it to HBM, read it back, add $M$, write it back, read it back, compute Softmax... we traverse the "Memory Wall" 3 times.
FlashAttention & Inductor: Modern kernels (like FlashAttention or Inductor's fused attention) keep the Score Matrix $S$ in the GPU's SRAM (Shared Memory). The addition of the broadcasted mask $M$ happens in SRAM/Registers. This eliminates the HBM traffic for $S$ entirely for intermediate steps.14
5.2 SwishLayerNorm: The Fusion Imperative
The SwishLayerNorm pattern ($LayerNorm(Swish(x))$) is a prime candidate for bandwidth optimization via fusion.
Component Operations:
Swish (SiLU): $y = x \cdot \sigma(x)$. Element-wise.
LayerNorm: $\mu = \text{mean}(y)$, $\sigma^2 = \text{var}(y)$, $z = \frac{y - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$. Requires reduction (mean/var) and element-wise apply.
Non-Fused Execution (Standard PyTorch):
Kernel 1 (Swish): Read $x$ (500MB), Write $y$ (500MB).
Kernel 2 (Mean/Var): Read $y$ (500MB), Write stats (small).
Kernel 3 (Norm): Read $y$ (500MB), Read stats, Write $z$ (500MB).
Total Data Movement: ~2.5 GB transferred.
Bandwidth Utilization: High.
Fused Execution (Inductor/Triton): Inductor generates a single kernel 16:
Load: Read $x$ (500MB) into registers/SRAM.
Compute Swish: Perform $x \cdot \sigma(x)$ in registers.
Compute Stats: Perform Welford's online algorithm or multi-pass reduction in Shared Memory to get $\mu, \sigma^2$ without writing $y$ to global memory.
Broadcast Parameters: Load $\gamma, \beta$ (broadcasted from small vectors) into Shared Memory.
Normalize: Apply normalization in registers.
Store: Write $z$ (500MB).
Total Data Movement: ~1.0 GB transferred (1 Read, 1 Write).
Performance Gain: 2.5x speedup in bandwidth-limited scenarios.
Broadcasting Role: The parameters $\gamma$ and $\beta$ are broadcasted implicitly. The fused kernel handles this efficiently by loading them once per thread block.
6. Control Flow and Graph Optimization: The Hoisting Strategy
The user is facing a specific "code smell" involving conditional execution based on training configuration (BF16).

Python


# Before
if bf16_training:
    result = func(tensor.unsqueeze(1).to(bf16))
else:
    result = func(tensor.unsqueeze(1))


This pattern introduces complexity and potential graph breaks.
6.1 The Mechanics of Graph Breaks in Conditionals
When torch.compile traces this code, it evaluates the condition if bf16_training.
Static Condition: If bf16_training is a global constant or a hyperparameter known at compile time, Dynamo acts as a partial evaluator. It traces only the branch that is taken. The if statement disappears from the generated graph. In this case, there is no runtime graph break, but we have code duplication in the source.
Dynamic Condition: If bf16_training varies at runtime (e.g., changes between iterations or is a tensor result), Dynamo cannot predict the path. It inserts a Graph Break. It compiles the code before the if and the code inside the branches separately.
Cost: The graph break forces the materialization of tensor and synchronization with the host Python interpreter.
6.2 Research Question 5: Systematic Identification and Hoisting
Strategy for Hoisting: We want to extract Loop-Invariant (or Branch-Invariant) operations. This is a classic compiler optimization called Code Motion.18
Systematic Identification Steps:
AST Analysis (Mental or Automated): Examining the Abstract Syntax Tree of the if/else block.
Let $S_{true}$ be the set of operations in the if block.
Let $S_{false}$ be the set of operations in the else block.
Identify the longest common prefix $P = S_{true} \cap S_{false}$ starting from the input variables.
In the user example, tensor.unsqueeze(1) is common to both.
Dependency Check: Verify that the common operations do not depend on variables defined only within the specific branch logic (excluding the condition itself).
Side-Effect Check: Ensure the operations are pure (no printing, no global state mutation). unsqueeze is pure.
Refactoring (Hoisting):

Python


# Step 1: Hoist the common view operation
# This is valid regardless of bf16_training
tensor_view = tensor.unsqueeze(1)

# Step 2: Handle the divergent dtype logic
# Option A: Use Autocast (Preferred for mixed precision)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=bf16_training):
    # Autocast handles the casting of weights/inputs automatically
    result = func(tensor_view)

# Option B: Explicit Functional Control Flow (if manual control needed)
# This keeps it graph-traceable if bf16_training is a tensor
target_dtype = torch.bfloat16 if bf16_training else tensor.dtype
tensor_ready = tensor_view.to(target_dtype)
result = func(tensor_ready)


Benefits of Hoisting:
Reduced Graph Complexity: The graph captured by Dynamo is simpler. The unsqueeze op is now a permanent node feeding into the conditional logic.
Enhanced Fusion: By hoisting unsqueeze, it becomes visible to the producer of tensor. Inductor can potentially fuse the unsqueeze (as a stride metadata change) into the kernel that generated tensor, avoiding a standalone kernel launch or metadata manipulation overhead during the critical path.
Elimination of Redundant Code: Reduces maintenance burden and the risk of divergence between training modes.
6.3 Validating the Hoist with Profiling tools
To confirm that the hoisting works and graph breaks are eliminated, use TORCH_LOGS.
Command: TORCH_LOGS="graph_breaks,output_code" python train.py
Observation:
Before: You might see log entries like Graph break: if bf16_training.
After: The logs should show a continuous trace or a graph that includes torch.ops.aten.unsqueeze followed by the casting logic, fused into a single Triton kernel where possible.
7. Empirical Validation: Memory Footprint and Bandwidth
To rigorously validate the impact of these optimizations, one must employ profiling.
7.1 Calculating the "Tax" of Pattern A
Using PyTorch Profiler 20, we can quantify the cost.
Table 2: Estimated Costs for Batch 256K (Hidden=1024, BF16)
Metric
Pattern A (Materialized)
Pattern B (Broadcasted)
Diff
Allocation Size
500 MB
0 MB
+500 MB
HBM Read
1000 MB (A + B_exp)
500 MB (A)
+500 MB
HBM Write
500 MB
500 MB
0 MB
L2 Cache Hit Rate
Low (Thrashing)
High (99% for B)
Critical
Latency (A100)
~1.0 ms
~0.65 ms
~35% Slower

Note: The latency calculation assumes ideal bandwidth. In practice, the thrashing of the L2 cache by reading the massive materialized tensor B further degrades performance, making the gap wider.
7.2 Profiling Strategy
To detect if you are suffering from Pattern A materialization:
Run PyTorch Profiler:
Python
with torch.profiler.profile(
    activities=,
    record_shapes=True,
    profile_memory=True
) as prof:
    model(input)


Analyze Memory Timeline: Look for "Allocations" that match the size of your batch (500 MB). If you see an allocation immediately followed by an aten::copy_ or aten::contiguous, you have confirmed a materialization event.
Analyze Bandwidth: Use Nsight Compute to look at "DRAM Throughput". If the throughput is saturated but the FLOPs are low during the element-wise phases, you are strictly bandwidth-bound.
8. Conclusion and Recommendations
For Deep Learning Recommendation Models operating at the scale of 256,000 batch size, memory bandwidth is the scarce resource that dictates performance. Our analysis leads to the following definitive conclusions and recommendations:
Adopt Implicit Broadcasting (Pattern B): While semantically similar to expand() in ideal scenarios, Implicit Broadcasting is structurally robust against accidental materialization. It relies on the kernel's ability to handle stride-0 access via the L2 cache, saving 500 MB of allocation and 33% of memory bandwidth per operation compared to a materialized expand.
Leverage torch.compile: TorchInductor normalizes both patterns into identical, highly optimized Triton kernels. However, this optimization is fragile in the face of graph breaks.
Hoist Invariants: Aggressively hoist operations like unsqueeze out of conditional branches. This is not just code cleanup; it is essential for enabling the compiler to capture larger, uninterrupted regions of the graph, thereby maximizing the scope for vertical fusion.
Fuse Aggressively: Ensure Swish and LayerNorm are fused. The 2.5x reduction in global memory traffic is critical for alleviating the bottleneck at this batch scale.
By strictly adhering to these principles—preferring implicit broadcasting, maintaining graph continuity, and verifying fusion via profiling—engineers can reclaim the memory bandwidth wasted on redundant data movement, directly translating hardware capability into training throughput.
Works cited
NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog, accessed January 30, 2026, https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
Computing GPU memory bandwidth with Deep Learning Benchmarks - Paperspace Blog, accessed January 30, 2026, https://blog.paperspace.com/understanding-memory-bandwidth-benchmarks/
[D] Understanding Optimal Batch Size Calculation - Arithmetic Intensity : r/MachineLearning, accessed January 30, 2026, https://www.reddit.com/r/MachineLearning/comments/1lrc7vh/d_understanding_optimal_batch_size_calculation/
Arithmetic Intensity : Understand Op limits — Memory or Compute | by Jaideep Ray, accessed January 30, 2026, https://medium.com/better-ml/arithmetic-intensity-understand-op-limits-memory-or-compute-342fd15342bb
How to Perform an Expand Operation in PyTorch? | by Amit Yadav | Data Scientist's Diary, accessed January 30, 2026, https://medium.com/data-scientists-diary/how-to-perform-an-expand-operation-in-pytorch-bf1c07532dc1
torch.Tensor.expand — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html
PyTorch .expand() - Tensor Operations - Codecademy, accessed January 30, 2026, https://www.codecademy.com/resources/docs/pytorch/tensor-operations/expand
What's the difference between `reshape()` and `view()` in PyTorch? - Stack Overflow, accessed January 30, 2026, https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
Expand() memory savings - PyTorch Forums, accessed January 30, 2026, https://discuss.pytorch.org/t/expand-memory-savings/170074
Inductor notes – Ian's Blog, accessed January 30, 2026, https://ianbarber.blog/2024/01/16/inductor-notes/
TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes, accessed January 30, 2026, https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
Maximizing AI/ML Model Performance with PyTorch Compilation | by Chaim Rand - Medium, accessed January 30, 2026, https://chaimrand.medium.com/maximizing-ai-ml-model-performance-with-pytorch-compilation-7cdf840202e6
Working with Graph Breaks — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html
Out of the box acceleration and memory savings of decoder models with PyTorch 2.0, accessed January 30, 2026, https://pytorch.org/blog/out-of-the-box-acceleration/
Faster and Memory-Efficient Training of Sequential Recommendation Models for Large Catalogs - arXiv, accessed January 30, 2026, https://arxiv.org/html/2509.09682v5
Liger-Kernel: Efficient Triton Kernels for LLM Training - OpenReview, accessed January 30, 2026, https://openreview.net/pdf/b1356543e4a1517b82b8924c24dd5d075dbd81a9.pdf
liger-kernel 0.4.0 - PyPI, accessed January 30, 2026, https://pypi.org/project/liger-kernel/0.4.0/
pytorch/torch/fx/README.md at main - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/blob/main/torch/fx/README.md
Is Loop Hoisting still a valid manual optimization for C code? - Stack Overflow, accessed January 30, 2026, https://stackoverflow.com/questions/1982131/is-loop-hoisting-still-a-valid-manual-optimization-for-c-code
Understanding GPU Memory 1: Visualizing All Allocations over Time - PyTorch, accessed January 30, 2026, https://pytorch.org/blog/understanding-gpu-memory-1/
Efficient Metric Collection in PyTorch: Avoiding the Performance Pitfalls of TorchMetrics, accessed January 30, 2026, https://towardsdatascience.com/efficient-metric-collection-in-pytorch-avoiding-the-performance-pitfalls-of-torchmetrics/
