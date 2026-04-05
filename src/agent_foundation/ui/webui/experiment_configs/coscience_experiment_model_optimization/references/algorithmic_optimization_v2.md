

CMSL - Algorithmic Optimization V2
TL;DR:
To bring the CMSL architecture into the practical constraints of production deployment, the v-team executed a targeted set of CMSL performance optimizations. Details of the V1 optimizations were shared in previous notes. In V2, further optimizations include:
Module-Level Algorithmic Refinements: This involved the systematic elimination of graph breaks, the implementation of robust dynamic shape handling strategies to prevent JIT-compilation thrashing, and the removal of CPU-GPU synchronization points. These changes generalized the architecture, enabling scalable deployment across CFR, IFR, and CMSL FAM model variants.
Pipeline Re-architecture ("SDD Lite"): The development of a novel "SDD Lite" pipeline that decouples memory consumption from throughput. By leveraging asynchronous pre-fetching and overlapping CMSL dense computations with sparse embedding lookups, this architecture reduces pipeline memory overhead from a prohibitive 12% to a negligible 1%, while preserving a +4–5% QPS advantage.
Kernel Fusion and Micro-optimization: The application of iterative mutation and PT2-based compilation to fuse granular GPU kernels. This resulted in an 87% reduction in kernel launches for critical modules (e.g., Loss) , significantly mitigating the kernel launch latency bottleneck inherent in massive H100 clusters.
GPU Memory Optimization: Most of our models are bound by the GPU memory, especially on hardware with smaller GPU memory like H100. We adopted a few techniques to reduce the peak GPU memory usage so that we can fit our model in existing hardware configuration without adding more machines or reducing batch size. .
Collectively, these interventions yielded an additional >35% gain in QPS for the CMSL architectures.
1/ Motivation
CMSL introduces a materially new architecture for learning across organic and ads data, requiring end-to-end optimization across algorithm design, distributed sharding, and kernel-level optimization. Specifically, we have identified three critical areas for improvement:
Dynamic sequence length: The dynamic nature of recommendation queries (variable sequence lengths) causes many optimization challenges such as torch.compile recompilation, graph break, GPU to CPU synchronization and so on.
Memory Overhead: The standard data pipelines consumed too much HBM for buffering, limiting the batch sizes and model depths that could be deployed.
Execution Fragmentation: The execution graph was fragmented into thousands of tiny kernels, incurring massive CPU overhead and failing to fully utilize the GPU's massive parallelism capacity.
The following sections detail the systematic resolution of each of these bottlenecks.
2/ Module-Level Optimizations
The first vector of optimization focused on the CMSL algorithmic modules. The v-team leveraged the capabilities of PyTorch 2.0 (PT2) and its compiler to modernize the execution model. To unblock the full potential of PT2, we have upgraded the entire CMSL code base to eliminate inefficient factors.
2.1 Eliminating Graph Breaks
The central feature of PyTorch 2.0 is torch.compile, which captures the PyTorch program as an intermediate representation (FX Graph) and compiles it into optimized machine code (Triton/CUDA). When the compiler encounters unsupported Python constructs, it forces a "Graph Break," falling back to the slow Python eager mode execution. In the vanilla CMSL codebase, graph breaks were pervasive. They were caused by:
Data-Dependent Control Flow: Code such as tensor.sum().item() requires the CPU to inspect the value of a tensor. This halts the GPU, transfers data to the CPU, and breaks the compilation graph.
Non-Torch Functions: Calls to libraries like NumPy or Python's native list manipulations.
Unsupported Kernels: Certain code such as some triton kernels doesn’t support PT2 compilation well so it had to be executed in eager mode
The Optimization Strategy
A rigorous algorithmic optimization was executed:
Symbolic Tracing Refinement: Control flow was rewritten using PyTorch primitives that can be symbolically traced. This allows the compiler to capture both branches of the logic in a single graph, enabling global optimization.
Computation Graph Refactoring: The team analyzes the computation graph and restructures it to remove breaks. This involves pushing data-dependent logic out of the hot path or transforming it into mask-based operations that execute entirely on the GPU.
Holistic Compilation: By eliminating breaks, the team enabled TorchInductor to see the entire module as a single graph. This unlocks optimizations like global buffer reuse (reducing memory footprint) and cross-sub-module operator fusion, which are impossible when the graph is fragmented.
2.2 Optimizing Dynamic Tensor Shapes
Recommendation systems must handle extreme variability in input data. A batch might contain users with histories ranging from 1 to 1000 actions.
Standard PyTorch 2.0 compilation specializes kernels for specific input shapes. If the input shape changes, the system triggers a recompilation.  Compilation is expensive (seconds to minutes). In a production environment serving live traffic, a recompilation trigger causes a massive latency spike.


We adopted a simple approach to use a torch._dynamo.mark_dynamic to specify the dynamic dimensions of our CMSL tensors across all of our models. This enables the compiler to generate generic code (which is less efficiency) when absolutely necessary.
2.3 Eliminating CPU-GPU Synchronization in CMSL modules
The final module-level optimization was the removal of synchronization points.
The Issue: Vanilla code often contained implicit synchronizations—debug prints, assertions on tensor values, or data transfers that blocked the CPU until the GPU finished.
The Fix: The v-team adopted a fully asynchronous execution model. All validity checks were moved to asynchronous GPU-side assertions. The CPU's role was reduced to simply queuing work (Kernel Launches) into the CUDA stream.
Impact: This mitigates GPU starvation, and keeping the instruction queue full is paramount. Removing sync points allows the GPU work scheduler to look ahead and optimize instruction dispatch, hiding memory latency behind the computer.
3/ Pipeline Optimization: The "SDD Lite" Architecture
While module optimizations addressed the computational efficiency, the data delivery pipeline remained a bottleneck. The "SDD Lite" architecture represents a fundamental innovation in how CMSL data is supplied to the GPU.
3.1 The "Full SDD" Baseline and its Limitations
The standard Sparse Data Distribution (SDD) pipeline is designed to handle the massive embedding lookups required by recommendation models. In a typical flow:
Prefetch: The system fetches embeddings for the next batch.
Compute: The GPU processes the current batch.
To hide the latency of the fetch (which is slow due to random access), the "Full SDD" pipeline utilizes a large pre-fetch buffer. This buffering consumes approximately 12% of the GPU's memory capacity.
3.2 "SDD Lite": Decoupling Memory from Throughput
The "SDD Lite" pipeline breaks this coupling. It achieves the high throughput of the Full SDD pipeline without the massive memory tax.
Asynchronous Micro-Batching
SDD Lite employs a more granular, streaming approach to data delivery. Instead of buffering entire large batches, it streams data in "micro-batches" that flow directly into the L2 cache or a small staging buffer in HBM.
Mechanism: By utilizing Tensor Memory Accelerator (TMA), the system can issue asynchronous copy commands that move data from global memory to shared memory/registers at the exact moment it is needed by the compute kernel.
Overlap: The computation of Layer $N$ is perfectly overlapped with the data fetch for Layer $N+1$.
SDD lite leads to 12% to 1% reduction, preserving QPS. The pipeline maintained the +4–5% QPS advantage of the Full SDD approach. This implies that the "Lite" pipeline is not "lazy" (waiting for data) but "just-in-time" (data arrives exactly when needed).
4/ Inplace Data Input Copy
Historically, data input has often been overlooked in training efficiency optimization, as it was not considered a significant contributor to GPU memory usage. However, recent findings in CMSL optimization reveal that the conventional approach of using a separate CUDA stream for data transfer from CPU to GPU can reserve an additional 3% to 6% of GPU memory, thereby reducing the memory available for efficient training. To address this inefficiency, a new method—inplace data input from CPU to GPU—was developed, which does not require reserving a separate memory buffer in CMSL training, but instead, directly writes to a designated location on the GPU. When combined with input pinned-memory improvements (which is a prerequisite for the non-blocking H2D copy) for the IFR model, a 3% reduction in GPU peak memory usage on the IFR model was demonstrated, with no QPS regression.


5/ Kernel Fusion and Micro-Optimization
The final layer of optimization descended to the bare metal. Our models contain thousands of small GPU kernels in just the first path, which both saturate the Cuda kernel buffer (which is 1024) and cause the performance to be CPU bound. The v-team employed aggressive kernel fusion to minimize small GPU kernel launches.
5.1 The Iterative Mutation Strategy
"Iterative Mutation" combined with PT2-based compilation refinements for GPU kernel optimization
Methodology: Explore different fusion strategies (e.g., tile sizes, thread block configurations, loop unrolling factors) to find the optimal configuration for the H100's SMs (Streaming Multiprocessors).
PT2 Refinements: The team tuned the TorchInductor compiler to recognize specific patterns in the CMSL graph and map them to highly optimized Triton kernels.
5.2 Case Study: The Loss Module (87% Reduction)
The Context: The "CMSL Loss" calculation (e.g., Cross Entropy or specialized ranking losses) usually occurs at the end of the forward pass. In unoptimized PyTorch, this is a sequence of element-wise operations:
Logits -> Softmax (Kernel 1, Read/Write HBM)
Logarithm (Kernel 2, Read/Write HBM)
Gather (Select target class) (Kernel 3, Read/Write HBM)
Masking (Ignore padding) (Kernel 4, Read/Write HBM)
Reduction (Sum/Mean) (Kernel 5, Read/Write HBM)
The Fusion: The v-team fused all these operations into several bigger kernels. The logits are read from memory once. All intermediate steps (Softmax, Log, Mask) happen in the fast registers of the GPU. Only the final scalar loss value is written back to memory.
Impact:
Bandwidth: Reduces memory traffic by 5x (since intermediate data is never written to HBM).
Latency: Eliminates the CPU overhead of launching many separate kernels. On B200, where GPU kernel execution is incredibly fast, the launch overhead can often exceed the execution time for small kernels. Fusing them removes this "bubble" in the pipeline.
5.3 Simplification of the Execution Graph
Kernel fusion does more than just save bandwidth; it simplifies the execution graph.
Before: A graph with 1,000 nodes (small kernels). The runtime overhead of traversing this graph is high.
After: A graph with 100 nodes (fused kernels). The runtime overhead is negligible. This simplification is particularly important for the CMSL Ads model , which contains complex, branching logic that can generate thousands of tiny kernels if left unfused.
6/ GPU Memory Optimization
For both IFR and CFR models, the GPU memory is very tight when running on H100 hardware. Our optimization strategy is

Use Python Function Scope
We noticed excessive GPU peak memory usage was caused by a long forward function that kept intermediate tensors alive throughout forward and backward passes. As a practical solution, we conduct CMSL algorithmic optimizations and break the forward logic into smaller functions, limiting tensor lifetimes to only where they are needed. This released memory earlier and reduced GPU peak memory usage by up to 6%.
The peak memory usage before and after this change can be seen in the following picture.



Replace PadCompile
In recommendation models, user sequence lengths vary naturally. A common workaround is to pad all sequences to a fixed length, enabling the use of CUDA Graphs for significant execution speedups. However, this approach, PadCompile, comes at the cost of inflated GPU memory usage.
To reduce this overhead, we explored replacing padding with true variable-length tensors. This change removed padding-related memory inflation but introduced new challenges: dynamic sequence lengths degraded both compilation and execution efficiency. By further optimizing fuse CMSL kernels and reducing kernel launch overhead, we achieved a 13% reduction in GPU memory usage with less than 1% QPS regression. The figure below illustrates the peak GPU memory usage before and after this change.


V-Team
MRS: Tao Jia, Junjie Yang Jijie Wei, Li Sheng, Yujia Hao, Kaan Sancak, Yan Li, Keke Zhai, Linfeng Liu, Haicheng Wang, Zefeng Zhang, Haoyue Tang, Tai Guo, Yujunrong Ma, Zikun Cui, Renzhi Wu, Tony Chen, Yu Zheng , Xiong Zhang, Chenglin Wei, Sameer Pawar, Yanzun Huang, Yuting Zhang, Matt Ma, Hao Wang, Wei Zhao, Yifan Shao, Yuedong Zhang, Jiyuan Zhang, Hong Li previous note
AI infra: Huanyu He, Shuangping Liu, Haoyu Zhang, Jeremy Hadidjojo, Hongtao Yu, Manman Ren, Ying Liu
CFR: Zheng Wu, Xinyue Shen, Yizhou Qian, Ji Qi
FM: Honghao Wei, Hang Wang, Pu Zhang, Xinzhe Chai, Jeff Wang, Mingda Li, Jianwu Xu, Harry Huang, Li Yu
IFR: Wanli Ma, Wenshun Liu, Xiaoyi Zhang, Yue Weng
FBR: Johann Dong, Baokui Yang, Srivatsan Ramanujam, Zhen Hang Jiang, Jugal Marfatia,Yihuan Huang, Kuen Ching
DS: Lingxiao Zhai, Michael Li, Shan Huang, Ke Gong
PMs: Neeraj Bhatia, Vijayant Bhatnagar, Deepak Vijaywargi
Acknowledgement
Thanks Hong Yan for guiding the team toward an inspiring technical path. Thanks Hong Yan and Lars Backstrom, for fostering a fast-moving culture that empowers us to boldly pursue new ideas.  Thanks Sri Reddy, Xinyao Hu,  Sophia (Xueyao) Liang, Neeraj Bhatia, Nipun Mathur for the strong leadership and EM support.


PS: previous note


CMSL - Algorithmic Optimization
TL; DR; Constructive Multi-Sequence Learning (CMSL) is our modeling effort aimed at capturing structured user interests from behavior x-surfaces and x-domains. In this work, we share our journey in optimizing training QPS. Through a series of targeted improvements—including minimizing CUDA synchronization points, efficient algorithm implementation and deduplicating features —we achieved a 40% increase in training QPS, which highlights ROI-aware algorithm design that can deliver sizable efficiency gains. The techniques employed are broadly applicable and can be leveraged in other modeling efforts.
Remove CUDA Synchronization Points
A synchronization point is when the CPU has to wait for the GPU to finish its current work before continuing.
Normally, PyTorch runs asynchronously when using CUDA:
The CPU schedules operations to the GPU and moves on immediately.
The GPU executes those operations in parallel, in its own time.
This allows the CPU and GPU to work in parallel, which is very efficient.
A sync point forces the CPU to wait until all queued GPU operations are done, which stalls execution, destroys parallelism between CPU and GPU.
No photo description available.
The first set of optimization is to reduce the unnecessary Sync point.

1. Constant Tensors
When a constant tensor is being repeatedly created during the forward pass, like this:
def forward(self, x):
constant = torch.tensor([1.0, 2.0, 3.0], device='cuda') # created every time
return x + constant
This is inefficient for two reasons:
Performance Overhead: Creating GPU tensors involves communication between the CPU and GPU, which can be expensive.
CUDA Synchronization Point: Each time you allocate a tensor on the GPU, it can force a CUDA sync, stalling the CPU until the GPU is ready — this is a common performance bottleneck.
The solution is to create it during __init__ and reuse it, which avoids repeated allocation, removes unnecessary sync points, and is more memory and compute efficient.
class MyModule(nn.Module):
def __init__(self):
super().__init__()
constant = torch.tensor([1.0, 2.0, 3.0], device='cuda') # created once
self.register_buffer('constant',constant)
def forward(self, x):
return x + self.constant # reused every time

2. Use of torch.nonzero operator
In PyTorch, the nonzero operator (e.g. nonzero_indices = tensor.nonzero()) introduces a CUDA synchronization point because the size of its output is not known in advance. As a result, PyTorch must compute the non-zero elements on the GPU, then wait for the result to transfer to the CPU to allocate the necessary memory — which stalls execution.
The solution is to replace torch.nonzero with torch.nonzero_static(size: int) if we already know the output size. In the above example, we do know the output size so this can be handled.
Sometimes we select elements from a tensor based on a mask of binary values. Under the hood this triggers a torch.nonzero operator and causes the same synchronization problem as mentioned above. When it is not immediately clear what’s the output size so we cannot use the same approach with torch.nonzero_static. In this case, we can optimize the code to produce a fixed output tensor size.
3. Convert tensor to int
There are many places where we need to get the max (or sum) of a sequence length GPU tensor and use it as an int type, for example max_len = seq_lens.max().item()
.item() is called on a CUDA tensor, PyTorch will wait for all preceding GPU operations to finish (this is the sync), transfer the value from GPU memory to CPU memory, and return it as a Python scalar. This can stall the CPU and GPU pipelines — especially bad inside tight loops or performance-critical code.
Since those sequence lengths are directly coming from the model’s preproc, we just need to precompute such values on the CPU in the data preparation pipeline stage so that it is transferred to the GPU in a batch fashion to minimize the overhead.

Result
After removing all the identified CUDA synchronization points, we observed about 12% training QPS boost and 20% eval QPS gain:
No photo description available.
As we have moved the bottleneck, it also opens the door for further optimization such as quantization etc.
Algorithms Optimization
Optimized algorithms can lead to significant efficiency gains. For example, in CMSL, we need reversing the order of features within each sub-list—as illustrated in the following image:

No photo description available.
The initial naive implementation is to convert the list to dense format with some padded zeros so that it can be easily flipped. Then we can index through the dense format to remove the padded zeros to keep only the original results.
No photo description available.
While this is straightforward to implement, it is quite inefficient in the following aspects:
We’ll have to allocate a lot of extra memory to store the dense formatted tensor as well as the mask.
To convert this to dense format, we need to know the maximum sequence length, which might trigger a CUDA synchronization point.
As shown before, indexing into an array with a mask is slow as we don’t know the output size and this causes a CUDA synchronization point.

Here we designed a much better algorithm that does not involve padding extra zeros and avoids the cost of synchronization points. In this algorithm, it only takes some basic math computations to get the desired position of each element from the original tensor mapping to the final result. Then we can use this position index and torch.gather to generate the output efficiently as shown below:
No photo description available.

By just re-writing this flip method, we were able to achieve a QPS gain of 12%:

No photo description available.
Input Deduplication
There are many duplicated user side features as shown below, by deduplicating the same feature values, it saves embedding look-up time and compute time.
No photo description available.

When this is applied, we observed about 11% training QPS improvement.
No photo description available.

Acknowledgement
We would like to thank the support from PyTorch team to improve the training efficiency of CMSL model: Ning Wang,Xiaodong Wang, Huanyu He, Chuanhao Zhuge, Yue Dong, Haoyu Zhang Shuqi Yang, Yogesh Upadhyay
This work is also closely done with MRS sister teams: Chloe Liu, Jiyuan Zhang, Jijie Wei, Li Lu, Zikun Cui, Siqi Yan, Min Ni
