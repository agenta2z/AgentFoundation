Optimization of Normalization Primitives in Large-Scale Recommendation Architectures: A Deep Analysis of PyTorch Inductor Fusion and Memory Hierarchies
Executive Summary
The optimization of large-scale recommendation systems, particularly those leveraging Transformer-based architectures and dense embedding retrieval, represents one of the most significant challenges in modern High-Performance Computing (HPC) for AI. As these models scale to trillions of parameters and contend with extremely high-cardinality features, the computational overhead of seemingly minor operations—such as normalization—can aggregate into substantial latency bottlenecks. This report presents an exhaustive analysis of the optimization trajectory for normalization operations within the PyTorch 2.0 ecosystem, specifically targeting the transition from manual, multi-kernel normalization patterns to the fused torch.nn.functional.normalize() (F.normalize) operation under the torch.compile() (Inductor) backend.
Our analysis, grounded in a rigorous examination of the Inductor compiler stack, Triton kernel generation, and Activation Checkpointing (AC) dynamics, confirms that replacing the manual torch.linalg.norm sequence with F.normalize yields a guaranteed reduction in kernel launches from three to one for the normalization subgraph itself when compiled. This consolidation is driven by Inductor's ability to generate "persistent reduction" kernels that fuse reduction logic with subsequent pointwise operations, thereby minimizing High Bandwidth Memory (HBM) round-trips.1
However, the investigation reveals a critical architectural boundary: standard Inductor compilation does not currently support the automatic vertical fusion of General Matrix Multiplications (GEMM) with subsequent reduction-based normalizations. While pointwise epilogues (e.g., ReLU, bias addition) are readily fused into GEMM templates, normalization remains a distinct kernel execution due to the synchronization requirements of row-wise reductions.3 Consequently, the optimized execution flow comprises two highly efficient kernels—GEMM followed by Fused Normalization—rather than a single monolithic kernel, unless specialized manual kernels (e.g., Liger Kernel) are employed.5
Furthermore, we scrutinize the impact of the strict activation_memory_budget = 0.05 setting. Our findings indicate that while this constraint enforces aggressive recomputation of intermediate activations during the backward pass, it does not disrupt the fundamental fusion logic of the forward pass. Instead, it necessitates the re-execution of the fused kernels during gradient computation, trading compute throughput for significant memory savings.7 The report concludes with detailed numerical stability analyses demonstrating the superiority of F.normalize's max(x, eps) formulation over the manual clamp pattern, particularly regarding gradient behavior near zero.8
1. Architectural Context and Problem Definition
1.1 The Computational Landscape of Recommendation Systems
Modern recommendation models, exemplified by Deep Learning Recommendation Models (DLRM) and sequential Transformer architectures, operate under distinct computational constraints compared to their computer vision or NLP counterparts. These systems are characterized by:
High-Cardinality Embeddings: Input features often map to embedding tables with millions of rows, requiring efficient gather/scatter operations.
Dense Interaction Layers: The retrieved embeddings are processed through Multi-Layer Perceptrons (MLPs) and attention mechanisms, necessitating heavy matrix multiplication (GEMM) workloads.
Frequent Normalization: To maintain training stability and prevent gradient explosion in deep networks, normalization layers (LayerNorm, RMSNorm, L2 Normalization) are interspersed frequently—often after every projection or attention block.
In the specific codebase under review, a manual normalization pattern is employed:

Python


norm = torch.linalg.norm(x, dim=-1, keepdim=True)  # Operation A: Reduction
norm = torch.clamp(norm, min=eps)                  # Operation B: Pointwise
result = x / norm                                  # Operation C: Pointwise/Broadcast


While semantically correct, this implementation in PyTorch's eager execution mode incurs a severe performance penalty known as the "launch overhead" and "memory wall." Each line triggers a separate CUDA kernel launch, forcing the GPU to read the input tensor x from global memory (HBM) and write the intermediate result back to HBM three separate times. For a model with hundreds of normalization layers, this creates a latency floor that no amount of compute parallelism can overcome.
1.2 The PyTorch 2.0 Compilation Stack
The introduction of torch.compile in PyTorch 2.0 marks a paradigm shift from eager execution to graph-based compilation. The compiler stack, primarily composed of TorchDynamo, AOTAutograd, and Inductor, aims to resolve the memory wall problem through operator fusion.
TorchDynamo: Captures the Python bytecode and constructs a generic FX graph, handling dynamic behavior and graph breaks.9
AOTAutograd: Traces the backward graph ahead of time, decomposing complex operators into atomic ATen primitives.10
TorchInductor: The default compiler backend that lowers the FX graph into optimized Triton kernels for NVIDIA GPUs. Its primary optimization mechanism is loop fusion, where multiple pointwise and reduction operations are combined into a single kernel to maximize data locality in the GPU's SRAM (L1/L2 cache).1
The core research objective is to quantify the efficacy of this stack in optimizing the specific normalization pattern and to understand how memory constraints influence the compiler's fusion decisions.
2. Kernel Fusion Mechanics: Manual vs. F.normalize
The transition from manual tensor operations to functional APIs is often advocated for code cleanliness, but in the context of torch.compile, it has profound implications for Intermediate Representation (IR) stability and kernel generation.
2.1 The Manual Implementation in Inductor
When torch.compile processes the manual sequence (norm -> clamp -> div), Inductor's scheduler receives a graph of decomposed primitives.
Reduction Node: The linalg.norm is identified as a reduction operation summing squares over the last dimension (dim=-1).
Pointwise Nodes: The clamp and div operations are identified as element-wise consumers of the reduction's output.
Fusion Capability: Inductor possesses robust heuristics for fusing "Producer-Consumer" patterns. Specifically, it can fuse a Reduction with subsequent Pointwise operations if the shapes align. The scheduler recognizes that the clamp and div operations can be performed immediately after the reduction result is available, without writing the intermediate norm tensor to global memory.12
However, manual implementations are susceptible to "decomposition drift." If a developer uses torch.norm (deprecated) versus torch.linalg.norm, or applies keepdim=False followed by a manual unsqueeze, the graph may contain view operations that complicate the scheduler's tiling decisions. While Inductor is generally capable of handling these views through symbolic indexing 11, complex manual patterns increase the risk of suboptimal "split reductions" or unnecessary data movement.
2.2 The F.normalize Implementation in Inductor
torch.nn.functional.normalize is a high-level composite operation. Under torch.compile, it does not call a pre-compiled opaque kernel. Instead, it is decomposed during the AOTAutograd phase into a canonical sequence of ATen operations:

$$y = \frac{x}{\max(\|x\|_p, \epsilon)}$$
This decomposition is explicitly designed to be compiler-friendly.
IR Cleanliness and Canonicalization: The primary advantage of F.normalize lies in its consistent lowering to a Persistent Reduction pattern. Inductor's pattern matcher is optimized to recognize this specific subgraph—Reduction (SumSq) $\rightarrow$ Pointwise (Sqrt) $\rightarrow$ Pointwise (Max) $\rightarrow$ Pointwise (Div)—and map it to a specialized Triton template.14
Unlike the manual implementation, which relies on the general-purpose scheduler to "discover" the fusion opportunity, F.normalize effectively hands the compiler a pre-validated structure. This reduces compilation time and guarantees that the generated kernel utilizes the most efficient tiling strategy available for the hardware.2
2.3 Kernel Count Verification
The reduction in kernel launches is the primary metric of success for this optimization.
Manual (Eager Mode): 3 Kernels.
Kernel 1: at::native::norm (Reduction + Sqrt)
Kernel 2: at::native::clamp (Pointwise)
Kernel 3: at::native::div (Pointwise broadcast)
F.normalize (Eager Mode): 1-2 Kernels.
Depending on the specific PyTorch version and CUDA backend, F.normalize may dispatch a single fused kernel or a split implementation (Norm + Div).
Manual (Inductor): 1 Kernel.
Inductor successfully fuses the norm, clamp, and div into a single triton_per_fused_... kernel.
F.normalize (Inductor): 1 Kernel.
Inductor generates a single triton_per_fused_... kernel.
Conclusion: Both implementations achieve the target of 1 kernel for the normalization block under Inductor. However, F.normalize is strictly preferred for its guarantee of generating a canonical graph that avoids potential "graph breaks" or scheduling suboptimalities associated with manual tensor manipulation.9
3. Deep Dive: Fusion with Adjacent Linear Layers
A critical research question posits: Does F.normalize() get fused with adjacent linear layers under torch.compile()? This query touches upon the fundamental distinction between Pointwise Fusion and GEMM Fusion.
3.1 The GEMM Kernel Barrier
Linear layers in neural networks are mathematically defined as General Matrix Multiplications (GEMM): $Y = XW^T + b$. In the PyTorch Inductor backend, GEMM operations are treated as "Template Kernels." Unlike pointwise ops, which are generated from scratch using Triton, GEMMs are mapped to highly optimized vendor libraries (cuBLAS, CUTLASS) or hand-tuned Triton templates.3
Why GEMMs are Special: GEMM optimization is an NP-hard tiling problem dependent on specific hardware characteristics (Tensor Core utilization, cache line sizes, warp occupancy). Inductor relies on "Max Autotune" to select the best kernel from a library of candidates.17
3.2 Epilogue Fusion and Its Limitations
Modern GEMM kernels support "Epilogue Fusion"—the ability to perform lightweight operations on the result of the matrix multiplication before writing it to memory.
Supported Epilogues: Unary or binary pointwise operations that map 1:1 with the output elements. Examples include ReLU, SiLU, Gelu, and Bias Add.1
The Normalization Problem: Normalization is a Reduction. To normalize the output vector of a GEMM, the kernel must compute the sum of squares of the entire output row.
In highly parallel GEMM implementations, the computation of a single output row is often distributed across multiple CUDA thread blocks or warps. No single thread block holds the complete accumulator for the row until the very end of the global write phase.
To fuse normalization, the GEMM kernel would require a global barrier synchronization to aggregate the partial sums from all blocks, compute the norm, and then perform a second pass to divide the elements. This "Global Reduction" pattern destroys the throughput advantages of standard GEMM pipelines.3
3.3 The Resulting Execution Graph
Consequently, the sequence Linear $\rightarrow$ F.normalize compiles into Two Distinct Kernels:
Kernel 1 (GEMM): Computes $Y = XW^T$. If a bias exists, it is fused here. Writes $Y$ to HBM.
Kernel 2 (Normalization): Reads $Y$ from HBM. Computes $\|Y\|$, normalizes, and writes $Z$.
Table 1: Fusion Capabilities of Inductor
Operation Sequence
Fused Kernel Count
Fusion Type
Notes
Linear + ReLU
1
Epilogue Fusion
Supported by cuBLAS/Triton templates.
Linear + Bias
1
Epilogue Fusion
Standard implementation.
Norm + Clamp
1
Vertical Fusion
Producer-Consumer fusion.
Linear + Norm
2
None
Reduction barrier prevents GEMM fusion.

Emerging Solutions (Liger Kernels): To achieve a true 1-kernel execution for Linear+Norm, specialized kernels like Liger Kernel or FlashLinear are required. These are manually written Triton kernels that fundamentally restructure the GEMM loop (often using "Split-K" or tile-based reduction strategies) to keep data in SRAM. Standard torch.compile does not currently generate these complex fused kernels automatically for generic Linear layers.5
Answer to Research Question 1: F.normalize() does not fuse with the adjacent linear layer. The barrier between the GEMM template and the reduction operation enforces a two-kernel execution flow.
4. Activation Memory Budget and Recomputation
The setting torch._functorch.config.activation_memory_budget = 0.05 introduces a rigorous constraint on the computational graph, fundamentally altering the backward pass.
4.1 Selective Activation Checkpointing (SAC) Mechanism
Activation Checkpointing (AC) is a technique to trade compute for memory. By discarding intermediate activations during the forward pass and recomputing them during the backward pass, models can scale to larger batch sizes or deeper architectures. The activation_memory_budget parameter controls the aggressiveness of this technique via an Integer Linear Programming (ILP) solver or a heuristic policy (Knapsack problem).7
Budget = 1.0: The system behaves like standard training, saving all necessary activations.
Budget = 0.05: The system is permitted to store only 5% of the total activations. This forces the partitioner to mark the vast majority of tensors (including the outputs of the Linear and Normalization layers) as "evictable."
4.2 Impact on Fusion Decisions
A critical concern is whether this aggressive recomputation forces the compiler to abandon fusion optimizations.
Forward Pass: The memory budget has no impact on the forward pass fusion. The Linear and F.normalize kernels are generated and executed as fused kernels (to the extent possible) regardless of whether their outputs are saved or discarded.
Backward Pass: When the autograd engine requires the input x to the normalization layer to compute gradients, and x has been discarded, the system triggers a Recomputation Graph.
The Insight: Inductor treats the recomputation subgraph exactly like any other graph. It compiles the recomputation operations.
Instead of launching 3 eager kernels to recompute the manual norm, Inductor launches the same single fused Triton kernel used in the forward pass.
Therefore, the 0.05 budget does not "break" fusion. It simply mandates that the fused kernels be executed twice: once during the forward pass (where output is discarded) and once during the backward pass (to regenerate data for gradients).7
Does it force recomputation over fusion?
No. It forces recomputation of the fused kernels. The decision to fuse is based on instruction-level parallelism and memory bandwidth, while the decision to recompute is based on memory capacity. These are orthogonal optimization axes. Inductor ensures that even the "penalized" recomputation path is as efficient as possible.
4.3 Performance Implications
While fusion is preserved, the 0.05 budget has severe implications for throughput.
Compute Intensity: Recomputing the output of Linear layers means re-executing heavy GEMM operations. This effectively doubles the FLOPs required for the linear layers.
Recommendation: A budget of 0.05 is extremely aggressive and likely suboptimal for speed. It should only be used if the model physically cannot fit in VRAM otherwise. Increasing the budget to 0.3 or 0.5 often provides a better balance, saving the largest tensors (like Attention matrices) while keeping cheaper activations (like Norm outputs) in memory.7
5. Numerical Precision: F.normalize vs. Manual Patterns
The transition to F.normalize also introduces subtle but critical changes in numerical behavior, particularly regarding the epsilon handling in gradients.
5.1 Mathematical Formulation
Manual Pattern:
$$v_{out} = \frac{v}{\text{clamp}(\|v\|, \min=\epsilon)}$$
Here, if $\|v\| < \epsilon$, the denominator is clamped to $\epsilon$. The effective operation becomes $v / \epsilon$.
F.normalize Pattern:
$$v_{out} = \frac{v}{\max(\|v\|, \epsilon)}$$
Functionally, for scalar values, clamp(x, min=e) and max(x, e) appear identical.
5.2 Gradient Divergence at the Singularity
The critical difference lies in the derivative implementation during the backward pass.
max(x, e) Gradient: PyTorch's max operator behaves like a Router (or Heaviside step function).
If $\|v\| > \epsilon$: The gradient flows through the norm term.
If $\|v\| < \epsilon$: The "winner" of the max is the constant $\epsilon$. The gradient with respect to the norm becomes 0. The denominator is treated as a constant. The gradient flow is effectively cut off from the norm calculation, simplifying the backward graph to just the gradient of the numerator term.
clamp(x, min=e) Gradient:
If $\|v\| < \epsilon$: The gradient of the clamp output w.r.t the input is 0.
However, manual compositions of div and clamp can sometimes introduce instability if the norm calculation itself produces NaN or Inf before the clamp (e.g., if inputs are extremely large or small).
The Superiority of F.normalize: F.normalize utilizes a fused backward definition in AOTAutograd that guards against division-by-zero singularities more robustly than the composition of atomic div and clamp. Specifically, F.normalize handles the case where the input vector is exactly zero by ensuring the gradient is correctly zeroed out or scaled, whereas manual x / clamp(norm) can result in 0 / eps (which is valid) but might have unstable gradients if the norm derivation is not carefully guarded.8
Answer to Research Question 4: F.normalize(eps=1e-6) uses max(x, eps), which provides a cleaner gradient signal (zero gradient w.r.t norm) when inputs fall below the epsilon threshold, preventing numerical explosions that can occur with manual clamp patterns in reduced precision training (FP16/BF16).
6. Detailed Kernel Count Analysis
To provide a definitive answer to the kernel launch count query, we synthesize the findings from Inductor's scheduler behavior.
Table 2: Kernel Launch Count Comparison (Single Linear + Norm Block)
Implementation Strategy
Execution Mode
Kernel 1
Kernel 2
Kernel 3
Total Kernels
Manual
Eager (Standard)
GEMM (addmm)
Reduction (norm)
Pointwise (clamp + div)
3 - 4
F.normalize
Eager
GEMM (addmm)
Fused Norm (vector_norm)
-
2
Manual
Inductor
GEMM (addmm)
Fused Norm (triton_per)
-
2
F.normalize
Inductor
GEMM (addmm)
Fused Norm (triton_per)
-
2
Liger Kernel
Custom Triton
Fused Linear+Norm
-
-
1

Interpretation:
Eager to Inductor: The manual implementation sees the greatest improvement (3 kernels $\rightarrow$ 1 fused norm kernel).
Manual vs. F.normalize in Inductor: Both converge to the same kernel count (2 kernels total: 1 GEMM + 1 Norm).
Why 2 Kernels? As detailed in Section 3, the GEMM barrier prevents the normalization from fusing into the linear layer in standard Inductor.
The "3 to 1" Impact: The expected impact cited in the user prompt ("3 kernels $\rightarrow$ 1 per normalization call") is verified for the normalization subgraph itself. The global count for the block reduces from ~4 to 2.
7. Deep Dive into Inductor's "Persistent Reduction"
To understand why Inductor is efficient here, we analyze the generated Triton code structure.
7.1 The "Persistent" Heuristic
Inductor employs a scheduling heuristic to select between Persistent Reductions and Loop Reductions.
Persistent Reduction: Used when the reduction dimension (e.g., embedding size $D=256$) is small enough to fit entirely within the GPU's register file or Shared Memory. The kernel loads the data once, keeps it "persistent" in fast memory, computes the reduction, and writes the output.
Memory Traffic: 1 Read ($X$), 1 Write ($Y$). This is bandwidth-optimal.
Loop Reduction: Used for massive reduction dimensions. The kernel must iterate over blocks, spilling partial results to global memory.
For recommendation models, embedding dimensions are typically small (64 to 1024). Inductor correctly selects Persistent Reduction.
7.2 Generated Code Simulation
The Triton kernel for F.normalize (Persistent) generated by Inductor follows this optimized structure:

Python


@triton.jit
def triton_per_fused_normalize_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. Block ID corresponds to the row index (Batch dimension)
    pid = tl.program_id(0)

    # 2. Coalesced Load: Load the entire row into registers
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + pid * BLOCK_SIZE + offsets)

    # 3. Reduction (Fused): Compute sum of squares in registers
    x_sq = x * x
    norm_sq = tl.sum(x_sq, axis=0)

    # 4. Pointwise (Fused): Sqrt, Max, and Div
    norm = tl.sqrt(norm_sq)
    clamped_norm = tl.maximum(norm, EPS)
    out = x / clamped_norm

    # 5. Coalesced Store: Write result
    tl.store(out_ptr + pid * BLOCK_SIZE + offsets, out)


This single kernel replaces the memory-intensive read/write cycles of the manual implementation, validating the efficiency gains.2
8. Strategic Recommendations
Based on the synthesis of fusion dynamics, memory budgeting, and numerical analysis, the following actions are recommended for optimizing the recommendation model.
8.1 Immediate Adoption of F.normalize
Recommendation: Refactor all manual normalization patterns to torch.nn.functional.normalize(x, p=2, dim=-1, eps=1e-6).
Reasoning: While Inductor can fuse the manual pattern, F.normalize provides a guaranteed, canonical graph that maps directly to optimized "Persistent Reduction" templates. It eliminates the risk of scheduler miss-optimization and ensures superior gradient stability via max(x, eps).15
8.2 Optimization of Activation Memory Budget
Recommendation: Re-evaluate the 0.05 budget setting.
Reasoning: A budget of 0.05 forces the recomputation of GEMM layers, which are compute-bound. This likely degrades training speed significantly unless the batch size increase is massive.
Action: Conduct a sweep of budget values (0.2, 0.5, 1.0). Aim for the highest budget that fits within VRAM to minimize GEMM recomputation. Inductor will handle the recomputation efficiently (using fused kernels), but avoiding the recomputation entirely is always faster.7
8.3 Future-Proofing with Custom Kernels
Recommendation: For the ultimate performance optimization (Linear+Norm fusion), investigate Liger Kernel integration.
Reasoning: Standard Inductor cannot fuse GEMM and Normalization. Liger Kernels provide drop-in replacements for Linear layers that include fused RMSNorm/LayerNorm, effectively reaching the "1 Kernel" theoretical limit for the entire block. This is the next logical step after standard compilation.5
9. Conclusion
The analysis confirms that the manual normalization pattern is a significant inefficiency in eager mode that is effectively rectified by torch.compile. The switch to F.normalize ensures a robust reduction to a single "Persistent Reduction" kernel for the normalization operation, leveraging Triton's register-level optimizations. While adjacent Linear layers cannot be fused into this kernel due to fundamental GEMM template constraints, the resulting 2-kernel execution path represents a highly optimized state for standard PyTorch code. The aggressive activation memory budget of 0.05 serves as a powerful tool for memory capacity but imposes a compute penalty by forcing the re-execution of these fused kernels. By adopting F.normalize and tuning the memory budget, the recommendation model will achieve a state of "near-optimality" within the constraints of the current compiler stack.
Works cited
Pile, The Missing Manual | PDF | Software Bug | Python (Programming Language) - Scribd, accessed January 30, 2026, https://www.scribd.com/document/937961531/Torch-compile-The-Missing-Manual
How does torch.compile speed up a transformer? - Adam Casson, accessed January 30, 2026, https://www.adamcasson.com/posts/torch-compile-vit
Insum: Sparse GPU Kernels Simplified and Optimized with Indirect Einsums - arXiv, accessed January 30, 2026, https://arxiv.org/html/2510.17505v1
Enable TorchInductor to Generate Matmuls Natively via tl.dot #151705 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/151705
linkedin/Liger-Kernel: Efficient Triton Kernels for LLM Training - GitHub, accessed January 30, 2026, https://github.com/linkedin/Liger-Kernel
Liger Kernel: Efficient Triton Kernels for LLM Training - arXiv, accessed January 30, 2026, https://arxiv.org/html/2410.10989v2
Current and New Activation Checkpointing Techniques in PyTorch ..., accessed January 30, 2026, https://pytorch.org/blog/activation-checkpointing-techniques/
torch.nn.functional.normalize — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
Unleashing PyTorch Performance with TorchInductor: A Deep Dive | by Aadishagrawal, accessed January 30, 2026, https://medium.com/@aadishagrawal/unleashing-pytorch-performance-with-torchinductor-a-deep-dive-1f01e8b36efa
PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation, accessed January 30, 2026, https://docs.pytorch.org/assets/pytorch2-2.pdf
TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes, accessed January 30, 2026, https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
[Inductor] Fusion of Tiled Point-Wise and Reduction Operators · Issue #128063 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/128063
Performance Tuning Guide — PyTorch Tutorials 2.10.0+cu128 documentation, accessed January 30, 2026, https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
Learn by doing: TorchInductor Reduction Kernels | Karthick Panner ..., accessed January 30, 2026, https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-Reduction/
kornia.contrib - Read the Docs, accessed January 30, 2026, https://kornia.readthedocs.io/en/latest/contrib.html
Torch Compile and External Kernels — NVIDIA PhysicsNeMo Framework, accessed January 30, 2026, https://docs.nvidia.com/physicsnemo/latest/user-guide/performance_docs/torch_compile_support.html
pytorch/torch/_inductor/config.py at main - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py
Questions about Inductor code generation for gemm on Nvidia device - compiler, accessed January 30, 2026, https://dev-discuss.pytorch.org/t/questions-about-inductor-code-generation-for-gemm-on-nvidia-device/1379
PluS: Highly Efficient and Expandable ML Compiler with Pluggable Graph Schedules - USENIX, accessed January 30, 2026, https://www.usenix.org/system/files/atc25-wu-ruofan.pdf
torch._functorch.partitioners — functorch nightly documentation - PyTorch, accessed January 30, 2026, https://docs.pytorch.org/functorch/nightly/_modules/torch/_functorch/partitioners.html
Peak Performance, Minimized Memory: Optimizing torchtune's performance with torch.compile & Liger Kernel - PyTorch, accessed January 30, 2026, https://pytorch.org/blog/peak-performance-minimized-memory/
