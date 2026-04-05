Optimizing L2 Normalization with PyTorch Inductor and Checkpointing
Fusion of F.normalize() with Linear Layers

PyTorch’s new compiler backend (TorchInductor) aggressively fuses operations for efficiency. In general, Inductor can merge chained pointwise and reduction ops into single kernels. For instance, the sequence of computing an L2 norm, clamping it, and dividing can be treated as one fused operation (similar to how softmax’s multi-step reduction is fused into one kernel). However, fusing an L2 normalization directly with an adjacent linear layer (matrix multiplication) is not straightforward. Linear layers (GEMM operations) are typically executed via highly-optimized libraries (cuBLAS/CUTLASS) and only support limited “epilogue” fusions (like adding bias or simple activation). The L2 norm involves a reduction across features followed by elementwise division, which doesn’t fit into a GEMM’s epilogue. In practice, Inductor will compile the linear layer and the normalization as separate fused kernels – the linear layer remains its own kernel (with its bias fused if applicable), and the F.normalize() sequence is fused independently. This means you won’t get a single giant kernel for linear→normalize→linear, but the normalization itself can be executed in a single fused kernel (instead of three). Notably, third-party implementations (e.g. FlashAttention’s fused RMSNorm) show the benefit of such fusion by reducing memory and overhead versus unfused PyTorch code.
Effect of activation_memory_budget=0.05 on Fusion Decisions

Setting torch._functorch.config.activation_memory_budget = 0.05 enables an automatic activation checkpointing mode that aggressively trades compute for lower memory usage. A lower budget (closer to 0) means the compiler will recompute more intermediate activations (especially cheap pointwise ops) instead of storing them, to stay within the memory limit. This impacts kernel fusion because the graph gets partitioned into smaller segments to manage memory. In other words, the compiler may introduce additional graph cuts to drop activations, which can prevent some ops from fusing together if they end up in different segments. By default, TorchInductor’s partitioner tries to minimize recomputation and only recompute simple fusible ops for speed. But at a budget as low as 0.05 (very memory-constrained), it will recompute nearly everything feasible, prioritizing memory savings over maximal fusion. This can lead to more total kernel launches than an unconstrained compilation because each recomputed region might be executed as its own fused kernel. Essentially, the compiler leans toward a “checkpoint everything” strategy (similar to manual activation checkpointing), which inherently limits the size of each fused region. The end result is that fusion opportunities across larger ops may be sacrificed – the model is broken into many recompute-friendly chunks, so you might observe additional kernel launches and a hit to performance (in exchange for significantly lower memory usage). In summary, a tiny activation_memory_budget forces Inductor to favor recomputation of pointwise ops over keeping them fused in memory, reducing memory at the cost of fragmentation of the fused kernels.
Kernel Launch Counts: Manual vs. F.normalize() vs. Fused

Baseline (Manual): In eager mode (no compilation), the manual normalization code

norm = torch.linalg.norm(x, dim=-1, keepdim=True)
norm = torch.clamp(norm, min=eps)
result = x / norm

will perform three separate GPU kernels – one for the norm reduction, one for the clamp, and one for the division (each high-level op triggers its own kernel launch in vanilla PyTorch). This per-op launching is costly, especially when repeated hundreds of times.

Using F.normalize(): The functional API call is essentially doing the same three operations under the hood. In eager execution, F.normalize(x, dim, eps=1e-6) does not magically fuse them – it will still launch the equivalent sequence of kernels (norm + clamp + divide). In other words, without torch.compile, F.normalize likely also results in 3 kernel launches (the implementation computes the norm, applies epsilon, then divides). The benefit of F.normalize comes when combined with TorchInductor: it expresses the pattern as a single high-level op, which Inductor can recognize and optimize.

Under TorchInductor (torch.compile): The compiler will fuse the normalization’s sub-operations into a single kernel (or at most, a couple of kernels) for each normalization call. Instead of three separate launches, the fused kernel computes the norm and performs the division in one pass. This has a huge impact on kernel counts. For example, if your model had N such normalization steps, eager mode would launch 3N kernels for them; with Inductor fusion it can drop to ~N kernels. In a large model with “hundreds of normalizations,” this reduction is dramatic. In one internal experiment, fusing many small ops (in a loss computation) reduced the kernel launches from 221 down to 30 – a similar order-of-magnitude cut in kernels. We can expect a comparable improvement here: manual vs F.normalize (compiled) will go from 3 kernels per instance down to 1 per instance. In a fully fused scenario under torch.compile, each normalization incurs roughly one kernel launch (the norm+clamp+divide fused), versus three launches in the manual unfused approach. The linear layers remain separate kernels, but overall GPU launch overhead is greatly reduced by using the fused F.normalize in compiled mode.
Numerical Precision Considerations (Manual vs. Fused Normalize)

When it comes to numerical results, both approaches should be nearly identical for practical purposes – F.normalize(x, eps=1e-6) computes $y = \frac{x}{\max(\lVert x\rVert_2,;10^{-6})}$, which is exactly what the manual code does. If eps is the same, the output values and gradients will match up to floating-point rounding error. However, there are a few minor precision nuances to be aware of:

    Floating-Point Accumulation: Computing the norm involves summing $x^2$ across dimensions. In eager mode this sum is done using PyTorch’s standard routines (typically in the input tensor’s dtype). Under Inductor’s fusion, the reduction might be performed with different tiling or ordering. This can introduce tiny differences (on the order of machine epsilon) due to the non-associativity of floating-point addition. The fused kernel may sum elements in a different order or parallel pattern than the default, so you might see minuscule discrepancies (e.g. in the last few bits of a float) compared to the manual approach. These differences are very small and generally negligible for model quality, but they exist.

    Half-Precision Behavior: If using FP16/BF16, PyTorch eager sometimes internally upsamples certain ops to higher precision for stability (or inserts casts around operations). TorchInductor by default does not insert extra upcasts/downcasts for fused operations, which means the fused kernel might keep all computations in lower precision. This could lead to slight numeric divergence. For example, in eager mode the torch.linalg.norm might accumulate in FP32 even if the input is FP16 (ensuring more precision), whereas an Inductor-fused kernel might accumulate in FP16 throughout. As a result, the fused version could have a bit more rounding error in low precision. PyTorch provides a config to emulate eager’s precision-casting if exact parity is needed, but by default Inductor chooses performance and allows minor precision differences.

    Epsilon Value: The choice of epsilon (here $10^{-6}$) affects results only for very small norms. Both manual and F.normalize apply the same clamp, so they will produce the same output given the same eps. If your manual code was using a different epsilon (say $10^{-12}$ as in PyTorch’s default), switching to $1\mathrm{e}{-6}$ will make tiny vectors normalize to a larger floor norm, slightly altering those outputs. This is a deliberate change for numerical stability with float16 on newer GPUs (since $10^{-12}$ might underflow in FP16). As long as you use the same epsilon in both methods, there’s no discrepancy – just be mindful if you changed the value.

In summary, no significant numerical precision differences should exist between the manual normalization and F.normalize(eps=1e-6) beyond what’s expected from floating-point math. The fused operation in Inductor will yield results that are almost the same as the manual sequence. Any differences come from implementation details like reduction order or precision of intermediate steps, which can cause minor rounding variation. Importantly, Inductor’s fused normalize does not degrade the mathematical correctness – it maintains stability by clamping to eps just as the manual code does. So you can expect equivalent normalization behavior with the benefit of faster execution.

Sources: PyTorch Inductor documentation and blogs
Citations

Introduction to torch.compile and How It Works with vLLM | vLLM Blog
https://blog.vllm.ai/2025/08/20/torch-compile.html

Learn by doing: TorchInductor Reduction Kernels | Karthick Panner Selvam
https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-Reduction/

Reduced memory requirements of fused RMSNorm kernel · Issue #570 · Dao-AILab/flash-attention · GitHub
https://github.com/Dao-AILab/flash-attention/issues/570

Current and New Activation Checkpointing Techniques in PyTorch – PyTorch
https://pytorch.org/blog/activation-checkpointing-techniques/

Current and New Activation Checkpointing Techniques in PyTorch – PyTorch
https://pytorch.org/blog/activation-checkpointing-techniques/

Accelerating PyTorch Models: Inside torch.compile’s Kernel Optimization | Abhik Sarkar
https://www.abhik.ai/articles/compiling-pytorch-kernel

How does torch.compile speed up a transformer? | Adam Casson
https://www.adamcasson.com/posts/torch-compile-vit

Accelerating PyTorch Models: Inside torch.compile’s Kernel Optimization | Abhik Sarkar
https://www.abhik.ai/articles/compiling-pytorch-kernel

torch.nn.functional.normalize — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html

State of torch.compile for training (August 2025) : ezyang's blog
https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
All Sources
blog.vllm
karthick
github
pytorch
abhik
adamcasson
docs.pytorch
blog.ezyang
