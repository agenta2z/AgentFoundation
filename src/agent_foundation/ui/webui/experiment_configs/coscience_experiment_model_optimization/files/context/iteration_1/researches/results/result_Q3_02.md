Optimization and Integrity of Gradient Flow in Compiled Recommendation Architectures
The transition of the PyTorch framework from a purely imperative execution model to a compiled, graph-centric paradigm represents one of the most significant shifts in high-performance machine learning engineering. With the introduction of the modern compiler, developers are now able to leverage just-in-time (JIT) kernel fusion and specialized backend optimizations, such as those provided by the TorchInductor engine.1 However, this shift introduces complex interactions between the autograd system—which traditionally operated on a dynamic tape-based mechanism—and the static requirements of graph capture.3 In the context of large-scale recommendation systems, where categorical feature embeddings often represent the vast majority of model parameters, maintaining the integrity of the backward pass is both critical and increasingly difficult to verify.5 The identification of silent bugs, particularly those arising from the interaction between gradient-disabling decorators and symbolic tracing, necessitates a deep architectural understanding of how TorchDynamo and AOTAutograd construct and partition computation graphs.3
Mechanisms of Graph Capture and the Evolution of TorchDynamo
The fundamental objective of the PyTorch compiler is to transform standard Python bytecode into an optimized intermediate representation (IR), which can then be lowered into high-performance kernels.7 This process is orchestrated primarily by TorchDynamo, which utilizes the CPython Frame Evaluation API to intercept code execution at the level of the Python interpreter.7 Unlike previous attempts at compilation, such as TorchScript, which required restrictive type annotations and often failed on complex Python features, TorchDynamo is designed to trace arbitrary Python code by creating "graph breaks" when it encounters unsupported operations.1
A graph break occurs when the symbolic tracer determines that a particular segment of code cannot be safely represented as a static FX graph.1 Common triggers for graph breaks include data-dependent control flow, such as if-statements that branch based on the value of a tensor, and calls to non-PyTorch functions or unsupported C-extensions.10 When a break is triggered, the compiler finalizes the current graph, executes the problematic code in the standard eager Python runtime, and then attempts to resume tracing for the subsequent portion of the model.1 While this mechanism ensures that the code remains executable, it introduces significant overhead due to the transition between optimized kernels and the Python interpreter, effectively fragmenting the execution flow and limiting the potential for kernel fusion.9
The Role of AOTAutograd in Training Workloads
For models undergoing training, the compilation process must account for the backward pass. This is handled by AOTAutograd, which traces the autograd engine's logic to generate a dedicated backward graph segment for every forward graph segment captured by TorchDynamo.2 This ahead-of-time approach allows the compiler to perform global optimizations across both passes, such as min-cut partitioning.3 Min-cut partitioning is a sophisticated optimization strategy that identifies the minimal set of intermediate activations that must be saved during the forward pass to compute gradients during the backward pass, thereby reducing memory pressure in large-scale recommendation models.3
However, the efficacy of AOTAutograd is entirely dependent on the presence of a valid autograd tape.3 If a section of the forward pass is wrapped in a mechanism that disables gradient tracking, such as torch.no_grad(), the symbolic tracer perceives those operations as having no impact on the backward pass.13 Consequently, no backward graph is generated for those segments, leading to the "silent" failure of gradient propagation to the upstream components of the model, such as the encoder or embedding layers.14
Compilation Mode
Primary Objective
Optimization Strategy
Trade-offs
Default
Balanced performance
Standard fusion and overhead reduction
Baseline memory and speed
Reduce-Overhead
Minimal latency
CUDA Graphs to bypass Python overhead
Increased memory usage
Max-Autotune
Maximum throughput
Triton-based template searching
Longest compilation time
Max-Autotune-No-Cudagraphs
Throughput without memory spikes
Triton kernels without static graph overhead
High compute, moderate memory

9
Architectural Analysis of the Gradient Disabling Bug
The critical bug identified in large recommendation models involves the misuse of the @torch.no_grad() decorator on functions performing index selection logic.5 In these systems, $seq\_embeddings$ are often the output of an encoder that requires training. The function get_nro_embeddings performs an indexing operation, which is fundamentally a selection process $Y = S[i]$, where $S$ is the sequence of embeddings and $i$ are the indices.5 While the computation of the indices $i$ themselves may be non-differentiable and thus suitable for a no-grad scope, the act of selecting from $S$ is a differentiable operation in PyTorch.5
Decorator Scope and the Symbolic Tracer
When @torch.no_grad() is applied as a decorator, it wraps the entire function call in a gradient-disabling context.18 For the symbolic tracer in TorchDynamo, this decorator acts as a signal that every operation within the function should be executed without tracking gradients.13 This has two catastrophic effects. First, it ensures that the output tensor $Y$ has the property $requires\_grad=False$, regardless of whether the input $S$ has $requires\_grad=True$.13 Second, it creates a graph break at the entry and exit points of the function, as the compiler must transition into a specialized no-grad state.1
Mathematically, the autograd system builds a graph of functions representing the chain rule. If we have a loss $\mathcal{L}$, the gradient with respect to the sequence embeddings should be calculated as $\frac{\partial \mathcal{L}}{\partial S} = \frac{\partial \mathcal{L}}{\partial Y} \frac{\partial Y}{\partial S}$. By performing the indexing $Y = S[i]$ inside a decorated function, the connection between $Y$ and $S$ is deleted from the autograd tape.14 The $grad\_fn$ of $Y$ becomes $None$, effectively treating $Y$ as a leaf node in the graph with no history.16 During the backward pass, the gradient $\frac{\partial \mathcal{L}}{\partial Y}$ is computed correctly, but it has no further path to flow back to $S$, leaving the encoder weights frozen.14
Caching and State Persistence
Further complicating this issue is the potential for caching within the autograd and compiler engines. Some research suggests that tensors originating from a no_grad block can continue to be affected by gradient disabling due to internal caching of the $requires\_grad$ state.21 In a compiled environment, if the compiler optimizes a path once under a no-grad context, it may reuse that optimized path in subsequent iterations, even if the surrounding context has changed, leading to non-deterministic gradient flow failures.21 This is particularly problematic in recommendation models where different batches may trigger different paths through the embedding logic.5
Graph Break Quantification and Performance Implications
The impact of torch.no_grad() on compilation is not merely semantic; it is a major performance bottleneck. Every instance of a gradient-disabling context manager or decorator within a compiled function typically results in at least one graph break.1 In a typical forward pass, if a developer uses localized no-grad blocks across multiple sub-modules, the number of distinct graphs can grow linearly with the number of such blocks.9
Analysis of compilation metrics through tools like tlparse or TORCH_LOGS="graph_breaks" reveals that any graph break within a nested function can generate a number of graphs proportional to the depth of the user stack.12 For a recommendation model with deep hierarchies of feature transformations, a single misplaced decorator can fragment a potentially unified 100-operation graph into 10 or more isolated kernels.12 This fragmentation prevents the compiler from performing cross-operation optimizations such as horizontal fusion (combining independent operations) or vertical fusion (merging sequential operations into a single loop).1
Operation Type
Impact on Graph
Overhead Level
Reason for Break
torch.no_grad()
Graph Break
Moderate
Transition to gradient-disabled state
.item()
Graph Break
Severe
GPU-to-CPU synchronization required
if tensor > 0:
Graph Break
High
Data-dependent branching
print(tensor)
Graph Break
Low
Side-effect that cannot be traced
Python Loops
Traceable
Variable
Traces every iteration (unrolling)

9
Inference Mode vs. No-Grad: A Compiler Compatibility Study
The introduction of torch.inference_mode() in PyTorch 1.9 provided a theoretically superior alternative to torch.no_grad() for evaluating models.24 Inference mode offers better performance by disabling view tracking and version counter updates, which are necessary for eager-mode autograd to detect in-place mutations.24 For index computation in recommendation systems, which is purely an inference-like task even during training, the question arises whether inference_mode should replace no_grad.
Theoretical Benefits and Practical Obstacles
In eager mode, the performance difference is clear: inference_mode removes more CPU overhead than no_grad.24 However, within the context of torch.compile, the distinction becomes negligible because the compiler already generates low-level code that avoids the Python-side view tracking logic.26 Furthermore, inference_mode produces tensors with a specialized metadata state that has historically caused compatibility issues with the compiler's backend.26
Specifically, in PyTorch versions 2.1 through 2.3, using inference_mode within a compiled region frequently triggered backend errors or caused the compiler to fallback to eager mode entirely, resulting in performance that was often worse than non-compiled code.26 Developers reported that torch.no_grad() remained the only stable way to disable gradients while maintaining significant compilation speedups.26 While version 2.4 and 2.5 nightlies have addressed many of these bugs, the professional recommendation remains to use torch.no_grad() for localized blocks within a compiled training script to ensure maximum stability across heterogeneous hardware backends.26
Feature
torch.no_grad()
torch.inference_mode()
Mechanism
Disables gradient recording
Disables gradients + view tracking + versioning
CPU Overhead
Low
Lowest (in eager mode)
Compiler Stability
High
Historically Moderate (improved in 2.4+)
Metadata Impact
requires_grad=False
Inference Tensor (stricter)
In-place safety
Tracked via versioning
Errors if mutation occurs

24
The Correct Pattern for Indexing and Embedding Logic
To resolve the silent gradient bug while maintaining compilation efficiency, the model must be refactored to isolate non-differentiable logic.13 The correct pattern avoids function decorators and instead utilizes the torch.no_grad() context manager strictly around the index generation phase.13
Differentiable Selection Mechanics
Consider the mathematical representation of the selection. If $S$ is the input tensor and $i$ are the indices, the indexing operation $Y = S[i]$ is equivalent to a gathering operation. The gradient of the loss $\mathcal{L}$ with respect to $S$ is:

$$\frac{\partial \mathcal{L}}{\partial S_j} = \sum_{k} \frac{\partial \mathcal{L}}{\partial Y_k} \delta(i_k, j)$$
Where $\delta$ is the Kronecker delta. This summation is natively handled by the autograd engine's scatter-add logic during the backward pass.4 By placing the indexing operation $S[i]$ outside the no_grad context, the autograd tape records the dependency between $Y$ and $S$.4

Python


def get_nro_embeddings(seq_embeddings,...):
    # This function should NOT have a @torch.no_grad decorator
    with torch.no_grad():
        # Only non-differentiable logic goes here
        indices = compute_indices(...)

    # The actual selection happens here, outside the block.
    # The compiler traces this as a differentiable 'aten::index' node.
    return seq_embeddings[indices]


This pattern ensures that the symbolic tracer captures a graph where the embedding lookup is fully differentiable, while the complex, potentially branch-heavy index logic is treated as a constant or a separate graph segment.1 This approach also minimizes graph breaks by keeping the "no-grad" scope localized to the smallest possible set of operations.1
Detection Strategies for Silent Gradient Bugs
Silent bugs are among the most difficult to diagnose in large-scale machine learning because they do not trigger exceptions; rather, they lead to subtle training plateaus or failures to converge.6 In recommendation models, where sparse embedding updates are the norm, a total lack of gradients in a sub-module might be mistaken for poor hyperparameter choice or data quality issues.6
Telemetry through PyTorch Hooks
The most robust mechanism for detecting gradient flow interruptions in large models is the use of PyTorch hooks.30 Hooks act as sensors installed on the model's internal "assembly lines," allowing developers to monitor activations and gradients without altering the core logic.30
Backward Hooks: By registering a hook on the encoder's output or the embedding table, developers can verify if gradients are actually reaching those parameters.30 If the grad_output in a backward hook is None or all zeros consistently across batches, it is a definitive sign of a broken autograd chain.30
Forward Hooks: Can be used to monitor the $requires\_grad$ attribute of intermediate tensors. A forward hook can assert that the output of the embedding retrieval function still tracks gradients.30
Adam State and Mathematical Paradoxes
For models using the Adam optimizer, silent bugs can be detected by inspecting the optimizer's internal state.15 Adam maintains a running average of gradients ($exp\_avg$) and squared gradients ($exp\_avg\_sq$).15 A classic symptom of the decorator bug is a mathematical paradox: the encoder parameters show non-zero gradients in their .grad attribute (potentially due to noise or other paths), but the optimizer's second moment ($exp\_avg\_sq$) remains zero, or the weights themselves do not change after optimizer.step().15
In the case of certain hardware backends (like Apple's MPS), silent failures have been observed where non-contiguous memory layouts caused the addcmul_ kernel in Adam to fail silently, preventing weight updates even when gradients were flowing correctly.15 Verifying tensor contiguity using .is_contiguous() and testing on a CPU fallback are essential steps in the isolation of such low-level bugs.15
Detection Technique
Target Symptom
Implementation
Performance Impact
Backward Hook
Interrupted gradient flow
register_full_backward_hook
Low
detect_anomaly()
NaNs / Disconnected graphs
with torch.autograd.detect_anomaly():
Very High
Weight Inspection
Frozen parameters
torch.allclose(w_old, w_new)
Moderate
Adam State Check
Mismatch between grad and update
optimizer.state[p]['exp_avg_sq']
Low
PYSIASSIST
Misplaced no_grad calls
Static code analysis rules
None

6
The Hazard of Error Suppression in Production
The decision to set torch._dynamo.config.suppress_errors = True is often motivated by a desire for production stability, ensuring that if the compiler fails, the model falls back to a working—if slower—eager state.33 However, in the context of debugging gradient flow, this flag is a significant liability.
When errors are suppressed, TorchDynamo may encounter a compilation error related to the @torch.no_grad() decorator or a specialized embedding kernel and silently revert to eager mode.35 This removes the very error messages that would alert the developer to a graph capture failure.12 Furthermore, if the compiler succeeds in capturing a partial graph but fails on the "problematic" function, the model may run in a hybrid state where some optimizations are applied but the gradient semantics are subtly altered.21 Best practices for large-scale models dictate that suppress_errors should be strictly disabled during the training phase and only enabled in inference serving environments after rigorous validation.39
Optimization Strategies for Recommendation Embeddings
Large recommendation systems present a unique "compute vs. memory" trade-off. Unlike dense convolutional or transformer layers, embedding lookups are typically memory-bandwidth bound rather than compute-bound.41 Consequently, the speedups obtained from torch.compile on embedding layers are often less dramatic than those seen in the "over" and "under" MLP networks of a DLRM.5
Selective Compilation and Fused Kernels
Given the complexity of sharding massive embedding tables across multiple GPUs, standard graph compilation can sometimes struggle with model parallelism.41 The most sophisticated recommendation frameworks, such as TorchRec, utilize Table Batched Embeddings (TBE).5 TBE combines multiple embedding lookups into a single kernel call and often fuses the optimizer update directly into the backward pass to reduce memory traffic.5
In such cases, the recommended strategy is to "disable" compilation for the embedding lookup modules using torch.compiler.disable while allowing the rest of the model to be compiled.41 This prevents the compiler from attempting to trace through the complex distributed sharding logic, which often triggers multiple graph breaks, while still allowing the compute-heavy dense layers to benefit from kernel fusion.41
Handling Data-Dependent Operations
In functions like get_nro_embeddings, the use of torch.fx.wrap is a common strategy to tell the compiler to treat a function as a black box.42 While this avoids compilation errors, it does not solve the gradient flow problem if the wrapped function contains a no_grad decorator. A better approach for complex, non-differentiable logic is the use of torch.cond or registering the logic as a custom operator.1 Custom operators allow the developer to explicitly define the autograd behavior, ensuring that even if the forward pass is a "black box" to the tracer, the gradient flow remains intact and theoretically sound.2
Strategic Conclusions and Engineering Recommendations
The management of gradient flow within a compiled environment requires a shift from intuitive Python programming to a more rigorous understanding of symbolic tracing and graph partitioning. The identified failure of the decorator pattern in large recommendation models serves as a primary example of how localized optimizations can have global repercussions on model convergence.
Recommendation 1: Refactor Disabling Logic
Engineers should systematically replace function decorators with localized context managers. The correct placement of torch.no_grad() must be verified to ensure that every differentiable operation—particularly those involving parameters with $requires\_grad=True$—occurs outside the gradient-disabling scope. This preserves the $grad\_fn$ chain and ensures that AOTAutograd can correctly generate the corresponding backward graph.
Recommendation 2: Assertive Validation with Hooks
Relying on the absence of runtime errors is insufficient for confirming the integrity of a training pipeline. Production-grade models should incorporate gradient telemetry using backward hooks. These hooks should be configured to sample gradient magnitudes across critical boundaries, such as the interface between the encoder and the embedding retrieval logic, and trigger alerts if gradients vanish or are consistently nullified.
Recommendation 3: Controlled Error Management
The suppress_errors configuration must be treated as a deployment-only flag. During the training and debugging lifecycle, the compiler should be run with fullgraph=True on individual modules to force the identification of graph breaks. By eliminating these breaks, developers can not only fix silent bugs but also unlock the full performance potential of kernel fusion and CUDA graphs.
Recommendation 4: Selective RecSys Compilation
For recommendation systems specifically, the "over" and "under" dense layers should be compiled using mode="reduce-overhead" to minimize Python latency. The embedding lookups, particularly when using distributed sharding via TorchRec, should be evaluated for performance; if the compiler introduces excessive graph breaks or fails to fuse lookups effectively, these modules should be excluded from compilation in favor of specialized, hand-optimized fused kernels like TBE.
The successful integration of torch.compile into the recommendation pipeline hinges on this duality: utilizing the compiler for what it does best—fusing dense, compute-heavy operations—while maintaining strict, manual control over the autograd tape in the sparse, distributed sections of the model. Through rigorous monitoring and adherence to compiler-safe differentiable patterns, developers can achieve the dual goals of maximum hardware utilization and mathematical correctness in their learning systems.
Works cited
Introduction to torch.compile — PyTorch Tutorials 2.10.0+cu128 documentation, accessed January 30, 2026, https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
Compiling Deepchem Torch Models, accessed January 30, 2026, https://deepchem.io/tutorials/compiling-deepchem-torch-models/
Frequently Asked Questions — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html
Ultimate guide to PyTorch library in Python - Deepnote, accessed January 30, 2026, https://deepnote.com/blog/ultimate-guide-to-pytorch-library-in-python
Introduction to TorchRec — PyTorch Tutorials 2.10.0+cu128 documentation, accessed January 30, 2026, https://docs.pytorch.org/tutorials/intermediate/torchrec_intro_tutorial.html
Investigating and Detecting Silent Bugs in PyTorch Programs - Xiang Gao, accessed January 30, 2026, https://gaoxiang9430.github.io/papers/saner24a.pdf
Introduction to torch.compile - 파이토치, accessed January 30, 2026, https://tutorials.pytorch.kr/intermediate/torch_compile_tutorial.html
torch.export — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
Everything You Need to Know About PyTorch Compile | by LambdaFlux | Medium, accessed January 30, 2026, https://medium.com/@lambdafluxofficial/everything-you-need-to-know-about-pytorch-compile-3d7fd94ce701
Common Graph Breaks — PyTorch 2.9 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/compile/programming_model.common_graph_breaks.html
ML4LM — PyTorch — What Not to Do in PyTorch Models for Better Performance (dynamo), accessed January 30, 2026, https://hoyath.medium.com/ml4lm-pytorch-what-not-to-do-in-pytorch-models-for-better-performance-dynamo-2e5c675dbec2
torch.compile Troubleshooting — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html
no_grad — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html
What happens when we use torch.no_grad() in the middle of forward pass? - autograd, accessed January 30, 2026, https://discuss.pytorch.org/t/what-happens-when-we-use-torch-no-grad-in-the-middle-of-forward-pass/131863
the bug that taught me more about PyTorch than years of using it ..., accessed January 30, 2026, https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/
no_grad causing problem in Pytorch NN model - Stack Overflow, accessed January 30, 2026, https://stackoverflow.com/questions/79652783/no-grad-causing-problem-in-pytorch-nn-model
torch.compile — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/generated/torch.compile.html
What Does Torch No Grad Do? - ProjectPro, accessed January 30, 2026, https://www.projectpro.io/recipes/what-does-torch-no-grad-do
What is the difference between '''@torch.no_grad()''' and '''with torch.no_grad()''', accessed January 30, 2026, https://stackoverflow.com/questions/77875298/what-is-the-difference-between-torch-no-grad-and-with-torch-no-grad
Torch.no_grad not functioning? - autograd - PyTorch Forums, accessed January 30, 2026, https://discuss.pytorch.org/t/torch-no-grad-not-functioning/24678
Gradient Issue after using torch.no_grad - Diffusers - Hugging Face Forums, accessed January 30, 2026, https://discuss.huggingface.co/t/gradient-issue-after-using-torch-no-grad/164699
[Release] LoRA-Safe TorchCompile Node for ComfyUI — drop-in speed-up that retains LoRA functionality : r/StableDiffusion - Reddit, accessed January 30, 2026, https://www.reddit.com/r/StableDiffusion/comments/1l3aetp/release_lorasafe_torchcompile_node_for_comfyui/
[user empathy day 2][based] torch.compile issues #128071 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/128071
PyTorch `torch.no_grad` vs `torch.inference_mode` - Stack Overflow, accessed January 30, 2026, https://stackoverflow.com/questions/69543907/pytorch-torch-no-grad-vs-torch-inference-mode
Top 140 PyTorch Interview Questions and Answers - HackMD, accessed January 30, 2026, https://hackmd.io/@husseinsheikho/pytorch-interview
Performance of `torch.compile` is significantly slowed down under ..., accessed January 30, 2026, https://discuss.pytorch.org/t/performance-of-torch-compile-is-significantly-slowed-down-under-torch-inference-mode/191939
Performance of `torch.compile` is significantly slowed down under `torch.inference_mode` - #2 by bdhirsh - PyTorch Forums, accessed January 30, 2026, https://discuss.pytorch.org/t/performance-of-torch-compile-is-significantly-slowed-down-under-torch-inference-mode/191939/2
`torch.compile` + `torch.no_grad` not working for Mask R-CNN · Issue #97340 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/97340
Torch compile error goes on for 2000 lines when using capture_scaler_output #160800, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/160800
Mastering PyTorch Hooks: Debug, Monitor, and Control Your Model Like a Pro | by Nithin Bharadwaj | Jan, 2026 | TechKoala Insights - Medium, accessed January 30, 2026, https://medium.com/techkoala-insights/mastering-pytorch-hooks-debug-monitor-and-control-your-model-like-a-pro-a30382c1bbf4
PyTorch 101: Understanding Hooks - DigitalOcean, accessed January 30, 2026, https://www.digitalocean.com/community/tutorials/pytorch-hooks-gradient-clipping-debugging
Debug your model (intermediate) — PyTorch Lightning 2.6.0 documentation, accessed January 30, 2026, https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html
plz help me anyone i installed triton : r/comfyui - Reddit, accessed January 30, 2026, https://www.reddit.com/r/comfyui/comments/1i7c4zs/plz_help_me_anyone_i_installed_triton/
google/gemma-2-2b-it · how to solve this error - Hugging Face, accessed January 30, 2026, https://huggingface.co/google/gemma-2-2b-it/discussions/64
Torch.compile raises error when compiled function calls other functions - PyTorch Forums, accessed January 30, 2026, https://discuss.pytorch.org/t/torch-compile-raises-error-when-compiled-function-calls-other-functions/177398
[Bug] glm-4v-9b在昇腾卡推理报错 · Issue #2819 · InternLM/lmdeploy - GitHub, accessed January 30, 2026, https://github.com/InternLM/lmdeploy/issues/2819
Automatic installation of Triton and SageAttention into Comfy v2.0 : r/StableDiffusion - Reddit, accessed January 30, 2026, https://www.reddit.com/r/StableDiffusion/comments/1iyt7d7/automatic_installation_of_triton_and/
Fixing Torch.compile With DLPack On CUDA, accessed January 30, 2026, https://st.splendapp.com/blog/fixing-torch-compile-with-dlpack
pytorch/torch/_dynamo/config.py at main - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/config.py
`torch._dynamo.config.suppress_errors` may not be properly reset · Issue #113606 - GitHub, accessed January 30, 2026, https://github.com/pytorch/pytorch/issues/113606
Ways to use torch.compile - ezyang's blog, accessed January 30, 2026, https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/
torch.fx — PyTorch 2.10 documentation, accessed January 30, 2026, https://docs.pytorch.org/docs/stable/fx.html
accessed January 30, 2026, https://gitee.com/ascend/torchair/pulls/1347.diff?skip_mobile=true
Update modeling_gemma.py · d-matrix/gemma-2b at 8745eab - Hugging Face, accessed January 30, 2026, https://huggingface.co/d-matrix/gemma-2b/commit/8745eabe8f78faab5d2c89d2ed20787c46d85520
How to solve the graph break happen in torch.compile - PyTorch Forums, accessed January 30, 2026, https://discuss.pytorch.org/t/how-to-solve-the-graph-break-happen-in-torch-compile/216858
torch.export Tutorial - PyTorch, accessed January 30, 2026, https://pytorch-cn.com/tutorials/intermediate/torch_export_tutorial.html
Torch Compile and External Kernels — NVIDIA PhysicsNeMo Framework, accessed January 30, 2026, https://docs.nvidia.com/physicsnemo/latest/user-guide/performance_docs/torch_compile_support.html
