SDPA Backend & Custom Triton Kernel Selection
(Attention Optimization)
PyTorch SDPA Backends Overview
PyTorch’s scaled_dot_product_attention (SDPA) operator can transparently dispatch to optimized attention
kernels. By default, three CUDA backends are enabled: FlashAttention, Memory-Efficient Attention, and
the standard “math” implementation . FlashAttention is based on the FlashAttention paper’s fused
kernel (fast, $O(N)$ memory usage), Memory-Efficient Attention comes from the Facebook xFormers library
(trades off some speed for lower memory), and the math backend is PyTorch’s native $O(N^2)$
implementation (a generic fallback) . A fourth backend, CuDNN’s fused attention, was introduced
later (PyTorch 2.5) to leverage NVIDIA’s optimized kernels on newer GPUs like H100 . By default, PyTorch
will automatically select the fastest available backend that meets the input constraints, falling back to
others only if needed . (For example, on Ampere GPUs with half-precision inputs, FlashAttention is
usually chosen, whereas on older GPUs or unsupported shapes, xFormers or math might be used.)
Verifying Which SDPA Backend is Used at Runtime
Currently, PyTorch does not provide a direct return value to indicate which backend was picked for a given
call. However, you can infer or control it via the backend flags and context managers. For instance,
torch.backends.cuda.flash_sdp_enabled() , mem_efficient_sdp_enabled() , and
math_sdp_enabled() report whether each SDPA backend is allowed in the current context . By
default all three are enabled (True), meaning PyTorch’s heuristics will choose the optimal one. To verify or
force a particular backend, you can use the SDPA context manager. For example:
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False,
enable_mem_efficient=False):
# Only FlashAttention is enabled in this block
out = F.scaled_dot_product_attention(q, k, v, ...)
Inside this context, flash_sdp_enabled() will be True while the others are False . This way, you can
test performance or ensure FlashAttention is being used. If the specified backend can’t run, PyTorch will
emit a warning explaining the fallback reason (e.g. “Flash attention kernel not used because…”), or raise
an error if no kernel is available . These warnings help to determine which kernel was ultimately
selected or why a certain backend was skipped.
Forcing Flash Attention with Context Managers
In general, you do not need to manually force FlashAttention – PyTorch will choose it by default when
conditions permit . However, for debugging or performance experiments you can override the
1
1 2
3
4 5
6
7
8 9
4 5
1
selection using the context manager or the higher-level API in torch.nn.attention . For example,
torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION) will restrict SDPA to
FlashAttention only . Forcing FlashAttention ensures that the fastest kernel is used (avoiding any fallback
to the math implementation) as long as your inputs meet its requirements. This can be beneficial if you
know your workload should be compatible – it guarantees you’re getting the FlashAttention memory and
speed benefits. In practice, developers might use with
torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False,
enable_math=False): ... around critical attention calls . Generally, if you trust PyTorch’s
heuristics, explicit forcing isn’t necessary; the context manager is mainly an advanced tool for
debugging and tuning . Keep in mind that if you disable the fallback math kernel and your inputs
violate FlashAttention’s constraints, you’ll get a runtime error (no kernel available), so use this only when
confident in the input compatibility.
Conditions That Trigger Fallback to the Math Backend
PyTorch will fall back to the slow math implementation whenever the faster fused kernels (Flash or
Memory-Efficient) cannot be applied. Key conditions that cause such fallbacks include:
Unsupported data type or precision: FlashAttention only supports FP16 or BF16 inputs (16-bit) .
If you use float32 or other dtypes, FlashAttention is not applicable. (The xFormers efficient backend
supports FP32 as well, but may require different alignment; see head dimensions below.) If neither
fused kernel can run due to dtype, PyTorch uses the math path.
Head dimension requirements: The fused kernels impose limits on the per-head dimension (the
embedding size per attention head). FlashAttention requires the head dimension to be a multiple of
8 (for FP16/BF16) or a multiple of 4 (for FP32) . Additionally, PyTorch’s implementation capped
FlashAttention support at head_dim ≤ 128 in early versions . The Memory-Efficient (xFormers)
kernel also has alignment requirements; for example, it required the last dimension to be divisible by
8 in some PyTorch releases . If your model uses an odd head size (e.g. 80 or 96 as in some
models) on hardware that doesn’t support it, SDPA may fall back. Notably, on Nvidia Ampere (SM80)
GPUs, some non-power-of-two head sizes became supported by FlashAttention, whereas on older GPUs
those would default to math .
GPU architecture: FlashAttention is only available on NVIDIA GPUs with SM80 or higher (Ampere or
newer) . If running on older CUDA architectures (e.g. V100 or Turing GPUs), PyTorch cannot use
FlashAttention. In those cases, it will try the xFormers efficient kernel (which only needs SM50+, so it
works on a broad range of GPUs) . If the device doesn’t meet either kernel’s requirements, math
is used.
Attention mask or pattern: As of PyTorch 2.0, the fused kernels had limited support for arbitrary
attention masks. Only causal masks (upper-triangular masks for autoregressive attention) were
supported via the is_causal=True flag, and any other mask or masking pattern forced a
fallback . For example, providing a custom attn_mask or key padding mask would disable
FlashAttention/xFormers in early implementations. (The recommendation was to combine masks or
use the is_causal flag for supported cases .) If an unsupported mask is given, SDPA prints a
warning and uses the math path.
Dropout in memory-efficient backend: In PyTorch 2.0, the memory-efficient xFormers kernel did
not support dropout (dropout had to be 0 for that backend) . If you were training with a non-
zero dropout in attention, PyTorch would prefer FlashAttention (which does support dropout) . If
FlashAttention wasn’t available (say due to dtype or device), then dropout would force a fallback to
10
7
11
• 12
•
12
12
13 14
15
•
16
16
•
17 18
17
•
19
19
2
math (since xFormers couldn’t be used with dropout enabled). In later versions, this may be relaxed,
but it was a known limitation initially.
Other special cases: Certain shapes (e.g. very small sequence lengths or a batch of 1 with singleton
dimensions) had bugs or unsupported edge cases in fused kernels in some versions . In such
scenarios, PyTorch might emit warnings (“kernel not used because…”) and revert to math for
correctness. Additionally, if the model requests attention output weights (e.g.
need_weights=True in MultiheadAttention ), the fused kernels can’t provide that (since they
don’t form the full attention matrix) – thus PyTorch would use the standard implementation
when need_weights=True unless you disable that requirement.
Summary: Any time the queries/keys/values don’t meet the fused kernel’s constraints (data type, device
capability, head dimension alignment, supported mask/dropout usage), PyTorch will default to the safe C++
math implementation . These fallbacks ensure correctness and generality, at the cost of speed and
memory. The library does log diagnostic warnings explaining the reason for a fallback (e.g. “Mem efficient
attention requires last dimension to be divisible by 8” or “Flash attention not used because X”) – so
checking the console output during execution can reveal why the fast path was skipped in any given run.
SDPA on Newer GPUs (A100 vs H100 and Beyond)
On NVIDIA A100 (Ampere) GPUs, both FlashAttention and memory-efficient kernels are available and
commonly used. Ampere’s large shared memory and Tensor Cores make FlashAttention especially effective
(up to several-fold speedups for moderate head sizes) . However, if your model had unusual head
dimensions (e.g. 80, 96, 256), Ampere might still fall back to math or use xFormers unless kernels improved.
For NVIDIA H100 (Hopper) and later GPUs (“H200” presumably referring to the next generation), PyTorch
introduced an even faster fused backend via CuDNN. In PyTorch 2.5, a CuDNN FlashAttention kernel is
enabled by default on H100s, yielding up to 75% speedup over the earlier FlashAttention v2 implementation
. This means on H100, the SDPA dispatch may choose the CuDNN backend (denoted as
SDPBackend.CUDNN_ATTENTION ) automatically for optimal performance . Users don’t need to do
anything special – if the program is running on H100 or newer, PyTorch will leverage the CuDNN fused
kernel out-of-the-box . The presence of this CuDNN backend further reduces the need to force
FlashAttention manually, since on the latest hardware the library will pick CuDNN’s optimized kernel which
is tuned for that GPU. We can expect that H100 and future Hopper-next (H200) GPUs will continue to have
improved support, either via CuDNN or updated FlashAttention kernels, to handle larger head sizes and
other constraints more efficiently. (For example, the CuDNN backend likely lifts some limitations and is
highly optimized for Hopper’s architecture.) In summary, Ampere GPUs use the Triton-based
FlashAttention path, while Hopper GPUs get an additional boost from CuDNN’s fused attention – all
managed by PyTorch’s backend selection logic .
Custom Triton Kernels & torch.compile() Graph Capture
Beyond PyTorch’s built-in SDPA, this model uses custom Triton GPU kernels for sequence processing tasks
not covered by PyTorch. When integrating these custom kernels with PyTorch 2.x’s torch.compile() (the
TorchDynamo/Inductor graph capture), there are important considerations. By default, a raw Python call to
a Triton JIT kernel (e.g. launching a Triton function via kernel[grid](...) ) can interrupt graph
•
20 21
22
5
13 14
23 24
3
25
3
3
3
capture, because TorchDynamo may treat it as an unsupported Python operation. In practice, you have a
couple of strategies to make custom Triton ops play nicely with torch.compile() :
Wrap Triton code as a PyTorch custom operator (with autograd): The recommended approach (as
of PyTorch 2.3+) is to use the torch.library.triton_op API to register your Triton kernel as a
first-class op . This involves defining your Triton kernel in a Python function decorated with
@triton.jit , then wrapping it with @triton_op("your_namespace::op_name") . You can
then call this torch.ops.your_namespace.op_name(tensors...) in your model. PyTorch’s
compiler will treat it as a single opaque op (just like a native aten op) and include it in the FX graph
rather than breaking out. In the background, Inductor can even fuse around it or schedule it, and
the Triton JIT compilation will happen as needed. The PyTorch tutorial demonstrates that a custom
triton_op will “work with torch.compile and AOTInductor”, producing correct results . This way,
your custom kernel becomes just another node in the optimized graph.
Provide autograd (backward) for custom ops: If your Triton kernel is used in training, you should
register a backward formula so that torch.compile can capture gradients properly. The
triton_op API allows attaching an autograd function via
op.register_autograd(backward_fn, setup_context_fn) . This is preferred over
using a raw torch.autograd.Function , because torch.compile has some known difficulties
(or “footguns”) with tracing through custom autograd.Functions in Python . By registering a
backward composed of torch operations (or even calling another Triton op for the backward pass),
you ensure the entire forward and backward are traceable by the compiler. In short, make your
Triton kernels “compiler-friendly” by using the provided hooks – this avoids graph breaks.
Marking or allowing graph capture: In cases where wrapping as a custom op is not feasible,
PyTorch provides an escape hatch torch._dynamo.allow_in_graph that can decorate a function
to treat it as a graph-able call (it will not be inlined, but treated as an atomic graph node) . This
can sometimes be applied to a custom autograd function or a simple Triton call to prevent Dynamo
from trying to trace inside it. However, using allow_in_graph requires caution – you must ensure
the function is side-effect free and behaves like an op. The custom op approach is more robust.
If you don’t take these measures, you might encounter Dynamo graph breaks or errors when using
torch.compile with custom Triton code. For example, users have reported that an autotuned Triton
kernel inside an autograd.Function caused Dynamo to fall back to eager mode (unable to trace) . The
solution was to integrate that kernel properly as described above. The good news is that as of PyTorch 2.3,
user-defined Triton ops with dynamic shapes and autograd are supported in the compiler path . PyTorch
2.6 further expanded this with torch.library.triton_op for even better integration with custom
tensor subclasses, etc. . In summary, custom Triton kernels can be made fully compatible with
torch.compile() , but you should register them as custom ops so that the compiler “sees” them as
single units . This allows your entire model (PyTorch SDPA + custom Triton parts) to be captured in one
graph for optimization. With this approach, you get the best of both worlds: you can use Triton for niche
operations while still benefiting from TorchInductor’s graph-level optimizations and fusion on the overall
model.
•
26 27
26
•
28 29
28
•
30
31 32
33
34
26
4
JIT vs Pre-Compiled Triton Kernels: Performance Trade-offs
The model design uses two variants of each Triton kernel – a standard JIT-compiled version for training
and a CUDA-compiled version for inference. The reasoning comes down to startup latency vs. flexibility:
Triton JIT (Just-In-Time) compilation: When you call a Triton kernel via the Python API, the Triton
compiler will generate PTX code for your specific kernel and launch configuration at runtime (the
first time you call it for given shapes). This JIT process incurs some initial overhead – compiling and
possibly auto-tuning the kernel launch parameters. In training, this one-time cost is usually
acceptable: you typically warm up your model, and the kernel is reused for many iterations,
amortizing the compile cost. The JIT approach also makes it easier to support autograd (you can
write or generate a backward kernel similarly) and dynamic shapes. The standard (training) variant
of your kernel likely uses Triton JIT so that it can integrate with PyTorch autograd and handle any
shape or dtype flexibility needed during training. Once compiled, Triton caches the kernel (in
memory, and newer versions can even cache to disk) so subsequent calls of the same configuration
are fast . However, if your training uses many different sequence lengths or configurations,
Triton might end up compiling multiple variants, which can add to overhead. Generally, the JIT
variant optimizes for developer convenience and adaptability, at the cost of some upfront
latency.
Pre-compiled CUDA kernel: For inference deployment, the priority is minimal latency and fast
startup. Here, the model uses a pre-compiled CUDA kernel variant – which likely means the Triton
code was ahead-of-time compiled (perhaps using Triton’s compilation utility or rewritten in CUDA C+
+), and the binary (PTX or cubin) is loaded directly. By using a pre-built kernel, you skip the runtime
compilation step entirely. This leads to faster process startup or model initialization, which is crucial
in production inference scenarios (no waiting for JIT). The trade-off is that this variant is less flexible:
it’s probably compiled for a specific GPU architecture (say, sm80 or sm90) and for specific input
dimensions or a range of dimensions. It may also omit autograd support (since for inference we
don’t need gradients, the kernel might be simplified or have no backward implementation). In
essence, the pre-compiled CUDA kernels are frozen implementations optimized for inference, giving
you consistent performance from the first inference call. There is no auto-tuning or compilation
happening at runtime, so you avoid any jitter. The downside is if you need to run on a different GPU
architecture or change the sequence length beyond what was anticipated, you’d have to recompile
the kernel separately.
Performance considerations: In practice, a JIT Triton kernel after warm-up can run just as fast as a pre-
compiled kernel, since they ultimately generate similar low-level code. The difference is in startup and
adaptability. For long-running training jobs, the JIT overhead is negligible compared to the total training
time, and the benefit is easier integration with autograd and varying shapes. For inference, especially in
environments where a model might be loaded on demand (serverless or microservice contexts), the JIT
compile time can be a significant fraction of the first query’s latency. By using the pre-compiled version in
inference, you ensure there’s no pause to compile – the kernel launches immediately. This can be important
for latency-sensitive applications. Moreover, not having autograd in the inference kernel can slightly reduce
its complexity and overhead (no need to save context for backward, etc.).
Another subtle point: Triton’s auto-tuning. The training/JIT variant might be using Triton’s autotuner to
find the best launch config for your specific hardware and problem size. This yields an optimized kernel, but
•
35 36
•
5
the tuning process itself can take a bit of time on first run (trying several configs). The inference variant, on
the other hand, likely has a chosen launch configuration baked in (based on prior tuning or developer
optimization). This removes any tuning overhead at runtime. The result is a known-good performance
without any trial runs.
In summary, the JIT Triton kernels prioritize flexibility (dynamic shapes, ease of development,
autograd support) at the cost of an initial compile cost, whereas the pre-compiled (CUDA) kernels
prioritize immediate execution and streamlined inference. For this model, using the JIT version during
training means developers can iterate and let PyTorch/Triton handle compilation and gradients. When
deploying to production, swapping in the pre-compiled kernel means users on (say) A100/H100 GPUs get
the optimized kernel with zero compile delay and slightly faster startup. This dual approach ensures fast
training iterations and optimal inference throughput. The overall performance difference at steady-
state runtime between the two should be minimal (they likely achieve similar kernel efficiency); it’s the
startup and usage context where they differ. As a best practice, you might time your inference throughput/
latency with both approaches – if the JIT warm-up is acceptable and happens rarely, you could simplify by
using the same kernel for both. But in high-scale systems, eliminating any JIT step (and having
deterministic, pre-tuned performance) is often worth the extra effort of maintaining a pre-compiled kernel.
Conclusion: Integrating SDPA and Custom Kernels
In this scenario, we see PyTorch’s native optimized attention working in tandem with custom Triton
kernels to maximize performance for sequence modeling. PyTorch’s SDPA covers the multi-head attention
computation with highly optimized backends (FlashAttention, xFormers, or CuDNN) that automatically
exploit the hardware capabilities . At the same time, custom Triton kernels handle other sequence
processing tasks that PyTorch doesn’t optimize out-of-the-box. To get the most out of this hybrid approach,
we verify which kernels are used and ensure they run under ideal conditions (using context managers for
debugging and tuning), and we integrate everything smoothly with PyTorch 2.x’s compiler. By registering
Triton kernels as custom ops, the entire model – attention + custom logic – can be captured and fused by
torch.compile , yielding end-to-end speedups without losing the benefits of either component .
On modern accelerators like A100, H100, and beyond, this approach leverages all available optimizations:
FlashAttention or CuDNN fused kernels for attention, and Triton’s efficient GPU code for bespoke operations. The
result is a highly optimized sequence model pipeline that minimizes memory overhead and maximizes
throughput, across both training and inference deployments.
Sources: The descriptions of SDPA backends and their behaviors are based on PyTorch documentation and
forums , as well as the PyTorch 2.x release notes for newer features (CuDNN backend) .
Details on Triton kernel integration come from official tutorials and user experiences integrating Triton with
torch.compile . These sources provide insight into how PyTorch chooses kernels and how to
extend it with custom GPU ops.
python - What is the difference between various backends in torch.nn.attention.SDPBackend, and
what do they mean? - Stack Overflow
https://stackoverflow.com/questions/79167465/what-is-the-difference-between-various-backends-in-torch-nn-attention-
sdpbackend
24 3
26 37
1 38 39 3
26 28
1 2 25
6
PyTorch 2.5 Release Blog – PyTorch
https://pytorch.org/blog/pytorch2-5/
torch.nn.functional.scaled_dot_product_attention — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
Accelerated PyTorch 2 Transformers – PyTorch
https://pytorch.org/blog/accelerated-pytorch-2/
Flash Attention - PyTorch Forums
https://discuss.pytorch.org/t/flash-attention/174955
SDPA memory efficient and flash attention kernels don't work with singleton dimensions ·
Issue #127523 · pytorch/pytorch · GitHub
https://github.com/pytorch/pytorch/issues/127523
torch.nn.attention.sdpa_kernel — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html
Memory efficient attention for tensors where the last dimension is not divisible by 8 · Issue #110630
· pytorch/pytorch · GitHub
https://github.com/pytorch/pytorch/issues/110630
Out of the box acceleration and memory savings of decoder models with PyTorch 2.0 – PyTorch
https://pytorch.org/blog/out-of-the-box-acceleration/
Using User-Defined Triton Kernels with torch.compile — PyTorch Tutorials
2.10.0+cu128 documentation
https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
Frequently Asked Questions — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html
Torch.compile with custom Triton kernel - torch.compile - PyTorch Forums
https://discuss.pytorch.org/t/torch-compile-with-custom-triton-kernel/192876
Understanding Triton Cache: Optimizing GPU Kernel Compilation
https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/
3
4 38
5 11 12 16 17 18 19 22 37 39
6 7
8 9 20 21
10
13 14
15 23 24
26 27 28 29 33 34
30
31 32
35 36
7
