Autocast Dtype Casting & Kernel Dispatch Optimization
Removing Redundant Casts Under Autocast

When using automatic mixed precision (AMP) with torch.autocast, you should avoid manual dtype casts on inputs or models inside the autocast region. Autocast automatically casts operations to the specified lower precision (e.g. bfloat16) when beneficial, so an explicit x = x.to(torch.bfloat16) inside the context is redundant and can introduce extra overhead. In fact, the PyTorch docs recommend not calling .half() or .bfloat16() on your models or inputs when using autocast. Each manual cast launches a cast kernel, which is usually unnecessary since autocast will handle casting for eligible ops. Moreover, autocast has an internal cache for casting: it will cache the bfloat16/float16 version of a tensor (such as model parameters) after the first cast, so subsequent uses in the same forward pass reuse the cached half-precision tensor. Manually casting tensors yourself can bypass or duplicate this mechanism, leading to avoidable kernel launches. In summary, remove explicit tensor.to(bfloat16) calls under an autocast context – let autocast perform the casting. This cleanup reduces redundant casts and ensures the system uses the optimized casting and caching that AMP provides.
Verifying Which SDPA Backend Is Used

PyTorch’s scaled dot-product attention (SDPA) operator will automatically choose among Flash Attention, Memory-Efficient Attention, or the standard Math implementation based on hardware and dtype. To verify which backend is actually being used at runtime, you can leverage the torch.backends.cuda.sdp_kernel context manager. By enabling or disabling specific backends in this context, you can force a particular backend and observe behavior or performance. For example, to force Flash Attention and disable others, one can do:

with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=True)

Inside this block, PyTorch will attempt to use Flash Attention exclusively. If Flash Attention is not available for the given inputs/hardware, the operation will fall back (potentially raising an error if all other backends are disabled). Similarly, you could enable only memory-efficient or only math in the context manager. This technique helps confirm which kernel is in use by measuring performance or checking for errors when others are turned off. The Hugging Face documentation also suggests using this context manager to check backend availability – if the operation succeeds with only Flash enabled, you know Flash was used; if it fails or runs only when math is enabled, then it was falling back to math, etc. Additionally, PyTorch provides flags like torch.backends.cuda.flash_sdp_enabled() (and similar for mem-efficient and math) that indicate if a given backend is currently allowed. By inspecting these within the context (as shown above), you can see which backends are active. In short, use the torch.backends.cuda.sdp_kernel context or backend flags to probe and ensure the desired SDPA kernel is being employed.
Dtype Conditions for SDPA Backend Selection (and Math Fallback)

The SDPA kernel dispatch is heavily influenced by tensor dtype and hardware capabilities. Key conditions include:

    Flash Attention: This fused kernel requires 16-bit inputs – specifically FP16 or BF16 tensors – and is only available on NVIDIA GPUs with SM80 or newer (Ampere A100, Hopper H100, etc.). All query, key, and value matrices must be of type float16 or bfloat16 for Flash to be chosen. If this condition isn’t met (e.g. inputs are float32 or an older GPU), Flash Attention won’t be used.

    Memory-Efficient Attention: This backend supports a broader range of dtypes (both FP16/BF16 and FP32) and works on a wider range of GPUs (SM50 and above). On pre-SM80 GPUs (like V100 or 1080 Ti), Flash is unavailable but the memory-efficient implementation can often be used for 16-bit or 32-bit attention. If inputs are float32 on such GPUs, the kernel will typically default to the memory-efficient path (since Flash isn’t an option but mem-efficient can handle FP32).

    Math (Fallback) Implementation: This is the default, unoptimized attention (the O(N²) memory version) used when none of the conditions for Flash or mem-efficient are satisfied. Several scenarios trigger a forced fallback to math:

        Unsupported dtype combinations: If inputs aren’t all the same supported dtype, the dispatcher may not find a suitable fused kernel. For example, if any of the Q/K/V tensors is float64 or if there’s a type mismatch (one tensor in FP32 and others in FP16 without autocast promoting them), the safe route is to use the math implementation. Autocast usually ensures dtypes match by casting to the lower precision, but any leftover mismatch or unsupported type (e.g. FP64) will cause a fallback to math.

        Use of unsupported features: As of PyTorch 2.0, certain features disable the fused kernels. Notably, providing a custom attention mask (other than a causal mask) will cause SDPA to fall back to the standard math implementation. The fused flash/mem-efficient kernels currently only support the causal mask via is_causal=True; any other attn_mask or key padding mask (if not folded into a NestedTensor) triggers the fallback. Similarly, if dropout is used during training, support depends on the kernel: Flash Attention in PyTorch 2.0 supports arbitrary dropout, but the memory-efficient kernel did not support dropout (required dropout_p=0.0) in that initial release. Thus, a non-zero dropout probability could force a switch to math if only the mem-efficient path was available for a given dtype. (Flash does handle dropout, so on SM80+ it might still use Flash in training; on older GPUs, dropout would make mem-efficient ineligible, leaving math as the fallback.)

        Incompatible dimensions or hardware: Flash Attention has constraints on the attention head size (head dimension must be a multiple of 8 for FP16/BF16, and was limited to ≤128 in PyTorch 2.0). If these constraints are violated (e.g. very large head dimension that the flash kernel doesn’t support, or sequence lengths that exceed what can be handled in shared memory), PyTorch will fall back to another kernel. Often it will try the mem-efficient kernel in such cases (which is more flexible), but if that also cannot handle the scenario, the math kernel is used as a last resort. In essence, whenever the higher-performance kernels cannot be applied due to dtype or other constraints, the library uses the safe “catch-all” math implementation.

In practice, to ensure the fastest kernel is used, you should keep inputs in the supported dtypes and satisfy the constraints. For example, use FP16 or BF16 for attention on Ampere/Hopper GPUs so that Flash or mem-efficient attention can run (FP32 would work on mem-efficient but not flash). Also avoid unnecessary dtype mismatches – if one part of your model produces FP32 while the rest is autocast to BF16, reconcile them (autocast or manual cast) so that the attention sees uniform types. By cleaning up redundant casts (as discussed earlier) and adhering to these requirements, you can prevent unintentional fallbacks to the slow math kernel and ensure optimal performance.

(Note: NVIDIA A100 (SM80) and H100 (SM90) GPUs both have strong BF16/FP16 support for these fused kernels. H100, being newer, can handle even larger workloads and may lift some limitations in newer PyTorch versions. Future GPUs (e.g., a hypothetical “H200”) will likely continue this trend, potentially extending supported head dimensions or adding even more efficient attention kernels. Always refer to the latest PyTorch release notes for updated kernel capability on new hardware.)
Autocast Interaction with Custom Triton Kernels

Custom ops or kernels (for example, those written in Triton or other libraries) require special consideration with autocast. Autocast by default only knows how to handle built-in operations; it has a whitelist of ops for which it will cast inputs to the target dtype. If you call a custom Triton kernel directly, autocast might not automatically cast the inputs to BF16/FP16, even inside an autocast region, unless that custom op is integrated with AMP. This means that if your Triton kernel expects a certain dtype (say it’s only implemented for float16 for performance), you could end up feeding it float32 tensors under autocast (because autocast won’t downcast them on its own if it doesn’t recognize the op). The result could be that the kernel either runs in a slower precision or fails if it strictly requires 16-bit inputs.

To address this, PyTorch provides a way to register autocast behavior for custom operations. Using torch.library.register_autocast (or similar mechanisms), you can tell autocast to cast the inputs of your custom op to a given dtype when in an autocast region. In essence, you inform AMP that “when this op is called under autocast, cast its float inputs to bfloat16 (or float16)”. If writing a custom C++/Python op, you should use this to ensure compatibility with AMP. If that's not done, a safer approach is to manually handle the casting: for example, call my_triton_kernel(x.half(), y.half()) inside the autocast block if you know it needs half precision. Another approach is to disable autocast around the custom kernel call and manage precision manually for that part.

A concrete example is FlashAttention 2 (an advanced Triton-based attention kernel released by HazyResearch). It isn’t part of PyTorch by default, so if you install and use it, you must ensure the inputs are half-precision. In fact, the Hugging Face integration for FlashAttention-2 requires the model to be cast to torch.float16 or torch.bfloat16 beforehand, otherwise it won’t activate the kernel. This highlights that custom Triton kernels often won’t automatically cooperate with autocast – the developer or user has to arrange the correct dtype. In summary, when using custom kernels, make sure to meet their dtype requirements explicitly. You can register the op with AMP or do manual casts, but do not assume autocast will handle it unless it’s an official part of the autocast op list. By doing so, you ensure that your custom fused kernels run in the intended precision and you get the maximum benefit from them, without dtype mismatches or unintended fallbacks.
Caching Dtypes and Restoring Precision After Autocast

When exiting an autocast region, you often want to return results to a higher precision (e.g., back to float32) for the rest of your workflow. A common pattern is to cache the original dtype of a tensor before entering the autocast block, so that you can convert outputs back to that dtype afterward. For example:

orig_dtype = seq_embeddings.dtype  # e.g. torch.float32
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16_training):
    out = cross_attn(seq_embeddings, ...)  # runs in bf16
# Restore to original precision:
out = out.to(orig_dtype)

This approach is recommended for a couple of reasons. First, it makes it explicit what dtype you expect to return to. In the code above, orig_dtype would typically be float32, so the output of the attention (which was computed in BF16 for speed) gets cast back to FP32. If you hadn’t saved orig_dtype, you might be tempted to do something like out = out.to(seq_embeddings.dtype). However, note that seq_embeddings.dtype is still float32 in this case (since autocast doesn’t actually change the stored dtype of seq_embeddings; it only affects the computation), so that would work in this particular scenario. But if the input tensor itself had been converted to BF16 inside the autocast region (e.g., some code erroneously did seq_embeddings = seq_embeddings.to(torch.bfloat16) inside), then by the end of the block seq_embeddings.dtype would have changed. Caching the dtype before any autocast (or casting) side-effects ensures you have the true original type.

More generally, any tensor produced inside an autocast region may be in lower precision (FP16/BF16). If you immediately use that tensor in subsequent operations outside autocast (which might expect FP32), you could get dtype mismatch errors. The PyTorch AMP documentation notes that if you have tensors of different dtypes (e.g. a FP16 output and a FP32 tensor) interacting, you should cast the autocast output back to float32 (or whatever higher precision is needed). Therefore, it’s good practice to do this casting back explicitly after the autocast block. Using a cached dtype makes the intent clear and avoids any confusion.

In summary, cache the original dtype before entering an autocast block whenever you need to convert results back. After the autocast region, use that cached dtype to cast outputs to the desired precision. This way, you safely restore the precision for subsequent computations and avoid any type mismatch issues. It also documents the expected dtype transition in your code, which is helpful for maintenance. If the original tensor’s dtype is easily accessible and unchanged, you could technically use it directly after the block – but caching upfront is a simple, defensive step that guarantees you’re using the right type. This complements the earlier guidance: don’t mutate dtypes inside autocast, and do restore precision afterward as needed. Following this pattern will keep your mixed-precision code stable and correct, while still reaping the performance benefits of autocast during the critical sections.

Sources:

    PyTorch Forums – Flash Attention usage and SDPA backend control

    PyTorch Documentation – Automatic Mixed Precision Guidelines

    PyTorch 2.0 Release Blog – SDPA backends (Flash, Mem-Eff, Math) and constraints

    Hugging Face Transformers Performance Guide – Autocast caching and FlashAttention-2 requirements

Citations

Automatic Mixed Precision package - torch.amp — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/amp.html

Performance and Scalability: How To Fit a Bigger Model and Train It Faster
https://huggingface.co/docs/transformers/v4.15.0/en/performance

Flash Attention - PyTorch Forums
https://discuss.pytorch.org/t/flash-attention/174955

GPU inference
https://huggingface.co/docs/transformers/v4.39.0/en/perf_infer_gpu_one

Flash Attention - PyTorch Forums
https://discuss.pytorch.org/t/flash-attention/174955

Accelerated PyTorch 2 Transformers – PyTorch
https://pytorch.org/blog/accelerated-pytorch-2/

Accelerated PyTorch 2 Transformers – PyTorch
https://pytorch.org/blog/accelerated-pytorch-2/

Accelerated PyTorch 2 Transformers – PyTorch
https://pytorch.org/blog/accelerated-pytorch-2/

Accelerated PyTorch 2 Transformers – PyTorch
https://pytorch.org/blog/accelerated-pytorch-2/

Accelerated PyTorch 2 Transformers – PyTorch
https://pytorch.org/blog/accelerated-pytorch-2/

Accelerated PyTorch 2 Transformers – PyTorch
https://pytorch.org/blog/accelerated-pytorch-2/

torch.library — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/library.html

GPU inference
https://huggingface.co/docs/transformers/v4.39.0/en/perf_infer_gpu_one

Automatic Mixed Precision package - torch.amp — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/amp.html

Accelerated PyTorch 2 Transformers – PyTorch
https://pytorch.org/blog/accelerated-pytorch-2/
All Sources
docs.pytorch
huggingface
discuss.pytorch
pytorch
