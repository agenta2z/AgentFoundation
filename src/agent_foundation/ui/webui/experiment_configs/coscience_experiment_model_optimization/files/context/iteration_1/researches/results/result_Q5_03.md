# PyTorch autocast, SDPA dispatch, and mixed-precision internals

Explicit `.to(bfloat16)` calls inside autocast blocks are **redundant but not harmful**—autocast checks tensor dtypes before casting and skips tensors already in the target format. The pattern wastes an explicit cast kernel when autocast would have handled it automatically. For SDPA, Flash Attention requires **fp16/bf16 only**, head dimensions **divisible by 8**, and **no custom attention masks**—fp32 inputs or arbitrary masks trigger fallback to the slower Math kernel. Your dtype caching pattern is **correct and necessary** because outputs from autocast blocks may have different dtypes than inputs.

## Autocast performs per-operation casting, not global tensor conversion

PyTorch's `torch.autocast` works at the **operator dispatch level**, not by modifying tensor dtypes globally. When an eligible operation (like `torch.mm`, `conv2d`, or `linear`) executes inside an autocast block, the dispatcher intercepts the call and casts inputs to lower precision before execution. The critical implementation detail from `autocast_mode.cpp`:

```cpp
Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, DeviceType device_type) {
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    // Cast only if dtype differs
    return arg.to(to_type);
  }
  return arg;  // NO CAST if already correct dtype
}
```

When `arg.scalar_type() != to_type` is false, **no cast operation occurs**. This means your explicit `.to(bfloat16)` launches a cast kernel, and then autocast's subsequent check finds the tensor already in bf16 and does nothing. You've done work autocast would have done automatically. The official documentation explicitly warns: "You should not call `half()` or `bfloat16()` on your model(s) or inputs when using autocasting."

**Double-casting overhead is rare** but possible. If you explicitly cast to a *different* dtype than autocast wants, you get two casts:
```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x = input.to(torch.float32)  # Cast 1: explicit to fp32
    y = torch.mm(x, x)           # Cast 2: autocast to bf16 for matmul
```

For model weights, autocast **caches** the lower-precision versions of float32 parameters with `requires_grad=True`, avoiding repeated casting across forward passes.

## Your dtype caching pattern is correct and necessary

The pattern you're questioning is valid:
```python
input_dtype = seq_embeddings.dtype  # Cache before
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16_training):
    out = cross_attn(seq_embeddings, ...)
out = out.to(input_dtype)  # Restore original precision
```

**Autocast does not modify tensor `.dtype` attributes**—accessing `tensor.dtype` always returns the actual storage dtype. However, **output tensors from autocasted operations may have different dtypes than inputs**:

```python
x = torch.randn(10, 10, device='cuda')  # float32
with torch.autocast(device_type='cuda', dtype=torch.float16):
    y = torch.mm(x, x)
    print(y.dtype)  # torch.float16 — output in lower precision!
    print(x.dtype)  # torch.float32 — input unchanged
```

The caching pattern ensures you can restore the original precision for downstream operations that expect float32. Without caching, you'd need to determine the "correct" dtype to restore, which may not be obvious if the code path varies.

## SDPA backend verification uses can_use_* functions and profiling

PyTorch provides multiple methods to verify which SDPA backend executes at runtime. The most reliable approach uses **`can_use_*` functions with `debug=True`**, which logs specific failure reasons:

```python
from torch.nn.attention import SDPAParams
import torch

query = torch.randn(2, 8, 128, 64, dtype=torch.float16, device="cuda")
key = torch.randn(2, 8, 128, 64, dtype=torch.float16, device="cuda")
value = torch.randn(2, 8, 128, 64, dtype=torch.float16, device="cuda")

params = SDPAParams(query, key, value, attn_mask=None, dropout=0.0, is_causal=False)

print("Flash:", torch.backends.cuda.can_use_flash_attention(params, debug=True))
print("Efficient:", torch.backends.cuda.can_use_efficient_attention(params, debug=True))
```

**Force specific backends** using `torch.nn.attention.sdpa_kernel()` (PyTorch 2.3+):
```python
from torch.nn.attention import SDPBackend, sdpa_kernel

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(query, key, value)
```

**Profile to identify active kernels**—kernel names reveal the backend:

| Kernel Pattern | Backend |
|----------------|---------|
| `pytorch_flash::flash_fwd_kernel` | Flash Attention |
| `fmha_*`, `efficient_attention_*` | Memory-Efficient |
| `aten::bmm`, `aten::softmax` | Math fallback |

Environment variables provide global control: `TORCH_SDPA_FLASH_ENABLED=0` disables Flash Attention entirely.

## SDPA dtype and shape requirements determine backend selection

Flash Attention has **strict requirements**—violating any triggers fallback:

| Requirement | Flash Attention | Memory-Efficient | Math Kernel |
|-------------|-----------------|------------------|-------------|
| **fp16** | ✅ | ✅ | ✅ |
| **bf16** | ✅ (SM80+) | ✅ | ✅ |
| **fp32** | ❌ Falls back | ✅ | ✅ |
| **fp64** | ❌ | ❌ | ✅ (unique) |
| **Head dim** | Divisible by 8, ≤256 | Divisible by 8 (fp16/bf16) or 4 (fp32) | No restriction |
| **Custom attn_mask** | ❌ Must be None | ✅ Additive masks | ✅ All masks |
| **GPU arch** | SM80+ (Ampere/Ada/Hopper) | SM50+ | Any |

**A100-specific considerations**: Native bf16 support, head dimensions up to 256 supported in backward pass (flash-attn 2.5.5+), and optimal FlashAttention-2 performance. The CUDA grid limit of **65,535 blocks** means `batch_size × num_heads` cannot exceed this value.

**Critical contiguity requirement**: Both Flash and Memory-Efficient backends require the **last dimension to be contiguous** (`stride(-1) == 1`). Non-contiguous key/value tensors with custom masks can produce incorrect results (PyTorch issue #112577).

Common fallback triggers in practice:
- Using `need_weights=True` (forces attention weight computation)
- Head dimension of 1 (singleton dimensions)
- Nested tensors during training (PyTorch 2.0)
- Pre-Ampere GPUs (T4, RTX 20xx)

## Triton kernels ignore autocast—explicit dtype handling required

**Triton kernels do not automatically respect autocast context.** Unlike PyTorch's built-in operations with registered dispatch rules, Triton kernels are opaque to the dispatcher:

```python
x = torch.randn(1024, 1024, device='cuda')  # float32

with torch.autocast(device_type='cuda', dtype=torch.float16):
    # Triton kernel still receives float32—autocast doesn't auto-cast
    # inputs to non-PyTorch ops
    output = my_triton_kernel_wrapper(x)  # sees float32
```

However, if preceding PyTorch ops under autocast produced float16 tensors, the Triton kernel receives float16:

```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    x = torch.mm(a, b)  # x is now float16 (autocast op)
    output = my_triton_kernel_wrapper(x)  # sees float16
```

**Best practice for Triton kernels under autocast**:

1. **Handle dtype explicitly inside kernels**—cast to float32 for numerical stability, store back in input dtype:
```python
@triton.jit
def my_kernel(X, Y, N, BLOCK: tl.constexpr):
    x = tl.load(X + offsets).to(tl.float32)  # Load and upcast
    # ... compute in float32 ...
    tl.store(Y + offsets, result.to(X.dtype.element_ty))  # Store in input dtype
```

2. **Use `torch.library.triton_op`** (PyTorch 2.6+) for torch.compile compatibility:
```python
from torch.library import triton_op, wrap_triton

@triton_op("mylib::my_op", mutates_args={})
def my_op(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    wrap_triton(my_kernel)[grid](x, out, x.numel())
    return out
```

3. **Use `custom_fwd`/`custom_bwd` decorators** for autograd Functions to inherit autocast state:
```python
from torch.amp import custom_fwd, custom_bwd

class MyTritonOp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x):
        # Runs with caller's autocast state
        ...
```

## Recommended code patterns for your A100 optimization

**Remove redundant explicit casts**:
```python
# Before (redundant)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x = x.to(dtype=torch.bfloat16)  # Unnecessary
    output = model(x)

# After (correct)
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(x)  # Autocast handles casting automatically
```

**Keep your dtype caching pattern** when you need original precision after autocast:
```python
input_dtype = tensor.dtype
with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16_training):
    out = model(tensor)
out = out.to(input_dtype)  # Valid—output may be bf16
```

**Verify SDPA backend at development time**:
```python
def verify_flash_attention(q, k, v):
    params = SDPAParams(q, k, v, None, 0.0, False)
    if not torch.backends.cuda.can_use_flash_attention(params, debug=True):
        raise RuntimeError("Flash Attention constraints not met")
```

**For Triton kernels**, always check input dtype and handle conversion explicitly—don't assume autocast will pre-cast your inputs.

## Conclusion

The explicit `.to(bfloat16)` in your code is **safe to remove**—autocast handles casting automatically and skips already-cast tensors. Your dtype caching pattern before autocast blocks is **correct** because output dtypes may differ from inputs. For SDPA on A100 with bf16, ensure head dimensions are **divisible by 8**, use `is_causal=True` instead of custom masks for Flash Attention eligibility, and verify backend selection using `can_use_flash_attention(params, debug=True)`. Custom Triton kernels require **explicit dtype handling** since they bypass autocast's dispatch mechanism entirely.
