# Maximizing torch.compile for Recommendation Models on H100 GPUs

Embedding-heavy recommendation models require a **hybrid compilation strategy**: compile dense components aggressively while excluding sparse architectures from compilation. Direct torch.compile on full DLRM-style models yields limited benefits due to graph breaks from sparse operations, with practitioners reporting "essentially zero difference" on large embedding tables. The optimal approach combines TorchRec/FBGEMM for embeddings with torch.compile on MLP interaction layers, achieving **20-50% latency reductions** on the dense portions while preserving FBGEMM's hand-tuned sparse kernels.

H100 GPUs unlock significant additional gains through FP8 Tensor Cores (**6x throughput** vs A100 FP16), TMA for **59% memory bandwidth improvement**, and `max-autotune` mode with CUDA graphs. However, the sparse embedding lookup—typically 60-80% of recommendation model compute—cannot leverage these features through compilation alone. Production deployments must focus torch.compile on the remaining dense architecture while implementing robust caching, warm-up procedures, and recompilation monitoring.

---

## Graph breaks fragment optimization in recommendation models

Graph breaks occur when TorchDynamo encounters untraceable code, fragmenting the computation into isolated graphs with Python overhead between them. Recommendation systems face particularly severe challenges because their core operations—embedding lookups with variable-length sequences, data-dependent control flow for feature processing, and custom CUDA kernels—all trigger breaks.

**Critical break causes for RecSys:**

```python
# Data-dependent control flow - BREAKS GRAPH
def forward(self, x):
    if x.sum() > 0:  # Tensor-dependent branch
        return self.layer_a(x)
    return self.layer_b(x)

# Fix with torch.cond
def forward(self, x):
    return torch.cond(
        x.sum() > 0,
        lambda x: self.layer_a(x),
        lambda x: self.layer_b(x),
        (x,)
    )
```

Sparse embeddings with `sparse=True` fail outright in Inductor with an assertion error—the backend explicitly checks `assert not sparse`. Standard `torch.nn.Embedding` with dense gradients compiles successfully, but TorchRec's specialized `EmbeddingBagCollection` requires the `@torch.compiler.disable` decorator on the sparse forward path. The `.item()` call, commonly used to extract scalar values for logging or dynamic decisions, forces immediate graph termination and Python fallback.

Dynamic shapes from variable-length user histories trigger **guard failures** and recompilation rather than true graph breaks. Each new sequence length combination generates fresh compiled code until hitting `recompile_limit` (default: 8 per function). Production systems serving millions of users with diverse history lengths can exhaust this limit within minutes, degrading to eager execution.

---

## Detection tools reveal compilation bottlenecks

The primary diagnostic is `fullgraph=True`, which converts silent fallbacks into explicit errors:

```python
# Immediately surfaces any graph breaks as exceptions
model = torch.compile(model, fullgraph=True)
```

For detailed analysis, environment variables expose Dynamo's internal decisions:

```bash
# Show graph break locations and causes
TORCH_LOGS=graph_breaks python train.py

# Track recompilation triggers from shape changes
TORCH_LOGS=recompiles,guards python train.py

# Maximum verbosity for debugging
TORCH_LOGS="+dynamo,aot,inductor" python train.py
```

The `torch._dynamo.explain()` function provides structured output identifying break count and causes:

```python
explanation = torch._dynamo.explain(model)(sample_input)
print(f"Graph breaks: {explanation.graph_break_count}")
print(explanation.break_reasons)
```

Production monitoring should track `torch._dynamo.utils.counters` for recompilation rates—excessive recompiles indicate shape variability requiring `mark_dynamic()` intervention or input bucketing.

---

## Elimination patterns transform dynamic operations to static alternatives

The most effective pattern for recommendation models wraps the sparse architecture with `@torch.compiler.disable` while allowing compilation on the dense interaction and ranking layers:

```python
class DLRM(nn.Module):
    def __init__(self):
        self.sparse_arch = EmbeddingBagCollection(...)
        self.dense_arch = nn.Sequential(...)
        self.interaction = InteractionArch(...)
        self.over_arch = nn.Sequential(...)

    @torch.compiler.disable
    def embed(self, kjt):
        return self.sparse_arch(kjt)  # Runs eager

    def forward(self, dense_features, sparse_features):
        sparse_emb = self.embed(sparse_features)
        dense_out = self.dense_arch(dense_features)  # Compiled
        interaction = self.interaction(dense_out, sparse_emb)  # Compiled
        return self.over_arch(interaction)  # Compiled

model = torch.compile(DLRM(), mode="max-autotune")
```

For variable batch sizes, mark the batch dimension as dynamic rather than triggering recompilation:

```python
batch_input = torch.randn(batch_size, feature_dim)
torch._dynamo.mark_dynamic(batch_input, 0)  # Batch dimension varies
torch._dynamo.mark_dynamic(batch_input, 0, min=1, max=512)  # With bounds
```

Custom CUDA kernels require FakeTensor registration to participate in compilation:

```python
from torch.library import custom_op

@custom_op("mylib::fused_embedding", mutates_args=())
def fused_embedding(indices: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return custom_cuda_impl(indices, table)

@fused_embedding.register_fake
def _(indices, table):
    batch_size = indices.shape[0]
    embed_dim = table.shape[1]
    return torch.empty(batch_size, embed_dim, device=indices.device)
```

---

## H100 backend selection maximizes Hopper architecture benefits

The Inductor backend generates Triton kernels that exploit H100's architectural advances. **FP8 Tensor Cores deliver 1,979 TFLOPS**—6x the A100's FP16 throughput—making `torchao.float8` integration compelling for dense layers:

```python
from torchao.float8 import convert_to_float8_training

# Only convert dense layers (embedding tables stay FP32/FP16)
def filter_fn(mod, fqn):
    if isinstance(mod, torch.nn.Linear):
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
    return False

convert_to_float8_training(model, module_filter_fn=filter_fn)
compiled = torch.compile(model, mode="max-autotune")
```

TMA (Tensor Memory Accelerator) enables **1.45 TB/s** effective memory throughput versus 910 GB/s without—a 59% improvement that Triton kernels can leverage for memory-bound operations common in recommendation inference.

| Mode | Compile Time | Runtime | Memory | Use Case |
|------|-------------|---------|--------|----------|
| `default` | 1-5 min | Good | Baseline | Development |
| `reduce-overhead` | 2-8 min | Better (small batch) | +10-20% | Inference |
| `max-autotune` | 10-60+ min | Best | Variable | Production |

For inference serving with predictable shapes:

```python
compiled = torch.compile(
    model,
    mode="max-autotune",
    options={
        "epilogue_fusion": True,
        "max_autotune": True,
        "shape_padding": True,  # Tensor core alignment
        "triton.cudagraphs": True,
    }
)
```

CUDA graphs (`reduce-overhead` mode) eliminate Python kernel launch overhead but require static shapes. Recommendation systems with variable batch sizes should either bucket inputs to discrete sizes or use `max-autotune-no-cudagraphs`.

---

## TorchRec embeddings compile partially at best

TorchRec 1.0 added torch.compile compatibility, but significant limitations persist. Sharded embeddings (`ShardedEmbeddingBagCollection`) rely on custom collective operations not fully traced by Dynamo, causing graph breaks or long compilation times. The library's own troubleshooting documentation states that "the sparse arch is difficult to compile."

FBGEMM's `SplitTableBatchedEmbeddingBagsCodegen`—the kernel powering production embedding lookups—operates as an opaque operation within compiled graphs. While registered with PyTorch's dispatcher, Inductor cannot fuse or optimize these operations further. The benefit of keeping embeddings in the compiled region is minimal since FBGEMM kernels are already heavily optimized.

Jagged tensors for variable-length sequences work with the `torch.nested` layout:

```python
# Native nested tensors compile with fullgraph=True
nt = torch.nested.nested_tensor([a, b], layout=torch.jagged)

@torch.compile(fullgraph=True)
def process(x):
    return x.sin() + 1

output = process(nt)  # Single graph handles ragged structure
```

For KeyedJaggedTensor inputs common in TorchRec, mark dynamic dimensions on both values and lengths tensors:

```python
torch._dynamo.mark_dynamic(kjt.values(), 0)
torch._dynamo.mark_dynamic(kjt.lengths(), 0)
```

---

## Production deployment requires caching and monitoring infrastructure

Compilation caching eliminates cold-start penalties across deployments:

```bash
# Persistent cache location
export TORCHINDUCTOR_CACHE_DIR=/persistent/cache/torchinductor
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
```

For containerized deployments, save portable cache artifacts:

```python
# After warm-up, export cache
artifacts = torch.compiler.save_cache_artifacts()
artifact_bytes, cache_info = artifacts
with open("torch_cache.bin", "wb") as f:
    f.write(artifact_bytes)

# On new deployment instances
with open("torch_cache.bin", "rb") as f:
    torch.compiler.load_cache_artifacts(f.read())
```

Critical production configuration settings:

```python
# Recompilation limits prevent runaway compilation
torch.compiler.config.recompile_limit = 8
torch.compiler.config.accumulated_recompile_limit = 256
torch.compiler.config.fail_on_recompile_limit_hit = True  # Alert on limit

# Dynamic shape handling
torch.compiler.config.automatic_dynamic_shapes = True
torch.compiler.config.assume_static_by_default = True
```

Warm-up procedures must complete before serving traffic—**first inference triggers full compilation** taking 10-60+ minutes for large models:

```python
def warm_up_model(model, sample_inputs, warmup_iterations=5):
    model.eval()
    compiled = torch.compile(model, mode="max-autotune")
    with torch.no_grad():
        for i in range(warmup_iterations):
            _ = compiled(*sample_inputs)
            if i == 0:
                print("Initial compilation complete")
    torch.cuda.synchronize()
    return compiled
```

Monitor recompilations and cache effectiveness:

```python
import torch._dynamo

counters = torch._dynamo.utils.counters
metrics.gauge("pytorch.recompiles", sum(counters['recompiles'].values()))
metrics.gauge("pytorch.graph_breaks", sum(counters['graph_break'].values()))
```

---

## Industry deployments validate hybrid compilation approaches

Meta's production recommendation systems at **trillion-parameter scale** rely on TorchRec for distributed embeddings rather than direct torch.compile. The PyTorch Conference 2025 featured Pinterest's talk on "Scaling Inference of O(10K)-length Sequence Recommendation Models" using CUDA graphs and Triton kernels—complementary technologies to torch.compile rather than replacements.

Amazon teams achieved **30-40% latency reduction** on diffusion models using Inductor, with key lessons applicable to RecSys:
- Forward pass must be a pure function—no state mutations
- Use only torch.Tensor types as inputs—lift non-tensor configs to module properties
- Minimize branching complexity by pushing to initialization
- Always test for numerical differences (compiled models are not bitwise exact)

AWS benchmarks on Graviton processors show **2x speedup** for model inference including DLRM benchmarks in TorchBench. The torch.compile performance dashboard reports **43% average speedup** across 163 models with 93% compatibility rate—though recommendation models with heavy sparse operations fall below these averages.

Netflix's ML platform uses a custom "Trace Framework" achieving 20% improvement on target benchmarks through meta-level optimization. Their approach to recommendation systems emphasizes foundation models with shared member preference embeddings—architectures where dense components dominate and torch.compile benefits compound.

---

## Conclusion

Recommendation model compilation effectiveness depends on architectural partitioning rather than wholesale model compilation. The 60-80% of compute spent in embedding lookups benefits from FBGEMM's specialized kernels, not Inductor—attempting to compile these operations yields graph breaks or negligible gains. **Dense MLP components achieve 2-3x speedups** with `max-autotune` mode and H100 FP8 integration.

Production deployments require infrastructure investment in persistent caching, pre-warming procedures (10-60 minutes for initial compilation), and recompilation monitoring. The emerging pattern from Meta, Pinterest, and Netflix points toward domain-specific libraries (TorchRec) handling sparse operations while torch.compile optimizes dense feature interaction layers.

For immediate implementation: apply `@torch.compiler.disable` to embedding lookups, compile interaction and ranking layers with `mode="max-autotune"`, enable FP8 on Linear layers divisible by 16, and implement cache persistence using `torch.compiler.save_cache_artifacts()`. This hybrid approach captures **80% of available speedup** while avoiding the graph break challenges that have stalled full-model compilation attempts.
