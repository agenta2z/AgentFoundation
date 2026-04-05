# Embedding table optimization for Meta-scale recommendation training

Large-scale recommendation model training with **100M-1B+ embedding entries** faces a fundamental memory-compute tradeoff: embedding tables can consume **50-500GB** of GPU memory, yet only a fraction of entries are accessed per batch. Modern solutions combine aggressive compression, intelligent caching, and fused operators to achieve training within tight memory budgets. For MTML ROO architectures with HSTU and **5% activation memory constraints**, the most impactful optimizations are FBGEMM's fused backward-optimizer (eliminating gradient materialization), software-managed UVM caching (enabling **1-5% GPU memory** utilization), and compositional embeddings (achieving **10-1000× compression**) with maintained model quality.

## Memory profile reveals embeddings dominate GPU consumption

For a single embedding table with **100M entries and 128-dimension** in FP32, the parameter footprint alone reaches **51.2GB**—exceeding most GPU HBM capacities. Training memory follows this formula:

**Total Memory = Parameters + Optimizer States + Gradients + Activations**

With Adam optimizer in mixed precision, per-parameter costs break down to **6 bytes** for parameters (FP16 + FP32 master), **8 bytes** for optimizer states, and **4 bytes** for gradients—totaling **18 bytes per parameter**. However, FBGEMM's `SplitTableBatchedEmbeddingBagsCodegen` with fused optimizers eliminates gradient materialization entirely, saving memory equal to parameter size.

For HSTU architectures with **512-1024 sequence lengths**, activation memory per attention layer reaches approximately **B × L × D × 2 bytes**. A configuration with batch 512, sequence 1024, and dimension 256 consumes ~537MB per layer. Within a **5% activation budget** on an 80GB GPU (~4GB), this constrains depth to 4-6 layers without gradient checkpointing.

| Embedding Scale | Dimension | FP32 Memory | FP16 Memory |
|-----------------|-----------|-------------|-------------|
| 100M entries    | 64        | 25.6 GB     | 12.8 GB     |
| 100M entries    | 128       | 51.2 GB     | 25.6 GB     |
| 1B entries      | 64        | 256 GB      | 128 GB      |
| 1B entries      | 128       | 512 GB      | 256 GB      |

The **KeyedJaggedTensor** format efficiently represents variable-length sparse features with flattened values and lengths tensors. Memory formula: `KJT_mem = values_count × dtype_size + lengths_count × offset_dtype_size`. For training bottleneck analysis, embedding lookups are **memory-bound, not compute-bound**, with all-to-all communication becoming co-dominant at scale.

## Compression techniques achieve 10-1000× reduction during training

Multiple compression approaches work during training while maintaining model quality. **Mixed-dimension embeddings (MDE)** assign smaller dimensions to rare features, achieving **2-16× compression** with maintained or improved accuracy due to regularization effects on long-tail features. Implementation requires sorting features by popularity and partitioning into blocks with decreasing dimensions.

**Quotient-Remainder (QR) embeddings** decompose feature ID `i` into quotient `q = i // M` and remainder `r = i % M`, looking up from two smaller tables and combining via element-wise multiplication. This achieves **10-15× compression** for 1B features with negligible accuracy loss. The approach is fully differentiable and requires minimal code changes:

```python
# QR decomposition: e_i = E_q[i // M] ⊙ E_r[i % M]
# Original: O(N × D) → QR: O(2√N × D)
```

**TT-Rec (Tensor Train decomposition)** replaces embedding matrices with sequences of smaller tensor cores, achieving **112× compression on Criteo Terabyte with no accuracy loss**. Meta's FBTT-Embedding library provides drop-in PyTorch replacement with LFU caching for frequently-accessed vectors.

**ROBE (Random Offset Block Embedding)** uses a single shared parameter array with block-based universal hashing, achieving **1000× compression** (100GB → 100MB) while meeting MLPerf target AUC of 0.8025. Block-wise access improves cache performance and reduces hash variance compared to standard feature hashing.

**Quantization-aware training (QAT)** with INT4 provides **8× reduction** while the quantization acts as **strong regularization** that can actually improve accuracy by mitigating DLRM overfitting. FBGEMM supports FP16, INT8, INT4, and INT2 weight precisions.

| Technique | Compression | Quality Impact | Training-Compatible |
|-----------|-------------|----------------|-------------------|
| Mixed-Dimension (MDE) | 2-16× | Maintains/improves | ✓ |
| QR Embeddings | 10-15× | Negligible | ✓ |
| TT-Rec | 100-200× | No loss | ✓ |
| ROBE | 1000× | No loss | ✓ |
| QAT INT4 | 8× | Regularization benefit | ✓ |

## FBGEMM operators enable fused, memory-efficient training

The core FBGEMM operator `SplitTableBatchedEmbeddingBagsCodegen` provides batched embedding lookups with integrated optimizer updates. Key parameters include `EmbeddingLocation` (DEVICE for HBM, MANAGED for UVM, MANAGED_CACHING for UVM with HBM cache) and fused optimizers (`EXACT_ROWWISE_ADAGRAD`, `ADAM`, `LAMB`).

```python
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import EmbeddingLocation

tbe = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=[
        (100_000_000, 128, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA)
    ],
    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
    cache_load_factor=0.2,  # 20% cached in HBM
    prefetch_pipeline=True,
)
```

TorchRec wraps FBGEMM with high-level APIs: **EmbeddingBagCollection** for pooled embeddings and **EmbeddingCollection** for sequence embeddings (critical for HSTU attention). The **FusedEBC** variant combines table batching, fused optimizer, and UVM caching—achieving **>2× performance** over standard PyTorch EmbeddingBag. For distributed training, **DistributedModelParallel** handles sharding automatically via the **EmbeddingShardingPlanner**.

GPU kernel optimizations include table batching (reducing kernel launch overhead), coalesced memory reads (embedding dimensions aligned to 4), and stochastic rounding for FP16 stability. The fused backward-optimizer pattern eliminates gradient tensor allocation entirely—gradients are applied directly during backward propagation.

## Caching strategies enable training beyond GPU memory limits

For embeddings exceeding GPU memory, **UVM caching** places the full table in unified virtual memory with hot rows cached in HBM. FBGEMM's `MANAGED_CACHING` location with `cache_load_factor=0.2` keeps 20% of rows in fast memory. Cache policies include **LRU** (adaptive to changing patterns) and **LFU** (better for stable recommendation access patterns).

**Software-managed caching** outperforms hardware UVM paging. ColossalAI's frequency-aware cache achieves training with **only 1.5-5% of embeddings in GPU memory**—for a 91GB table, only 3.75GB CUDA memory is required. Implementation pre-analyzes dataset for frequency distribution and warms the cache with highest-frequency IDs before training.

**ScratchPipe's "look-forward" cache** knows exactly which embeddings will be accessed in upcoming batches, achieving ~100% cache hit rate. This requires only cache size equal to the working set of the current batch. Performance reaches **2.8× average speedup** (up to 4.2×) versus prior GPU embedding systems.

**Prefetch pipelining** overlaps cache operations with compute: cache-insert for batch_{i+1} executes in parallel with forward/backward of batch_i. Critical implementation details include preventing immature eviction via `lxu_cache_locking_counter` and handling cache invalidation before backward pass for correct gradient writes.

For **5% activation memory budget**, the recommended architecture is:
- **3-4%** for embedding cache (software-managed LFU)
- **1%** for prefetch buffer (3-5 batch lookahead)
- Remaining for forward activations with gradient checkpointing
- Use fused optimizers to eliminate gradient memory entirely

## int32 optimization delivers immediate 50% savings on sparse tensors

Switching from int64 to int32 for embedding indices and offsets provides **50% memory reduction** on sparse tensor components. FBGEMM v1.1.0 explicitly introduced int32 support for TBE training, and TorchRec defaults to `lengths_dtype=torch.int32`.

**Safety thresholds**: int32 is safe when embedding table cardinality is under **2.1 billion** and cumulative offsets (batch_size × num_features × max_sequence_length) remain under int32 max. For production recommendation systems, feature IDs are typically hashed/modulated to fit within table sizes, making int32 universally applicable.

```python
# Optimized KeyedJaggedTensor construction
sparse_features = KeyedJaggedTensor(
    keys=["user_id", "item_id", "category"],
    values=torch.tensor([...], dtype=torch.int32),    # 50% savings
    lengths=torch.tensor([...], dtype=torch.int32),   # Default
    weights=torch.tensor([...], dtype=torch.float16), # 50% savings on scores
)

# FBGEMM TBE configuration
tbe = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs,
    embedding_table_index_type=torch.int32,
    embedding_table_offset_type=torch.int32,
    weights_precision=SparseType.FP16,
)
```

Additional micro-optimizations include **FP16/BF16 weights** for id_score_list features (50% savings on scores), **INT8/INT4 embedding table quantization** (75-87.5% savings), and using `optimizer.zero_grad(set_to_none=True)` to free gradient memory immediately.

## Industry practices converge on hybrid parallelism and real-time updates

**Meta's TorchRec** employs hybrid parallelism: model parallelism for embedding tables (sharded via table-wise, row-wise, or column-wise strategies) and data parallelism for dense MLP layers. The EmbeddingShardingPlanner automatically generates optimal sharding plans based on device topology and memory constraints. Production deployments reach **1.25 trillion parameters** with UVM enabling tables larger than GPU memory.

**ByteDance's Monolith** introduces collisionless embedding tables via Cuckoo hashing with expirable embeddings (removed after inactivity) and frequency filtering (minimum interaction thresholds). Real-time online training achieves sub-minute parameter synchronization through incremental updates of only touched embeddings.

**ByteDance's Persia** scales to **100 trillion parameters** through hybrid synchronous/asynchronous training: embedding layers update asynchronously (memory-intensive, 99.99%+ of parameters) while dense networks update synchronously (compute-intensive). This achieves **3.8× higher throughput** versus fully synchronous mode with **7.12× speedup** over baseline systems.

**Gradient compression** reduces communication overhead dramatically. Deep Gradient Compression (DGC) demonstrates that **99.9% of gradient exchange is redundant**, achieving 270-600× compression without accuracy loss. TorchRec's Qcomm library enables quantized all-to-all and all-reduce at 4 bits without accuracy degradation.

For **MTML ROO with HSTU** specifically, recommended practices include:

- Use FusedEmbeddingBagCollection with `EXACT_ROWWISE_ADAGRAD` for sparse features
- Enable `prefetch_pipeline=True` with software-managed LFU caching
- Apply QR embeddings or TT-Rec for largest tables (user/item embeddings)
- Use mixed-precision (BF16) for HSTU attention with gradient checkpointing
- Implement int32 indices throughout the sparse feature pipeline
- Consider Monolith-style expirable embeddings for dynamic feature spaces

## Conclusion

Optimizing embedding tables at Meta production scale requires a multi-pronged approach. The most impactful interventions are: **(1)** FBGEMM fused backward-optimizer eliminating gradient memory, **(2)** software-managed UVM caching with frequency-aware policies enabling 1-5% GPU memory utilization, **(3)** compositional embeddings (QR, TT-Rec) providing 10-100× compression during training, and **(4)** int32 index optimization for 50% sparse tensor savings. For the **5% activation memory constraint**, combine aggressive caching with prefetch pipelining and gradient checkpointing. The convergence of industry approaches toward hybrid sync/async training, real-time updates, and LLM integration suggests these optimization patterns will remain relevant as recommendation architectures continue evolving toward transformer-based sequential models like HSTU.
