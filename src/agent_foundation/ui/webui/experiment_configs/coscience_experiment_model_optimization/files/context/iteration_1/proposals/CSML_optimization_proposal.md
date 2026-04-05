# Consolidated CSML Optimization Proposals (Q1-Q6)

**Document Version**: 2.0
**Date**: 2026-01-10
**Author**: CSML Optimization Team
**Status**: Consolidated Proposal
**Target Codebase**: `fbs_1ce9d8_jia` (commit `1ce9d80b`)

---

## Executive Summary

This document consolidates all optimization proposals from Q1-Q6 research, filtering out **already implemented** optimizations and merging duplicate items. The proposals are organized by priority based on:
- **Easiness** (implementation complexity)
- **Expected Impact** (QPS/memory improvement)
- **Risk Level** (regression potential)
- **Success Probability** (likelihood of achieving expected gains)

### What's Already Implemented (V2 Codebase)

The following optimizations are **ALREADY IMPLEMENTED** and should NOT be re-implemented:

| Optimization | Commit | Impact | Status |
|--------------|--------|--------|--------|
| NumCandidatesInfo Pattern | `f7b3c5cf` | 2-8% QPS | ✅ Done |
| PT2 Input Preprocessor | `fde87298` | Significant | ✅ Done |
| PT2 HSTU Loss | `1bed91e0` | Kernel fusion (221→30 kernels) | ✅ Done |
| PT2 on SIM | `d03af38d` | Significant | ✅ Done |
| SDD Lite Pipeline | `f220c6d5` | 4-5% QPS | ⚠️ **DISABLED** - Code exists, production config commented out |
| Inplace Data Copy | `f220c6d5` | -3% peak memory | ⚠️ **DISABLED** - Part of SDD, config commented out |
| bf16 on task_arch | `1f2108d0` | Memory/speed | ✅ Done |
| int32 id_score_list | `1ce9d80b` | 50% indices savings | ✅ Done |
| activation_memory_budget = 0.05 | V2 config | Enables recomputation | ✅ Done |
| `gradient_as_bucket_view=True` | TorchRec default | ~4GB savings | ✅ Done |
| `pin_memory=True` | DPP config | 16% faster transfers | ✅ Done |
| `cache_load_factor=0.1` | Prod config | UVM caching | ✅ Done |
| `CacheAlgorithm.LFU` | Prod config | Optimal for rec workloads | ✅ Done |
| `automatic_dynamic_shapes` | PyTorch default | Shape handling | ✅ Done |

---

## Research Background: Q1-Q6 Optimization Studies

This consolidated proposal synthesizes findings from **six independent deep research studies** (Q1-Q6), each focusing on a specific optimization domain for recommendation model training. The individual proposal documents provide deeper technical analysis for each area.

### Research Directions Overview

| Study | Focus Area | Key Techniques | Individual Proposal |
|-------|------------|----------------|---------------------|
| **Q1** | GPU-CPU Synchronization | Eliminate `.item()` calls, NumCandidatesInfo/NumTargetsInfo pattern, DataPrefetcher, `no_sync()` | `proposal_Q1.md` |
| **Q2** | torch.compile Strategies | Hybrid compilation, `mark_dynamic()`, `mode="reduce-overhead"`, compilation caching | `proposal_Q2.md` |
| **Q3** | Activation Memory | FlashAttention/SDPA verification, selective checkpointing, `activation_memory_budget` tuning, memory profiler | `proposal_Q3.md` |
| **Q4** | Kernel Fusion & Launch | TorchInductor config, CUDA graphs, epilogue fusion, kernel count profiling | `proposal_Q4.md` |
| **Q5** | TorchRec Data Pipeline | SDD Lite re-enablement, DPP tuning, buffer pre-allocation, `collate_fn` optimization, `zero_grad` | `proposal_Q5.md` |
| **Q6** | Embedding Optimization | INT8/FP16 precision, TBE `prefetch_pipeline`, fused optimizers, dimension alignment, MDE, CCE | `proposal_Q6.md` |

### Research Methodology

Each Q study followed a consistent methodology:
1. **Deep Research Phase**: Claude + Gemini AI analysis of academic literature and industry best practices
2. **Codebase Investigation**: Verification against actual V2 codebase (`fbs_1ce9d8_jia`, commit `1ce9d80b`)
3. **Proposal Generation**: Prioritized recommendations with assessment criteria (Easiness, Complexity, Risk, Success Probability)

### How to Use This Document

- **For Quick Reference**: Use the priority tables (Tier 1-4) for implementation planning
- **For Deep Dives**: Refer to the Detailed Proposal Specifications section or individual Q1-Q6 documents
- **For Dependencies**: Check the Dependencies Between Proposals table before starting any optimization

---

## Consolidated Priority Table

### Tier 1: Low-Hanging Fruits 🍎 (IMPLEMENT FIRST)

| Priority | ID | Proposal | Source | Easiness | Impact | Risk | Effort | Status |
|----------|-----|----------|--------|----------|--------|------|--------|--------|
| **P0-0** | LHF-SDD | **Re-enable SDD Lite Pipeline** | Q5/Q6 | 5/5 | **4-5% QPS** | Low | 30min | TODO |
| **P0-1** | LHF-COMPILE | Add `mode="reduce-overhead"` to torch.compile | Q1, Q4 | 5/5 | 10-30% | Low | 30m | TODO |
| **P0-2** | LHF-SDPA | Verify FlashAttention backend for SDPA | Q3 | 5/5 | 10-20× attention memory | Low | 0.5d | TODO |
| **P0-3** | LHF-FP16W | FP16/BF16 weights for id_score_list | Q6 | 5/5 | 50% score memory | Low | 0.5d | TODO |
| **P0-4** | LHF-NONBLOCK | Fix `non_blocking=False` in benchmark config | Q1 | 5/5 | 5-10% **benchmark only** | Low | 30m | TODO |
| **P0-5** | LHF-ALIGN | Embedding dimension alignment check (% 4) | Q6 | 5/5 | Optimal FBGEMM perf | Low | 0.5d | TODO |
| **P0-6** | LHF-CACHE-VERIFY | Verify aiplatform compiler_cache launcher integration | Q2 | 5/5 | Enable cache persistence | Low | 2h | TODO |
| **P0-7** | LHF-PREFETCH | Enable `prefetch_pipeline=True` in TBE | Q6 | 4/5 | Better overlap | Low | 1d | TODO |
| **P0-8** | LHF-FUSEDOPT | Verify fused optimizer is enabled | Q6 | 4/5 | Eliminate grad storage | Low | 1d | TODO |
| **P0-9** | LHF-HYBRID | Hybrid Sparse/Dense compile (`@torch.compiler.disable` on sparse) | Q2 | 4/5 | 20-50% dense speedup | Low | 2-3d | TODO |
| **P0-10** | LHF-BUFFER | Audit register_buffer() usage for constant tensors | Q3 | 5/5 | Minor perf + less fragmentation | Low | 2h | TODO |
| **P0-11** | LHF-TARGETS | Extend NumCandidatesInfo to `num_targets` tensor | Q1 | 4/5 | 1-3% | Low | 0.5-1d | TODO |
| **P0-12** | LHF-ZEROG | `zero_grad(set_to_none=True)` | Q5 | 5/5 | Minor speedup | Low | 0.5d | TODO |
| **P0-13** | LHF-DYNAMIC | Add `mark_dynamic()` guards in forward() | Q2 | 4/5 | Stable compilation | Low | 1d | TODO |
| **P0-14** | LHF-SYNC | Sync Debug Mode Audit (diagnostic) | Q1 | 5/5 | Find issues | None | 2h | TODO |

> **Note on LHF-NONBLOCK**: This fix only affects benchmark configurations (`benchmark/main_feed_mtml_config.py`), NOT production training. The impact is limited to benchmark accuracy.

**Total Tier 1 Effort: ~8-10 days**

### Tier 2: High Priority (After Low-Hanging Fruits)

| Priority | ID | Proposal | Source | Easiness | Impact | Risk | Effort | Status |
|----------|-----|----------|--------|----------|--------|------|--------|--------|
| **P1-1** | HP-KERNELPROF | **Kernel count profiling** (DIAGNOSTIC FIRST) | Q4 | 5/5 | Identifies opportunities | None | 2-4h | TODO |
| **P1-2** | HP-MEMPROFILER | **Memory profiling hooks** (DIAGNOSTIC) | Q3 | 4/5 | Enables data-driven decisions | None | 1d | TODO |
| **P1-3** | HP-CHECKPOINT | Enable `activation_checkpointing=True` in DeepCrossNet | Q3 | 3/5 | 40-60% activation memory | Low | 2-3d | TODO |

> **Note on HP-CHECKPOINT**: Current codebase has mixed settings:
> - `model_roo_v0.py:373,390`: `activation_checkpointing=False` (explicit False)
> - `model_configs.py:441`: `dcn_activation_checkpointing: bool = False` (default)
> - `flattened_*_datafm.py:2556`: `enable_activation_checkpointing=True` (some configs)
>
> **Action**: Audit which configs need the change; some may already have it enabled.

| **P1-4** | HP-BUFFERREUSE | Enable `allow_buffer_reuse=True` in Inductor | Q3 | 3/5 | 5-15% memory | Medium | 1d | TODO |
| **P1-5** | HP-BUDGET | Tune `activation_memory_budget` (0.05 vs 0.1 vs 0.3) | Q3 | 5/5 | 5-20% memory/throughput tradeoff | Low | 1d | TODO |

> **Note on HP-BUFFERREUSE**: Easiness downgraded to 3/5 because `allow_buffer_reuse=False` was explicitly set at `model_roo_v0.py:75` for an unknown reason. **Must investigate why it was disabled before enabling.** Possible correctness issue.

| **P1-6** | HP-INDUCTOR | TorchInductor config optimization (`max_autotune`, etc.) | Q4 | 5/5 | 5-10% marginal | Low | 2-4h | TODO |
| **P1-7** | HP-EPILOGUE | GEMM epilogue fusion verification (SwishLayerNorm) | Q4 | 4/5 | 2-5% | Low | 0.5d | TODO |
| **P1-8** | HP-PREALLOCATE | Buffer pre-allocation warmup | Q5 | 4/5 | Prevent OOM | Low | 1d | TODO |
| **P1-9** | HP-DPP | DPP `prefetch_queue_size` & `client_thread_count` tuning | Q5 | 4/5 | Data loading | Low | 1d | TODO |

> **Note**: HP-KERNELPROF and HP-MEMPROFILER are diagnostics that should run **before** other Tier 2 optimizations to inform which actions to prioritize.

**Total Tier 2 Effort: ~10-12 days**

### Tier 3: Medium Priority (Week 3-4)

| Priority | ID | Proposal | Source | Easiness | Impact | Risk | Effort | Status |
|----------|-----|----------|--------|----------|--------|------|--------|--------|
| **P2-1** | MP-LOSSBATCH | Further loss batching (30→15 kernels) | Q2, Q4 | 3/5 | 5-10% loss speedup | Medium | 3-4d | TODO |
| **P2-2** | MP-NOSYNC | DDP `no_sync()` for gradient accumulation | Q1 | 3/5 | 25-100% (if applicable) | Low | 2-3d | TODO |

> **Prerequisite for MP-NOSYNC**: This optimization ONLY applies if gradient accumulation is used. Verify gradient accumulation is enabled in training config before implementing. If no gradient accumulation, `no_sync()` provides **zero benefit**.

| **P2-3** | MP-GRAPHBREAK | Graph Break Elimination Audit | Q2 | 2/5 | Enable fullgraph | Low | 3-5d | TODO |
| **P2-4** | MP-INT8 | INT8 Embedding Quantization (QAT) | Q6 | 3/5 | 4× reduction | Medium | 5-7d | TODO |
| **P2-5** | MP-GRADHOOK | Post-accumulate gradient hooks for dense | Q5 | 3/5 | ~20% peak memory | Medium | 3-4d | TODO |
| **P2-6** | MP-FUSEDSDD | Fused SDD for optimizer overlap | Q5 | 3/5 | 5-10% QPS | Medium | 3-5d | TODO |
| **P2-7** | MP-COLLATE | Custom collate_fn Optimization | Q5 | 3/5 | 10-15% data loading | Medium | 3-4d | TODO |
| **P2-8** | MP-CACHEWARM | Software-Managed LFU Cache Warming | Q6 | 2/5 | Train with 1.5-5% GPU | Low | 7-10d | TODO |
| **P2-9** | MP-CCE | Clustered Compositional Embeddings | Q6 | 2/5 | 10-50× compression | Medium | 10-15d | TODO |
| **P2-10** | MP-SSDOFF | SSD Offloading (FlashNeuron/Legend) | Q6 | 1/5 | Enable multi-TB models | High | 20+d | TODO |

**Total Tier 3 Effort: ~60-80 days**

### Tier 4: Lower Priority / Research (Requires Significant Investment)

| Priority | ID | Proposal | Source | Easiness | Impact | Risk | Effort | Status |
|----------|-----|----------|--------|----------|--------|------|--------|--------|
| **P4-1** | FP-TMA | H100 TMA Optimization | Q4 | 1/5 | 10-15% memory BW | High | 2-3w | TODO |
| **P4-2** | FP-ZEROBUBBLE | Zero Bubble Pipeline Parallelism | Q1 | 1/5 | Near-theoretical PP throughput | High | 2-4w | TODO |
| **P4-3** | FP-MONOLITH | Monolith-Style Expirable Embeddings | Q6 | 1/5 | Dynamic table sizing | High | 15+d | TODO |
| **P4-4** | FP-ROBE | ROBE (Random Offset Block Embedding) | Q6 | 1/5 | 1000× compression | High | 20+d | TODO |

> **Note on Tier 4**: These proposals require significant investment (1/5 Easiness, High Risk) and should only be pursued after Tiers 1-3 are exhausted and profiling indicates clear bottlenecks they would address.

**Total Tier 4 Effort: 60+ days**

### Recommended Week 1-2 Schedule (Tier 1)

| Time | Proposal | Expected Outcome |
|------|----------|------------------|
| **Day 1 AM** | **P0-0: LHF-SDD** - Re-enable SDD Lite Pipeline | **4-5% QPS** (highest impact, 30min) |
| **Day 1 AM** | **P0-1: LHF-COMPILE** - Add `mode="reduce-overhead"` | 10-30% potential |
| **Day 1 PM** | **P0-2: LHF-SDPA** - Verify FlashAttention backend | 10-20× attention memory |
| **Day 1 PM** | **P0-3: LHF-FP16W** - FP16 weights for scores | 50% score memory |
| **Day 2 AM** | **P0-4: LHF-NONBLOCK** - Fix benchmark config | 5-10% benchmark |
| **Day 2 AM** | **P0-5: LHF-ALIGN** - Embedding dimension check | Optimal FBGEMM perf |
| **Day 2 PM** | **P0-6: LHF-CACHE-VERIFY** - Verify compiler cache | Enable persistence |
| **Day 3** | **P0-7: LHF-PREFETCH** - TBE prefetch pipeline | Better overlap |
| **Day 4** | **P0-8: LHF-FUSEDOPT** - Verify fused optimizer | Eliminate gradient storage |
| **Day 5-7** | **P0-9: LHF-HYBRID** - Sparse/Dense compilation | 20-50% dense speedup |
| **Day 8** | **P0-10: LHF-BUFFER** - Register buffer audit | Reduced fragmentation |
| **Day 8** | **P0-11: LHF-TARGETS** - NumTargetsInfo pattern | 1-3% QPS |
| **Day 9** | **P0-12: LHF-ZEROG** - `set_to_none=True` | Minor speedup |
| **Day 9** | **P0-13: LHF-DYNAMIC** - `mark_dynamic()` guards | Prevent recompilation |
| **Day 10** | **P0-14: LHF-SYNC** - Sync debug audit (diagnostic) | List remaining sync points |

### Week 2-3: High Priority (~10-12 days)

| Days | Tasks | Expected Outcome |
|------|-------|------------------|
| Day 11 | **HP-KERNELPROF** (P1-1, DIAGNOSTIC) | Identify kernel count & opportunities |
| Day 12 | **HP-MEMPROFILER** (P1-2, DIAGNOSTIC) | Enable data-driven memory decisions |
| Day 13-15 | HP-CHECKPOINT | 40-60% activation memory |
| Day 16 | HP-BUFFERREUSE (investigate first!) | 5-15% memory |
| Day 17 | HP-BUDGET tuning | Optimal memory/throughput tradeoff |
| Day 18 | HP-INDUCTOR | Config optimization |
| Day 19 | HP-EPILOGUE | 2-5% from fusion |
| Day 20-21 | HP-PREALLOCATE | Prevent OOM |
| Day 22 | HP-DPP | Data loading optimization |

### Week 4+: Medium & Lower Priority

Continue with Tier 3 and Tier 4 proposals based on profiling results.

---

## Expected Combined Impact

> **⚠️ IMPORTANT IMPACT DISCLAIMER**: V2 codebase already has extensive optimizations (~65-75% QPS over V1 baseline). Many "10-30%" gains cited in literature are for unoptimized baselines. **Realistic marginal gains** for V2 may be significantly lower. Impact estimates below include both **optimistic** (theoretical maximum) and **realistic** estimates.

| Phase | Optimistic Estimate | **Realistic Estimate** |
|-------|---------------------|------------------------|
| **Baseline (V2)** | ~65-75% QPS over V1 | ~65-75% QPS over V1 |
| **After Tier 1** | +15-35% QPS | **+10-20% QPS** |
| **After Tier 2** | +10-20% additional | **+5-15% additional** |
| **After Tier 3** | +10-15% additional | **+5-10% additional** |
| **After Tier 4** | +10-20% additional (variable) | **+5-15% additional (variable)** |
| **Total Potential** | ~50-100% over V2 | **~25-50% over V2** |

**⚠️ CRITICAL CAVEATS**:
1. **Gains are NOT additive** - many optimizations address overlapping bottlenecks
2. Actual impact depends on current bottleneck distribution (profile first!)
3. Hardware-specific (H100 vs B200), batch size, and sequence length affect results
4. Distributed training configuration affects which optimizations apply
5. The 50-100% "optimistic" range is a theoretical maximum, not a realistic target
6. **Recommend targeting 25-35% improvement** as a realistic goal for Tier 1+2

---

## Appendix

### Appendix A: Files to Modify

#### Tier 1 Files
| File | Changes |
|------|---------|
| `flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py` | **LHF-SDD**: Uncomment `train_pipeline_provider_fqn` (line 1124) |
| `pytorch_modules.py` | NumTargetsInfo pattern (lines 4935-4936) |
| `benchmark/main_feed_mtml_config.py` | `non_blocking=True` (line 137) |
| `model_roo_v0.py` | `mark_dynamic()` in forward (line 1136) |
| `pytorch_modules.py` | `@torch.compiler.disable` on SparseArch |
| Training config | `mode="reduce-overhead"` |
| KJT construction | FP16 weights for scores |

#### Tier 2 Files
| File | Changes |
|------|---------|
| `model_roo_v0.py` | `activation_checkpointing=True` (lines 373, 390) |
| `model_roo_v0.py` | `allow_buffer_reuse=True` (line 75) |
| `mast_roo_trainer_config.py` | DPP tuning |
| New file | Memory profiler utility |

---

### Appendix B: Verification Status

The following claims from source proposals require verification against the actual codebase:

| Claim | Verified? | Evidence / Notes |
|-------|-----------|------------------|
| NumCandidatesInfo (19+ locations) | ⬜ Pending | Need grep verification in `pytorch_modules_roo.py` |
| SDD Lite Pipeline enabled | ❌ **DISABLED** | Production config has `train_pipeline_provider_fqn` commented out (`flattened_*_prod_*.py:1124`) |
| Inplace Data Copy enabled | ❌ **DISABLED** | Part of SDD Lite, also disabled |
| `gradient_as_bucket_view=True` | ✅ Verified | TorchRec `DefaultDataParallelWrapper` (line 128) |
| `cache_load_factor=0.1` | ✅ Verified | Prod config line 1119, May Cargo config line 1301 |
| `aiplatform/compiler_cache` integrated | ⬜ Pending | Code exists, need to verify launcher integration |
| `allow_buffer_reuse=False` reason | ⬜ Pending | Line 75 in model_roo_v0.py - investigate why disabled |
| `enable_torch_compile` flag usage | ⬜ Pending | Need to trace across configs |
| FlashAttention dispatch for SDPA | ⬜ Pending | SDPA used at `pytorch_modules_roo.py:746`, backend not verified |
| HSTU uses Hammer Triton kernels | ✅ Verified | `hammer/v2/ops/hstu_attention_template.py` |
| `epilogue_fusion=False` in inference | ✅ Verified | `online_trainer_roo_config.py:127`, `local_roo_trainer_config.py:158` |

**Critical Finding**: SDD Lite Pipeline code exists (commit `f220c6d5`) but is **not enabled** in production. The pipeline provider is commented out:
```python
# File: flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py:1124
# train_pipeline_provider_fqn="...MainFeedMTMLROOSDDTrainPipelineProvider",
#                            ↑ COMMENTED OUT
```

---

### Appendix C: Dependencies Between Proposals

The following proposals have dependencies that affect implementation order:

| Proposal | Depends On | Rationale |
|----------|------------|-----------|
| LHF-COMPILE | Verify `enable_torch_compile` flag status | Must confirm torch.compile is actually enabled before adding mode parameter |
| HP-BUFFERREUSE | Test in isolation | May have been disabled for correctness - validate before enabling |
| FP-CUDAGRAPH | LHF-COMPILE | `mode="reduce-overhead"` is prerequisite for automatic CUDA graphs |
| MP-INT8 | LHF-FP16W | Lower precision cascade - validate FP16 first |
| FP-FLEXATTN | HP-KERNELPROF | Need to understand HSTU's Hammer Triton kernels first |
| MP-LOSSBATCH | HP-KERNELPROF | Identify which loss kernels remain before further batching |
| FP-REGIONAL | LHF-HYBRID | Understand compilation boundaries before regional compilation |
| MP-MDE | Feature frequency data | Requires offline feature frequency analysis infrastructure |
| FP-CACHEWARM | Feature frequency data | Same infrastructure as MDE |

---

*Document generated: 2026-01-10*
*Consolidated from: Q1, Q2, Q3, Q4, Q5, Q6 proposals*
*Target codebase: fbs_1ce9d8_jia (commit 1ce9d80b)*

---

## Detailed Proposal Specifications

This section provides comprehensive implementation details for each proposal, extracted and consolidated from the Q1-Q6 source documents. Each proposal includes technical analysis, code examples, file locations, and assessment criteria.

---

### Tier 1: Low-Hanging Fruits - Detailed Specifications

---

#### P0-0: LHF-SDD - Re-enable SDD Lite Pipeline

**Source**: Q5, Q6

**Technical Analysis**:

The SDD (Sparse Data Distribution) Lite Pipeline was implemented in commit `f220c6d5` but is currently **disabled** in production configs. The code exists but the pipeline provider is commented out:

```python
# File: flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py:1124
# train_pipeline_provider_fqn="minimal_viable_ai.models.main_feed_mtml.train_pipeline_roo.MainFeedMTMLROOSDDTrainPipelineProvider",
#                            ↑ COMMENTED OUT
```

**Why SDD Lite matters**:
- SDD Lite provides 2-stage overlap (memcpy + dist + compute)
- Memory overhead: ~1.01× batch (vs ~3× for full SDD)
- QPS gain: +4-5%

**Implementation**:

```python
# File: flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py
# UNCOMMENT this line:
train_pipeline_provider_fqn="minimal_viable_ai.models.main_feed_mtml.train_pipeline_roo.MainFeedMTMLROOSDDTrainPipelineProvider",
```

**Files to Modify**:
- `minimal_viable_ai/models/main_feed_mtml/conf/flattened_main_feed_mtml_model_keeper_prod_roo_hstu_datafm.py`
- Any other production configs with commented-out SDD pipeline

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Uncomment one line |
| Complexity | 1/5 | Config change only |
| Success Probability | 90% | Was working before, just disabled |
| Risk Level | **Low-Medium** | **Verify why it was disabled before re-enabling** |

**⚠️ Investigation Note**: SDD Lite was deliberately disabled in production. Before re-enabling:
1. Check commit history for `f220c6d5` to understand why it was commented out
2. Verify no known issues/bugs caused the disablement
3. Test on a small-scale experiment first

**Expected Impact**: 4-5% QPS gain
**Effort Estimate**: 30 minutes

---

#### P0-1: LHF-COMPILE - Add `mode="reduce-overhead"` to torch.compile

**Source**: Q1, Q4

**⚠️ PREREQUISITE VERIFICATION**: Before adding `mode="reduce-overhead"`, verify torch.compile is actually enabled:
```python
# Check if torch.compile is enabled in current config
# Search for: enable_torch_compile=True in training configs
# If NOT enabled, this optimization has no effect
```

**Technical Analysis**:

From Claude Q4 research: "H100 CUDA graphs via `reduce-overhead` mode provides 1.3x-3x speedup for small batch workloads." The `mode="reduce-overhead"` auto-enables CUDA graphs which capture and replay kernel sequences with a single CPU-side launch.

**Current State**:
- V2 uses `enable_torch_compile=True` flag
- But does NOT use `mode="reduce-overhead"` which auto-enables CUDA graphs
- This is a simple config change with potentially large impact

**Implementation**:

```python
# Option 1: In training config (preferred)
# Add to PT2Config or compilation settings:
torch.compile(model, mode="reduce-overhead")

# Option 2: In model initialization
# File: model_roo_v0.py or training framework
compiled_model = torch.compile(
    model,
    mode="reduce-overhead",  # Auto-enables CUDA graphs
    # Optionally add:
    # fullgraph=True,  # If no graph breaks
)
```

**Files to Modify**:
- Training config / PT2Config
- `minimal_viable_ai/models/main_feed_mtml/model_roo_v0.py` (if module-level compile needed)

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Single config parameter |
| Complexity | 1/5 | Standard PyTorch option |
| Success Probability | 85% | 10-30% for compute-bound workloads |
| Risk Level | Low | Can be toggled off |

**Expected Impact**: 10-30% QPS (for compute-bound portions)
**Effort Estimate**: 30 minutes
**Dependencies**: Verify `enable_torch_compile` flag is actually enabled

---

#### P0-2: LHF-SDPA - Verify FlashAttention Backend for SDPA

**Source**: Q3

**Technical Analysis**:

PyTorch's `scaled_dot_product_attention` (SDPA) automatically dispatches to FlashAttention-2 on Ampere+ GPUs with fp16/bf16. This eliminates the O(N²) attention matrix materialization, replacing it with O(N) tiled computation.

**Current State**:
- SDPA is used in `pytorch_modules_roo.py:746`
- BUT: HSTU uses **Hammer Triton kernels**, not SDPA
- FlashAttention verification is relevant for **cross-attention modules** (non-HSTU)

**Implementation**:

```python
# Verification code - add to debug/profiling script
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

print(f"FlashAttention enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Memory-efficient enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"Math backend enabled: {torch.backends.cuda.math_sdp_enabled()}")

# Force FlashAttention backend when supported
with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
    attention_output = F.scaled_dot_product_attention(
        query, key, value,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=False
    )
```

**Files to Modify**:
- `torchrec/fb/ml_foundation/modules/res_gating.py` - `MultiHeadTalkingAttentionNetworkROOV2`
- `minimal_viable_ai/models/main_feed_mtml/pytorch_modules.py` - Custom attention layers

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Verification + potential one-line change |
| Complexity | 1/5 | Drop-in replacement |
| Success Probability | 80% | Requires verification of actual backend dispatch |
| Risk Level | Low | Standard PyTorch API |

**Expected Impact**: 10-20× attention memory reduction if FlashAttention not already active
**Effort Estimate**: 0.5 day

---

#### P0-3: LHF-FP16W - FP16/BF16 Weights for id_score_list Features

**Source**: Q6

**Technical Analysis**:

The `id_score_list` features contain weight/score values that represent feature importance. Currently stored in FP32, but FP16/BF16 is sufficient:

1. Scores are typically normalized values (0-1 range)
2. FP16 has sufficient precision (±65,504 range, 3-4 decimal precision)
3. Already using bf16 on task_arch (commit `1f2108d0`)

**Implementation**:

```python
# In KeyedJaggedTensor construction (pytorch_modules_roo.py)
# BEFORE (implicit FP32):
sparse_features = KeyedJaggedTensor(
    keys=["feature_with_scores"],
    values=torch.tensor([...], dtype=torch.int32),     # Already optimized
    lengths=torch.tensor([...], dtype=torch.int32),    # Already optimized
    weights=torch.tensor([...], dtype=torch.float32),  # CURRENT: FP32
)

# AFTER (50% savings):
sparse_features = KeyedJaggedTensor(
    keys=["feature_with_scores"],
    values=torch.tensor([...], dtype=torch.int32),
    lengths=torch.tensor([...], dtype=torch.int32),
    weights=torch.tensor([...], dtype=torch.float16),  # NEW: 50% savings
    # OR: dtype=torch.bfloat16 for better numerical stability
)
```

**Files to Modify**:
- `pytorch_modules_roo.py`: Sparse feature processing
- Data preprocessing pipeline (if applicable)

**Validation**:
```python
def validate_fp16_scores(fp32_kjt, fp16_kjt):
    with torch.no_grad():
        out_fp32 = model(fp32_kjt)
        out_fp16 = model(fp16_kjt)
        relative_error = torch.abs(out_fp32 - out_fp16) / (torch.abs(out_fp32) + 1e-8)
        assert relative_error.max() < 0.01, f"Error too high: {relative_error.max()}"
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | dtype change only |
| Complexity | 1/5 | No logic changes |
| Success Probability | 95% | Precision sufficient for scores |
| Risk Level | Low | Same pattern as bf16 task_arch |

**Expected Impact**: 50% memory savings on score tensors
**Effort Estimate**: 0.5 day

---

#### P0-4: LHF-NONBLOCK - Fix `non_blocking=False` in Benchmark Config

**Source**: Q1

**Technical Analysis**:

All CPU→GPU transfers should use `non_blocking=True` when used with pinned memory. The benchmark config has this set incorrectly:

```python
# File: benchmark/main_feed_mtml_config.py:137
# CURRENT (ANTI-PATTERN):
train_batch.to(device=self.device, non_blocking=False)
```

**⚠️ NOTE**: This only affects **benchmark** configurations, NOT production training.

**Implementation**:

```python
# File: benchmark/main_feed_mtml_config.py:137
# FIXED:
train_batch.to(device=self.device, non_blocking=True)
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Single boolean change |
| Complexity | 1/5 | Standard PyTorch option |
| Success Probability | High | Documented 5-10% gains |
| Risk Level | Very Low | Well-tested option |

**Expected Impact**: 5-10% **benchmark** improvement (NOT production)
**Effort Estimate**: 30 minutes

---

#### P0-5: LHF-ALIGN - Embedding Dimension Alignment Check

**Source**: Q6

**Technical Analysis**:

FBGEMM's TBE kernels achieve optimal performance when embedding dimensions are aligned to 4 bytes (divisible by 4 for FP32, 8 for FP16). Unaligned dimensions cause suboptimal memory access patterns.

**Implementation**:

```python
def verify_embedding_alignment(model):
    """Verify all embedding dimensions are aligned for FBGEMM."""
    issues = []

    for name, module in model.named_modules():
        if hasattr(module, 'embedding_dim'):
            dim = module.embedding_dim
            if dim % 4 != 0:
                issues.append(f"{name}: dim={dim} not aligned to 4")

    return issues

# Usage
issues = verify_embedding_alignment(model)
if issues:
    print("WARNING: Unaligned embedding dimensions:")
    for issue in issues:
        print(f"  - {issue}")
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Audit + potential config change |
| Complexity | 1/5 | Dimension verification |
| Success Probability | 95% | Standard FBGEMM optimization |
| Risk Level | Low | No algorithmic changes |

**Expected Impact**: Optimal FBGEMM performance
**Effort Estimate**: 0.5 day

---

#### P0-6: LHF-CACHE-VERIFY - Verify aiplatform compiler_cache Launcher Integration

**Source**: Q2

**Technical Analysis**:

Cache artifact management is **already implemented** at the AI platform level in `aiplatform/compiler_cache/cache.py`:

```python
from torch.compiler import load_cache_artifacts, save_cache_artifacts

def save_pt2_cache_artifacts_impl(...):
    artifacts = save_cache_artifacts()
    # ... saves to Manifold

def load_pt2_cache_artifacts(...):
    cache_info_opt = load_cache_artifacts(decompressed_artifacts)
    # ... loads from Manifold
```

**Verification Steps**:
1. Check if `aiplatform.compiler_cache` is integrated into the training launcher
2. Verify Manifold bucket access and permissions
3. This is an **infrastructure configuration issue**, not a code change

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Verification only |
| Complexity | 1/5 | Infrastructure check |
| Success Probability | 90% | Already exists at platform level |
| Risk Level | None | Verification only |

**Expected Impact**: Enable cache persistence across deployments
**Effort Estimate**: 2 hours

---

#### P0-7: LHF-PREFETCH - Enable `prefetch_pipeline=True` in TBE

**Source**: Q6

**Technical Analysis**:

FBGEMM's TBE supports prefetch pipelining which overlaps cache operations with compute:

```
Without prefetch_pipeline:
Batch N:   [Cache Miss Check] → [Fetch from UVM] → [Forward] → [Backward]
Batch N+1:                                                     [Cache Miss Check] → ...

With prefetch_pipeline=True:
Batch N:   [Cache Miss Check] → [Fetch from UVM] → [Forward] → [Backward]
Batch N+1:                      [Cache Insert ←─────────────────] [Forward] → ...
                                 ↑ Overlapped with batch N compute
```

**Current State** (verified):
```python
# File: minimal_viable_ai/core/model_family_api/configs.py:513
prefetch_pipeline: bool = False  # ❌ Default is OFF
```

**Implementation**:

```python
# In TBE/TorchRec configuration
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)

tbe = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=embedding_specs,
    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
    prefetch_pipeline=True,  # ENABLE THIS
)
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Config parameter |
| Complexity | 2/5 | May need pipeline compatibility check |
| Success Probability | 85% | Standard FBGEMM feature |
| Risk Level | Low | Well-documented option |

**Expected Impact**: Better cache overlap, reduced effective latency
**Effort Estimate**: 1 day

---

#### P0-8: LHF-FUSEDOPT - Verify Fused Optimizer is Enabled

**Source**: Q6

**Technical Analysis**:

FBGEMM's fused backward-optimizer pattern eliminates gradient materialization:

```
Standard backward-then-optimizer:
Backward: Compute gradients → Store all gradients in memory
Memory:   [grad_emb_1][grad_emb_2]...[grad_emb_N]  ← All stored!
Optimizer: Read gradients → Update weights → Free gradients

Fused backward-optimizer (FBGEMM):
Backward: Compute grad_emb_i → IMMEDIATELY apply optimizer → Free
Memory:   Never stores full embedding gradients!
```

**Memory savings**: Equal to entire embedding table parameter size (e.g., 51.2GB for 100M × 128D × 4 bytes)

**Audit Script**:

```python
def audit_fused_optimizer(model):
    """Check if model uses fused optimizers for embeddings."""
    issues = []

    for name, module in model.named_modules():
        # Check for non-fused EmbeddingBag
        if isinstance(module, nn.EmbeddingBag):
            issues.append(f"Non-fused EmbeddingBag found: {name}")

        # Verify TBE configuration
        if hasattr(module, '_embedding_bags'):
            for tbe_name, tbe in module._embedding_bags.items():
                if hasattr(tbe, 'optimizer'):
                    if tbe.optimizer == OptimType.SGD:
                        issues.append(f"Non-fused SGD in {tbe_name}")

    return issues
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Audit and verify |
| Complexity | 2/5 | May need migration if not fused |
| Success Probability | 90% | Numerically equivalent |
| Risk Level | Low | Standard pattern |

**Expected Impact**: Eliminate gradient storage for all embeddings
**Effort Estimate**: 1 day

---

#### P0-9: LHF-HYBRID - Hybrid Sparse/Dense Compilation Strategy

**Source**: Q2

**Technical Analysis**:

From Claude research: "Direct torch.compile on full DLRM-style models yields essentially zero difference due to graph breaks from sparse operations."

The optimal approach:
1. **Sparse Architecture** (60-80% of compute): Already optimized via FBGEMM - attempting to compile yields graph breaks
2. **Dense Architecture** (20-40% of compute): MLP interaction layers benefit significantly from Inductor fusion

**Implementation**:

```python
# In pytorch_modules.py - Explicitly disable compilation on sparse lookups
class SparseArch(nn.Module):
    @torch.compiler.disable  # Don't compile sparse lookups - FBGEMM already optimized
    def forward(
        self,
        id_list_features: KeyedJaggedTensor,
        id_score_list_features: KeyedJaggedTensor,
    ) -> Tuple[KeyedTensor, KeyedTensor]:
        with record_function(f"## {self.__class__.__name__}: forward ##"):
            with record_function("## ebc: forward ##"):
                embeddings = self.ebc(id_list_features)
            return embeddings

# In model_roo_v0.py - Compile dense interaction layers with max-autotune
class MainFeedMTMLROO(nn.Module):
    def __init__(self, ...):
        ...
        # Compile dense interaction architecture
        self.interaction_arch = torch.compile(
            InteractionArch(...),
            mode="max-autotune",
            options={"triton.cudagraphs": False}  # Avoid shape issues initially
        )

        # Compile task architecture (dense MLP)
        self.task_arch = torch.compile(
            TaskArch(...),
            mode="max-autotune"
        )
```

**Files to Modify**:
- `minimal_viable_ai/models/main_feed_mtml/model_roo_v0.py`
- `minimal_viable_ai/models/main_feed_mtml/pytorch_modules.py` (SparseArch at line ~3747)

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Decorator-based approach |
| Complexity | 2/5 | Clear sparse/dense separation exists |
| Success Probability | 80% | Validated by Meta, Pinterest |
| Risk Level | Low | Conservative disable on sparse |

**Expected Impact**: 20-50% dense layer speedup
**Effort Estimate**: 2-3 days

---

#### P0-10: LHF-BUFFER - Audit register_buffer() Usage for Constant Tensors

**Source**: Q3

**Technical Analysis**:

Creating tensors inside `forward()` causes repeated GPU allocations, potential CUDA sync points, and memory fragmentation. Using `register_buffer()` creates tensors once in `__init__()` and reuses them.

**Pattern to Fix**:

```python
# BEFORE (in forward - bad):
def forward(self, x):
    mask = torch.ones(x.size(0), device=x.device)  # Created every call!
    return x * mask

# AFTER (with register_buffer - good):
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('ones_template', torch.tensor(1.0))

    def forward(self, x):
        mask = self.ones_template.expand(x.size(0))  # Reuses buffer
        return x * mask
```

**Search Pattern**:
```bash
grep -rn "torch.tensor.*device" minimal_viable_ai/models/main_feed_mtml/
grep -rn "torch.ones.*device" minimal_viable_ai/models/main_feed_mtml/
grep -rn "torch.zeros.*device" minimal_viable_ai/models/main_feed_mtml/
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Simple refactoring |
| Complexity | 1/5 | No algorithmic changes |
| Success Probability | 98% | Standard PyTorch best practice |
| Risk Level | Low | Well-tested pattern |

**Expected Impact**: Minor QPS improvement, reduced memory fragmentation
**Effort Estimate**: 2 hours

---

#### P0-11: LHF-TARGETS - Extend NumCandidatesInfo to `num_targets` Tensor

**Source**: Q1

**Technical Analysis**:

While `num_candidates` is already optimized via NumCandidatesInfo, the `num_targets` tensor still has `.item()` calls:

```python
# File: pytorch_modules.py:4935-4936
# CURRENT (ANTI-PATTERN - syncs every forward pass!):
max_num_targets = num_targets.max().item()  # GPU→CPU sync #1
num_candidates_sum = num_targets.sum().item()  # GPU→CPU sync #2
```

**Implementation**:

```python
# Add NumTargetsInfo following the NumCandidatesInfo pattern
class NumTargetsInfo(NamedTuple):
    num_targets: torch.Tensor
    num_targets_cpu: torch.Tensor  # Single sync at creation
    num_targets_sum: int           # Precomputed
    num_targets_max: int           # Precomputed

def build_num_targets_info(num_targets: torch.Tensor) -> NumTargetsInfo:
    num_targets_cpu = num_targets.cpu()  # Single GPU→CPU transfer
    return NumTargetsInfo(
        num_targets=num_targets,
        num_targets_cpu=num_targets_cpu,
        num_targets_sum=int(num_targets_cpu.sum()),
        num_targets_max=int(num_targets_cpu.max()),
    )
```

**Files to Modify**:
1. Add `NumTargetsInfo` to `num_candidates_info.py` (or new file)
2. Update `pytorch_modules.py` lines 4925-4950

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Pattern already exists |
| Complexity | 2/5 | Need to trace and propagate |
| Success Probability | High | Directly eliminates known sync points |
| Risk Level | Low | Same numerical result |

**Expected Impact**: 1-3% QPS (only 1 hot path location)
**Effort Estimate**: 0.5-1 day

---

#### P0-12: LHF-ZEROG - `zero_grad(set_to_none=True)`

**Source**: Q5

**Technical Analysis**:

Standard `zero_grad()` writes zeros to gradient memory, requiring memory allocation and write operations. `set_to_none=True` uses Python assignment instead:

1. Sets `.grad = None` (no memory operation)
2. Gradients allocated lazily on next backward
3. Faster and avoids unnecessary memory operations

**Implementation**:

```python
# In training loop or optimizer wrapper
optimizer.zero_grad(set_to_none=True)
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Single parameter change |
| Complexity | 1/5 | No code restructuring |
| Success Probability | 95% | Standard PyTorch API |
| Risk Level | Low | Well-documented option |

**Expected Impact**: Minor speedup, cleaner memory management
**Effort Estimate**: 0.5 day

---

#### P0-13: LHF-DYNAMIC - Add `mark_dynamic()` Guards in forward()

**Source**: Q2

**Technical Analysis**:

Variable batch sizes trigger guard failures and recompilation. The `mark_dynamic()` API prevents shape specialization:

```python
# From Gemini research:
# "torch._dynamo.mark_dynamic(tensor, dim) explicitly tells Dynamo not to specialize
# on a specific dimension, forcing generation of a dynamic kernel."
```

**Implementation**:

```python
# In MainFeedMTMLROOTrain.forward() (line 1136 of model_roo_v0.py)
def forward(
    self, train_input: MainFeedMTMLROOTrainBatch
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Add dynamic shape guards at start of forward
    torch._dynamo.mark_dynamic(train_input.ro_float_features, 0)  # Batch dim
    torch._dynamo.mark_dynamic(train_input.nro_float_features, 0)  # Batch dim
    torch._dynamo.mark_dynamic(train_input.num_candidates, 0)      # Variable length

    # Existing forward logic follows...
```

**File**: `model_roo_v0.py` (MainFeedMTMLROOTrain.forward, line 1136)

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Add mark_dynamic calls |
| Complexity | 2/5 | Identify key tensors |
| Success Probability | 90% | Proven pattern from vLLM |
| Risk Level | Low | Non-invasive |

**Expected Impact**: Prevent recompilation thrashing, stable compilation
**Effort Estimate**: 1 day

---

#### P0-14: LHF-SYNC - Sync Debug Mode Audit (Diagnostic)

**Source**: Q1

**Technical Analysis**:

Enable PyTorch sync debug mode to detect ALL hidden GPU→CPU synchronizations:

```python
# Add to training script temporarily
torch.cuda.set_sync_debug_mode(1)  # Warns on implicit sync
# or
torch.cuda.set_sync_debug_mode(2)  # Errors on implicit sync
```

**Remaining `.item()` calls found in V2**:
```
pytorch_modules.py:4935-4936  # ⚠️ Hot path (num_targets)
pytorch_modules_roo.py:808    # ℹ️ Validation only (devserver)
model_configs.py:227          # ℹ️ Random data gen only
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | One line of code |
| Complexity | 1/5 | Just diagnostic |
| Success Probability | Very High | Will find any remaining syncs |
| Risk Level | None | Diagnostic only |

**Expected Impact**: Identifies remaining sync points for 2-10% additional gains
**Effort Estimate**: 2 hours for audit; fixes vary

---

### Tier 2: High Priority - Detailed Specifications

---

#### P1-1: HP-KERNELPROF - Kernel Count Profiling (DIAGNOSTIC)

**Source**: Q4

**Technical Analysis**:

V2 achieved 221→30 kernels for loss. What's the **total training step kernel count**?

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    train_step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
prof.export_chrome_trace("training_trace.json")
```

**Alternative: Nsight Systems**:
```bash
nsys profile -w true -t cuda,nvtx python train_step.py
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Diagnostic only |
| Complexity | 1/5 | Standard profiling |
| Success Probability | Very High | Will identify opportunities |
| Risk Level | None | No production impact |

**Expected Impact**: Identifies 5-15% additional optimization opportunities
**Effort Estimate**: 2-4 hours

---

#### P1-2: HP-MEMPROFILER - Add Memory Profiling Hooks Utility (DIAGNOSTIC)

**Source**: Q3

**Technical Analysis**:

Add memory snapshot capabilities for production debugging:

**Implementation**:

```python
# New file: minimal_viable_ai/models/main_feed_mtml/memory_profiler.py

import torch
import os

class MemoryProfiler:
    """Memory profiling utility for CSML ROO model training."""

    def __init__(
        self,
        enabled: bool = False,
        snapshot_dir: str = "/tmp/memory_snapshots",
        profile_interval: int = 1000,
    ):
        self.enabled = enabled
        self.snapshot_dir = snapshot_dir
        self.profile_interval = profile_interval
        os.makedirs(snapshot_dir, exist_ok=True)

        if enabled:
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self._setup_oom_observer()

    def _setup_oom_observer(self):
        """Automatically capture snapshot on OOM."""
        def oom_observer(device, alloc, device_alloc, device_free):
            snapshot_path = f"{self.snapshot_dir}/oom_snapshot.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_path)
            print(f"OOM Snapshot saved to: {snapshot_path}")

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def get_peak_memory_gb(self) -> float:
        """Get peak allocated memory in GB."""
        return torch.cuda.max_memory_allocated() / (1024**3)
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Add utility module |
| Complexity | 2/5 | Integration with training loop |
| Success Probability | 95% | Standard debugging |
| Risk Level | Low | Diagnostic only |

**Expected Impact**: Enables data-driven optimization decisions
**Effort Estimate**: 1 day

---

#### P1-3: HP-CHECKPOINT - Enable Activation Checkpointing in DeepCrossNet

**Source**: Q3

**Technical Analysis**:

Current code has `activation_checkpointing=False` in DeepCrossNet modules. Selective checkpointing saves compute-intensive operations while recomputing cheap operations.

**Current State** (model_roo_v0.py lines 362-393):
```python
ro_head_modules = nn.ModuleList([
    DeepCrossNet(
        input_dim=...,
        low_rank_dim=512,
        num_layers=2,
        ...
        activation_checkpointing=False,  # <-- Opportunity!
    )
    for _ in range(mhta_num_ro_heads)
])
```

**Implementation - Selective Checkpointing Policy**:

```python
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts, CheckpointPolicy

def csml_checkpoint_policy(ctx, op, *args, **kwargs):
    """
    Custom checkpointing policy for CSML ROO model.
    Save: Matrix multiplications, attention operations (expensive)
    Recompute: Activations, layer norms, dropout (cheap)
    """
    compute_intensive_ops = {
        torch.ops.aten.mm,
        torch.ops.aten.bmm,
        torch.ops.aten.addmm,
        torch.ops.aten.linear,
        torch.ops.aten._scaled_dot_product_flash_attention,
    }

    if op in compute_intensive_ops:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE
```

**Files to Modify**:
- `model_roo_v0.py` - Change `activation_checkpointing=False` → `True` (lines 373, 390)

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 3/5 | May need policy function |
| Complexity | 2/5 | Need to identify operations |
| Success Probability | 85% | Well-understood technique |
| Risk Level | Low | 10-20% overhead |

**Expected Impact**: 40-60% activation memory reduction
**Effort Estimate**: 2-3 days

---

#### P1-4: HP-BUFFERREUSE - Enable `allow_buffer_reuse=True` in Inductor

**Source**: Q3

**Technical Analysis**:

`allow_buffer_reuse = False` was explicitly set at `model_roo_v0.py:75` for an unknown reason. Modern PyTorch 2.5+ has fixed most issues that previously required this.

**Current Setting**:
```python
torch._inductor.config.allow_buffer_reuse = False  # Line 75
```

**⚠️ WARNING**: Must investigate why it was disabled before enabling.

**Testing Protocol**:
1. Run with `allow_buffer_reuse = True` on small validation set
2. Compare numerical outputs with baseline
3. If numerical differences detected, investigate specific ops
4. Use `torch._inductor.config.debug = True` to trace buffer reuse decisions

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 3/5 | Requires investigation |
| Complexity | 2/5 | Single flag but needs validation |
| Success Probability | 70% | May have correctness issues |
| Risk Level | Medium | Was disabled for a reason |

**Expected Impact**: 5-15% memory reduction
**Effort Estimate**: 1 day

---

#### P1-5: HP-BUDGET - Tune `activation_memory_budget`

**Source**: Q3

**Technical Analysis**:

Currently set to 0.05 (5%), which is aggressive. Research shows values 0.1-0.5 provide good memory-compute trade-offs.

**Current Configuration** (model_roo_v0.py lines 69-77):
```python
torch._functorch.config.activation_memory_budget = 0.05  # Very aggressive
torch._functorch.config.activation_memory_budget_runtime_estimator = "flops"
torch._functorch.config.activation_memory_budget_solver = "dp"
```

**Experimentation Matrix**:

| Budget Value | Expected Memory | Expected Overhead | Use Case |
|--------------|-----------------|-------------------|----------|
| 0.05 (current) | Minimum | High (~30-40%) | Memory-critical |
| 0.1 | Low | Moderate (~20-30%) | Balanced |
| 0.3 | Moderate | Low (~10-15%) | Speed-focused |
| 0.5 | Higher | Minimal (~5%) | Maximum throughput |

**Visualization**:
```python
torch._functorch.config.visualize_memory_budget_pareto = True
# Dumps SVG showing memory-runtime trade-off curve
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Configuration change |
| Complexity | 1/5 | Single parameter |
| Success Probability | 90% | Requires profiling |
| Risk Level | Low | Reversible |

**Expected Impact**: 5-20% memory variation
**Effort Estimate**: 1 day

---

#### P1-6: HP-INDUCTOR - TorchInductor Config Optimization

**Source**: Q4

**Technical Analysis**:

Enable aggressive fusion and autotune configurations:

```python
import torch._inductor.config as config

# Enable aggressive fusion
config.max_autotune = True              # Profile multiple kernel configurations
config.epilogue_fusion = True           # Fuse bias+activation into GEMM
config.aggressive_fusion = True         # Fuse even without shared memory benefit
config.triton.cudagraphs = True         # Combine with CUDA graphs

# For H100 specifically
config.coordinate_descent_tuning = True # Better kernel selection
```

**⚠️ NOTE**: `epilogue_fusion` is currently `False` in inference configs - investigate why before enabling.

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 5/5 | Config changes only |
| Complexity | 1/5 | Standard PyTorch options |
| Success Probability | Medium-High | 5-10% marginal |
| Risk Level | Low | Can be toggled off |

**Expected Impact**: 5-10% marginal improvement
**Effort Estimate**: 2-4 hours

---

#### P1-7: HP-EPILOGUE - GEMM Epilogue Fusion Verification

**Source**: Q4

**Technical Analysis**:

Linear layers followed by bias/activation may not be auto-fused. The MTML ROO model uses **SwishLayerNorm**, NOT GELU:

```python
# From model_roo_v0.py line 397-404:
nn.Sequential(
    nn.Linear(..., mhta_head_dim),
    SwishLayerNorm(input_dims=[mhta_head_dim]),  # Custom activation!
)
```

**Verification Steps**:
1. Run profiler on a single forward pass
2. Look for separate `aten::linear` and `SwishLayerNorm` kernels
3. If separate, consider restructuring or enabling `config.epilogue_fusion = True`

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Verification + potential config |
| Complexity | 2/5 | May need model restructuring |
| Success Probability | High | 3x speedup documented |
| Risk Level | Low | Well-tested optimization |

**Expected Impact**: 2-5% from epilogue fusion
**Effort Estimate**: 0.5 day

---

#### P1-8: HP-PREALLOCATE - Buffer Pre-allocation Warmup

**Source**: Q5

**Technical Analysis**:

The fragmentation problem:
```
Iteration 1: Allocate [====Batch 1====]
Iteration 2: Allocate [==Batch 2==] (smaller)
Free Batch 1: [    hole    ][==Batch 2==]
Iteration 3: Need [======Batch 3======] but hole too small!
             → Force cudaMalloc → Fragmentation → OOM
```

**Solution**: Pre-allocate maximum-size buffers during warmup.

**Implementation**:

```python
def preallocate_buffers(model, dataloader, device):
    """Warmup pass with max-size batch to cache memory allocations."""
    print("Pre-allocating GPU memory buffers...")

    # Get a representative max-size batch
    for batch in dataloader:
        max_batch = batch
        break

    # Run forward/backward to allocate all buffers
    model.train()
    with torch.cuda.amp.autocast():
        output = model(max_batch.to(device))
        loss = output.sum()
        loss.backward()

    # Clear gradients but keep allocations
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    print(f"Pre-allocation complete. Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Add warmup function |
| Complexity | 2/5 | Need representative batch |
| Success Probability | 80% | Prevents fragmentation |
| Risk Level | Low | Standard practice |

**Expected Impact**: Prevent OOM from fragmentation
**Effort Estimate**: 1 day

---

#### P1-9: HP-DPP - DPP `prefetch_queue_size` & `client_thread_count` Tuning

**Source**: Q5

**Technical Analysis**:

MTML ROO uses DPP (Data Platform Pipeline), NOT standard PyTorch DataLoader:

| PyTorch DataLoader | DPP Equivalent | Current Value |
|-------------------|----------------|---------------|
| `prefetch_factor` | `prefetch_queue_size` | 32 |
| `num_workers` | `client_thread_count` | 24 |

**Current Config** (mast_roo_trainer_config.py):
```python
def _dataloader_config(self) -> DataLoaderConfig:
    return DisgDataLoaderConfig(
        client_thread_count=24,
        prefetch_queue_size=32,
        pin_memory=True,
    )
```

**Tuning Recommendation**: Consider reducing `prefetch_queue_size` to 16-24 to balance throughput vs memory.

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 4/5 | Config change with profiling |
| Complexity | 2/5 | May need profiling |
| Success Probability | 80% | Standard tuning |
| Risk Level | Low | Config change only |

**Expected Impact**: Reduced memory pressure, potentially faster data loading
**Effort Estimate**: 1 day

---

### Tier 3: Medium Priority - Detailed Specifications

---

#### P2-1: MP-LOSSBATCH - Further Loss Batching (30→15 kernels)

**Source**: Q2, Q4

**Technical Analysis**:

V2 achieved 221→30 kernels. Additional batching by loss type could yield incremental gains:

```python
@torch.compile(mode="max-autotune", fullgraph=True)
def batched_multi_task_loss(preds_dict, targets_dict, weights_dict):
    """Batch all same-type losses into single kernel calls."""

    # Group by loss type
    bce_preds, bce_targets, bce_weights = [], [], []
    mse_preds, mse_targets, mse_weights = [], [], []

    for task_name, pred in preds_dict.items():
        if task_name.endswith('_bce'):
            bce_preds.append(pred)
            bce_targets.append(targets_dict[task_name])
            bce_weights.append(weights_dict[task_name])

    # Vectorized BCE computation
    if bce_preds:
        bce_preds_cat = torch.stack(bce_preds)
        bce_targets_cat = torch.stack(bce_targets)
        bce_loss = F.binary_cross_entropy_with_logits(
            bce_preds_cat, bce_targets_cat, reduction='none'
        ).mean(dim=-1)

    return losses
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 3/5 | Tensor manipulation |
| Complexity | 3/5 | Must maintain numerical correctness |
| Success Probability | 75% | Incremental on top of existing |
| Risk Level | Medium | Numerical precision sensitive |

**Expected Impact**: 5-10% additional loss compute speedup
**Effort Estimate**: 3-4 days

---

#### P2-2: MP-NOSYNC - DDP `no_sync()` for Gradient Accumulation

**Source**: Q1

**Technical Analysis**:

`no_sync()` context manager disables gradient synchronization during accumulation steps, reducing AllReduce overhead by (accumulation_steps - 1)×.

**⚠️ PREREQUISITE**: This ONLY applies if gradient accumulation is used. Verify first!

```python
# Usage pattern
for i, batch in enumerate(dataloader):
    # Skip sync for accumulation steps
    context = model.no_sync() if (i + 1) % accumulation_steps != 0 else nullcontext()

    with context:
        loss = model(batch)
        loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 3/5 | Context manager usage |
| Complexity | 2/5 | Training loop modification |
| Success Probability | 75% (if applicable) | 0% if no gradient accumulation |
| Risk Level | Low | Standard DDP feature |

**Expected Impact**: 25-100% (if gradient accumulation is used)
**Effort Estimate**: 2-3 days

---

#### P2-3: MP-GRAPHBREAK - Graph Break Elimination Audit

**Source**: Q2

**Technical Analysis**:

Graph breaks fragment optimization. Main causes:
- Data-dependent control flow
- `.item()` calls
- Sparse embeddings with `sparse=True`
- Custom CUDA kernels without FakeTensor registration

**Audit Tool**:
```python
# Enable graph break logging
import torch._dynamo
torch._dynamo.config.log_level = logging.DEBUG

# Or use explain mode
explanation = torch._dynamo.explain(model)(sample_input)
print(f"Graph breaks: {explanation.graph_break_count}")
print(f"Break reasons: {explanation.break_reasons}")
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 2/5 | Requires investigation |
| Complexity | 3/5 | May need code restructuring |
| Success Probability | 70% | Depends on break causes |
| Risk Level | Low | Diagnostic + fixes |

**Expected Impact**: Enable fullgraph compilation
**Effort Estimate**: 3-5 days

---

#### P2-4: MP-INT8 - INT8 Embedding Quantization (QAT)

**Source**: Q6

**Technical Analysis**:

INT8 quantization provides 4× compression with quantization-aware training (QAT) to maintain accuracy:

```python
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode

tbe = SplitTableBatchedEmbeddingBagsCodegen(
    embedding_specs=[
        (num_embeddings, embedding_dim, EmbeddingLocation.MANAGED_CACHING, ComputeDevice.CUDA)
    ],
    weights_precision=SparseType.INT8,  # 4× compression
    optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
)
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 3/5 | Config + QAT setup |
| Complexity | 3/5 | Accuracy validation |
| Success Probability | 75% | May need tuning |
| Risk Level | Medium | Quality impact |

**Expected Impact**: 4× embedding table compression
**Effort Estimate**: 5-7 days

---

#### P2-5: MP-GRADHOOK - Post-accumulate Gradient Hooks for Dense

**Source**: Q5

**Technical Analysis**:

Apply optimizer step during backward pass for dense layers:

```python
def register_gradient_hooks(model, optimizer):
    """Register hooks to apply optimizer during backward."""

    def make_hook(param_group_idx, param_idx):
        def hook(grad):
            # Apply optimizer step immediately
            param = optimizer.param_groups[param_group_idx]['params'][param_idx]
            optimizer.step_single_param(param, grad)
            return None  # Don't store gradient
        return hook

    for pg_idx, param_group in enumerate(optimizer.param_groups):
        for p_idx, param in enumerate(param_group['params']):
            if param.requires_grad:
                param.register_hook(make_hook(pg_idx, p_idx))
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 3/5 | Hook implementation |
| Complexity | 3/5 | Optimizer modification |
| Success Probability | 70% | Complex interaction |
| Risk Level | Medium | Correctness validation |

**Expected Impact**: ~20% peak memory reduction
**Effort Estimate**: 3-4 days

---

#### P2-6: MP-FUSEDSDD - Fused SDD for Optimizer Overlap

**Source**: Q5

**Technical Analysis**:

Full SDD (not Lite) overlaps optimizer step with embedding lookup:

```
SDD Lite: [memcpy] → [dist] → [forward] → [backward]
Fused SDD: [memcpy] → [dist + optimizer overlap] → [forward] → [backward]
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 3/5 | Config change |
| Complexity | 3/5 | Memory overhead (~3× batch) |
| Success Probability | 75% | May hit memory limits |
| Risk Level | Medium | Memory increase |

**Expected Impact**: 5-10% QPS (with memory cost)
**Effort Estimate**: 3-5 days

---

#### P2-7: MP-COLLATE - Custom collate_fn Optimization

**Source**: Q5

**Technical Analysis**:

Optimize KeyedJaggedTensor collation with pre-allocation and vectorization:

```python
def optimized_kjt_collate(samples):
    """Optimized collate function for KeyedJaggedTensor batching."""
    keys = samples[0].keys()

    # Vectorized length computation
    all_lengths = {key: [] for key in keys}
    all_values = {key: [] for key in keys}

    for sample in samples:
        for key in keys:
            all_lengths[key].append(len(sample[key]))
            all_values[key].append(sample[key])

    # Concatenate with pre-allocation
    batched_values = {}
    batched_lengths = {}

    for key in keys:
        lengths = torch.tensor(all_lengths[key], dtype=torch.int32)
        total_len = lengths.sum().item()

        # Pre-allocate values tensor
        values = torch.empty(total_len, dtype=torch.int64)
        offset = 0
        for v in all_values[key]:
            values[offset:offset+len(v)] = torch.tensor(v)
            offset += len(v)

        batched_values[key] = values
        batched_lengths[key] = lengths

    return KeyedJaggedTensor(
        keys=list(keys),
        values=torch.cat([batched_values[k] for k in keys]),
        lengths=torch.cat([batched_lengths[k] for k in keys]),
    )
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 3/5 | Custom function |
| Complexity | 3/5 | KJT structure understanding |
| Success Probability | 65% | Must preserve semantics |
| Risk Level | Medium | Data loading critical path |

**Expected Impact**: 10-15% data loading speedup
**Effort Estimate**: 3-4 days

---

### Tier 4: Research - Detailed Specifications

---

#### P4-1: FP-TMA - H100 TMA Optimization

**Source**: Q4

**Technical Analysis**:

H100's Tensor Memory Accelerator (TMA) provides 59% memory throughput improvement by offloading address generation to dedicated hardware.

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 1/5 | Requires Triton expertise |
| Complexity | 5/5 | Custom kernel development |
| Success Probability | 60% | Hardware-specific |
| Risk Level | High | Novel implementation |

**Expected Impact**: 10-15% memory bandwidth improvement
**Effort Estimate**: 2-3 weeks

---

#### P4-2: FP-ZEROBUBBLE - Zero Bubble Pipeline Parallelism

**Source**: Q1

**Technical Analysis**:

Decouple B_input (critical path) from B_weight (can delay) to fill pipeline bubbles:

```python
class ZeroBubbleScheduler:
    """Zero Bubble Pipeline Parallelism Scheduler.

    Key optimization: Decouple B_input from B_weight.
    - B_input is computed immediately (critical path)
    - B_weight is delayed to fill idle bubbles
    """

    def schedule_microbatch(self, stage_id, microbatch_id, phase):
        if phase == "backward_input":
            return self.schedule_backward_input(stage_id, microbatch_id)
        elif phase == "backward_weight":
            return self.schedule_backward_weight(stage_id, microbatch_id)
```

**⚠️ NOTE**: Only relevant if pipeline parallelism is used. Verify first!

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 1/5 | Major architectural change |
| Complexity | 5/5 | Custom scheduler |
| Success Probability | 50% | Complex integration |
| Risk Level | High | Large change surface |

**Expected Impact**: Near-theoretical throughput for pipeline parallelism
**Effort Estimate**: 2-4 weeks

---

#### P4-3: FP-MONOLITH - Monolith-Style Expirable Embeddings

**Source**: Q6

**Technical Analysis**:

ByteDance's Monolith introduces:
1. Cuckoo hashing: Collision-free with O(1) lookup
2. Expirable embeddings: Remove after inactivity
3. Frequency filtering: Only create embeddings for IDs seen > N times

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 1/5 | Major infrastructure change |
| Complexity | 5/5 | Custom hash table |
| Success Probability | 40% | Novel architecture |
| Risk Level | High | Research-level |

**Expected Impact**: Dynamic table sizing, memory = active features
**Effort Estimate**: 15+ days

---

#### P4-4: FP-ROBE - ROBE (Random Offset Block Embedding)

**Source**: Q6

**Technical Analysis**:

ROBE uses a single shared parameter array with block-based universal hashing, achieving 1000× compression:

```
Single shared array: P ∈ R^M (M << N × D)

For feature ID i:
  offset = hash(i) % (M - block_size)
  embedding = P[offset : offset + block_size]
```

**Assessment**:

| Criterion | Rating | Justification |
|-----------|--------|---------------|
| Easiness | 1/5 | Novel research implementation |
| Complexity | 5/5 | Research prototype quality |
| Success Probability | 35% | Extreme compression |
| Risk Level | High | Quality risk |

**Expected Impact**: 1000× compression
**Effort Estimate**: 20+ days

---

*End of Detailed Proposal Specifications*
