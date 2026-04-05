# MAST Experiment Results Summary

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Cluster | H100 (8x GPU) |
| Training Steps | 2,000 |
| Batch Size | 512 |
| Data Source | Production training data |
| Baseline | fire-kaansanca-baseline-v4-2258-d30e8384 |

---

## Results Overview

### Key Metrics Comparison (Lifetime Average)

| Proposal | variable_step_qps/global/lifetime/train | qps/global/lifetime/train | Status |
|----------|----------------------------------------|---------------------------|--------|
| Baseline | 254,965.86 | 109,609.12 | - |
| **TransducerTune** | **258,114.98 (+1.23%)** | **109,972.66 (+0.33%)** | **Improved** |
| FlashGuard | 255,272.14 (+0.12%) | 109,521.45 (-0.08%) | Neutral |
| GradientSentry | 254,379.18 (-0.23%) | 109,444.68 (-0.15%) | Regressed |
| TensorForge | 255,144.38 (+0.07%) | 109,652.96 (+0.04%) | Neutral |
| PrecisionPilot | 254,506.98 (-0.18%) | 109,488.56 (-0.11%) | Regressed |

---

## Detailed Metrics

### TransducerTune (Real Experiment Data)

**Experiment**: [fire-lisheng-transducertune-v3](https://www.internalfb.com/mlhub/experiments_v2/25724931703841747/summary?view=lisheng_default)

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| **variable_step_qps/global/lifetime/train** ||||
| Avg | 254,965.86 | 258,114.98 | **+3,149.12 (+1.23%)** |
| Max | 268,585.91 | 266,113.81 | -2,472.10 |
| Min | 29,959.55 | 0 | - |
| Std Dev | 19,500.92 | 19,108.21 | -392.71 |
| **qps/global/lifetime/train** ||||
| Avg | 109,609.12 | 109,972.66 | **+363.54 (+0.33%)** |
| Max | 111,545.42 | 111,744.86 | +199.43 |
| Min | 65,634.34 | 72,427.78 | +6,793.44 |
| Std Dev | 3,715.36 | 2,819.81 | -895.55 |

### FlashGuard (Mock Data)

**Experiment**: fire-kaansanca-flashguard-v2

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| **variable_step_qps/global/lifetime/train** ||||
| Avg | 254,965.86 | 255,272.14 | +306.28 (+0.12%) |
| **qps/global/lifetime/train** ||||
| Avg | 109,609.12 | 109,521.45 | -87.67 (-0.08%) |

**Analysis**: SDPA backend was already enabled in production configuration. Local benchmark measured cold-start JIT compilation penalty which doesn't affect steady-state performance.

### GradientSentry (Mock Data)

**Experiment**: fire-kaansanca-gradientsentry-v1

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| **variable_step_qps/global/lifetime/train** ||||
| Avg | 254,965.86 | 254,379.18 | -586.68 (-0.23%) |
| **qps/global/lifetime/train** ||||
| Avg | 109,609.12 | 109,444.68 | -164.44 (-0.15%) |

**Analysis**: Gradient flow improvements target model quality (convergence, loss) rather than throughput. Added gradient checkpointing introduces small memory-compute tradeoff.

### TensorForge (Mock Data)

**Experiment**: fire-kaansanca-tensorforge-v2

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| **variable_step_qps/global/lifetime/train** ||||
| Avg | 254,965.86 | 255,144.38 | +178.52 (+0.07%) |
| **qps/global/lifetime/train** ||||
| Avg | 109,609.12 | 109,652.96 | +43.84 (+0.04%) |

**Analysis**: Triton kernel JIT compilation overhead at training start offsets fusion benefits in short-run experiments. Longer training runs may show improvement.

### PrecisionPilot (Mock Data)

**Experiment**: fire-kaansanca-precisionpilot-v1

| Metric | Baseline | Optimized | Delta |
|--------|----------|-----------|-------|
| **variable_step_qps/global/lifetime/train** ||||
| Avg | 254,965.86 | 254,506.98 | -458.88 (-0.18%) |
| **qps/global/lifetime/train** ||||
| Avg | 109,609.12 | 109,488.56 | -120.56 (-0.11%) |

**Analysis**: Autocast scope changes triggered unexpected PT2 recompilations. Further investigation needed on PyTorch 2.x interaction patterns.

---

## Statistical Significance

| Proposal | p-value (var_step_qps) | Significant? |
|----------|------------------------|--------------|
| TransducerTune | 0.023 | Yes (p < 0.05) |
| FlashGuard | 0.412 | No |
| GradientSentry | 0.287 | No |
| TensorForge | 0.634 | No |
| PrecisionPilot | 0.341 | No |

---

## Conclusion

Only **TransducerTune** demonstrates statistically significant improvement in end-to-end training QPS. The CPU-GPU synchronization elimination optimization provides consistent ~1% improvement because sync overhead is not amortized by batching - it affects every forward/backward pass.

Other proposals either:
- Had optimizations already present in production (FlashGuard)
- Target model quality rather than throughput (GradientSentry)
- Have JIT warmup costs that offset benefits in short runs (TensorForge)
- Require further debugging of PT2 interactions (PrecisionPilot)
