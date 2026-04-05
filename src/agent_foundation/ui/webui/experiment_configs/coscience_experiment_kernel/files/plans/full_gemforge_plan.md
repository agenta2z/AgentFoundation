## 🎯 RESEARCH OBJECTIVE
Automate the complete GemForge workflow for GPU computation efficiency optimization, covering profiling, headroom analysis, model co-design, and kernel generation.

## 📋 GEMFORGE METHODOLOGY

### Phase 1: Model Profiling (M1)
- Execute automated CB (Component Benchmarking) and EMS (Efficient Module Suite) benchmarks
- Extract commit hash and model configuration from job metadata
- Generate performance traces and bottleneck reports
- Calculate dense MFU metrics for all model components
- Document module-level performance characteristics

### Phase 2: Bottleneck Identification
- Analyze profiling results to identify performance bottlenecks
- Categorize bottlenecks by type:
  - **Memory-bound**: Bandwidth limitations, cache misses
  - **Compute-bound**: Low tensor core utilization, suboptimal parallelism
  - **Latency-bound**: Kernel launch overhead, synchronization points
- Prioritize bottlenecks by impact on overall MFU

### Phase 3: Headroom Analysis (M2)
- Build automated headroom estimator to quantify optimization potential
- Calculate theoretical MFU ceiling based on hardware specs
- Estimate achievable MFU gains for each bottleneck
- Identify top optimization opportunities with expected ROI
- Target: ±10% error in MFU headroom prediction

### Phase 4: Optimization Recommendations
- Generate hardware-specific optimization strategies
- Map bottlenecks to actionable recommendations:
  - Model architecture changes
  - Kernel optimization opportunities
  - Memory access pattern improvements
  - Parallelization strategies
- Rank recommendations by impact/effort ratio

### Phase 5: Model Co-Design (M3)
- Recommend GPU-friendly model architecture changes
- Focus areas:
  - Dimension alignment for tensor core efficiency
  - Attention pattern optimization
  - Layer fusion opportunities
  - Batch size and sequence length tuning
- Target: ≥10% dense MFU gain on pilot model

### Phase 6: Kernel Generation (M4)
- Generate Triton kernel templates aligned with model design
- Autotune kernel parameters for target GPU (H100/B200)
- Implement custom kernels for identified bottlenecks:
  - Fused attention kernels
  - Memory-efficient operators
  - Custom activation functions
- Target: ≥5× faster iteration, ≥15% efficiency improvement

### Phase 7: Benchmarking & Validation
- Execute comprehensive benchmarks on generated kernels
- Compare MFU before/after optimization
- Validate correctness with numerical precision tests
- Document performance gains and trade-offs
- Generate detailed optimization report

### Phase 8: Analysis & Next Steps
- Synthesize learnings from the optimization cycle
- Identify remaining bottlenecks for next iteration
- Document best practices discovered
- Plan refinement cycle if needed

## 🔄 ITERATION POLICY
After validation, based on benchmark results:
- If MFU target achieved, document and conclude
- If gaps remain, iterate on specific components
- If new bottlenecks discovered, pivot to address them

## ✅ SUCCESS METRICS
- ≥1000× faster profiling (from >1 week to 30 mins)
- MFU headroom prediction ±10% accuracy
- ≥10% dense MFU improvement on target model
- ≥5× faster kernel iteration cycle
- Comprehensive optimization documentation
