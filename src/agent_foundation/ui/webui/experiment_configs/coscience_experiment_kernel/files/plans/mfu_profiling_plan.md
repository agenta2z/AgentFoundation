## 🎯 RESEARCH OBJECTIVE
Automate GPU model profiling to detect performance bottlenecks and generate actionable optimization recommendations. Focus on Milestone 1 (Profiling) of the GemForge methodology.

## 📋 MFU PROFILING METHODOLOGY

### Phase 1: Job Information Collection
- Collect MAST job information from user
- Extract commit hash and model configuration from job metadata
- Identify target GPU hardware (H100/B200/AMD)
- Document model architecture details

### Phase 2: Automated Benchmarking
- Execute Component Benchmarking (CB) with kernel-level profiling
- Run Efficient Module Suite (EMS) benchmarks
- Collect performance traces using GPU profiling tools
- Generate raw benchmark data files

### Phase 3: Performance Analysis
- Calculate dense MFU for each model component
- Identify memory bandwidth utilization
- Measure tensor core efficiency
- Analyze kernel launch overhead
- Profile memory access patterns

### Phase 4: Bottleneck Detection
- Classify bottlenecks by category:
  - **Memory-bound operations**: embedding lookups, attention softmax
  - **Compute-bound operations**: matrix multiplications, linear layers
  - **I/O-bound operations**: data loading, checkpointing
- Rank bottlenecks by contribution to overall latency
- Identify low-hanging optimization opportunities

### Phase 5: Report Generation
- Generate comprehensive MFU analysis report
- Include visualizations:
  - Module-level MFU breakdown
  - Hardware utilization charts
  - Bottleneck impact analysis
- Provide actionable optimization recommendations
- Document hardware-specific insights

### Phase 6: Analysis
- Compare results with historical benchmarks
- Identify performance regression if applicable
- Summarize key findings
- Recommend next steps (co-design, kernel optimization, or further profiling)

## 🔄 ITERATION POLICY
After analysis, based on profiling findings:
- May proceed to Model Co-Design if architecture changes needed
- May proceed to Kernel Optimization if custom kernels needed
- May continue profiling with different configurations

## ✅ SUCCESS METRICS
- Complete automated profiling in <30 minutes (vs. >1 week manual)
- Accurate MFU calculation for all model components
- Clear bottleneck identification with severity ranking
- Actionable recommendations for each bottleneck
- Comprehensive documentation for team reference
