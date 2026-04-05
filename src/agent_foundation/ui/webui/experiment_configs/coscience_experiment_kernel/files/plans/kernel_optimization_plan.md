## 🎯 RESEARCH OBJECTIVE
Generate and optimize custom GPU kernels (Triton/CUDA) to maximize hardware utilization for specific bottleneck operations. Focus on Milestone 4 (Kernel Generation) of the GemForge methodology.

## 📋 KERNEL OPTIMIZATION METHODOLOGY

### Phase 1: Bottleneck Selection
- Review profiling results and identified bottlenecks
- Select target operations for kernel optimization:
  - High-latency operations
  - Low tensor core utilization operations
  - Memory-bound operations with optimization potential
- Prioritize by expected impact on overall MFU

### Phase 2: Kernel Design
- Design custom kernel architecture:
  - Thread block configuration
  - Shared memory usage strategy
  - Register pressure management
  - Memory coalescing patterns
- Document design decisions and trade-offs

### Phase 3: Triton Implementation
- Implement Triton kernel templates:
  - Fused attention kernels
  - Custom activation functions
  - Memory-efficient operators
- Follow best practices:
  - Block size tuning
  - Memory access patterns
  - Launch configuration

### Phase 4: CUDA Optimization (if needed)
- Implement CUDA kernels for operations requiring:
  - Fine-grained control
  - Hardware-specific optimizations
  - Features not available in Triton
- Apply optimization techniques:
  - Tensor core utilization
  - Async memory operations
  - Warp-level primitives

### Phase 5: Autotuning
- Implement autotuning framework:
  - Parameter search space definition
  - Benchmark harness
  - Configuration selection
- Tune for target hardware (H100/B200)
- Document optimal configurations

### Phase 6: Integration
- Integrate custom kernels into model:
  - PyTorch custom operator wrapping
  - Backward pass implementation
  - Numerical stability verification
- Validate integration correctness

### Phase 7: Benchmarking
- Execute comprehensive benchmarks:
  - Kernel-level latency comparison
  - End-to-end model MFU
  - Memory utilization
- Compare against baseline implementations
- Document performance improvements

### Phase 8: Analysis & Documentation
- Analyze benchmark results
- Document kernel design patterns
- Create usage guidelines for team
- Identify remaining optimization opportunities

## 🔄 ITERATION POLICY
After benchmarking, based on results:
- If performance target achieved, document and deploy
- If gaps remain, iterate on kernel design
- If new bottlenecks identified, return to profiling

## ✅ SUCCESS METRICS
- ≥5× faster kernel iteration cycle
- ≥15% efficiency improvement vs. baseline
- Numerically stable and correct implementation
- Well-documented and maintainable code
- Transferable knowledge for team
