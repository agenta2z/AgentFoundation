# CSML Optimization: Experiment Setup

## Hardware Configuration

### GPU Setup
- **Primary GPU**: NVIDIA H100 80GB
- **Comparison GPU**: NVIDIA B200
- **GPU Memory**: 80GB HBM3
- **Interconnect**: NVLink 4.0

### Cluster Configuration
- **Single Node**: 8x H100 GPUs
- **Multi-Node**: Up to 64 GPUs (8 nodes)
- **Network**: InfiniBand HDR

## Software Environment

### Framework Versions
- **PyTorch**: 2.x (latest stable)
- **CUDA**: 12.x
- **cuDNN**: 8.x
- **TorchRec**: Latest

### Compilation Settings
```python
torch.compile(
    model,
    mode="reduce-overhead",
    fullgraph=False,
    dynamic=True,
)
```

## Model Configuration

### HSTU Transducer CInt
- **Embedding Dimension**: 128
- **Number of Heads**: 8
- **Sequence Length**: 512
- **Vocabulary Size**: 10M

### Training Configuration
- **Batch Size**: 4096
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW
- **Precision**: BF16 mixed

## Benchmarking Methodology

### Warmup
- 100 iterations for JIT compilation
- Discard first 10 measurements

### Measurement
- 1000 iterations per configuration
- Report mean and std deviation
- Measure wall-clock time

### Metrics Collected
1. **QPS** (Queries Per Second)
2. **Latency** (P50, P90, P99)
3. **Memory** (Peak allocated, reserved)
4. **Kernel Count** (via profiler)

## Baseline Configuration

```python
# Baseline (no optimizations)
config = {
    "activation_memory_budget": 0.1,
    "bf16_on_task_arch": False,
    "sdd_lite": False,
    "torch_compile": False,
}
```

## Optimized Configuration

```python
# Optimized
config = {
    "activation_memory_budget": 0.05,
    "bf16_on_task_arch": True,
    "sdd_lite": True,
    "torch_compile": True,
    "compile_mode": "reduce-overhead",
}
```

## Profiling Tools

1. **PyTorch Profiler** - Kernel timing, memory
2. **Nsight Systems** - GPU utilization, sync points
3. **Nsight Compute** - Kernel analysis
4. **Custom Benchmark** - End-to-end QPS
