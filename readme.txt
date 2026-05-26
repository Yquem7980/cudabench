# FlashAttention CUDA Implementation

This folder contains a simplified CUDA implementation of FlashAttention and several optimization experiments.

本文件夹包含 FlashAttention 的简化 CUDA 实现及相关优化实验。

## Contents

The implementation includes:

- A naive attention implementation.
- An optimized softmax kernel.
- A double-buffered `qk_kernel` for QKᵀ computation.
- A tiled `pv_kernel` for PV computation.
- Benchmark and performance analysis for different kernel versions.

主要内容包括：

- naive attention 实现；
- softmax kernel 优化；
- 使用 double buffering 优化的 `qk_kernel`；
- 使用 tiling 优化的 `pv_kernel`；
- 不同 kernel 版本的 benchmark 与性能分析。

## Performance

Through these optimizations, the three-stage attention pipeline was improved from **267.33 μs** to **35.9 μs** in the tested configuration.

通过上述优化，在测试配置下，三段式 attention pipeline 从 **267.33 μs** 优化至 **35.9 μs**。

## Future Work

The current three-stage implementation still involves intermediate global memory reads and writes between QKᵀ, softmax, and PV. To further reduce global memory traffic, a fused FlashAttention-style kernel based on online softmax is under continuous development and optimization.

当前三段式实现仍然需要在 QKᵀ、softmax 和 PV 之间进行中间结果的 global memory 读写。为进一步减少 global memory traffic，正在持续开发和优化基于 online softmax 的 fused FlashAttention-style kernel。