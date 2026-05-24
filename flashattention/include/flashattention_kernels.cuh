#pragma once

#include <cuda_runtime.h>

void launch_naive_flashattention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int m,
    int n,
    int d,
    cudaStream_t stream);

// void launch_shared_gemm(
//     int M,
//     int N,
//     int K,
//     float alpha,
//     float* A,
//     float* B,
//     float beta,
//     float* C,
//     cudaStream_t stream);

// void launch_tile_gemm(
//     int M,
//     int N,
//     int K,
//     float alpha,
//     float* A,
//     float* B,
//     float beta,
//     float* C,
//     cudaStream_t stream);

// void launch_vector_gemm(
//     int M,
//     int N,
//     int K,
//     float alpha,
//     float* A,
//     float* B,
//     float beta,
//     float* C,
//     cudaStream_t stream);

// void launch_doublebuffering_gemm(
//     int M,
//     int N,
//     int K,
//     float alpha,
//     float* A,
//     float* B,
//     float beta,
//     float* C,
//     cudaStream_t stream);

// void launch_warp_gemm(
//     int M,
//     int N,
//     int K,
//     float alpha,
//     float* A,
//     float* B,
//     float beta,
//     float* C,
//     cudaStream_t stream);