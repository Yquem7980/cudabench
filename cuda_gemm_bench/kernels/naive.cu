#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include "gemm_kernels.cuh"

#define BLOCK_SIZE 32


__global__ void naive_kernel(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局x
  int gy = blockIdx.y * blockDim.y + threadIdx.y;  // 全局y

  if (gx >= N || gy >= M) return;

  float tmp = 0.0f;
  for (int i = 0; i < K; i++) {
    tmp += A[gy * K + i] * B[i * N + gx];  // 两次全局内存访问
  }
  C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}





void launch_naive_gemm(
    int M,
    int N,
    int K,
    float alpha,
    float* A,
    float* B,
    float beta,
    float* C,
    cudaStream_t stream)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y);

    naive_kernel<<<blocks, threads, 0, stream>>>(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        C);
}