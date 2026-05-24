#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include "gemm_kernels.cuh"

#define BLOCK_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void shared_kernel(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  const int BM = BLOCK_SIZE;
  const int BN = BLOCK_SIZE;
  const int BK = BLOCK_SIZE;

  int tx = threadIdx.x % BN;
  int ty = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

  float tmp = 0.;
  for (int k = 0; k < K; k += BK) {
    As[ty * BK + tx] = A[ty * K + tx];
    Bs[ty * BN + tx] = B[ty * N + tx];
    __syncthreads();
    A += BK;
    B += BK * N;
    for (int i = 0; i < BK; i++) {
      tmp += As[ty * BK + i] * Bs[i * BN + tx];
    }
    __syncthreads();
  }
  C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}


void launch_shared_gemm(
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
      dim3 blockDim(1024);
      dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));

    shared_kernel<<<gridDim, blockDim, 0, stream>>>(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        C);
}