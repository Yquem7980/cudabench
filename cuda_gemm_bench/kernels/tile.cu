#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include "gemm_kernels.cuh"

#define BLOCK_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void tile_kernel(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int block_row_thread = BN / TN;
  int block_col_thread = BM / TM;
  int thread_num = block_row_thread * block_col_thread;

  int tx = (threadIdx.x % block_row_thread) * TN;
  int ty = (threadIdx.x / block_row_thread) * TM;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

  int a_tile_row = threadIdx.x / BK;
  int a_tile_col = threadIdx.x % BK;
  int a_tile_stride = thread_num / BK;

  int b_tile_row = threadIdx.x / BN;
  int b_tile_col = threadIdx.x % BN;
  int b_tile_stride = thread_num / BN;

  float tmp[TM][TN] = {0.};
#pragma unroll
  for (int k = 0; k < K; k += BK) {
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
      As[(a_tile_row + i) * BK + a_tile_col] =
          A[(a_tile_row + i) * K + a_tile_col];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
      Bs[(b_tile_row + i) * BN + b_tile_col] =
          B[(b_tile_row + i) * N + b_tile_col];
    }
    __syncthreads();
    A += BK;
    B += BK * N;
#pragma unroll
    for (int i = 0; i < BK; i++) {
#pragma unroll
      for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TN; l++)
          tmp[j][l] += As[(ty + j) * BK + i] * Bs[tx + l + i * BN];
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < TM; j++) {
    for (int l = 0; l < TN; l++)
      C[(ty + j) * N + tx + l] =
          alpha * tmp[j][l] + beta * C[(ty + j) * N + tx + l];
  }
}

void launch_tile_gemm(
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
      dim3 blockDim(256);
      dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(N, 128));

    tile_kernel<128, 128, 8, 8, 8><<<gridDim, blockDim>>>(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        C);
}