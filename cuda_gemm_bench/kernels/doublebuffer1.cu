
#include <thrust/device_vector.h>


#define BLOCK_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(256)
    doublebuffering_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta,
               float *C) {
  int bx = blockIdx.x;
  int by = blockIdx.y;

  const int block_row_thread = BN / TN;
  const int block_col_thread = BM / TM;
  const int thread_num = block_row_thread * block_col_thread;

  int tx = (threadIdx.x % block_row_thread) * TN;
  int ty = (threadIdx.x / block_row_thread) * TM;

  __shared__ float As[2][BK * BM];
  __shared__ float Bs[2][BK * BN];

  const int ldg_a_num = BK * BM / thread_num / 4;
  const int ldg_b_num = BK * BN / thread_num / 4;

  int a_tile_row = threadIdx.x / (BK / 4);
  int a_tile_col = threadIdx.x % (BK / 4) * 4;
  int a_tile_stride = BM / ldg_a_num;

  int b_tile_row = threadIdx.x / (BN / 4);
  int b_tile_col = threadIdx.x % (BN / 4) * 4;
  int b_tile_stride = BK / ldg_b_num;

  float accum[TM][TN] = {0.};

  float ldg_a_reg[4 * ldg_a_num] = {0.};
  float ldg_b_reg[4 * ldg_b_num] = {0.};

  float a_frag[2][TM];
  float b_frag[2][TN];

  A = &A[by * BM * K];
  B = &B[bx * BN];
  C = &C[by * BM * N + bx * BN];

#pragma unroll
  for (int i = 0; i < BM; i += a_tile_stride) {
    int ldg_index = i / a_tile_stride * 4;
    FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
        FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
    As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
    As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] =
        ldg_a_reg[ldg_index + 1];
    As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] =
        ldg_a_reg[ldg_index + 2];
    As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] =
        ldg_a_reg[ldg_index + 3];
  }
#pragma unroll
  for (int i = 0; i < BK; i += b_tile_stride) {
    FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
        FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]);
  }
  __syncthreads();

#pragma unroll
  for (int m = 0; m < TM; m += 4) {
    FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]);
  }
#pragma unroll
  for (int n = 0; n < TN; n += 4) {
    FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]);
  }

  int write_index = 1;
  int load_index;
  int k = 0;
  do {
    k += BK;
    if (k < K) {
#pragma unroll
      for (int i = 0; i < BM; i += a_tile_stride) {
        int ldg_index = i / a_tile_stride * 4;
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
            FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
      }
#pragma unroll
      for (int i = 0; i < BK; i += b_tile_stride) {
        int ldg_index = i / b_tile_stride * 4;
        FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
            FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
      }
    }

    load_index = write_index ^ 1;
#pragma unroll
    for (int bk = 0; bk < BK - 1; bk++) {
      for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) =
            FETCH_FLOAT4(As[load_index][OFFSET(bk + 1, ty + m, BM)]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) =
            FETCH_FLOAT4(Bs[load_index][OFFSET(bk + 1, tx + n, BN)]);
      }
#pragma unroll
      for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n++) {
          accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
        }
      }
    }
    if (k < K) {
#pragma unroll
      for (int i = 0; i < BM; i += a_tile_stride) {
        int ldg_index = i / a_tile_stride * 4;
        As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] =
            ldg_a_reg[ldg_index];
        As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] =
            ldg_a_reg[ldg_index + 1];
        As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] =
            ldg_a_reg[ldg_index + 2];
        As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] =
            ldg_a_reg[ldg_index + 3];
      }
#pragma unroll
      for (int i = 0; i < BK; i += b_tile_stride) {
        int ldg_index = i / b_tile_stride * 4;
        FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
            FETCH_FLOAT4(ldg_b_reg[ldg_index]);
      }
      __syncthreads();
#pragma unroll
      for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[0][m]) =
            FETCH_FLOAT4(As[write_index][OFFSET(0, ty + m, BM)]);
      }
#pragma unroll
      for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[0][n]) =
            FETCH_FLOAT4(Bs[write_index][OFFSET(0, tx + n, BN)]);
      }

      write_index ^= 1;
    }
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
      for (int n = 0; n < TN; n++) {
        accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
      }
    }

  } while (k < K);

#pragma unroll
  for (int m = 0; m < TM; m++) {
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
      float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
      ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
      ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
      ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
      ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
      FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
    }
  }
}

int main()
{
    int M = 8192;
    int N = 8192;
    int K = 8192;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];

    for (int i = 0; i < M * K; i++)
        h_A[i] = 1.0f;

    for (int i = 0; i < K * N; i++)
        h_B[i] = 1.0f;

    for (int i = 0; i < M * N; i++)
        h_C[i] = 0.0f;

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice);

    dim3 blockDim(256);

    dim3 gridDim(
        CEIL_DIV(N, 128),
        CEIL_DIV(M, 128));

    doublebuffering_kernel<
        128, 128, 8,
        8, 8>
    <<<gridDim, blockDim>>>(
        M,
        N,
        K,
        1.0f,
        d_A,
        d_B,
        0.0f,
        d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_C,
        d_C,
        sizeC,
        cudaMemcpyDeviceToHost);

    std::cout << "C[0] = "
              << h_C[0]
              << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}