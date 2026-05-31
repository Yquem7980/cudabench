// #include <nvbench/nvbench.cuh>
// #include <thrust/device_vector.h>
// #include "flashattention_kernels.cuh"
#include <cstdlib>
#include <ctime>
#include <cmath> 
#include <algorithm> 
// #include <iostream>
// #include <vector>
// #include <random>
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// ==================== CPU 参考实现 ====================

// cpu  compute
void cpu_AB(const float* A, const float* B, float* C, int M, int N, int K) {

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum ;
        }
    }
}



// CPU GEMM reference
void cpu_GEMM(
    const float* A, const float* B,  float* C,
     int M, int N, int K
) {
    // std::vector<float> A(M * K);
    // std::vector<float> B(K * N);
    
    cpu_AB(A, B, C, M, N, K);
    
}

// ==================== 验证工具 ====================

// 比较两个数组，返回最大相对误差
float check_result(const float* ref, const float* test, int size, float rel_tol  = 1e-3f,float abs_tol  = 1e-4f) {
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int err_count = 0;
    
    for (int i = 0; i < size; i++) {
        float abs_diff = fabsf(ref[i] - test[i]);
        float rel_diff;
        float denom = fabsf(ref[i]);
        //float rel_diff = diff / (fabsf(ref[i]) + 1e-6f);
        // float abs_diff = fabsf(ref[i] - test[i]);
        if (denom> 1e-6f)
            rel_diff = abs_diff / denom;
        else
            rel_diff = abs_diff;

        max_abs_err = std::max(max_abs_err, abs_diff);
        max_rel_err = std::max(max_rel_err, rel_diff);
        
        if (abs_diff > abs_tol &&rel_diff > rel_tol) {
            if (err_count < 5) {  // 只打印前5个错误
                printf("  Error at [%d]: ref=%.8f, test=%.8f, abs_err=%.8f, rel_err=%.8f\n",
                    i, ref[i], test[i], abs_diff, rel_diff);
            }
            err_count++;
        }
    }
    
    if (err_count > 0) {
    printf("  Total errors: %d / %d (%.2f%%)\n", 
           err_count, size, 100.0f * err_count / size);
    printf("  max_abs_err=%.8f, max_rel_err=%.8f\n",
           max_abs_err, max_rel_err);
    } else {
        printf("  All passed! max_abs_err=%.8f, max_rel_err=%.8f\n",
            max_abs_err, max_rel_err);
    }
    
    return max_rel_err;
}


// ==================== CUDA Kernels ====================
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void AB_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{   __shared__ float As[BM * BK];//As= BM * BK，而blockdim是BM * BN，所以BN = BK才不会越界
    __shared__ float Bs[BK * BN];//As= BK * BN，而blockdim是BM * BN，所以BM = BK才不会越界

    int by = blockIdx.y;
    int bx = blockIdx.x;

    int block_row_thread = BM / TM;
    int block_col_thread = BN / TN;
    int thread_num = block_row_thread * block_col_thread;

    int ty = threadIdx.x / block_col_thread * TM;
    int tx = threadIdx.x % block_col_thread * TN;


    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK;

    int b_tile_row = threadIdx.x / BN;
    int b_tile_col = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    A = &A[by * BM * K];//每个block第一个线程处理的矩阵的左顶点位置
    B = &B[bx * BN];
    C = &C[by* BM * N + bx * BN];

    float tmp[TM][TN] = {0.};

  
    for(int bk = 0; bk < K;  bk += BK){
        for (int i = 0; i < BM; i += a_tile_stride){
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        for(int i = 0; i < BK; i += b_tile_stride){
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        
        __syncthreads();

        A += BK;
        B += BK * N;
    
        for (int i = 0 ; i < BK ; i++){
            for(int j = 0; j < TM ; j++){
                for(int l = 0; l < TN ; l++){
                    tmp[j][l] += As[(ty + j) * BK + i] * Bs[i * BN + tx + l];
                }
            }
            
        }
        __syncthreads();

    }

    for(int j = 0; j < TM ; j++){
        for(int l = 0; l < TN ; l++){
            C[(ty +j) * N + tx + l]= tmp[j][l];
        }
    }
}










int main()
{
    int M = 512;
    int N = 512;
    int K = 512;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);


    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    // float *h_C = new float[M * N];
    
    float *h_C_gpu = new float[M * N];   // GPU 结果
    float *h_C_cpu = new float[M * N];   // CPU 参考结果   


    srand(0);
    for (int i = 0; i < M * K; i++)
        h_A[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    for (int i = 0; i < K * N; i++)
        h_B[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

   
    for (int i = 0; i < M * N; i++)
        h_C_gpu[i] = 0.0f;
    for (int i = 0; i < M * N; i++)
        h_C_cpu[i] = 0.0f;


    // ========== CPU 参考计算 ==========
    printf("Running CPU reference...\n");
    cpu_GEMM(h_A, h_B, h_C_cpu, M, N, K);
    printf("CPU done. C_cpu[0] = %.6f\n\n", h_C_cpu[0]);

    // ========== GPU 计算 ==========
    printf("Running GPU kernels...\n");


    float *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));


    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC));

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    dim3 block(CEIL_DIV(BN,TN) * CEIL_DIV(BM,TN));

   

    dim3 grid(CEIL_DIV(N,BN),CEIL_DIV(M,BM));
    
    // Step 1: AB
    AB_kernel<BM,BN,BK,TM,TN><<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());    
    CHECK_CUDA(cudaDeviceSynchronize());


   

    

   
    // 拷贝结果    
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));
    printf("GPU done. C_gpu[0] = %.6f\n\n", h_C_gpu[0]);



        // ========== 验证 ==========
    printf("Checking AB intermediate...\n");
    check_result(h_C_cpu, h_C_gpu, M * N, 1e-3f, 1e-4f);

    // 释放
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
   

    delete[] h_A;
    delete[] h_B;
    // delete[] h_C;

    delete[] h_C_gpu;
    delete[] h_C_cpu;

    return 0;
}