#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void shared_kernel(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float tmp = 0.f;

    for (int k = 0; k < K; k += BLOCK_SIZE) {

        // load A tile
        As[ty][tx] = A[row * K + (k + tx)];

        // load B tile
        Bs[ty][tx] = B[(k + ty) * N + col];

        __syncthreads();

        // compute
        for (int i = 0; i < BLOCK_SIZE; i++) {
            tmp += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    C[row * N + col] =
        alpha * tmp + beta * C[row * N + col];
}

int main()
{
    int M = 1024;
    int N = 1024;
    int K = 1024;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *A, *B, *C;

    cudaMalloc(&A, sizeA);
    cudaMalloc(&B, sizeB);
    cudaMalloc(&C, sizeC);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(
        CEIL_DIV(N, BLOCK_SIZE),
        CEIL_DIV(M, BLOCK_SIZE));

    for (int i = 0; i < 10; i++) {

        shared_kernel<<<gridDim, blockDim>>>(
            M,
            N,
            K,
            1.f,
            A,
            B,
            0.f,
            C);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess) {
            std::cout << "Launch Error: "
                      << cudaGetErrorString(err)
                      << std::endl;
        }
    }

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cout << "CUDA Error: "
                  << cudaGetErrorString(err)
                  << std::endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}