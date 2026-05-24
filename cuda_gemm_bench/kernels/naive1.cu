#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32

__global__ void naive_kernel(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= N || gy >= M) return;

    float tmp = 0.0f;

    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx];
    }

    C[gy * N + gx] =
        alpha * tmp + beta * C[gy * N + gx];
}

void launch_naive_gemm(
    int M,
    int N,
    int K,
    float alpha,
    float* A,
    float* B,
    float beta,
    float* C)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    dim3 blocks(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    naive_kernel<<<blocks, threads>>>(
        M,
        N,
        K,
        alpha,
        A,
        B,
        beta,
        C);
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

    launch_naive_gemm(
        M,
        N,
        K,
        1.0f,
        d_A,
        d_B,
        0.0f,
        d_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "C[0] = " << h_C[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}