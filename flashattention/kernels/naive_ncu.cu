// #include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
// #include "flashattention_kernels.cuh"
#include <cstdlib>
#include <ctime>

#define BLOCK_SIZE 32
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))



__global__ void qk_kernel(
    const float* Q,
    const float* K,
    float* S,
    int m,
    int n,
    int d)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // n
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m

    if (row >= m || col >= n) return;

    float sum = 0.0f;

    for (int k = 0; k < d; k++) {
        sum += Q[row * d + k] * K[col * d + k];
    }

    S[row * n + col] = sum;
}

__global__ void softmax_kernel(
    const float* S,
    float* P,
    int m,
    int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= m || col >= n) return;

    // 1. find max
    float max_val = -1e30f;

    for (int j = 0; j < n; j++) {
        float val = S[row * n + j];
        if (val > max_val) {
            max_val = val;
        }
    }

    // 2. compute exp sum
    float sum = 0.0f;

    for (int j = 0; j < n; j++) {
        sum += expf(S[row * n + j] - max_val);
    }

    // 3. normalize
    float exp_val = expf(S[row * n + col] - max_val);

    P[row * n + col] = exp_val / sum;
}

__global__ void pv_kernel(
    const float* P,
    const float* V,
    float* O,
    int m,
    int n,
    int d)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // d
    int row = blockIdx.y * blockDim.y + threadIdx.y; // m

    if (row >= m || col >= d) return;

    float sum = 0.0f;

    for (int k = 0; k < n; k++) {
        sum += P[row * n + k] * V[k * d + col];
    }

    O[row * d + col] = sum;
}


// dim3 block(16, 16);

// dim3 grid_qk(
//     (n + block.x - 1) / block.x,
//     (m + block.y - 1) / block.y);

// dim3 grid_softmax(
//     (n + block.x - 1) / block.x,
//     (m + block.y - 1) / block.y);

// dim3 grid_pv(
//     (d + block.x - 1) / block.x,
//     (m + block.y - 1) / block.y);

// qk_kernel<<<grid_qk, block>>>(Q, K, S, m, n, d);

// softmax_kernel<<<grid_softmax, block>>>(S, P, m, n);

// pv_kernel<<<grid_pv, block>>>(P, V, O, m, n, d);



int main()
{
    int M = 512;
    int N = 512;
    int D = 128;

    size_t sizeQ = M * D * sizeof(float);
    size_t sizeK = N * D * sizeof(float);
    size_t sizeS = M * N * sizeof(float);
    size_t sizeP = M * N * sizeof(float);
    size_t sizeV = N * D * sizeof(float);
    size_t sizeO = M * D * sizeof(float);

    float *h_Q = new float[M * D];
    float *h_K = new float[N * D];
    float *h_S = new float[M * N];
    float *h_P = new float[M * N];
    float *h_V = new float[N * D];
    float *h_O = new float[M * D];


    srand(time(NULL));
    for (int i = 0; i < M * D; i++)
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    for (int i = 0; i < D * N; i++)
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    for (int i = 0; i < M * N; i++)
        h_S[i] = 0.0f;

    for (int i = 0; i < M * N; i++)
        h_P[i] = 0.0f;

    for (int i = 0; i < N * D; i++)
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    for (int i = 0; i < M * D; i++)
        h_O[i] = 0.0f;





    float *d_Q, *d_K, *d_S, *d_P, *d_V, *d_O;

    cudaMalloc(&d_Q, sizeQ);
    cudaMalloc(&d_K, sizeK);
    cudaMalloc(&d_S, sizeS);
    cudaMalloc(&d_P, sizeP);
    cudaMalloc(&d_V, sizeV);
    cudaMalloc(&d_O, sizeO);


    cudaMemcpy(d_Q, h_Q, sizeQ, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, sizeK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, sizeV, cudaMemcpyHostToDevice);

    dim3 block(16, 16);

    dim3 grid_qk(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    dim3 grid_softmax(
        (N + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    dim3 grid_pv(
        (D + block.x - 1) / block.x,
        (M + block.y - 1) / block.y);

    qk_kernel<<<grid_qk, block>>>(d_Q, d_K, d_S, M, N, D);
    
    cudaDeviceSynchronize();

    cudaMemcpy(
        h_S,
        d_S,
        sizeS,
        cudaMemcpyDeviceToHost);
    
    softmax_kernel<<<grid_softmax, block>>>(d_S, d_P, M, N);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_P,
        d_P,
        sizeP,
        cudaMemcpyDeviceToHost);

    pv_kernel<<<grid_pv, block>>>(d_P, d_V, d_O, M, N, D);

    cudaDeviceSynchronize();

    cudaMemcpy(
        h_O,
        d_O,
        sizeO,
        cudaMemcpyDeviceToHost);

    std::cout << "O[0] = "
              << h_O[0]
              << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_S);
    cudaFree(d_V);
    cudaFree(d_P);
    cudaFree(d_O);

    delete[] h_Q;
    delete[] h_K;
    delete[] h_S;
    delete[] h_V;
    delete[] h_P;
    delete[] h_O;

    return 0;
}