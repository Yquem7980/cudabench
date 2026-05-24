#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include "flashattention_kernels.cuh"

#define BLOCK_SIZE 32


// __global__ void naive_kernel(const float* Q, const float* K, const float* V, float* S, float* P, float* O, int m, int n, int d ) {

//     int gx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局x
//     int gy = blockIdx.y * blockDim.y + threadIdx.y;  // 全局y   
//     if (gx >= n || gy >= m) return; 

//     float tmp = 0.0f;
//     for (int k = 0; k < d; k++) {
//       tmp += Q[gy * d + k] * K[gx * d + k];  // 两次全局内存访问
//     }
//     S[gy * n + gx] = tmp;
    
//     // 1. 找当前行的最大值
//     float max_val = -1e30f;
//     for (int j = 0; j < n; j++) {
//         float val = S[gy * n + j];
//         if (val > max_val) max_val = val;
//     }

//     // 2. 计算 exp(x - max) 并累加求和
//     float exp_val = expf(S[gy * n + gx] - max_val);
//     float sum = 0.0f;
//     for (int j = 0; j < n; j++) {
//         sum += expf(S[gy * n + j] - max_val);
//     }

//     // 3. 归一化并写入结果
//     P[gy * n + gx] = exp_val / sum;

//     tmp = 0.0f;
//     for (int k = 0; k < d; k++) {
//       tmp += P[gy * n + k] * V[k * d + gx];  // 两次全局内存访问
//     }
//     O[gy * d + gx] = tmp;
// }

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



void launch_naive_flashattention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int m,
    int n,
    int d,
    cudaStream_t stream)
{
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    dim3 blocks(
        (n + threads.x - 1) / threads.x,
        (m + threads.y - 1) / threads.y);

    naive_kernel<<<blocks, threads, 0, stream>>>(
        Q,
        K,
        V,
        O,
        m,
        n,
        d);
}