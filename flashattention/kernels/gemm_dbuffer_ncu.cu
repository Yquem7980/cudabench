// #include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
// #include "flashattention_kernels.cuh"
#include <cstdlib>
#include <ctime>
#include <cmath> 
#include <algorithm> 
#include <iostream>
#include <vector>
#include <random>

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

// QK^T: (M,D) @ (N,D)^T -> (M,N)
void cpu_qk(const float* Q, const float* K, float* S, int M, int N, int D) {
    float scale = 1/sqrtf((float)D);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += Q[i * D + k] * K[j * D + k];
            }
            S[i * N + j] = sum * scale;
        }
    }
}

// Softmax: 逐行
void cpu_softmax(const float* S, float* P, int M, int N) {
    for (int i = 0; i < M; i++) {
        // 找 max
        float max_val = -1e30f;
        for (int j = 0; j < N; j++) {
            max_val = std::max(max_val, S[i * N + j]);
        }
        
        // exp sum
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += expf(S[i * N + j] - max_val);
        }
        
        // normalize
        for (int j = 0; j < N; j++) {
            P[i * N + j] = expf(S[i * N + j] - max_val) / sum;
        }
    }
}

// PV: (M,N) @ (N,D) -> (M,D)
void cpu_pv(const float* P, const float* V, float* O, int M, int N, int D) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < D; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += P[i * N + k] * V[k * D + j];
            }
            O[i * D + j] = sum;
        }
    }
}

// 完整的 CPU Flash Attention
void cpu_flashattention(
    const float* Q, const float* K, const float* V,
    float* O, int M, int N, int D
) {
    std::vector<float> S(M * N);
    std::vector<float> P(M * N);
    
    cpu_qk(Q, K, S.data(), M, N, D);
    cpu_softmax(S.data(), P.data(), M, N);
    cpu_pv(P.data(), V, O, M, N, D);
}

// ==================== 验证工具 ====================

// 比较两个数组，返回最大相对误差
float check_result(const float* ref, const float* test, int size, float tol = 1e-3f) {
    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    int err_count = 0;
    
    for (int i = 0; i < size; i++) {
        float diff = fabsf(ref[i] - test[i]);
        float rel_diff;
        //float rel_diff = diff / (fabsf(ref[i]) + 1e-6f);
        float abs_diff = fabsf(ref[i] - test[i]);
        if (fabsf(ref[i]) > 1e-3f)
            rel_diff = abs_diff / fabsf(ref[i]);
        else
            rel_diff = abs_diff;

        max_err = std::max(max_err, diff);
        max_rel_err = std::max(max_rel_err, rel_diff);
        
        if (rel_diff > tol) {
            if (err_count < 5) {  // 只打印前5个错误
                printf("  Error at [%d]: ref=%.6f, test=%.6f, rel_err=%.6f\n",
                       i, ref[i], test[i], rel_diff);
            }
            err_count++;
        }
    }
    
    if (err_count > 0) {
        printf("  Total errors: %d / %d (%.2f%%)\n", 
               err_count, size, 100.0f * err_count / size);
    } else {
        printf("  All passed! max_rel_err=%.6f\n", max_rel_err);
    }
    
    return max_rel_err;
}


// ==================== CUDA Kernels ====================
template <const int BM, const int BN, const int BD, const int TM, const int TN>
__global__ void qk_kernel(
    float* Q,
    float* K,
    float* S,
    int M,
    int N,
    int D)
{   
    float scale = 1/sqrtf((float)D);
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BM / TM;
    const int block_col_thread = BN / TN;    
    const int thread_num = block_row_thread * block_col_thread;

    int tx = threadIdx.x % block_col_thread * TN;
    int ty = threadIdx.x / block_col_thread * TM;
    
    __shared__ float Qs[2][BD * BM];
    __shared__ float Ks[2][BD * BN];

    const int ldg_q_num = BM * BD / thread_num / 4;
    const int ldg_k_num = BN * BD / thread_num / 4;

    int q_tile_row = threadIdx.x / (BD / 4);
    int q_tile_col = threadIdx.x % (BD / 4) * 4;
    int q_tile_stride = BM / ldg_q_num;

    int k_tile_row = threadIdx.x / (BD / 4);
    int k_tile_col = threadIdx.x % (BD / 4) * 4;
    int k_tile_stride = BN / ldg_k_num;

    float accum[TM][TN] = {0.};

    float ldg_q_reg[4 * ldg_q_num] = {0.};
    float ldg_k_reg[4 * ldg_k_num] = {0.};

    float q_frag[2][TM];
    float k_frag[2][TN];

    Q = &Q[by * BM * D];
    K = &K[bx * BN * D];
    S = &S[by * BM * N + bx * BN];

    for (int i = 0; i < BM; i += q_tile_stride) {
        int ldg_index = i / q_tile_stride * 4;
        FETCH_FLOAT4(ldg_q_reg[ldg_index]) = FETCH_FLOAT4(Q[OFFSET(q_tile_row + i, q_tile_col, D)]);
        Qs[0][OFFSET(q_tile_col, i + q_tile_row, BM)] = ldg_q_reg[ldg_index];
        Qs[0][OFFSET(q_tile_col + 1, i + q_tile_row, BM)] = ldg_q_reg[ldg_index + 1];
        Qs[0][OFFSET(q_tile_col + 2, i + q_tile_row, BM)] = ldg_q_reg[ldg_index + 2];
        Qs[0][OFFSET(q_tile_col + 3, i + q_tile_row, BM)] = ldg_q_reg[ldg_index + 3];
    }

    for (int i = 0; i < BN; i += k_tile_stride) {
        int ldg_index = i / k_tile_stride * 4;
        FETCH_FLOAT4(ldg_k_reg[ldg_index]) = FETCH_FLOAT4(K[OFFSET(k_tile_row + i, k_tile_col, D)]);
        Ks[0][OFFSET(k_tile_col, i + k_tile_row, BN)] = ldg_k_reg[ldg_index];
        Ks[0][OFFSET(k_tile_col + 1, i + k_tile_row, BN)] = ldg_k_reg[ldg_index + 1];
        Ks[0][OFFSET(k_tile_col + 2, i + k_tile_row, BN)] = ldg_k_reg[ldg_index + 2];
        Ks[0][OFFSET(k_tile_col + 3, i + k_tile_row, BN)] = ldg_k_reg[ldg_index + 3];
    }
    __syncthreads();

    for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(q_frag[0][m]) = FETCH_FLOAT4(Qs[0][OFFSET(0, ty + m, BM)]);
    }
 
    for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(k_frag[0][n]) = FETCH_FLOAT4(Ks[0][OFFSET(0, tx + n, BN)]);
    }
    int write_index = 1;
    int load_index;
    int d = 0;
    do{
        d += BD;
        if(d < D){
            for (int i = 0; i < BM; i += q_tile_stride)
            {
                int ldg_index = i / q_tile_stride * 4;
                FETCH_FLOAT4(ldg_q_reg[ldg_index]) = FETCH_FLOAT4(Q[OFFSET(q_tile_row + i, d + q_tile_col, D)]);
            }
            for (int i = 0; i < BN; i += k_tile_stride)
            {
                int ldg_index = i / k_tile_stride * 4;
                FETCH_FLOAT4(ldg_k_reg[ldg_index]) = FETCH_FLOAT4(K[OFFSET(k_tile_row + i, d + k_tile_col, D)]);
            }
        }

        load_index = write_index ^ 1;
        for(int bd= 0; bd < BD - 1; bd++){
            for(int m = 0; m < TM; m += 4){
                FETCH_FLOAT4(q_frag[(bd + 1)%2][m]) = FETCH_FLOAT4(Qs[load_index][OFFSET(bd + 1, ty + m, BM)]);
            }
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(k_frag[(bd + 1)%2][n]) = FETCH_FLOAT4(Ks[load_index][OFFSET(bd + 1, tx + n, BN)]);
            }
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                accum[m][n] += q_frag[bd % 2][m] * k_frag[bd % 2][n];
                }
            }
        }
        if(d < D){
            for (int i = 0; i < BM; i += q_tile_stride) {
                int ldg_index = i / q_tile_stride * 4;
                Qs[write_index][OFFSET(q_tile_col, i + q_tile_row, BM)] = ldg_q_reg[ldg_index];
                Qs[write_index][OFFSET(q_tile_col + 1, i + q_tile_row, BM)] = ldg_q_reg[ldg_index + 1];
                Qs[write_index][OFFSET(q_tile_col + 2, i + q_tile_row, BM)] = ldg_q_reg[ldg_index + 2];
                Qs[write_index][OFFSET(q_tile_col + 3, i + q_tile_row, BM)] = ldg_q_reg[ldg_index + 3];
            }  
            for (int i = 0; i < BN; i += k_tile_stride) {
                int ldg_index = i / k_tile_stride * 4;
                Ks[write_index][OFFSET(k_tile_col, i + k_tile_row, BN)] = ldg_k_reg[ldg_index];
                Ks[write_index][OFFSET(k_tile_col + 1, i + k_tile_row, BN)] = ldg_k_reg[ldg_index + 1];
                Ks[write_index][OFFSET(k_tile_col + 2, i + k_tile_row, BN)] = ldg_k_reg[ldg_index + 2];
                Ks[write_index][OFFSET(k_tile_col + 3, i + k_tile_row, BN)] = ldg_k_reg[ldg_index + 3];
            }
            __syncthreads();
            for (int m = 0; m < TM; m += 4) {           
                FETCH_FLOAT4(q_frag[0][m]) = FETCH_FLOAT4(Qs[write_index][OFFSET(0, ty + m, BM)]);
            }

            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(k_frag[0][n]) = FETCH_FLOAT4(Ks[write_index][OFFSET(0, tx + n, BN)]);
            }


            write_index ^= 1;  
        }
        for (int m = 0; m < TM; m++) {
            for (int n = 0; n < TN; n++) {
                accum[m][n] += q_frag[(BD - 1) % 2][m] * k_frag[(BD - 1) % 2][n];
            }
        }
    }while (d < D);

    for (int m = 0; m < TM; m++) {
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(S[OFFSET(ty + m, tx + n, N)]);
            ctmp.x = accum[m][n] * scale;
            ctmp.y = accum[m][n + 1] * scale;
            ctmp.z = accum[m][n + 2] * scale;
            ctmp.w = accum[m][n + 3] * scale;
            FETCH_FLOAT4(S[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}

__global__ void softmax_kernel_v2(
    const float* S,
    float* P,
    int m,
    int n)
{
    int tid = threadIdx.x;
    int row = blockIdx.x;
    if (row >= m) return;
    
    extern __shared__ float smem[];

    // 1.parallel max
    float local_max = -1e30f;

    for (int j = tid; j < n; j += blockDim.x) {
        local_max = max(local_max, S[row * n + j]);
    }
    smem[tid] = local_max;
    __syncthreads();

    // reduction max
    for(int stride = blockDim.x/2; stride > 0; stride>>=1){
        if(tid < stride){
            smem[tid] = max(smem[tid],smem[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = smem[0];
    
    // 2. parallel exp sum
    float local_sum  = 0.0f;

    for (int j = tid; j < n; j += blockDim.x) {
        float val = expf(S[row * n + j] - max_val);
        P[row * n + j] = val;
        local_sum  += val;
    }

    smem[tid] = local_sum;
    __syncthreads();

    // reduction sum
    for(int stride = blockDim.x/2; stride > 0; stride>>=1){
        if(tid < stride){
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float sum = smem[0];

    // 3. normalize
    for (int j = tid; j < n; j += blockDim.x) {
        P[row * n + j] /=  sum;
    }
}

template <const int BM, const int BD, const int BN, const int TM, const int TD>
__global__ void pv_kernel(
    const float* P,
    const float* V,
    float* O,
    int M,
    int N,
    int D)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x; // d
    // int row = blockIdx.y * blockDim.y + threadIdx.y; // m

    int block_row_thread = BM / TM;
    int block_col_thread = BD / TD;
    int thread_num = block_row_thread * block_col_thread;

    int tx = threadIdx.x % block_col_thread * TD;
    int ty = threadIdx.x / block_col_thread * TM;

    __shared__ float Ps[BN * BM];
    __shared__ float Vs[BN * BD];

    P = &P[by * BM * N];
    V = &V[bx * BD];
    O = &O[by * BM * D + bx * BD];

    int p_tile_row = threadIdx.x / BD;
    int p_tile_col = threadIdx.x % BN;
    int p_tile_stride = thread_num / BM;

    int v_tile_row = threadIdx.x / BD;
    int v_tile_col = threadIdx.x % BD;
    int v_tile_stride = thread_num / BD;

    float tmp[TM][TD] = {0.};

    for (int n = 0; n < N; n += BN)
    {
        for (int i = 0; i < BM; i += p_tile_stride)
        {
            Ps[(p_tile_row + i) * BN + p_tile_col] = P[(p_tile_row + i) * N + p_tile_col];
        }
        for (int i = 0; i < BN; i += v_tile_stride)
        {
            Vs[(v_tile_row + i) * BD + v_tile_col] = V[(v_tile_row + i) * D + v_tile_col];
        }
        __syncthreads();//wait data from global to shared
        P += BN;
        V += BN * D;

        for(int i = 0; i < BN; i++){
            for(int j = 0 ; j < TM; j++){
                for(int l = 0; l < TD; l++){
                    tmp[j][l] += Ps[(ty + j) * BN + i] * Vs[i * BD + tx + l];
                }
            }
        }
         __syncthreads();
    }
    for (int j = 0; j < TM; j++) {
        for (int l = 0; l < TD; l++)
            O[(ty + j) * D + tx + l] = tmp[j][l] ;
    }
        
    
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
    float *h_O_gpu = new float[M * D];   // GPU 结果
    float *h_O_cpu = new float[M * D];   // CPU 参考结果   

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
        h_O_gpu[i] = 0.0f;
    for (int i = 0; i < M * D; i++)
        h_O_cpu[i] = 0.0f;

    // ========== CPU 参考计算 ==========
    printf("Running CPU reference...\n");
    cpu_flashattention(h_Q, h_K, h_V, h_O_cpu, M, N, D);
    printf("CPU done. O_cpu[0] = %.6f\n\n", h_O_cpu[0]);

    // ========== GPU 计算 ==========
    printf("Running GPU kernels...\n");


    float *d_Q, *d_K, *d_S, *d_P, *d_V, *d_O;

    CHECK_CUDA(cudaMalloc(&d_Q, sizeQ));
    CHECK_CUDA(cudaMalloc(&d_K, sizeK));
    CHECK_CUDA(cudaMalloc(&d_S, sizeS));
    CHECK_CUDA(cudaMalloc(&d_P, sizeP));
    CHECK_CUDA(cudaMalloc(&d_V, sizeV));
    CHECK_CUDA(cudaMalloc(&d_O, sizeO));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, sizeQ, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, sizeK, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, sizeV, cudaMemcpyHostToDevice));

    constexpr int BM = 16;
    constexpr int BD = 16;
    constexpr int BN = 16;
    constexpr int TM = 2;
    constexpr int TD = 2;
    dim3 block(16, 16);

    dim3 blockqk(64);
    dim3 blocksoftmax(256);
    dim3 blockkv((BM/TM) * (BD/TD));
    size_t smem_size = blocksoftmax.x * sizeof(float);

    dim3 grid_qk(CEIL_DIV(N, 32), CEIL_DIV(M, 32));

    dim3 grid_softmax(M);

    dim3 grid_pv(
        CEIL_DIV(D, BD),CEIL_DIV(M, BM));

    // Step 1: QK^T
    qk_kernel<32,32,8,4,4><<<grid_qk, blockqk>>>(d_Q, d_K, d_S, M, N, D);
    CHECK_CUDA(cudaGetLastError());    
    CHECK_CUDA(cudaDeviceSynchronize());
    // CHECK_CUDA(cudaMemcpy(h_S, d_S, sizeS, cudaMemcpyDeviceToHost));

    // Step 2: Softmax
    softmax_kernel_v2<<<grid_softmax, blocksoftmax, smem_size>>>(d_S, d_P, M, N);
    CHECK_CUDA(cudaGetLastError());    
    CHECK_CUDA(cudaDeviceSynchronize());
    // CHECK_CUDA(cudaMemcpy(h_P, d_P, sizeP, cudaMemcpyDeviceToHost));

    // Step 3: PV
    pv_kernel<BM, BD, BN, TM, TD><<<grid_pv, blockkv>>>(d_P, d_V, d_O, M, N, D);
    CHECK_CUDA(cudaGetLastError());    
    CHECK_CUDA(cudaDeviceSynchronize());

    // 拷贝结果    
    CHECK_CUDA(cudaMemcpy(h_O_gpu, d_O, sizeO, cudaMemcpyDeviceToHost));
    printf("GPU done. O_gpu[0] = %.6f\n\n", h_O_gpu[0]);


        // ========== 验证 ==========
    printf("Checking QK^T intermediate...\n");
    std::vector<float> h_S_gpu(M * N);
    CHECK_CUDA(cudaMemcpy(h_S_gpu.data(), d_S, sizeS, cudaMemcpyDeviceToHost));

    std::vector<float> h_S_cpu(M * N);
    cpu_qk(h_Q, h_K, h_S_cpu.data(), M, N, D);
    check_result(h_S_cpu.data(), h_S_gpu.data(), M * N, 1e-4f);

    printf("\nChecking Softmax intermediate...\n");
    std::vector<float> h_P_gpu(M * N);
    CHECK_CUDA(cudaMemcpy(h_P_gpu.data(), d_P, sizeP, cudaMemcpyDeviceToHost));

    std::vector<float> h_P_cpu(M * N);
    cpu_softmax(h_S_cpu.data(), h_P_cpu.data(), M, N);
    check_result(h_P_cpu.data(), h_P_gpu.data(), M * N, 1e-4f);

    printf("\nChecking final output O...\n");
    check_result(h_O_cpu, h_O_gpu, M * D, 1e-3f);

    // 释放
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_P));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O));

    delete[] h_Q;
    delete[] h_K;
    delete[] h_S;
    delete[] h_V;
    delete[] h_P;
    delete[] h_O_gpu;
    delete[] h_O_cpu;

    return 0;
}