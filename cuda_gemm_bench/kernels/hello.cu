// test_ncu.cu

#include <stdio.h>

__global__ void hello() {
    printf("hello cuda\n");
}

int main() {
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}