cmake_minimum_required(VERSION 3.30)

project(cuda_gemm_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

############################################################
# CUDA ARCH
############################################################

set(CMAKE_CUDA_ARCHITECTURES 89)

############################################################
# Find NVBench
############################################################

add_subdirectory(../nvbench build_nvbench)

############################################################
# Include Directories
############################################################

include_directories(include)

############################################################
# GEMM Benchmark
############################################################

add_executable(gemm_bench
    bench/gemm_bench.cu
    kernels/naive.cu
    kernels/shared.cu
    kernels/tile.cu
    kernels/vector.cu
    kernels/doublebuffering.cu
    kernels/warp.cu
)

target_link_libraries(gemm_bench
    PRIVATE
    nvbench::main
)

set_target_properties(gemm_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)