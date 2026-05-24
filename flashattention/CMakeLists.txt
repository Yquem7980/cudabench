cmake_minimum_required(VERSION 3.30)

project(cuda_flashattention_bench LANGUAGES CXX CUDA)

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

add_executable(flashattention
    bench/flashattention_bench.cu
    kernels/naive.cu
#    kernels/shared.cu
#    kernels/tile.cu
#    kernels/vector.cu
#    kernels/doublebuffering.cu
#    kernels/warp.cu
)

target_link_libraries(flashattention
    PRIVATE
    nvbench::main
)

set_target_properties(flashattention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)