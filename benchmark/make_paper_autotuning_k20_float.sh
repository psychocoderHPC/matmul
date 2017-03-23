#!/bin/sh
cmake -DMATMUL_ELEMENT_TYPE_DOUBLE=ON -DMATMUL_INDEX_TYPE=int -Dalpaka_DIR=$ALPAKA_ROOT -DMATMUL_BENCHMARK_VERIFY_RESULT=OFF -DCUDA_NVCC_FLAGS="-lineinfo -DGPU_TILE_SIZE=$2 -DGPU_THREADS_NUM=$1 --ptxas-options=-v" -DMATMUL_BENCHMARK_ALPAKA_TILING_KERNEL=ON -DMATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL=OFF -DMATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL=OFF -DMATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA=ON -DMATMUL_BENCHMARK_PRINT_GFLOPS=ON -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -DNDEBUG" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Wno-deprecated-declarations" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_FLAGS="-mtune=native -march=native -Wno-deprecated-declarations" -DALPAKA_CUDA_ARCH="sm_60"  `dirname $0` 2>&1 | tee cmake_log.txt
#cmake -Dalpaka_DIR=$ALPAKA_ROOT -DCUDA_NVCC_FLAGS="-lineinfo -DGPU_ELEM_NUM=$1 --ptxas-options=-v" -DMATMUL_BENCHMARK_ALPAKA_TILING_KERNEL=ON -DMATMUL_BENCHMARK_ALPAKA_OMPNATIVE_KERNEL=OFF -DMATMUL_BENCHMARK_ALPAKA_CUDASDK_KERNEL=OFF -DMATMUL_BENCHMARK_PAR_ALPAKA_ACC_GPU_CUDA=ON -DMATMUL_BENCHMARK_PRINT_GFLOPS=ON -DCMAKE_CXX_FLAGS_RELEASE="-Ofast -DNDEBUG" -DCMAKE_CXX_FLAGS="-mtune=native -march=native -Wno-deprecated-declarations" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_FLAGS="-mtune=native -march=native -Wno-deprecated-declarations" -DALPAKA_CUDA_ARCH="sm_35" -DMATMUL_ELEMENT_TYPE_DOUBLE=OFF `dirname $0` 2>&1 | tee cmake_log.txt
make -j 2>&1 | tee make_log.txt 
