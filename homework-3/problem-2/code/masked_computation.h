#ifndef MASKED_COMPUTATION_H
#define MASKED_COMPUTATION_H

#include <cuda_runtime.h>

// Baseline kernel with if/else - causes warp divergence
__global__ void maskedComputationBaseline(float* A, float* B, int N);

// Optimized kernel without warp divergence - separate even/odd handling
__global__ void maskedComputationEven(float* A, float* B, int N);
__global__ void maskedComputationOdd(float* A, float* B, int N);

// Host functions
void runBaseline(float* h_A, float* h_B, int N, dim3 blocks, dim3 threads, float* time_ms);
void runOptimized(float* h_A, float* h_B, int N, dim3 blocks_even, dim3 threads_even, dim3 blocks_odd, dim3 threads_odd, float* time_ms);

#endif
