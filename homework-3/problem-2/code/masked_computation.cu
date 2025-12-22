#include <cuda_runtime.h>
#include "masked_computation.h"

// Baseline kernel with if/else - causes warp divergence
__global__ void maskedComputationBaseline(float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (i % 2 == 0) {
            // Even indices: multiply by 10
            B[i] = A[i] * 10.0f;
        } else {
            // Odd indices: copy as is
            B[i] = A[i];
        }
    }
}

// Optimized kernel for even indices only
__global__ void maskedComputationEven(float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Map to even indices only: 0, 2, 4, 6, ...
    int actual_i = i * 2;

    if (actual_i < N) {
        B[actual_i] = A[actual_i] * 10.0f;
    }
}

// Optimized kernel for odd indices only
__global__ void maskedComputationOdd(float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Map to odd indices only: 1, 3, 5, 7, ...
    int actual_i = i * 2 + 1;

    if (actual_i < N) {
        B[actual_i] = A[actual_i];
    }
}

// Host function for baseline implementation
void runBaseline(float* h_A, float* h_B, int N, dim3 blocks, dim3 threads, float* time_ms) {
    float *d_A, *d_B;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    // Copy input to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    cudaEventRecord(start);
    maskedComputationBaseline<<<blocks, threads>>>(d_A, d_B, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get elapsed time
    cudaEventElapsedTime(time_ms, start, stop);

    // Copy result back
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
}

// Host function for optimized implementation
void runOptimized(float* h_A, float* h_B, int N, dim3 blocks_even, dim3 threads_even, dim3 blocks_odd, dim3 threads_odd, float* time_ms) {
    float *d_A, *d_B;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    // Copy input to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernels
    cudaEventRecord(start);
    maskedComputationEven<<<blocks_even, threads_even>>>(d_A, d_B, N);
    maskedComputationOdd<<<blocks_odd, threads_odd>>>(d_A, d_B, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get elapsed time
    cudaEventElapsedTime(time_ms, start, stop);

    // Copy result back
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
}
