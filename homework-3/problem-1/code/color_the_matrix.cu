#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "color_the_matrix.h"

// CUDA kernel for block-based image coloring
__global__ void colorMatrixKernel(Pixel* matrix, int N, unsigned int seed) {
    // Calculate global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check bounds
    if (x >= N || y >= N) return;

    // Each block gets a unique color based on block index
    // Use blockIdx.x + blockIdx.y * gridDim.x as seed for this block
    unsigned int block_seed = seed + blockIdx.x + blockIdx.y * gridDim.x;

    // Generate random color for this block (same for all threads in block)
    // Simple LCG random number generator
    block_seed = (block_seed * 1103515245 + 12345) % (1U << 31);
    unsigned char r = (block_seed >> 16) & 0xFF;
    block_seed = (block_seed * 1103515245 + 12345) % (1U << 31);
    unsigned char g = (block_seed >> 16) & 0xFF;
    block_seed = (block_seed * 1103515245 + 12345) % (1U << 31);
    unsigned char b = (block_seed >> 16) & 0xFF;

    // Set pixel color
    int idx = y * N + x;
    matrix[idx].r = r;
    matrix[idx].g = g;
    matrix[idx].b = b;
}

// Host function to launch the CUDA kernel
void colorMatrixCUDA(Pixel* h_matrix, int N, dim3 blocks, dim3 threads) {
    Pixel* d_matrix;
    size_t size = N * N * sizeof(Pixel);

    // Allocate device memory
    cudaMalloc((void**)&d_matrix, size);

    // Copy input data to device (if needed)
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    // Launch kernel with current time as seed for randomness
    unsigned int seed = (unsigned int)time(NULL);
    colorMatrixKernel<<<blocks, threads>>>(d_matrix, N, seed);

    // Copy result back to host
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_matrix);
}

void color(Pixel* matrix, int N)
{
    FILE* fp = fopen("output.ppm", "wb");
    if (!fp) {
        perror("Failed to open output file");
        return;
    }

    // PPM header
    fprintf(fp, "P6\n%d %d\n255\n", N, N);

    // Write pixel data
    fwrite(matrix, sizeof(Pixel), N * N, fp);

    fclose(fp);
}
