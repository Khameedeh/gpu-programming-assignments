#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "rasterization.h"

extern "C" {
    void draw_shape(int *arr, int N);
}

int main(int argc, char* argv[]) {
    int N = 512; // Default size
    int test_case = 0; // Default: draw all shapes
    int block_size = 16; // Default block size for 2D kernels

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        test_case = atoi(argv[2]);
    }
    if (argc > 3) {
        block_size = atoi(argv[3]);
    }

    printf("Problem 3: Parallel Rasterization\n");
    printf("Image size: %dx%d\n", N, N);
    printf("Block size: %dx%d\n", block_size, block_size);

    // Allocate pixel array
    int* pixels = (int*)malloc(N * N * sizeof(int));
    if (!pixels) {
        perror("Failed to allocate pixel array");
        return 1;
    }

    // Clear pixel array
    memset(pixels, 0, N * N * sizeof(int));

    // Configure CUDA blocks and threads
    dim3 blocks((N + block_size - 1) / block_size, (N + block_size - 1) / block_size);
    dim3 threads(block_size, block_size);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time_ms = 0.0f;

    cudaEventRecord(start);

    switch (test_case) {
        case 0:
            // Draw all shapes
            printf("Test case 0: Drawing all shapes (line, circle, ellipse)\n");

            // Draw a diagonal line
            drawLineCUDA(pixels, N, 50, 50, N-50, N-50, 3.0f, blocks, threads);

            // Draw a circle
            drawCircleCUDA(pixels, N, N/2, N/2, N/4, 2.0f, blocks, threads);

            // Draw an ellipse
            drawEllipseCUDA(pixels, N, N/4, 3*N/4, N/6, N/8, 2.0f, blocks, threads);
            break;

        case 1:
            // Draw only line
            printf("Test case 1: Drawing line\n");
            drawLineCUDA(pixels, N, 100, 100, N-100, N-100, 5.0f, blocks, threads);
            break;

        case 2:
            // Draw only circle
            printf("Test case 2: Drawing circle\n");
            drawCircleCUDA(pixels, N, N/2, N/2, N/3, 3.0f, blocks, threads);
            break;

        case 3:
            // Draw only ellipse
            printf("Test case 3: Drawing ellipse\n");
            drawEllipseCUDA(pixels, N, N/2, N/2, N/3, N/5, 3.0f, blocks, threads);
            break;

        case 4:
            // Performance test: draw multiple circles
            printf("Test case 4: Performance test - multiple circles\n");
            for (int i = 0; i < 10; i++) {
                int cx = N/2 + (i-5) * 30;
                int cy = N/2 + (i-5) * 20;
                int r = 20 + i * 5;
                drawCircleCUDA(pixels, N, cx, cy, r, 1.0f, blocks, threads);
            }
            break;

        default:
            printf("Invalid test case. Use 0-4\n");
            free(pixels);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return 1;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    printf("CUDA drawing time: %.3f ms\n", time_ms);

    // Generate output image
    draw_shape(pixels, N);

    printf("Image saved as output.ppm\n");

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(pixels);

    return 0;
}
