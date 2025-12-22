#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "color_the_matrix.h"

int main(int argc, char* argv[]) {
    int N = 512; // Default size
    int test_case = 0; // Default: 1 thread per block

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        test_case = atoi(argv[2]);
    }

    printf("Problem 1: Block-Based Image Coloring\n");
    printf("Image size: %dx%d\n", N, N);

    // Allocate host memory
    Pixel* h_matrix = (Pixel*)malloc(N * N * sizeof(Pixel));
    if (!h_matrix) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Initialize matrix to black
    memset(h_matrix, 0, N * N * sizeof(Pixel));

    dim3 blocks, threads;

    switch (test_case) {
        case 0:
            // Test case 0: 1 thread per block, N×N blocks (simple setup)
            printf("Test case 0: 1 thread per block, %dx%d blocks\n", N, N);
            blocks = dim3(N, N);
            threads = dim3(1, 1);
            break;

        case 1:
            // Test case 1: 16x16 threads per block
            printf("Test case 1: 16x16 threads per block\n");
            blocks = dim3((N + 15) / 16, (N + 15) / 16);
            threads = dim3(16, 16);
            break;

        case 2:
            // Test case 2: 32x32 threads per block
            printf("Test case 2: 32x32 threads per block\n");
            blocks = dim3((N + 31) / 32, (N + 31) / 32);
            threads = dim3(32, 32);
            break;

        case 3:
            // Test case 3: 8x8 threads per block
            printf("Test case 3: 8x8 threads per block\n");
            blocks = dim3((N + 7) / 8, (N + 7) / 8);
            threads = dim3(8, 8);
            break;

        default:
            printf("Invalid test case. Using default (case 0)\n");
            blocks = dim3(N, N);
            threads = dim3(1, 1);
            break;
    }

    printf("Blocks: (%d, %d), Threads per block: (%d, %d)\n",
           blocks.x, blocks.y, threads.x, threads.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Launch CUDA kernel
    colorMatrixCUDA(h_matrix, N, blocks, threads);

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Save the image
    char filename[256];
    sprintf(filename, "output_p1_test%d.ppm", test_case);
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Failed to open output file");
        free(h_matrix);
        return 1;
    }

    // PPM header
    fprintf(fp, "P6\n%d %d\n255\n", N, N);
    // Write pixel data
    fwrite(h_matrix, sizeof(Pixel), N * N, fp);
    fclose(fp);

    printf("Image saved as %s\n", filename);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_matrix);

    return 0;
}
