#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "masked_computation.h"

// Function to initialize array with random values
void initializeArray(float* arr, int N) {
    for (int i = 0; i < N; i++) {
        arr[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

// Function to verify results
int verifyResults(float* A, float* B, int N) {
    for (int i = 0; i < N; i++) {
        float expected = (i % 2 == 0) ? A[i] * 10.0f : A[i];
        if (fabs(B[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: expected %f, got %f\n", i, expected, B[i]);
            return 0;
        }
    }
    return 1;
}

int main(int argc, char* argv[]) {
    int N = 1000000; // Default size: 1M elements
    int test_case = 0; // Default: baseline implementation
    int block_size = 256; // Default block size

    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2) {
        test_case = atoi(argv[2]);
    }
    if (argc > 3) {
        block_size = atoi(argv[3]);
    }

    printf("Problem 2: Masked Computation & Divergence Analysis\n");
    printf("Array size: %d elements\n", N);

    // Allocate host memory
    float* h_A = (float*)malloc(N * sizeof(float));
    float* h_B = (float*)malloc(N * sizeof(float));

    if (!h_A || !h_B) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }

    // Initialize input array
    srand(time(NULL));
    initializeArray(h_A, N);

    float time_ms = 0.0f;

    if (test_case == 0) {
        // Baseline implementation with if/else (warp divergence)
        printf("Test case 0: Baseline implementation with if/else (warp divergence)\n");
        printf("Block size: %d\n", block_size);

        dim3 blocks((N + block_size - 1) / block_size);
        dim3 threads(block_size);

        runBaseline(h_A, h_B, N, blocks, threads, &time_ms);

        printf("Kernel execution time: %.3f ms\n", time_ms);

        // Verify results
        if (verifyResults(h_A, h_B, N)) {
            printf("Results verified successfully\n");
        } else {
            printf("Results verification failed\n");
        }

    } else if (test_case == 1) {
        // Optimized implementation without warp divergence
        printf("Test case 1: Optimized implementation without warp divergence\n");
        printf("Block size: %d\n", block_size);

        // Calculate dimensions for even and odd kernels
        int half_N = (N + 1) / 2; // Number of even indices (rounded up)
        int odd_N = N / 2;        // Number of odd indices

        dim3 blocks_even((half_N + block_size - 1) / block_size);
        dim3 threads_even(block_size);
        dim3 blocks_odd((odd_N + block_size - 1) / block_size);
        dim3 threads_odd(block_size);

        printf("Even indices kernel: %d blocks, %d threads/block\n", blocks_even.x, threads_even.x);
        printf("Odd indices kernel: %d blocks, %d threads/block\n", blocks_odd.x, threads_odd.x);

        runOptimized(h_A, h_B, N, blocks_even, threads_even, blocks_odd, threads_odd, &time_ms);

        printf("Total kernel execution time: %.3f ms\n", time_ms);

        // Verify results
        if (verifyResults(h_A, h_B, N)) {
            printf("Results verified successfully\n");
        } else {
            printf("Results verification failed\n");
        }

    } else {
        printf("Invalid test case. Use 0 for baseline or 1 for optimized\n");
        free(h_A);
        free(h_B);
        return 1;
    }

    // Cleanup
    free(h_A);
    free(h_B);

    return 0;
}
