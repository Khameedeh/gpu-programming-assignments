#ifndef COLOR_THE_MATRIX_H
#define COLOR_THE_MATRIX_H

#include <cuda_runtime.h>

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

/*
 * CUDA kernel for block-based image coloring
 * Each block chooses one random color and all threads in the block use it
 */
__global__ void colorMatrixKernel(Pixel* matrix, int N, unsigned int seed);

/*
 * Host function to launch the CUDA kernel
 */
void colorMatrixCUDA(Pixel* h_matrix, int N, dim3 blocks, dim3 threads);

/*
 * Writes an N×N image using the RGB data in matrix
 * Output format: PPM (P6)
 */
void color(Pixel* matrix, int N);

#endif
