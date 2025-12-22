#ifndef RASTERIZATION_H
#define RASTERIZATION_H

#include <cuda_runtime.h>

// CUDA kernel for drawing a line
__global__ void drawLineKernel(int* pixels, int N, int x1, int y1, int x2, int y2, float thickness);

// CUDA kernel for drawing a circle
__global__ void drawCircleKernel(int* pixels, int N, int centerX, int centerY, int radius, float thickness);

// CUDA kernel for drawing an ellipse
__global__ void drawEllipseKernel(int* pixels, int N, int centerX, int centerY, int radiusX, int radiusY, float thickness);

// Host function to draw a line using CUDA
void drawLineCUDA(int* h_pixels, int N, int x1, int y1, int x2, int y2, float thickness, dim3 blocks, dim3 threads);

// Host function to draw a circle using CUDA
void drawCircleCUDA(int* h_pixels, int N, int centerX, int centerY, int radius, float thickness, dim3 blocks, dim3 threads);

// Host function to draw an ellipse using CUDA
void drawEllipseCUDA(int* h_pixels, int N, int centerX, int centerY, int radiusX, int radiusY, float thickness, dim3 blocks, dim3 threads);

// Host function to clear the pixel array
void clearPixelsCUDA(int* h_pixels, int N, dim3 blocks, dim3 threads);

// Helper functions for CPU-based drawing (for comparison)
void drawLineCPU(int* pixels, int N, int x1, int y1, int x2, int y2, float thickness);
void drawCircleCPU(int* pixels, int N, int centerX, int centerY, int radius, float thickness);
void drawEllipseCPU(int* pixels, int N, int centerX, int centerY, int radiusX, int radiusY, float thickness);

#endif
