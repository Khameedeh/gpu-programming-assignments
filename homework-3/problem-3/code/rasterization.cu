#include <cuda_runtime.h>
#include <math.h>
#include "rasterization.h"

// Device function to check if a point is within bounds
__device__ bool inBounds(int x, int y, int N) {
    return x >= 0 && x < N && y >= 0 && y < N;
}

// CUDA kernel for clearing the pixel array
__global__ void clearPixelsKernel(int* pixels, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < N) {
        pixels[y * N + x] = 0;
    }
}

// CUDA kernel for drawing a line using Bresenham's algorithm
__global__ void drawLineKernel(int* pixels, int N, int x1, int y1, int x2, int y2, float thickness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N) return;

    // For each pixel, check if it's close to the line
    // Using distance from point to line formula
    int dx = x2 - x1;
    int dy = y2 - y1;

    // Avoid division by zero
    if (dx == 0 && dy == 0) return;

    // Distance from point (x,y) to line defined by (x1,y1) to (x2,y2)
    float numerator = abs(dy * (x - x1) - dx * (y - y1));
    float denominator = sqrtf((float)(dx * dx + dy * dy));
    float distance = numerator / denominator;

    // Check if pixel is within thickness/2 of the line
    if (distance <= thickness / 2.0f) {
        pixels[y * N + x] = 1;
    }
}

// CUDA kernel for drawing a circle
__global__ void drawCircleKernel(int* pixels, int N, int centerX, int centerY, int radius, float thickness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N) return;

    // Calculate distance from center
    float dx = (float)x - centerX;
    float dy = (float)y - centerY;
    float distance = sqrtf(dx * dx + dy * dy);

    // Check if pixel is on the circle boundary (within thickness)
    float half_thickness = thickness / 2.0f;
    if (distance >= (float)radius - half_thickness && distance <= (float)radius + half_thickness) {
        pixels[y * N + x] = 1;
    }
}

// CUDA kernel for drawing an ellipse
__global__ void drawEllipseKernel(int* pixels, int N, int centerX, int centerY, int radiusX, int radiusY, float thickness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N) return;

    // Calculate normalized coordinates
    float nx = ((float)x - centerX) / (float)radiusX;
    float ny = ((float)y - centerY) / (float)radiusY;

    // Check if point is on ellipse boundary
    float distance = sqrtf(nx * nx + ny * ny);
    float half_thickness = thickness / 2.0f;

    // Adjust thickness based on the ellipse scaling
    float scaled_thickness = half_thickness / fmaxf((float)radiusX, (float)radiusY) * fminf((float)radiusX, (float)radiusY);

    if (fabs(distance - 1.0f) <= scaled_thickness) {
        pixels[y * N + x] = 1;
    }
}

// Host function to clear pixels using CUDA
void clearPixelsCUDA(int* h_pixels, int N, dim3 blocks, dim3 threads) {
    int* d_pixels;
    size_t size = N * N * sizeof(int);

    cudaMalloc((void**)&d_pixels, size);
    cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);

    clearPixelsKernel<<<blocks, threads>>>(d_pixels, N);

    cudaMemcpy(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

// Host function to draw a line using CUDA
void drawLineCUDA(int* h_pixels, int N, int x1, int y1, int x2, int y2, float thickness, dim3 blocks, dim3 threads) {
    int* d_pixels;
    size_t size = N * N * sizeof(int);

    cudaMalloc((void**)&d_pixels, size);
    cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);

    drawLineKernel<<<blocks, threads>>>(d_pixels, N, x1, y1, x2, y2, thickness);

    cudaMemcpy(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

// Host function to draw a circle using CUDA
void drawCircleCUDA(int* h_pixels, int N, int centerX, int centerY, int radius, float thickness, dim3 blocks, dim3 threads) {
    int* d_pixels;
    size_t size = N * N * sizeof(int);

    cudaMalloc((void**)&d_pixels, size);
    cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);

    drawCircleKernel<<<blocks, threads>>>(d_pixels, N, centerX, centerY, radius, thickness);

    cudaMemcpy(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

// Host function to draw an ellipse using CUDA
void drawEllipseCUDA(int* h_pixels, int N, int centerX, int centerY, int radiusX, int radiusY, float thickness, dim3 blocks, dim3 threads) {
    int* d_pixels;
    size_t size = N * N * sizeof(int);

    cudaMalloc((void**)&d_pixels, size);
    cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);

    drawEllipseKernel<<<blocks, threads>>>(d_pixels, N, centerX, centerY, radiusX, radiusY, thickness);

    cudaMemcpy(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

// CPU-based line drawing for comparison
void drawLineCPU(int* pixels, int N, int x1, int y1, int x2, int y2, float thickness) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int steps = max(dx, dy);

    for (int i = 0; i <= steps; i++) {
        float t = (float)i / (float)steps;
        int x = (int)(x1 + t * (x2 - x1) + 0.5f);
        int y = (int)(y1 + t * (y2 - y1) + 0.5f);

        int half_thickness = (int)(thickness / 2.0f + 0.5f);
        for (int dy = -half_thickness; dy <= half_thickness; dy++) {
            for (int dx = -half_thickness; dx <= half_thickness; dx++) {
                int px = x + dx;
                int py = y + dy;
                if (px >= 0 && px < N && py >= 0 && py < N) {
                    pixels[py * N + px] = 1;
                }
            }
        }
    }
}

// CPU-based circle drawing for comparison
void drawCircleCPU(int* pixels, int N, int centerX, int centerY, int radius, float thickness) {
    float half_thickness = thickness / 2.0f;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float dx = (float)x - centerX;
            float dy = (float)y - centerY;
            float distance = sqrtf(dx * dx + dy * dy);

            if (distance >= (float)radius - half_thickness && distance <= (float)radius + half_thickness) {
                pixels[y * N + x] = 1;
            }
        }
    }
}

// CPU-based ellipse drawing for comparison
void drawEllipseCPU(int* pixels, int N, int centerX, int centerY, int radiusX, int radiusY, float thickness) {
    float half_thickness = thickness / 2.0f;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            float nx = ((float)x - centerX) / (float)radiusX;
            float ny = ((float)y - centerY) / (float)radiusY;
            float distance = sqrtf(nx * nx + ny * ny);

            if (fabs(distance - 1.0f) <= half_thickness / fmaxf((float)radiusX, (float)radiusY)) {
                pixels[y * N + x] = 1;
            }
        }
    }
}
