# Homework 4: Image Processing Pipeline using CUDA
**Prepared & Supported by:** Sina Hakimzadeh  
**Due Date:** january 2, 2025

## Problem 2: Implementing Grayscale and Edge Detection Algorithms in CUDA

### Objective
In this homework, you will implement a CUDA-based image processing pipeline. You will create two main kernels:
1.  **Grayscale Conversion:** Convert a color image (RGB format) to grayscale.
2.  **Edge Detection:** Apply Sobel edge detection to the grayscale image.

You will then process multiple images in sequence using this pipeline, measuring the time taken for each step and evaluating the speedup of the CUDA implementation compared to a serial CPU implementation.



### Problem Description

#### 1. Grayscale Conversion (CUDA Kernel 1)
Implement a CUDA kernel that converts an image from RGB format to grayscale using the following formula:

$$Grayscale(i, j) = 0.299 \times R(i, j) + 0.587 \times G(i, j) + 0.114 \times B(i, j)$$

*Note: Each pixel in the resulting grayscale image will be a single intensity value.*

#### 2. Edge Detection (CUDA Kernel 2)
Implement the **Sobel edge detection operator** using two 3x3 kernels for detecting horizontal and vertical edges. The Sobel operator consists of the following kernels:

**Horizontal edge kernel ($G_x$):**
$$
G_x = \begin{bmatrix} 
-1 & 0 & 1 \\ 
-2 & 0 & 2 \\ 
-1 & 0 & 1 
\end{bmatrix}
$$

**Vertical edge kernel ($G_y$):**
$$
G_y = \begin{bmatrix} 
-1 & -2 & -1 \\ 
0 & 0 & 0 \\ 
1 & 2 & 1 
\end{bmatrix}
$$

**Algorithm:**
1.  Convolve both kernels with the grayscale image to compute the gradients in the horizontal ($G_x$) and vertical ($G_y$) directions.
2.  Calculate the magnitude of the gradient at each pixel using the formula:
    $$G = \sqrt{G_x^2 + G_y^2}$$
3.  Apply a threshold to classify pixels as edges.



### Tasks

#### CUDA Implementation
* **Kernels:** You must write two separate kernels:
    * `Kernel 1`: RGB to Grayscale conversion.
    * `Kernel 2`: Sobel edge detection.
* **Optimization:** * Use **shared memory** where possible to optimize performance (especially for the convolution stencil).
    * Ensure memory access patterns are optimized for **coalesced memory access**.

#### Multi-Image Pipeline
* Your code should be capable of processing multiple images.
* Implement a function that can process a sequence of images and output the results for each one.
* The images should be processed in parallel on the GPU, and the results should be stored back in memory.

#### Performance Evaluation
* Measure the time taken for each CUDA kernel (grayscale conversion and edge detection).
* Compare the performance of the CUDA implementation against a CPU-based serial implementation.
* Report the **speedup** of the CUDA implementation compared to the CPU version.
* Use Nsight System in order to evaluate the pipeline performance and behavior.

#### Documentation
* Write clear and well-structured documentation explaining how your CUDA code works.
* Include instructions on how to compile and run your code.
* Include performance graphs comparing the CPU and CUDA implementations for different image sizes.



### Deliverables

#### 1. Code
Submit a ZIP file containing:
* `main.cu`: The main program where the kernels are launched.
* `grayscale_kernel.cu`: The CUDA kernel for grayscale conversion.
* `sobel_kernel.cu`: The CUDA kernel for Sobel edge detection.
* `Makefile` or `CMakeLists.txt`: To compile your project.
* Any additional utility files or header files.

#### 2. Report
A PDF report that includes:
* A description of your approach.
* Performance comparison between the CPU and CUDA implementations(ploted).
* Performance graphs.
* Any challenges you faced and how you solved them.
* Instructions on running your code.


## Ethics & Academic Integrity

This homework must reflect **your own work**. While discussions with classmates about general concepts are encouraged, all submitted code, scripts, reports, and analysis must be authored individually.

* **Do not copy** solutions, scripts, or reports from other students, online sources, or prior years.
* **Do not share** your own completed solutions with others before the submission deadline.
* **Always cite** external sources (papers, documentation, tutorials) if you use them to inform your work.
* **Profiling or performance results must be your own.** Running your program and collecting data on your own machine is part of the assignment; submitting fabricated or borrowed results is considered misconduct.

Violations will be treated as academic dishonesty and handled according to university policy.

> When in doubt: ask questions, collaborate conceptually, but write and submit your **own independent work**.