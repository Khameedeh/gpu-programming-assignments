# Homework 4: Reduction and Shared Memory Bank Conflicts
**Prepared & Supported by:** Raha Rahmanian  \
**Due Date:** january 2, 2025

## Overview

In this assignment, you will implement a parallel reduction (sum of an array) using CUDA, analyze shared memory bank conflicts on your own GPU, and optimize the kernel to reduce or eliminate those conflicts. In addition, you will gain hands-on experience with **different CUDA memory allocation and access models** and compare their impact on performance.



## Learning Objectives

* Understand block-level parallel reduction
* Use shared memory effectively
* Understand and compare different CUDA memory models
* Observe and reason about shared memory bank conflicts
* Measure and compare GPU performance



## Problem Statement

Given an array of **N single-precision floating-point values**, compute the sum of all elements using CUDA. Your implementation must use a **shared-memory reduction inside each block**. You will implement and compare multiple versions of the solution using different memory allocation and access models.



## Part 1 — GPU Information

Write a small CUDA program that prints the following information about the GPU:

* GPU name
* Compute capability
* Warp size
* Shared memory per block
* Number of shared memory banks (if available)

Include this information in your report and briefly explain how it relates to shared memory behavior and bank conflicts.



## Part 2 — Memory Allocation Models

To gain hands-on experience with CUDA memory models, implement the reduction using **three of the following memory allocation approaches**:

1. **Global Memory (Baseline)**

   * Allocate input and output arrays using `cudaMalloc`
   * Copy data using `cudaMemcpy`
   * Use this version as the baseline for correctness and performance

2. **Pinned (Page-Locked) Host Memory**

   * Allocate host memory using `cudaMallocHost`
   * Compare transfer times and overall runtime with pageable memory

3. **Unified Memory (Managed Memory)**

   * Allocate memory using `cudaMallocManaged`
   * Ensure correct synchronization using `cudaDeviceSynchronize`
   * Discuss ease of programming vs. performance

For each memory model, verify correctness and record performance measurements.



## Part 3 — Naive Reduction Using Global and Shared Memory 

In this part, you will implement **two versions of a block-level reduction** to understand the performance impact of memory hierarchy choices.

### Part 3.1 — Global Memory Reduction (Baseline)

Implement a naive reduction that:

* Reads elements directly from **global (or managed) memory**
* Performs a tree-based reduction **without using shared memory**
* Writes one partial sum per block to global memory

This version is expected to be functionally correct but inefficient. Use it as a **baseline** to measure the cost of repeated global memory accesses.

Verify correctness for all tested memory allocation models before moving on.


### Part 3.2 — Shared Memory Reduction

Implement a block-level reduction that:

* Loads elements from global (or managed) memory into **shared memory**
* Performs a tree-based reduction inside the block using shared memory
* Writes one partial sum per block to global memory

Ensure correctness for all tested memory models. This version will be used for bank conflict analysis and optimization in later parts.


## Part 4 — Memory Access and Bank Conflict Analysis

Using profiling tools (e.g., Nsight Compute or Nsight Systems), analyze and compare the two reduction implementations from Part 3.

Answer the following questions:

### Global Memory Reduction Analysis

* How does the performance compare to the shared memory version?
* What inefficiencies arise due to repeated global memory accesses?
* Which GPU performance metrics highlight these inefficiencies?

### Shared Memory Reduction Analysis

* Do shared memory bank conflicts occur?
* During which reduction steps do they occur?
* Why do they occur on your GPU architecture?

Base your explanation on profiling results and/or detailed reasoning about memory layout, thread indexing, and shared memory bank organization.




## Part 5 — Optimized Reduction

Modify your kernel to reduce or eliminate shared memory bank conflicts. Possible approaches include:

* Padding shared memory arrays
* Changing indexing and access patterns

Verify correctness and profile the optimized kernel for each memory model used.



## Part 6 — Performance Evaluation

Compare **naive vs. optimized** implementations across different memory models:

* Kernel runtime
* Total runtime (including memory transfers)
* Speedup or slowdown

Test with **different block sizes** and summarize your results in tables or plots.



## Deliverables

* CUDA source code:

  * GPU information program
  * Naive reduction kernels
  * Optimized reduction kernels
  * Implementations using different memory allocation models
* Documentation/report including:

  * Explanation of each memory model
  * Bank conflict analysis
  * Performance comparison (tables, plots, discussion)
  * Final conclusions and lessons learned

## Ethics & Academic Integrity

This homework must reflect **your own work**. While discussions with classmates about general concepts are encouraged, all submitted code, scripts, reports, and analysis must be authored individually.

* **Do not copy** solutions, scripts, or reports from other students, online sources, or prior years.
* **Do not share** your own completed solutions with others before the submission deadline.
* **Always cite** external sources (papers, documentation, tutorials) if you use them to inform your work.
* **Profiling or performance results must be your own.** Running your program and collecting data on your own machine is part of the assignment; submitting fabricated or borrowed results is considered misconduct.

Violations will be treated as academic dishonesty and handled according to university policy.

> When in doubt: ask questions, collaborate conceptually, but write and submit your **own independent work**.