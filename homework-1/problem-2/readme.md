
# HW1-Matrix Multiplication: Profiling & Analysis
**Prepared & Supported by:**  Raha Rahmanian  
**Due date:** 17 October 2025
## Problem Overview
In the previous problem, you already explored how data type, problem size, and choice of algorithm can directly impact computation time. These factors influence how efficiently a program uses the CPU and memory resources, and you saw that different combinations can lead to very different performance results.

In this new problem, we explore the concept of CPU bound and I/O bound programs and examin the influence of other factors on performance. Matrix multiplication is a classic example of a CPU-bound program, because most of its execution time is spent doing arithmetic operations rather than waiting for input or output. Here, you will reuse the program provided earlier [matrix_multiplication.c](./../../../docs/02-Getting-started-with-profiling/matrix_multiplication.c) from the [“getting started with profiling tools”](./../../../docs/02-Getting-started-with-profiling/getting-started-with-profiling.md) and extend your analysis in two directions: first, by comparing CPU-bound versus I/O-bound workloads, and second, by studying the effect of cache coherence on performance.
## Task 1 – Profiling The Program
- Write a bash script or Python script that runs the program multiple times with both modes (CPU mode and I/O mode) and collects profiling data.

- Explain what is happening in each mode:

  -  Why might one of the modes exhibit an I/O bottleneck while the other will not?

  - Is the program CPU-bound or I/O-bound in each case?

> **Note**: All explanations must be backed up with profiling metrics.




## Task 2 – Modifying the Program to Become I/O-Bound
- Modify the program so that it becomes I/O-bound, meaning that performance is limited by I/O rather than CPU speed. For the purpose of this report, it will suffice to demonstrate that the time spent on I/O operations exceeds the time spent on CPU computations, as shown by your profiling results.
  - You are free to change any part of the program as you see fit to achieve this effect; however, if you are using perf, the program must still be compiled with the `-O0` flag, and simply eliminating the computation loop is not considered a valid solution. 
  - Additionally, adding meaningless or artificial I/O operations just to inflate I/O time is not allowed and will result in losing points for this task.


- Use profiling to prove your modification works: show that I/O now takes a measurable fraction of the program’s total execution time.



> **Note**: You are free to explore advanced flags for perf and gprof in your profiling.
## Task 3 – Exploring the Effect of Loop Orders


- The program allows different loop orders (ijk, ikj, jik) for matrix multiplication.

- Write a bash script that runs the program in CPU mode for each loop order and profiles them.

- Analyze the results and explain why some loop orders may perform better than others.

> **Hint**: this is related to cache coherence and memory access patterns.



### Deliverables
- Your scripts (bash or Python) that automate profiling.
- Your C program for task 2 (there is no need to submit this if the changes are not notable and are explained in your documentation)
- Profiling outputs 
- Documentation: Providing clean documentation with plots, tables, or annotated profiling results is a must.


