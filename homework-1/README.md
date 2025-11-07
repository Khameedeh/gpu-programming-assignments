# HW1 – Profiling & Performance Analysis

**Prepared & Supported by:**  
- Sina Hakimzadeh  
- Raha Rahmanian  


**Due date:** 24 October 2025  

You can upload your assignment using the following link: [Upload Form](http://forms.gle/7gK74kopnYZwvBy86)

## Overview
This homework focuses on understanding how **algorithms interact with hardware** by using profiling tools such as `perf` and `gprof`.  
It is divided into two parts:

- [**Problem 1 – Sorting Algorithms**  ](./problem-1)
  Analyze the performance of sorting algorithms under different configurations (algorithm, layout, type, input size). The goal is to connect theoretical complexity with real hardware performance and identify bottlenecks.  

- [**Problem 2 – Matrix Multiplication**    ](./problem-2)
  Study CPU-bound vs I/O-bound workloads and investigate the effect of loop ordering on cache performance.  

Both problems require automation (scripts + Makefile), profiling experiments, and clear documentation with figures/tables.  


## Pre-Reading & References
Before starting this homework, make sure you have reviewed the following materials:

- [Profiling: Essential for Efficient Code](./../../docs/01-Introduction-to-Profiling)
- [Getting started with profiling](./../../docs/02-Getting-started-with-profiling)
- [Learning Makefile](http://makefiletutorial.com/)





## Submission
Your submission should include:
- Documentation files
- Scripts (Bash/Python) for automation and profiling  
- Updated Makefile
- Collected metrics (CSV/JSON) + plots/tables  
- **All source code and any files needed to run your programs**, including modifications you made  
- **The exact commands you used** to run your experiments (include them in the report)




## Academic Integrity
All work must be your own. Profiling results must be collected on your own machine. Cite any references you use.
