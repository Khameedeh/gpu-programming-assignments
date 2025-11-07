# HW2 Problem 2: Multithreaded Monte Carlo Estimation of π

## Overview

This assignment implements the Monte Carlo method to estimate π using POSIX Threads (pthreads). The project explores four distinct synchronization strategies and analyzes how each approach impacts performance scaling across varying thread counts and workload sizes.

## Problem Statement

Estimate the value of π using the Monte Carlo method:
- Generate random points in a unit square [0,1] × [0,1]
- Count how many points fall within the inscribed unit circle (x² + y² ≤ 1)
- Use the ratio: π ≈ 4 × (points inside / total points)

Implement both sequential and parallel versions, exploring synchronization trade-offs and performance scaling.

## Project Structure

```
problem-2/
├── code/              # Source code and build files
│   ├── pi_seq.c
│   ├── pi_pthread.c
│   ├── pi_pthread_local_counters.c
│   ├── pi_pthread_byte_array.c
│   ├── Makefile
│   └── run_experiments.py
├── documents/         # Technical documentation
│   └── Latex/
│       └── main.tex (compiles to main.pdf)
├── results/           # Experimental results
│   └── pi_profiling_results.csv
└── README.md          # This file
```

## Implementations

### 1. Sequential Baseline (`code/pi_seq.c`)
- **Description:** Single-threaded reference implementation
- **Purpose:** Performance baseline for speedup calculations
- **Key Features:**
  - Uses `drand48()` for high-quality random number generation
  - Microsecond-precision timing with `gettimeofday()`
  - CSV-formatted output for easy parsing

**Usage:**
```bash
cd code
./pi_seq 1000000
```

### 2. Basic Multithreaded (`code/pi_pthread.c`)
- **Description:** Each thread generates samples independently, updates shared counter with mutex
- **Synchronization:** Global counter protected by `pthread_mutex_t`
- **Key Features:**
  - Simple implementation matching typical concurrent pattern
  - Each thread acquires mutex once (during final aggregation)
  - Minimal critical section (single integer addition)
  - Susceptible to mutex contention at high thread counts

**Design Pattern:**
```
Thread 0: Sample sampling → Lock mutex → Update counter → Unlock
Thread 1: Sample sampling → Lock mutex → Update counter → Unlock
...
```

**Usage:**
```bash
cd code
./pi_pthread 1000000 4  # 1M samples with 4 threads
```

### 3. Thread-Local Counters (`pi_pthread_local_counters.c`)
- **Description:** Each thread maintains local counter (no synchronization during sampling)
- **Synchronization:** None during sampling; aggregation after `pthread_join()`
- **Key Features:**
  - Eliminates mutex contention entirely
  - Each thread operates on its own stack (cache-friendly)
  - Minimal memory overhead (one counter per thread)
  - Aggregation is sequential but lock-free

**Design Pattern:**
```
Thread 0: Local sampling with local_count (no locks)
Thread 1: Local sampling with local_count (no locks)
...
Main:    Join all threads → Sum local_counts (sequential)
```

**Usage:**
```bash
cd code
./pi_pthread_local_counters 1000000 4
```

### 4. Byte-Array Partitioning (`code/pi_pthread_byte_array.c`)
- **Description:** Store individual sample results in array, eliminate synchronization
- **Synchronization:** None (static partitioning guarantees no false sharing)
- **Key Features:**
  - Zero synchronization (no locks, no atomic operations)
  - Eliminates false sharing through static partitioning
  - Higher memory usage: O(n) instead of O(1)
  - Suitable when memory is available and synchronization must be eliminated

**Design Pattern:**
```
Thread 0: Write results to array[0:250K]       (no synchronization)
Thread 1: Write results to array[250K:500K]    (no synchronization)
...
Main:    Join all threads → Sum array elements (sequential)
```

**Usage:**
```bash
cd code
./pi_pthread_byte_array 1000000 4
```

## Building and Testing

### Quick Build
```bash
cd code
make all        # Compile all implementations
make test       # Run quick sanity tests
make clean      # Remove binaries
```

### Comprehensive Profiling
```bash
cd code
make profile    # Run full 48-experiment suite
cd ..
cat results/pi_profiling_results.csv  # View results
```

### Manual Compilation
```bash
cd code
gcc -O2 -Wall -pthread -lm -o pi_seq pi_seq.c
gcc -O2 -Wall -pthread -lm -o pi_pthread pi_pthread.c
gcc -O2 -Wall -pthread -lm -o pi_pthread_local_counters pi_pthread_local_counters.c
gcc -O2 -Wall -pthread -lm -o pi_pthread_byte_array pi_pthread_byte_array.c
```

## Experimental Results

The comprehensive test suite (48 experiments) explores:
- **Sample Sizes:** 100K, 1M, 10M samples
- **Thread Counts:** 1, 2, 4, 8, 16 threads
- **Metrics:** Execution time, speedup, efficiency, accuracy

### Key Findings

1. **Synchronization Strategy Affects Performance, Not Correctness**
   - All implementations produce identical π estimates
   - Performance differences are purely in execution time

2. **Optimal Thread Count is System-Dependent**
   - 8 threads: ~75% efficiency
   - 16 threads: ~48% efficiency but higher absolute speedup
   - Diminishing returns beyond 8 cores

3. **Work Size Determines Optimal Strategy**
   - 100K samples: Overhead dominates; minimal speedup at 16 threads
   - 1M samples: Thread-local counters excel (6.07× vs. 5.37×)
   - 10M samples: Basic pthread peaks (7.60× vs. 6.90×)

4. **Byte-Array Trades Memory for Modest Performance**
   - Uses 100× more memory (10 MB vs. 100 bytes)
   - Performance gains marginal and sometimes negative
   - Memory bandwidth becomes limiting at 16 threads

5. **Thread Creation Overhead Matters for Small Workloads**
   - At 100K samples: 0.1-0.5 ms per thread overhead
   - At 10M samples: Overhead amortizes to negligible cost

### Performance Summary Table

| Implementation | Avg Time (s) | Best Speedup | Threads |
|---|---|---|---|
| Sequential | 0.03653 | 1.00× | 1 |
| Basic Pthread | 0.01523 | 7.60× | 16 (10M) |
| Thread-Local Counters | 0.01572 | 6.90× | 16 (10M) |
| Byte Array | 0.01661 | 5.98× | 16 (10M) |

**Efficiency at 10M Samples:**
- 2 threads: 1.00× efficiency (ideal scaling)
- 4 threads: 0.94× efficiency (minimal overhead)
- 8 threads: 0.75× efficiency (noticeable overhead)
- 16 threads: 0.48× efficiency (significant overhead)

## Results Files

- **`pi_profiling_results.csv`**: Complete experimental data
  - Columns: Implementation, Samples, Threads, Points_Inside, Pi_Estimate, Error, Execution_Time_Sec, Speedup
  - 48 rows (4 implementations × 12 configurations)

- **`documents/Latex/main.pdf`**: Technical report (10 pages)
  - Algorithm overview and mathematical foundation
  - Implementation design and synchronization strategies
  - Detailed performance analysis with tables
  - Key findings and conclusions
  - Reproducibility instructions

## Technical Details

### Random Number Generation
- Uses `drand48_r()` for thread-safe, high-quality random numbers
- Each thread seeded independently: `seed = time(NULL) + thread_id`
- Avoids correlation between thread streams

### Timing Precision
- Uses `gettimeofday()` for microsecond-level precision
- Measures wall-clock time from program start to completion
- Does not include compile time, only runtime

### Memory Layout
| Implementation | Memory Usage |
|---|---|
| Sequential | ~8 bytes (single counter) |
| Basic Pthread | ~8 bytes + mutex (~64 bytes) |
| Thread-Local | ~128 bytes (16 threads × 8 bytes) |
| Byte Array (10M) | ~10 MB (10M × 1 byte) |

### Synchronization Costs
- **Mutex lock/unlock:** ~100-200 ns per operation (platform-dependent)
- **Thread creation:** ~1-5 ms per thread (platform-dependent)
- **Cache miss:** ~100-300 cycles per miss (L3 cache)
- **Memory bandwidth:** ~50 GB/s typical (modern systems)

## Running Specific Configurations

### Single Experiment
```bash
./pi_pthread 10000000 8    # 10M samples, 8 threads
```

### Multiple Thread Counts (Same Sample Size)
```bash
for t in 1 2 4 8 16; do
    echo "Threads: $t"
    ./pi_pthread_local_counters 1000000 $t
done
```

### Comparative Analysis
```bash
SAMPLES=10000000
for impl in pi_seq pi_pthread pi_pthread_local_counters pi_pthread_byte_array; do
    if [ "$impl" = "pi_seq" ]; then
        ./$impl $SAMPLES
    else
        for t in 1 2 4 8; do
            ./$impl $SAMPLES $t
        done
    fi
done
```

## Customization and Extension

### Modifying Sample Size or Thread Count
Edit the test configuration in `run_experiments.py`:
```python
SAMPLE_COUNTS = [100000, 1000000, 10000000]  # Modify these
THREAD_COUNTS = [1, 2, 4, 8, 16]            # Modify these
```

### Adding New Implementations
1. Create `pi_pthread_variant.c` following the pattern of existing implementations
2. Add to `Makefile` build rules
3. Update `run_experiments.py` IMPLEMENTATIONS list
4. Run `make profile` to test

### Lock-Free Approaches
For advanced optimization, consider:
- **Atomic operations:** `__sync_fetch_and_add()` instead of mutexes
- **Compare-and-swap:** Implement lock-free counter with CAS
- **Hardware counters:** Performance counter integration with `perf`

## Analysis and Interpretation

### Understanding Speedup
- **Linear speedup (n×):** Ideal but rare due to overhead
- **Sub-linear speedup (0.5n - 0.9n):** Typical for well-optimized code
- **Super-linear speedup (>n×):** Unusual; check for cache effects or measurement artifacts

### Interpreting Results
- **Speedup < 1.0:** Thread overhead exceeds parallelization benefit (small workload)
- **Speedup = 2× at 4 threads:** 50% efficiency (typical)
- **Speedup = 6× at 8 threads:** 75% efficiency (good)
- **Speedup = 12× at 16 threads:** 75% efficiency (very good)

### Performance Bottlenecks
- **Synchronization-bound:** Mutex contention limits speedup (basic pthread at low sample sizes)
- **Memory-bound:** Memory bandwidth limits speedup (byte array at high thread counts)
- **CPU-bound:** Cache efficiency limits speedup (all implementations at 10M+ samples)

## Reproducibility

All experiments are fully reproducible:

1. **System Information:** Results depend on CPU cores, cache size, memory bandwidth
2. **Compiler Optimization:** Uses `-O2`; try `-O3` for potential improvements
3. **Operating System:** POSIX-compliant (Linux, macOS, BSD)
4. **Random Seed:** Uses `time()` for determinism; remove for reproducible π estimates

## Troubleshooting

### Compilation Errors
```
error: undefined reference to 'sin'
Solution: Ensure `-lm` flag is used (link math library)
```

### Poor Scaling (Low Speedup)
- **Small workload:** Use larger sample sizes (10M instead of 100K)
- **Many threads:** Reduce thread count to match CPU cores
- **Overheated CPU:** Let system cool before re-running

### Memory Exhaustion
- **Byte array with huge samples:** Reduce sample count or switch to counter-based approach
- **Typical limit:** 1GB RAM supports ~1B samples

## References

- POSIX Threads (pthreads): `man pthread`
- Random number generation: `man drand48`
- Performance profiling: `man time`, `man perf`
- Parallel efficiency analysis: OpenMP and MPI textbooks

## Files Included

```
problem-2/
├── pi_seq.c                          # Sequential baseline
├── pi_pthread.c                      # Basic multithreaded
├── pi_pthread_local_counters.c       # Thread-local optimization
├── pi_pthread_byte_array.c           # Byte-array partitioning
├── Makefile                          # Build configuration
├── run_experiments.py                # Profiling automation
├── pi_profiling_results.csv          # Experimental results
├── README.md                         # This file
└── documents/Latex/
    ├── main.tex                      # Technical report (LaTeX)
    └── main.pdf                      # Technical report (compiled)
```

## Academic Integrity

This work represents original implementation and analysis of the Monte Carlo method with different synchronization strategies. All profiling results were collected on the author's system. Code follows standard pthreads patterns and is appropriately commented to explain design decisions.

## Author Information

- **Student ID:** 40131873
- **Course:** GPU Programming / Parallel Computing
- **Assignment:** Homework 2, Problem 2
- **Submission Date:** November 2025

---

For detailed analysis, see `documents/Latex/main.pdf`.