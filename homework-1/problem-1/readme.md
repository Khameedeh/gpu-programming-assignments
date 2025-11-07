# HW1 – Sorting Algorithms: Profiling & Analysis

**Prepared & Supported by:**  Sina Hakimzadeh <br>
**Due date:** 17 October 2025

## Table of Contents

1. [Problem Overview](#1-problem-overview)  
2. [Program Overview](#2-program-overview)  
   - [Structure (modular)](#21-structure-modular)  
   - [What the code does](#22-what-the-code-does-current-behavior)  
   - [Build & Run](#23-build--run)  
   - [Input & Output formats](#24-input--output-formats)  
3. [Your Tasks](#3-your-tasks)  
   - [Profiling & Bottleneck Analysis](#31-profiling--bottleneck-analysis)  
   - [Automation (Script + Makefile)](#32-automation-script--makefile)  
   - [Required Experiments](#33-required-experiments--expected-reasoning)  
   - [Reporting Guidelines](#34-reporting-guidance-35-pages-incl-figures)  
   - [Submission Checklist](#35-submission-checklist)  
4. [Ethics & Academic Integrity](#4-ethics--academic-integrity)  


## 1) Problem Overview

In this homework, you are going to profile a **provided [C codebase](./code/)** that implements several sorting algorithms (`insertion`, `bubble`, and `merge`) on arrays and linked lists. Your task is not to re-implement sorting, but to use profiling tools like `perf` and `gprof` to measure performance, identify bottlenecks, and explain trade-offs. 

You will examine how factors such as algorithm choice, data layout, input size, data type, and hardware behavior affect execution. Along the way, you are expected to connect algorithmic complexity (e.g., O(N²) vs O(N log N)) with real hardware effects such as cache misses, branch mispredictions, and memory stalls. Finally, you will compare the strengths and limitations of `perf` and `gprof` in terms of methodology, overhead, and usability, and provide a clear, data-driven explanation of where and why bottlenecks occur.



## 2) Program overview

### 2.1 Structure (modular)
```
include/
  sort.h        # algorithms + node types (NodeInt, NodeDouble)
  io.h          # file readers, verification helpers, cleanup
  options.h     # CLI + minimal YAML (subset) parsing, output control flags
src/
  main.c        # orchestrates I/O → sort → outputs metadata (stdout) and metadata+sequence (file)
  sort.c        # array & linked-list sorting implementations
  io.c          # input parsing, verification, list/array utilities
  options.c     # command-line + YAML handling
Makefile        # portable build (O2, -std=c11)
```

### 2.2 What the code does (current behavior)
- Reads `N` and then `N values` from an input file (default: `input.txt`).
- Builds the dataset as `array` or `linked list`.
- Sorts using `Insertion`, `Bubble`, or `Merge`.
- Supports `-t int` or `-t double`.
- **Output semantics:**
  - **Stdout (terminal):** prints a **single metadata line** only, e.g.
    ```
    file=input.txt layout=array alg=merge type=int n=8 checksum=... first=... sorted=1
    ```
  - **Output file (when `--write-output` or `write_output: true`):**
    - **Line 1:** the same **metadata** line
    - **Line 2:** the **sorted sequence** (space‑separated)
- If `--verify` is given, the program checks sortedness and prints a **warning to stderr** if the sequence is not sorted. (The `sorted=` field appears in the metadata when `--verify` is used.)
- No timers or profiler hooks built in.

### 2.3 Build & Run

#### 1) Configure
Edit **`config.yaml`** to set defaults. Example:
```yaml
layout: array
alg: merge
type: int
file: input.txt
verify: true
output: output.txt
write_output: true
```
> Any option in the config file can be overridden at the command line.

#### 2) Build
Use the provided Makefile:
```bash
make
```
This creates the binary:
```
./layout_sort_file_profile
```

#### 3) Run with config file
```bash
./layout_sort_file_profile -config config.yaml
```
- **Terminal output:** one line of metadata.
- **Output file:** line 1 = metadata, line 2 = sorted sequence.

#### 4) Run with CLI overrides
If you ever need to override the configuration file in CLI, you can use the command in the following format:
```bash
# Override algorithm and type
./layout_sort_file_profile -config config.yaml -alg insertion -t double

# Or run entirely from CLI
./layout_sort_file_profile -layout array -alg merge -t int -file input.txt --verify --write-output -o output.txt
```

#### 5) Sanity check
Create a small `input.txt`:
```
5
3 1 4 1 5
```
Run:
```bash
./layout_sort_file_profile -layout array -alg merge -t int -file input.txt --verify --write-output -o output.txt
```
- **Terminal:** metadata only.
- **`output.txt`:** metadata line, then sorted sequence `1 1 3 4 5`.

### 2.4 Input & output formats
**Input file format** (`input.txt`):
```
N
v0 v1 v2 ... v{N-1}
```
- For `-t int`, all values must be valid integers.
- For `-t double`, values must be valid doubles (scientific notation allowed).

**Stdout (terminal):**
```
file=<path> layout=<array|list> alg=<insertion|bubble|merge> type=<int|double> n=<N> checksum=<...> first=<...> [sorted=0|1]
```

**Output file (when enabled):**
```
<metadata line>
<sorted_v0> <sorted_v1> ... <sorted_v{N-1}>
```




## 3) Your Tasks
Your task is to **profile the sorting program, collect performance data, and analyze results**. All steps must be automated with a script (input generation, builds, runs, and data collection) to ensure reproducibility and avoid manual execution. The final goal is to identify bottlenecks and present findings with clear plots, tables, and explanations.

### 3.1 Profiling & bottleneck analysis
1. Use `perf` and `gprof` to profile the program across multiple configurations (algorithm, layout, type, and a sensible range of ``N``).
2. Identify hotspots and explain observed bottlenecks. Support your claims using:
   - runtime statistics you collect,
   - relevant hardware counters (e.g., instructions, cycles, IPC, LLC-load-misses, dTLB-load-misses, branch-misses), and
   - call-graph evidence (from either tool) to attribute costs to functions/loops.
3. Present results clearly using **plots, tables, and/or figures** and tie the data back to algorithmic complexity and memory behavior.

> You may ignore the sorted sequence contents in your profiling pipeline (redirect stdout to `/dev/null` if convenient). Ensure that your measurements reflect **sorting work**, not file I/O (e.g., consider warm caches or tmpfs).

### 3.2 Automation (script + Makefile)
- Provide a **Python or Bash script** that:
  1) **Generates inputs**  for multiple ``N`` values and ranges; use a **fixed random seed** for reproducibility.
  2) **Builds** the program for each profiler as required by the Getting-Started handout (do **not** hardcode flags here; choose appropriate flags per profiler in your script/Makefile).
  3) **Runs all configurations** (algorithm × layout × type × `N`).
  4) **Collects metrics** into CSV/JSON for plotting and analysis. It is **your responsibility** to select the most relevant metrics for this homework.
  5) **Repeats each configuration multiple times**, records every run, and **computes averages** (and optionally variance/stddev) to reduce noise.
  6) **Exports all artifacts** — raw profiler outputs, processed metrics (CSV/JSON), and generated plots/tables — into a **timestamped results directory** (e.g., `results_2025-10-01_1530`) so runs are preserved and not overwritten.

- Update the **Makefile** to support repeatable builds. The Makefile may need **separate targets per profiler** (e.g., `make build_perf`, `make build_gprof`). Keep target names clear and document how to use them in your report/README.


### 3.3 Required experiments & expected reasoning
Run the experiments with increasing input sizes starting from ``N``=1024, doubling each time (e.g., 2048, 4096, 8192, …) until the runtime reaches a practical limit on your machine (ideally in the 0.2–10 s range). Remember that O(N²) sorts will hit this limit much sooner, so stop earlier for those algorithms.

1. **Algorithm scaling (`array` layout)**
    - Keep `layout=`array`` and `type=int`.
    - Test ``Insertion``, ``bubble``, and ``Merge`` for many input sizes ``N`` (e.g., 1024, 2048, 4096, …).
    - Deliverables: a plot of runtime vs ``N`` + a short note showing ``Insertion`/`bubble`` ≈ O(`N`^2) and ``Merge`` ≈ O(`N` log `N`).

2. **Layout impact**
    - Pick one algorithm and one ``N``, keep them fixed.
    - Compare ``array`` vs `list`
    - Deliverables: a small table with IPC, LLC-load-misses, dTLB-load-misses, and (optional) branch-misses.

3. **Data type impact**
    - Pick one algorithm (suggest `Merge`) and one layout.
    - Compare int vs double for several `N`.
    - Deliverables: plots/tables for runtime and key counters.

4. **Call-graph evidence**
    - Choose one heavy (slow) case for each algorithm.
    - Include a call-graph snippet and mark the hottest functions/loops.
    - Deliverables: a short explanation tying those hot spots to your measured stalls/misses/IPC to confirm your story.

> Note: The listed deliverables are only examples to guide you. They illustrate the kind of plots or notes we expect, but you should make your own decisions about which figures, tables, and explanations best demonstrate your findings.

### 3.4 Reporting guidance
Structure your report professionally and concisely:
1. **Setup**: CPU model, core/SMT count, compiler & flags, etc.
2. **Methodology**: input generation strategy; how you invoked each profiler; how you parsed and aggregated results.
3. **Results & Analysis**: figures/tables first, then explanation. For each required experiment, state the **observations**, the **metrics** that support them, and the **reasoning** (algorithmic + architectural). Avoid speculative claims—ground each statement in your data.
4. **bottlenecks**: In your report, clearly state the bottleneck points you identified (e.g., memory stalls, branch mispredictions, cache misses, algorithmic complexity) and justify each with profiling data.
5. **Comparison of tools**: summarize perf vs gprof (what each shows well/poorly, overhead, ease of attributing costs, metric breadth), with examples from your results.
6. **Takeaways**: concise bullets on when layout vs algorithm dominates, and when `int` vs `double` impacts performance.

### 3.5 Submission checklist
- [ ] Script (Bash/Python) that builds, runs, profiles, and collects metrics.
- [ ] Updated Makefile with clear targets and brief comments.
- [ ] Reproducible input generation (fixed seeds).
- [ ] CSV/JSON with collected metrics and any plotting code/notebooks.
- [ ] Report with required experiments, figures/tables, and analysis.
- [ ] Commands to reproduce your results in your report.


## 4) Ethics & Academic Integrity

This homework must reflect **your own work**. While discussions with classmates about general concepts are encouraged, all submitted code, scripts, reports, and analysis must be authored individually.  

- **Do not copy** solutions, scripts, or reports from other students, online sources, or prior years.  
- **Do not share** your own completed solutions with others before the submission deadline.  
- **Always cite** external sources (papers, documentation, tutorials) if you use them to inform your work.  
- **Profiling results must be your own.** Running the provided program and collecting data on your own machine is part of the assignment; submitting fabricated or borrowed results is considered misconduct.  

Violations will be treated as academic dishonesty and handled according to university policy.  

> When in doubt: ask questions, collaborate conceptually, but write and submit your **own independent work**.

