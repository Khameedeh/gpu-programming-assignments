CC      := cc
# -pg is added for gprof only during build_gprof
CFLAGS  := -std=c11 -Wall -Wextra
LDFLAGS := -lm

SRC     := matrix_multiplication.c
BIN     := matrix_multiplication
SCRIPT  := profile_runner.py
# Define the matrix sizes for documentation clarity
SIZES   := 1000 2500 5000

# Default build: optimized for performance testing with 'perf'
all: build_perf

# Optimized build (suitable for perf)
# Optimization level O2 is generally a good choice for real-world performance
build_perf:
	@echo "--- Compiling for perf (Optimized: -O2) ---"
	$(CC) $(CFLAGS) -O2 -o $(BIN) $(SRC) $(LDFLAGS)

# Build with gprof instrumentation and no optimization (-O0)
# -O0 is required for accurate gprof function timing, and -pg adds profiling instrumentation
build_gprof:
	@echo "--- Compiling for gprof (-O0 -pg) ---"
	$(CC) $(CFLAGS) -pg -O0 -o $(BIN) $(SRC) $(LDFLAGS)

# Master command: Runs the Python script to execute all experiments
run_all: build_perf build_gprof
	@echo "\n--- Starting all profiling experiments (Sizes: $(SIZES), 5 Runs Each) ---"
	# Pass the matrix sizes as arguments to the Python script
	python3 $(SCRIPT) $(SIZES)

clean:
	@echo "\n--- Cleaning up generated files and results ---"
	# Remove the compiled binary and gprof output
	rm -f $(BIN) gmon.out
	# Remove the I/O simulation files
	rm -f io_input_matrix.bin io_output_matrix.bin
	# Remove all generated results directories and the final CSV
	rm -rf results_* final_performance_report.csv

.PHONY: all build_perf build_gprof run_all clean
