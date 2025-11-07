# Simple, portable Makefile for profiling
# Build: make (default O2)
# Build for perf: make build_perf
# Build for gprof: make build_gprof
# Run all experiments: make run_all
# Clean: make clean

CC      := cc
CFLAGS  := -std=c11 -Wall -Wextra -Iinclude
LDFLAGS :=

SRC := src/main.c src/sort.c src/io.c src/options.c
BIN := layout_sort_file_profile
SCRIPT := profile_runner.py

all: build_perf

# Optimized build (default, suitable for perf)
build_perf:
	$(CC) $(CFLAGS) -O2 -o $(BIN) $(SRC) $(LDFLAGS)

# Build with gprof instrumentation (suitable for gprof)
build_gprof:
	$(CC) $(CFLAGS) -pg -O0 -o $(BIN) $(SRC) $(LDFLAGS)

# Target to run the automation script
run_all: build_perf build_gprof
	python3 $(SCRIPT)

clean:
	rm -f $(BIN) gmon.out
	rm -rf results_*

.PHONY: all build_perf build_gprof run_all clean