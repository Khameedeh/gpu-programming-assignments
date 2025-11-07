CC = gcc
CFLAGS = -O2 -Wall -Wextra -pthread -lm
CFLAGS_DEBUG = -O0 -g -Wall -Wextra -pthread -lm

# Targets
TARGETS = pi_seq pi_pthread pi_pthread_local_counters pi_pthread_byte_array

# Default target
all: $(TARGETS)

# Sequential implementation
pi_seq: pi_seq.c
	$(CC) $(CFLAGS) -o $@ $<

# Basic pthread version
pi_pthread: pi_pthread.c
	$(CC) $(CFLAGS) -o $@ $<

# Optimized with thread-local counters
pi_pthread_local_counters: pi_pthread_local_counters.c
	$(CC) $(CFLAGS) -o $@ $<

# Advanced byte-array version
pi_pthread_byte_array: pi_pthread_byte_array.c
	$(CC) $(CFLAGS) -o $@ $<

# Clean build artifacts
clean:
	rm -f $(TARGETS) *.o core vgcore.*

# Run basic test
test: all
	@echo "=== Sequential Test ==="
	./pi_seq 1000000
	@echo ""
	@echo "=== Pthread Test (2 threads) ==="
	./pi_pthread 1000000 2
	@echo ""
	@echo "=== Pthread Local Counters (2 threads) ==="
	./pi_pthread_local_counters 1000000 2
	@echo ""
	@echo "=== Byte Array Test (2 threads) ==="
	./pi_pthread_byte_array 1000000 2

# Run comprehensive profiling (requires Python for parsing)
profile: all
	python3 run_experiments.py

# Help target
help:
	@echo "Available targets:"
	@echo "  make all       - Build all versions"
	@echo "  make clean     - Remove built binaries"
	@echo "  make test      - Run quick sanity tests"
	@echo "  make profile   - Run comprehensive profiling (requires Python)"
	@echo "  make help      - Show this help message"

.PHONY: all clean test profile help