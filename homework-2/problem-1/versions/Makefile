CC = gcc
CFLAGS = -Wall -Wextra -pthread -O2 -g

# Targets for each version
VERSIONS = v1_initial v2_spurious_wakeup_fix v3_broadcast_fix v4_graceful_shutdown v5_final

all: $(VERSIONS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $<

v1: v1_initial
	./v1_initial

v2: v2_spurious_wakeup_fix
	./v2_spurious_wakeup_fix

v3: v3_broadcast_fix
	./v3_broadcast_fix

v4: v4_graceful_shutdown
	./v4_graceful_shutdown

v5: v5_final
	./v5_final

# Run all versions in sequence
run-all: $(VERSIONS)
	@echo "===== Running v1 (Initial - Buggy Version) ====="
	-timeout 10 ./v1_initial || echo "v1 interrupted or crashed (expected)"
	@echo "\n===== Running v2 (Spurious Wakeup Fix) ====="
	-timeout 10 ./v2_spurious_wakeup_fix || echo "v2 interrupted or crashed (expected)"
	@echo "\n===== Running v3 (Broadcast Fix) ====="
	-timeout 10 ./v3_broadcast_fix || echo "v3 interrupted or crashed (expected)"
	@echo "\n===== Running v4 (Graceful Shutdown) ====="
	./v4_graceful_shutdown
	@echo "\n===== Running v5 (Final Version) ====="
	./v5_final

clean:
	rm -f $(VERSIONS)
	rm -f *.o

rebuild: clean all

help:
	@echo "Available targets:"
	@echo "  make all          - Compile all versions"
	@echo "  make run-all      - Compile and run all versions"
	@echo "  make v1           - Compile and run v1_initial"
	@echo "  make v2           - Compile and run v2_spurious_wakeup_fix"
	@echo "  make v3           - Compile and run v3_broadcast_fix"
	@echo "  make v4           - Compile and run v4_graceful_shutdown"
	@echo "  make v5           - Compile and run v5_final"
	@echo "  make clean        - Remove compiled executables"
	@echo "  make rebuild      - Clean and recompile all"
	@echo "  make help         - Show this help message"

.PHONY: all run-all v1 v2 v3 v4 v5 clean rebuild help