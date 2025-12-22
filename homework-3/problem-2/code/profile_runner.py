#!/usr/bin/env python3

import subprocess
import os
import sys
import csv
import time
from datetime import datetime

def run_command(cmd, cwd=None):
    """Run a command and return the output"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=300)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def run_program(program_cmd):
    """Run the program and extract timing and verification from output"""
    returncode, stdout, stderr = run_command(program_cmd)

    if returncode != 0:
        print(f"Error running program: {stderr}")
        return {}

    # Extract timing and verification info
    metrics = {}

    lines = stdout.split('\n')
    for line in lines:
        if 'Kernel execution time:' in line or 'Total kernel execution time:' in line:
            try:
                time_str = line.split(':')[1].strip()
                time_ms = float(time_str.split()[0])
                metrics['duration_ms'] = time_ms
            except:
                pass
        elif 'Block size:' in line:
            try:
                block_size = int(line.split(':')[1].strip())
                metrics['block_size'] = block_size
            except:
                pass
        elif 'Results verified successfully' in line:
            metrics['verification'] = 'PASS'
        elif 'Results verification failed' in line:
            metrics['verification'] = 'FAIL'
        elif 'Even indices kernel:' in line:
            try:
                parts = line.split()
                blocks_even = int(parts[3])
                threads_even = int(parts[5])
                metrics['blocks_even'] = blocks_even
                metrics['threads_even'] = threads_even
            except:
                pass
        elif 'Odd indices kernel:' in line:
            try:
                parts = line.split()
                blocks_odd = int(parts[3])
                metrics['blocks_odd'] = blocks_odd
            except:
                pass

    return metrics

def main():
    sizes = [50000, 100000, 500000, 1000000, 5000000, 10000000]  # 50K, 100K, 500K, 1M, 5M, 10M elements
    test_cases = [
        (0, "baseline_if_else"),
        (1, "optimized_no_divergence")
    ]
    block_sizes = [64, 128, 256, 512, 1024]

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_file = f"{results_dir}/problem2_profiling_results_{timestamp}.csv"

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['size', 'test_case', 'description', 'block_size', 'duration_ms', 'verification', 'blocks_even', 'threads_even', 'blocks_odd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for size in sizes:
            for test_case, description in test_cases:
                for block_size in block_sizes:
                    print(f"\n=== Testing size {size}, test case {test_case}: {description}, block size {block_size} ===")

                    cmd = f"./masked_computation {size} {test_case} {block_size}"
                    metrics = run_program(cmd)

                    if metrics:
                        print(f"Duration: {metrics.get('duration_ms', 'N/A')} ms, Verification: {metrics.get('verification', 'N/A')}")

                    # Write results
                    result_row = {
                        'size': size,
                        'test_case': test_case,
                        'description': description,
                        'block_size': block_size,
                        'duration_ms': metrics.get('duration_ms', 'N/A'),
                        'verification': metrics.get('verification', 'N/A'),
                        'blocks_even': metrics.get('blocks_even', 'N/A'),
                        'threads_even': metrics.get('threads_even', 'N/A'),
                        'blocks_odd': metrics.get('blocks_odd', 'N/A')
                    }
                    writer.writerow(result_row)

    print(f"\nProfiling complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
