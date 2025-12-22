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

def profile_kernel(program_cmd, output_file):
    """Run the program and extract timing from its output"""
    print(f"Running: {program_cmd}")
    returncode, stdout, stderr = run_command(program_cmd)

    if returncode != 0:
        print(f"Error running program: {stderr}")
        return {}

    # Extract timing from program output
    metrics = {}

    lines = stdout.split('\n')
    for line in lines:
        if 'Kernel execution time:' in line:
            try:
                time_str = line.split(':')[1].strip()
                time_ms = float(time_str.split()[0])
                metrics['duration_ms'] = time_ms
            except:
                pass
        elif 'Blocks:' in line and 'Threads per block:' in line:
            try:
                # Parse both blocks and threads from the same line
                # Format: "Blocks: (X, Y), Threads per block: (A, B)"

                # Extract blocks part
                blocks_start = line.find('Blocks: (') + len('Blocks: (')
                blocks_end = line.find(')', blocks_start)
                blocks_part = line[blocks_start:blocks_end]
                bx, by = blocks_part.split(', ')
                metrics['blocks_x'] = int(bx)
                metrics['blocks_y'] = int(by)

                # Extract threads part
                threads_start = line.find('Threads per block: (') + len('Threads per block: (')
                threads_end = line.find(')', threads_start)
                threads_part = line[threads_start:threads_end]
                tx, ty = threads_part.split(', ')
                metrics['threads_x'] = int(tx)
                metrics['threads_y'] = int(ty)
            except:
                pass

    # Save full output
    with open(output_file, 'w') as f:
        f.write(stdout)
        if stderr:
            f.write("\nSTDERR:\n")
            f.write(stderr)

    return metrics

def main():
    sizes = [128, 256, 512, 1024, 2048]
    test_cases = [
        (0, "1_thread_per_block"),
        (1, "16x16_threads"),
        (2, "32x32_threads"),
        (3, "8x8_threads")
    ]

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_file = f"{results_dir}/problem1_profiling_results_{timestamp}.csv"

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['size', 'test_case', 'description', 'duration_ms', 'blocks_x', 'blocks_y', 'threads_x', 'threads_y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for size in sizes:
            for test_case, description in test_cases:
                print(f"\n=== Testing size {size}x{size}, test case {test_case}: {description} ===")

                # Run the program
                cmd = f"./color_matrix {size} {test_case}"
                output_file = f"{results_dir}/p1_size{size}_test{test_case}_output.txt"

                metrics = profile_kernel(cmd, output_file)

                # Write results
                result_row = {
                    'size': f"{size}x{size}",
                    'test_case': test_case,
                    'description': description,
                    'duration_ms': metrics.get('duration_ms', 'N/A'),
                    'blocks_x': metrics.get('blocks_x', 'N/A'),
                    'blocks_y': metrics.get('blocks_y', 'N/A'),
                    'threads_x': metrics.get('threads_x', 'N/A'),
                    'threads_y': metrics.get('threads_y', 'N/A')
                }
                writer.writerow(result_row)

    print(f"\nProfiling complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
