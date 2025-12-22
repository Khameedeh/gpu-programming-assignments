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
    """Run the rasterization program and extract timing from output"""
    returncode, stdout, stderr = run_command(program_cmd)

    if returncode != 0:
        print(f"Error running program: {stderr}")
        return {}

    # Extract timing and configuration info
    metrics = {}

    lines = stdout.split('\n')
    for line in lines:
        if 'CUDA drawing time:' in line:
            try:
                time_str = line.split(':')[1].strip()
                time_ms = float(time_str.split()[0])
                metrics['duration_ms'] = time_ms
            except:
                pass
        elif 'Block size:' in line:
            try:
                parts = line.split(':')[1].strip().split('x')
                block_x = int(parts[0])
                block_y = int(parts[1])
                metrics['block_size_x'] = block_x
                metrics['block_size_y'] = block_y
            except:
                pass
        elif 'Image size:' in line:
            try:
                size_str = line.split(':')[1].strip()
                size = int(size_str.split('x')[0])
                metrics['image_size'] = size
            except:
                pass
        elif 'Test case' in line:
            try:
                test_case = int(line.split(':')[0].split()[-1])
                metrics['test_case'] = test_case
            except:
                pass

    return metrics

def main():
    sizes = [256, 512, 1024]
    test_cases = [
        (0, "all_shapes"),
        (1, "line_only"),
        (2, "circle_only"),
        (3, "ellipse_only"),
        (4, "multiple_circles")
    ]
    block_sizes = [8, 16, 32]

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_file = f"{results_dir}/problem3_profiling_results_{timestamp}.csv"

    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['size', 'test_case', 'description', 'block_size_x', 'block_size_y', 'duration_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for size in sizes:
            for test_case, description in test_cases:
                for block_size in block_sizes:
                    print(f"\n=== Testing size {size}x{size}, test case {test_case}: {description}, block size {block_size}x{block_size} ===")

                    cmd = f"./rasterization {size} {test_case} {block_size}"
                    metrics = run_program(cmd)

                    if metrics:
                        print(f"Duration: {metrics.get('duration_ms', 'N/A')} ms")

                    # Write results
                    result_row = {
                        'size': f"{size}x{size}",
                        'test_case': test_case,
                        'description': description,
                        'block_size_x': block_size,
                        'block_size_y': block_size,
                        'duration_ms': metrics.get('duration_ms', 'N/A')
                    }
                    writer.writerow(result_row)

    print(f"\nProfiling complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
