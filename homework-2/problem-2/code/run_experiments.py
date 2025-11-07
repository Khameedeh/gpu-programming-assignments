#!/usr/bin/env python3
"""
Comprehensive testing and profiling script for Monte Carlo π estimation
Runs all implementations with varying thread counts and sample sizes
Collects results and generates CSV output for analysis
"""

import subprocess
import sys
import os
import csv
from pathlib import Path

# Configuration
SAMPLE_COUNTS = [100000, 1000000, 10000000]
THREAD_COUNTS = [1, 2, 4, 8, 16]

# Implementations to test
IMPLEMENTATIONS = [
    ('pi_seq', 'Sequential'),
    ('pi_pthread', 'Basic Pthread'),
    ('pi_pthread_local_counters', 'Thread-Local Counters'),
    ('pi_pthread_byte_array', 'Byte Array'),
]

# Math constant
PI = 3.14159265358979323846


def run_experiment(binary, samples, threads=None):
    """
    Run a single experiment and parse results
    
    Returns: dict with results or None on failure
    """
    try:
        if binary == 'pi_seq':
            cmd = [f'./{binary}', str(samples)]
        else:
            cmd = [f'./{binary}', str(samples), str(threads)]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  ERROR: {binary} failed with code {result.returncode}")
            return None
        
        # Parse CSV output
        data = {}
        for line in result.stdout.strip().split('\n'):
            if ',' in line:
                key, value = line.split(',', 1)
                try:
                    # Try to convert to float if possible
                    data[key] = float(value)
                except ValueError:
                    data[key] = value
        
        return data
    
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {binary} exceeded time limit")
        return None
    except Exception as e:
        print(f"  ERROR: {binary} - {e}")
        return None


def main():
    # Check if binaries exist
    print("Checking for compiled binaries...")
    for binary, name in IMPLEMENTATIONS:
        if not os.path.exists(f'./{binary}'):
            print(f"ERROR: {binary} not found. Run 'make all' first.")
            return 1
    
    print("✓ All binaries found\n")
    
    # Open output CSV file
    # Write to results/ folder if it exists, otherwise current directory
    results_dir = Path('../results')
    if results_dir.exists():
        output_file = str(results_dir / 'pi_profiling_results.csv')
    else:
        output_file = 'pi_profiling_results.csv'
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'Implementation', 'Samples', 'Threads',
            'Points_Inside', 'Pi_Estimate', 'Error', 
            'Execution_Time_Sec', 'Speedup'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sequential baseline results (for speedup calculation)
        sequential_times = {}
        
        # Run experiments
        total_experiments = 0
        for samples in SAMPLE_COUNTS:
            print(f"\n{'='*60}")
            print(f"Testing with {samples:,} samples")
            print(f"{'='*60}")
            
            # Sequential test (run once)
            if samples not in sequential_times:
                print(f"\nSequential Implementation:")
                data = run_experiment('pi_seq', samples)
                if data:
                    sequential_times[samples] = data.get('execution_time_sec', 0)
                    writer.writerow({
                        'Implementation': 'Sequential',
                        'Samples': samples,
                        'Threads': 1,
                        'Points_Inside': int(data.get('points_inside', 0)),
                        'Pi_Estimate': f"{data.get('pi_estimate', 0):.15f}",
                        'Error': f"{data.get('error', 0):.15f}",
                        'Execution_Time_Sec': f"{data.get('execution_time_sec', 0):.6f}",
                        'Speedup': 1.0
                    })
                    print(f"  π estimate: {data.get('pi_estimate', 0):.15f}")
                    print(f"  Error: {data.get('error', 0):.15f}")
                    print(f"  Time: {data.get('execution_time_sec', 0):.6f}s")
                    total_experiments += 1
                    csvfile.flush()
            
            # Parallel implementations
            for threads in THREAD_COUNTS:
                print(f"\nThreads: {threads}")
                
                for binary, name in IMPLEMENTATIONS[1:]:  # Skip sequential
                    print(f"  {name}...", end=' ', flush=True)
                    
                    data = run_experiment(binary, samples, threads)
                    if data:
                        seq_time = sequential_times.get(samples, 1.0)
                        speedup = seq_time / data.get('execution_time_sec', 1.0)
                        
                        writer.writerow({
                            'Implementation': name,
                            'Samples': samples,
                            'Threads': threads,
                            'Points_Inside': int(data.get('points_inside', 0)),
                            'Pi_Estimate': f"{data.get('pi_estimate', 0):.15f}",
                            'Error': f"{data.get('error', 0):.15f}",
                            'Execution_Time_Sec': f"{data.get('execution_time_sec', 0):.6f}",
                            'Speedup': f"{speedup:.2f}x"
                        })
                        
                        print(f"✓ {data.get('execution_time_sec', 0):.6f}s (speedup: {speedup:.2f}x)")
                        total_experiments += 1
                        csvfile.flush()
                    else:
                        print("✗")
    
    print(f"\n{'='*60}")
    print(f"Testing complete: {total_experiments} experiments")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")
    
    # Print summary
    try:
        import pandas as pd
        df = pd.read_csv(output_file)
        
        print("\nSummary by Implementation:")
        print(df.groupby('Implementation')[['Execution_Time_Sec']].mean())
        
    except ImportError:
        print("\n(Install pandas to see summary statistics)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())