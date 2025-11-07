import os
import subprocess
import time
import csv
import numpy as np
from collections import defaultdict

# --- Configuration for Matrix Multiplication Project ---
BINARY_NAME = './matrix_multiplication'

# Define the configurations for the profiling runs
# N: Matrix size (N x N). Updated as requested: 500, 1000, 2500, 5000, 7500.
# ORDER: Loop orders for matrix multiplication (CPU mode, Task 3).
# MODE: Execution mode (CPU-bound vs. I/O-bound, Task 1 & 2).
CONFIGS = {
    'N': [500, 1000, 1500, 2000, 2500], 
    'ORDER': ['ijk', 'ikj', 'jik'], 
    'MODE': ['cpu', 'io'],
}

REPETITIONS = 5
RESULTS_DIR = f"results_{time.strftime('%Y-%m-%d_%H%M%S')}"
CSV_DIR = os.path.join(RESULTS_DIR, 'csv')

# Perf counters relevant for CPU, memory, and cache analysis (Tasks 1, 2, & 3)
PERF_COUNTERS = [
    'task-clock',           # Total execution time
    'cycles',               # CPU cycles spent
    'instructions',         # Instructions executed
    'branches',             # Branch predictions
    'branch-misses',        # Branch mispredictions
    'L1-dcache-load-misses', # L1 cache misses (relevant for cache coherence)
    'dTLB-load-misses',     # Data TLB misses
    'cache-references',     # Total cache accesses (relevant for Task 3 analysis)
    'cache-misses'          # Total cache misses (relevant for Task 3 analysis)
]
PERF_COUNTERS_STR = ",".join(PERF_COUNTERS)


def parse_perf_output(file_path):
    """Parses the raw perf output file to extract metrics."""
    metrics = {}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Perf log not found at {file_path}")
        return {}

    for line in lines:
        for counter in PERF_COUNTERS:
            if counter in line:
                # Perf output is comma-separated, metric is the first field
                parts = line.split(',')
                try:
                    # Clean up the value and convert to float
                    value = float(parts[0].strip().replace(',', ''))
                    
                    # Convert task-clock from milliseconds to seconds
                    if 'task-clock' in counter and 'msec' in line:
                         value /= 1000.0
                         
                    metrics[f'Perf_{counter}'] = value
                except (IndexError, ValueError):
                    pass
    
    # Calculate Instructions Per Cycle (IPC)
    instr = metrics.get('Perf_instructions')
    cyc = metrics.get('Perf_cycles')
    if instr is not None and cyc is not None and cyc > 0:
        metrics['Perf_IPC'] = instr / cyc
    else:
        metrics['Perf_IPC'] = np.nan
        
    # Calculate Cache Miss Rate (Task 3 analysis)
    refs = metrics.get('Perf_cache-references')
    misses = metrics.get('Perf_cache-misses')
    if refs is not None and misses is not None and refs > 0:
        metrics['Perf_Cache_Miss_Rate'] = misses / refs
    else:
        metrics['Perf_Cache_Miss_Rate'] = np.nan
        
    return metrics

def run_single_config(N, order, mode, i):
    """Runs the C binary for a single configuration using perf and gprof."""
    
    # The C program expects arguments: <ijk|ikj|jik> <cpu|io> <N>
    base_cmd = [
        BINARY_NAME, 
        order, 
        mode, 
        str(N)
    ]
    
    config_name = f'N{N}_ORDER{order}_MODE{mode}'
    
    # --- 1. Run with Perf (Optimized binary: build_perf) ---
    # Perf command to run the program and collect hardware counters
    perf_cmd = [
        'perf', 'stat', '-x', ',', 
        '-e', PERF_COUNTERS_STR, 
        *base_cmd
    ]
    
    # Capture stderr which contains the perf output
    result = subprocess.run(perf_cmd, capture_output=True, text=True, check=False)
    
    perf_log_path = os.path.join(RESULTS_DIR, f'raw_perf_{config_name}_{i}.txt')
    with open(perf_log_path, 'w') as f:
        f.write(result.stderr)
        
    # --- 2. Run for Gprof (Instrumented binary: build_gprof) ---
    # Note: gprof requires the program to run normally (without perf)
    
    gprof_binary_name = BINARY_NAME # Use global BINARY_NAME for consistency
    gprof_cmd = [gprof_binary_name, order, mode, str(N)]
    
    start_time = time.time()
    # Execute the command (stdout/stderr are suppressed to keep console clean)
    subprocess.run(gprof_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    end_time = time.time()
    gprof_runtime = end_time - start_time 

    # --- 3. Generate Gprof Report ---
    gprof_log_path = None
    if os.path.exists('gmon.out'):
        gprof_report_cmd = ['gprof', gprof_binary_name, 'gmon.out']
        # The gprof output is large, so we capture and save it.
        gprof_report = subprocess.run(gprof_report_cmd, capture_output=True, text=True, check=False)
        
        gprof_log_path = os.path.join(RESULTS_DIR, f'raw_gprof_{config_name}_{i}.txt')
        with open(gprof_log_path, 'w') as f:
            f.write(gprof_report.stdout)

        # Cleanup the gmon.out file for the next run
        os.remove('gmon.out')
    else:
        # This will happen often for I/O mode if the runtime is very short 
        print(f"  Warning: gmon.out not generated for {config_name}. Gprof results will be missing.")


    return perf_log_path, gprof_runtime, gprof_log_path


def get_fieldnames():
    """Defines the field names for the output CSV files."""
    # Base fields for identification
    base_fields = ['N', 'Order', 'Mode', 'Repetition_ID', 'Run_Type']
    
    # Individual run metrics
    perf_metrics = [f'Perf_{c}' for c in PERF_COUNTERS] + ['Perf_IPC', 'Perf_Cache_Miss_Rate']
    individual_metrics = [f'Gprof_Runtime_s'] + perf_metrics
    
    # Mean metrics (prefixed for clarity)
    mean_metrics = ['Gprof_Runtime_Mean_s'] + [f for f in perf_metrics] 

    # Combine all fields, sorted
    universal_fieldnames = base_fields + sorted(list(set(individual_metrics + mean_metrics)), key=lambda x: ('Perf' not in x, x))
    
    # Ensure mean fieldnames only include the mean version of the runtime
    mean_fieldnames = base_fields + sorted(mean_metrics, key=lambda x: ('Perf' not in x, x))
    
    return universal_fieldnames, mean_fieldnames


def run_all_experiments(experiment_configs):
    """Executes all defined experiments."""
    
    all_raw_log_metadata = []
    
    # Ensure both binaries are built before starting
    print("Building 'perf' optimized binary...")
    subprocess.run(['make', 'build_perf'], check=True, stdout=subprocess.DEVNULL)
    print("Building 'gprof' instrumented binary...")
    subprocess.run(['make', 'build_gprof'], check=True, stdout=subprocess.DEVNULL)

    for N, order, mode in experiment_configs:
        print(f"--- Running N={N}, Order={order}, Mode={mode} ({REPETITIONS} reps) ---")
        
        for i in range(REPETITIONS):
            print(f"  -> Repetition {i+1}/{REPETITIONS}")
            perf_path, gprof_runtime, gprof_log_path = run_single_config(N, order, mode, i)
            
            all_raw_log_metadata.append({
                'N': N,
                'Order': order,
                'Mode': mode,
                'Repetition_ID': i + 1,
                'Perf_Log': perf_path,
                'Gprof_Log': gprof_log_path,
                'Gprof_Runtime_s': gprof_runtime 
            })
            
    return all_raw_log_metadata


def process_all_logs_and_save_csv(all_raw_log_metadata):
    """Processes raw logs, calculates means, and saves to CSV."""
    
    universal_data = []
    mean_data_only = []
    
    # Group runs by configuration
    config_groups = defaultdict(list)
    for run in all_raw_log_metadata:
        key = (run['N'], run['Order'], run['Mode'])
        config_groups[key].append(run)

    for (N, order, mode), runs in config_groups.items():
        print(f"--- Processing Logs for N={N}, Order={order}, Mode={mode} ---")
        
        metric_values = defaultdict(list)
        config_runs = []

        for run in runs:
            # 1. Parse Perf logs
            perf_metrics = parse_perf_output(run['Perf_Log'])
            
            # Prepare individual run data row
            run_data = {
                'N': N,
                'Order': order,
                'Mode': mode,
                'Repetition_ID': run['Repetition_ID'],
                'Run_Type': 'Individual',
                'Gprof_Runtime_s': run['Gprof_Runtime_s'],
            }
            run_data.update(perf_metrics)
            config_runs.append(run_data)
            
            # Collect values for calculating the mean
            metric_values['Gprof_Runtime_s'].append(run['Gprof_Runtime_s'])
            for k, v in perf_metrics.items():
                metric_values[k].append(v)
        
        # 2. Calculate Mean Data
        mean_data = {
            'N': N,
            'Order': order,
            'Mode': mode,
            'Repetition_ID': 'AVG',
            'Run_Type': 'Mean',
        }
        
        for metric, values in metric_values.items():
            if 'Gprof' in metric:
                 # Calculate mean for Gprof Runtime
                 mean_data[metric.replace('_s', '_Mean_s')] = np.mean(values)
            else:
                 # Calculate mean for Perf metrics
                 mean_data[metric] = np.mean(values) 

        # Filter mean data to only include mean/perf metrics
        mean_data_summary = {
            k: v for k, v in mean_data.items() 
            if 'Mean' in k or 'Perf' in k or k in ['N', 'Order', 'Mode', 'Repetition_ID', 'Run_Type']
        }
        
        universal_data.extend(config_runs)
        universal_data.append(mean_data_summary)
        mean_data_only.append(mean_data_summary)

    # 3. Save to CSV
    universal_fieldnames, mean_fieldnames = get_fieldnames()
    
    universal_csv_path = os.path.join(CSV_DIR, 'universal_metrics.csv')
    mean_csv_path = os.path.join(CSV_DIR, 'mean_metrics.csv')

    with open(universal_csv_path, 'w', newline='') as f_uni:
        writer = csv.DictWriter(f_uni, fieldnames=universal_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(universal_data)
        
    with open(mean_csv_path, 'w', newline='') as f_mean:
        writer = csv.DictWriter(f_mean, fieldnames=mean_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(mean_data_only)
    
    print(f"\nUniversal metrics (all runs + means) saved to: {universal_csv_path}")
    print(f"Mean metrics (mean only) saved to: {mean_csv_path}")


def main():

    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    print(f"Results (raw data) will be saved in: {RESULTS_DIR}")
    print(f"Metrics (CSV) will be saved in: {CSV_DIR}")
    
    # Generate all combinations of experiments
    # Note: Loop order only applies to 'cpu' mode, but we include it for 'io' too 
    # to maintain structure, though the C code ignores 'order' in 'io' mode.
    experiment_configs = [
        (N, order, mode)
        for N in CONFIGS['N']
        for order in CONFIGS['ORDER']
        for mode in CONFIGS['MODE']
    ]

    print("\n--- PHASE 1: RUNNING ALL EXPERIMENTS & SAVING RAW LOGS ---")
    all_raw_log_metadata = run_all_experiments(experiment_configs)

    print("\n--- PHASE 2: PROCESSING RAW LOGS & GENERATING CSV FILES ---")
    process_all_logs_and_save_csv(all_raw_log_metadata)

    print(f"\n--- ALL PROCESSES COMPLETE ---")
    print(f"Data is available in the '{CSV_DIR}' folder.")


if __name__ == '__main__':
    main()
