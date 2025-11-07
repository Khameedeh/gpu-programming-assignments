import os
import subprocess
import time
import random
import csv
import numpy as np
from collections import defaultdict

BINARY_NAME = './layout_sort_file_profile'
INPUT_FILE = 'input.txt'
CONFIGS = {
    'n': [32000, 64000, 128000, 256000],
    'layout': ['array', 'list'],
    'alg': ['insertion', 'bubble', 'merge'],
    'type': ['int', 'double'],
}
REPETITIONS = 5
FIXED_SEED = 42
RESULTS_DIR = f"results_{time.strftime('%Y-%m-%d_%H%M%S')}"
CSV_DIR = os.path.join(RESULTS_DIR, 'csv')

PERF_COUNTERS = [
    'task-clock',
    'cycles',
    'instructions',
    'branches',
    'branch-misses',
    'L1-dcache-load-misses',
    'dTLB-load-misses'
]
PERF_COUNTERS_STR = ",".join(PERF_COUNTERS)


def generate_input_data(N, data_type):
    random.seed(FIXED_SEED)
    
    if data_type == 'int':
        data = [random.randint(1, 100000) for _ in range(N)]
        fmt_str = '{}'
    elif data_type == 'double':
        data = [random.uniform(1.0, 100000.0) for _ in range(N)]
        fmt_str = '{:.6f}'
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    with open(INPUT_FILE, 'w') as f:
        f.write(f'{N}\n')
        f.write(' '.join(fmt_str.format(x) for x in data))

def parse_perf_output(file_path):
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
                parts = line.split(',')
                try:
                    value = float(parts[0].replace(',', ''))
                    if 'task-clock' in counter and 'msec' in line:
                         value /= 1000.0
                    metrics[f'Perf_{counter}'] = value
                except (IndexError, ValueError):
                    pass
    
    instr = metrics.get('Perf_instructions')
    cyc = metrics.get('Perf_cycles')
    if instr is not None and cyc is not None and cyc > 0:
        metrics['Perf_IPC'] = instr / cyc
    else:
        metrics['Perf_IPC'] = np.nan
        
    return metrics

def run_single_config(n, layout, alg, data_type, i):
    
    generate_input_data(n, data_type)
    
    base_cmd = [
        BINARY_NAME, 
        '-layout', layout, 
        '-alg', alg, 
        '-t', data_type, 
        '-file', INPUT_FILE
    ]
    config_name = f'N{n}_ALG{alg}_TYPE{data_type}_LAYOUT{layout}'
    
    perf_cmd = [
        'perf', 'stat', '-x', ',', 
        '-e', PERF_COUNTERS_STR, 
        *base_cmd
    ]
    
    result = subprocess.run(perf_cmd, capture_output=True, text=True, check=False)
    
    perf_log_path = os.path.join(RESULTS_DIR, f'raw_perf_{config_name}_{i}.txt')
    with open(perf_log_path, 'w') as f:
        f.write(result.stderr)
        
    start_time = time.time()
    subprocess.run(base_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    end_time = time.time()
    gprof_runtime = end_time - start_time 

    gprof_report_cmd = ['gprof', BINARY_NAME, 'gmon.out']
    gprof_report = subprocess.run(gprof_report_cmd, capture_output=True, text=True, check=False)
    
    gprof_log_path = os.path.join(RESULTS_DIR, f'raw_gprof_{config_name}_{i}.txt')
    with open(gprof_log_path, 'w') as f:
        f.write(gprof_report.stdout)

    if os.path.exists('gmon.out'):
         os.remove('gmon.out')

    return perf_log_path, gprof_runtime


def get_fieldnames():
    base_fields = ['N', 'Layout', 'Algorithm', 'Type', 'Repetition_ID', 'Run_Type']
    
    individual_metrics = [f'Gprof_Runtime_s'] + [f'Perf_{c}' for c in PERF_COUNTERS] + ['Perf_IPC']
    
    mean_metrics = ['Gprof_Runtime_Mean_s'] + [f for f in individual_metrics if f.startswith('Perf_')] 

    universal_fieldnames = base_fields + sorted(list(set(individual_metrics + mean_metrics)), key=lambda x: ('Perf' not in x, x))
    
    mean_fieldnames = base_fields + sorted(mean_metrics)
    
    return universal_fieldnames, mean_fieldnames


def run_all_experiments(experiment_configs):
    
    all_raw_log_metadata = []
    
    subprocess.run(['make', 'build_perf'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['make', 'build_gprof'], check=True, stdout=subprocess.DEVNULL)

    for n, layout, alg, data_type in experiment_configs:
        print(f"--- Running N={n}, Layout={layout}, Alg={alg}, Type={data_type} ({REPETITIONS} reps) ---")
        
        for i in range(REPETITIONS):
            print(f"  -> Repetition {i+1}/{REPETITIONS}")
            perf_path, gprof_runtime = run_single_config(n, layout, alg, data_type, i)
            
            all_raw_log_metadata.append({
                'N': n,
                'Layout': layout,
                'Algorithm': alg,
                'Type': data_type,
                'Repetition_ID': i + 1,
                'Perf_Log': perf_path,
                'Gprof_Runtime_s': gprof_runtime 
            })
            
    return all_raw_log_metadata


def process_all_logs_and_save_csv(all_raw_log_metadata):
    
    universal_data = []
    mean_data_only = []
    
    config_groups = defaultdict(list)
    for run in all_raw_log_metadata:
        key = (run['N'], run['Layout'], run['Algorithm'], run['Type'])
        config_groups[key].append(run)

    for (n, layout, alg, data_type), runs in config_groups.items():
        print(f"--- Processing Logs for N={n}, Layout={layout}, Alg={alg}, Type={data_type} ---")
        
        metric_values = defaultdict(list)
        config_runs = []

        for run in runs:
            perf_metrics = parse_perf_output(run['Perf_Log'])
            
            run_data = {
                'N': n,
                'Layout': layout,
                'Algorithm': alg,
                'Type': data_type,
                'Repetition_ID': run['Repetition_ID'],
                'Run_Type': 'Individual',
                'Gprof_Runtime_s': run['Gprof_Runtime_s'],
            }
            run_data.update(perf_metrics)
            config_runs.append(run_data)
            
            metric_values['Gprof_Runtime_s'].append(run['Gprof_Runtime_s'])
            for k, v in perf_metrics.items():
                metric_values[k].append(v)
        
        mean_data = {
            'N': n,
            'Layout': layout,
            'Algorithm': alg,
            'Type': data_type,
            'Repetition_ID': 'AVG',
            'Run_Type': 'Mean',
        }
        
        for metric, values in metric_values.items():
            if 'Gprof' in metric:
                 mean_data[metric.replace('_s', '_Mean_s')] = np.mean(values)
            else:
                 mean_data[metric] = np.mean(values) 

        mean_data_summary = {k: v for k, v in mean_data.items() if 'Mean' in k or 'Perf' in k or k in ['N', 'Layout', 'Algorithm', 'Type', 'Repetition_ID', 'Run_Type']}
        
        universal_data.extend(config_runs)
        universal_data.append(mean_data_summary)
        mean_data_only.append(mean_data_summary)

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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)
    print(f"Results (raw data) will be saved in: {RESULTS_DIR}")
    print(f"Metrics (CSV) will be saved in: {CSV_DIR}")
    
    experiment_configs = [
        (n, layout, alg, data_type)
        for n in CONFIGS['n']
        for layout in CONFIGS['layout']
        for alg in CONFIGS['alg']
        for data_type in CONFIGS['type']
    ]

    print("\n--- PHASE 1: RUNNING ALL EXPERIMENTS & SAVING RAW LOGS ---")
    all_raw_log_metadata = run_all_experiments(experiment_configs)

    print("\n--- PHASE 2: PROCESSING RAW LOGS & GENERATING CSV FILES ---")
    process_all_logs_and_save_csv(all_raw_log_metadata)

    print(f"\n--- ALL PROCESSES COMPLETE ---")
    print(f"Data is available in the '{CSV_DIR}' folder.")


if __name__ == '__main__':
    main()