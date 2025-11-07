import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import os
import glob
import numpy as np


def find_latest_results_dir():
    search_path = os.path.join(os.getcwd(), 'results_*')
    all_results_dirs = glob.glob(search_path)
    if not all_results_dirs:
        print("Error: Could not find any 'results_YYYY-MM-DD_HHMMSS' directory.")
        return None
    latest_dir = max(all_results_dirs, key=os.path.getctime)
    return latest_dir

def find_latest_csv(latest_results_dir, filename='universal_metrics.csv'):
    if not latest_results_dir:
        return None
    csv_path = os.path.join(latest_results_dir, 'csv', filename)
    if os.path.exists(csv_path):
        return csv_path
    return None


def create_single_plot(df_filtered, metric, plot_dir_path, n_values, group_by_column, title_prefix="", complexity_line=None):

    if df_filtered.empty:
        print(f"Skipping {title_prefix}{metric}: No data after filtering.")
        return

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    for key, group in df_filtered.groupby(group_by_column):
        plot_group = group.copy()
        plot_group[metric] = plot_group[metric].replace([np.inf, -np.inf], np.nan).dropna()
        
        if not plot_group.empty:
             plt.plot(plot_group['N'], plot_group[metric], label=key, marker='o', markersize=4)

    if complexity_line is not None:
        plt.plot(n_values, complexity_line, label=r'$O(N \log_2 N)$ (Scaled)', linestyle='--', color='red')
        
    metric_display_name = metric.replace('_', ' ')
    plt.title(f'{title_prefix}{metric_display_name}', fontsize=16)
    plt.xlabel('N', fontsize=14)
    plt.ylabel(metric_display_name, fontsize=14)
    plt.xscale('log') 
    
    formatter = ticker.ScalarFormatter(useOffset=False, useMathText=False)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticks(n_values)
    ax.set_xticklabels([f'{n}' for n in n_values])
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    
    log_n = np.log10(n_values)
    log_min = log_n.min()
    log_max = log_n.max()
    ax.set_xlim(10**(log_min - 0.05 * (log_max - log_min)), 
                10**(log_max + 0.05 * (log_max - log_min)))
    # ----------------------------------------------------

    plt.legend(title=group_by_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    base_file_name = title_prefix.lower().replace(': ', '_').replace(' ', '_').strip('_')
    if complexity_line is not None:
        base_file_name = 'complexity'
        
    file_name = f"{base_file_name}_{metric.lower().replace('perf_', '').replace('gprof_', '').replace('mean_s', 'runtime')}.png"
    plt.savefig(os.path.join(plot_dir_path, file_name))
    plt.close()
    print(f"  -> Generated {file_name}")


def plot_all_configs_vs_n(df_mean, n_values, plot_dir_path, metric_cols):
    
    for metric in metric_cols:
        create_single_plot(
            df_mean, 
            metric, 
            plot_dir_path, 
            n_values, 
            'Group_Key',
            title_prefix=""
        )

def plot_algorithm_scaling(df_mean, n_values, plot_dir_path):
    
    df_filtered = df_mean[
        (df_mean['Layout'] == 'array') & 
        (df_mean['Type'] == 'int')
    ]
    
    create_single_plot(
        df_filtered, 
        'Gprof_Runtime_Mean_s', 
        plot_dir_path, 
        n_values, 
        'Algorithm',
        title_prefix="Experiment 1: Algorithm Scaling "
    )

def plot_layout_impact(df_mean, n_values, plot_dir_path):
    
    df_filtered = df_mean[
        (df_mean['Algorithm'] == 'merge') & 
        (df_mean['Type'] == 'int')
    ]

    create_single_plot(
        df_filtered, 
        'Perf_IPC', 
        plot_dir_path, 
        n_values, 
        'Layout',
        title_prefix="Experiment 2: Layout Impact "
    )
    create_single_plot(
        df_filtered, 
        'Perf_L1-dcache-load-misses', 
        plot_dir_path, 
        n_values, 
        'Layout',
        title_prefix="Experiment 2: Layout Impact "
    )
    create_single_plot(
        df_filtered, 
        'Perf_dTLB-load-misses', 
        plot_dir_path, 
        n_values, 
        'Layout',
        title_prefix="Experiment 2: Layout Impact "
    )

def plot_data_type_impact(df_mean, n_values, plot_dir_path):
    
    df_filtered = df_mean[
        (df_mean['Algorithm'] == 'merge') & 
        (df_mean['Layout'] == 'array')
    ]

    create_single_plot(
        df_filtered, 
        'Gprof_Runtime_Mean_s', 
        plot_dir_path, 
        n_values, 
        'Type',
        title_prefix="Experiment 3: Data Type Impact "
    )
    create_single_plot(
        df_filtered, 
        'Perf_cycles', 
        plot_dir_path, 
        n_values, 
        'Type',
        title_prefix="Experiment 3: Data Type Impact "
    )

def plot_merge_sort_complexity(df_mean, n_values, plot_dir_path):
    
    df_merge_base = df_mean[
        (df_mean['Algorithm'] == 'merge') & 
        (df_mean['Layout'] == 'array') &
        (df_mean['Type'] == 'int')
    ].sort_values('N')
    
    if df_merge_base.empty:
        print("Skipping Complexity Plot: Cannot find Merge Sort (array, int) data.")
        return

    n_log_n = n_values * np.log2(n_values)
    
    last_measured_runtime = df_merge_base['Gprof_Runtime_Mean_s'].iloc[-1]
    last_n_log_n = n_log_n[-1]
    
    if last_n_log_n > 0:
        scaling_constant = last_measured_runtime / last_n_log_n
    else:
        scaling_constant = 0
    
    scaled_complexity = scaling_constant * n_log_n
    
    create_single_plot(
        df_merge_base, 
        'Gprof_Runtime_Mean_s', 
        plot_dir_path, 
        n_values, 
        'Group_Key',
        title_prefix="Merge Sort Runtime vs. ",
        complexity_line=scaled_complexity
    )



def generate_all_plots(csv_path, plot_dir_path):
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV file at {csv_path}. Error: {e}")
        return

    df_mean = df[df['Run_Type'] == 'Mean'].copy()
    
    if df_mean.empty:
        print("Error: Filtered data for Mean runs is empty.")
        return

    exclude_cols = ['N', 'Layout', 'Algorithm', 'Type', 'Repetition_ID', 'Run_Type', 'Gprof_Runtime_s']
    metric_cols = [col for col in df_mean.columns if col not in exclude_cols and ('Perf_' in col or 'Gprof_Runtime_Mean_s' in col)]

    df_mean['Group_Key'] = df_mean.apply(
        lambda row: f"{row['Algorithm']}-{row['Layout']}-{row['Type']}", axis=1
    )
    
    n_values = sorted(df_mean['N'].unique())
    
    os.makedirs(plot_dir_path, exist_ok=True)
    print(f"\nSaving plots to the directory: {plot_dir_path}")

    print("\n--- Generating Section 3.3 Experiment Plots ---")
    plot_algorithm_scaling(df_mean, n_values, plot_dir_path)
    plot_layout_impact(df_mean, n_values, plot_dir_path)
    plot_data_type_impact(df_mean, n_values, plot_dir_path)

    print("\n--- Generating Merge Sort Complexity Plot ---")
    plot_merge_sort_complexity(df_mean, n_values, plot_dir_path)
    
    print("\n--- Generating All Metrics Plots (Full Configuration) ---")
    plot_all_configs_vs_n(df_mean, n_values, plot_dir_path, metric_cols)


if __name__ == '__main__':
    latest_dir = find_latest_results_dir()

    csv_file = find_latest_csv(latest_dir)
    
    PLOT_DIR_PATH = os.path.join(latest_dir, 'plots')
    
    generate_all_plots(csv_file, PLOT_DIR_PATH)
