import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 
import os
import glob
import numpy as np

# --- Configuration ---
PLOT_DIR = 'plots'
CSV_FILENAME = 'mean_metrics.csv'

# Metrics of interest grouped by their relevance for analysis
TASK1_2_METRICS = [
    'Gprof_Runtime_Mean_s',  # Overall time (Task 1 & 2)
    'Perf_task-clock',       # Time spent running CPU/kernel (Task 1 & 2)
    'Perf_cycles',           # CPU activity (Task 1, high in CPU mode)
    'Perf_instructions',     # Total work done (Task 1, high in CPU mode)
]

TASK3_METRICS = [
    'Gprof_Runtime_Mean_s',  # Overall time for comparison
    'Perf_Cache_Miss_Rate',  # Key metric for cache analysis (Task 3)
    'Perf_L1-dcache-load-misses', # L1 miss count (Task 3)
    'Perf_IPC',              # Instructions per Cycle (Efficiency metric)
]

# --- Helper Functions ---

def find_latest_results_dir():
    """Finds the most recently created 'results_YYYY-MM-DD_HHMMSS' directory."""
    search_path = os.path.join(os.getcwd(), 'results_*')
    all_results_dirs = glob.glob(search_path)
    if not all_results_dirs:
        return None
    latest_dir = max(all_results_dirs, key=os.path.getctime)
    return latest_dir

def find_latest_csv(latest_results_dir, filename=CSV_FILENAME):
    """Constructs the path to the mean metrics CSV file."""
    if not latest_results_dir:
        return None
    csv_path = os.path.join(latest_results_dir, 'csv', filename)
    if os.path.exists(csv_path):
        return csv_path
    return None

def sanitize_filename(text):
    """Creates a safe filename part from a string."""
    return text.replace('Perf_', '').replace('_Mean_s', '').replace('/', '_').replace(':', '').replace(' ', '_').lower()

def get_plot_label(metric, units):
    """Generates the plot title and axis label."""
    metric_name = metric.replace('Perf_', '').replace('_Mean_s', ' Runtime (s)').replace('_', ' ')
    title = f"{metric_name} Scaling"
    y_label = f"{metric_name} ({units})"
    return title, y_label

def create_scaling_plot(df_filtered, metric, plot_dir_path, x_column, group_by_column, title_prefix=""):
    """
    Creates and saves a single plot showing how a metric scales with N, 
    grouped by a key column (Mode, Order, or combined Config).
    """

    if df_filtered.empty or metric not in df_filtered.columns:
        print(f"Skipping {title_prefix}{metric}: No valid data found.")
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Ensure all data types for the metric are numeric, replacing errors with NaN
    df_filtered[metric] = pd.to_numeric(df_filtered[metric], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    
    # Check if the metric column is now empty after coercing to numeric
    if df_filtered[metric].empty:
        plt.close()
        return

    for key, group in df_filtered.groupby(group_by_column):
        # Format the key for the legend (e.g., 'cpu_ijk' -> 'CPU ijk')
        legend_label = str(key).upper().replace('_', ' ').replace('PERF', 'Perf')
        
        # Sort by x-column (N) to ensure correct line plotting
        plot_group = group.sort_values(by=x_column)
        
        # Plot the line, only for groups with valid data
        if not plot_group.empty:
            plt.plot(plot_group[x_column], plot_group[metric], marker='o', linestyle='-', label=legend_label)

    # --- Plot Customization ---
    units = 'Seconds' if 'Runtime' in metric else ('%' if 'Rate' in metric else 'Count')
    
    # Determine if log scale is needed
    use_log_scale = any(m in metric for m in ['Runtime', 'task-clock', 'cycles', 'instructions', 'misses'])
    
    title_text, y_label_base = get_plot_label(metric, units)

    plt.title(f"{title_prefix}{title_text}")
    plt.xlabel(f"Matrix Size N ({x_column})")
    
    if use_log_scale:
        ax.set_yscale('log')
        y_label = f"Log Scale {y_label_base}"
    else:
        y_label = y_label_base

    plt.ylabel(y_label)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.legend(title=group_by_column.capitalize(), loc='best')
    plt.tight_layout()
    
    # --- FILENAME FIX ---
    safe_title_prefix = sanitize_filename(title_prefix)
    
    # Save the file
    filename = f"{safe_title_prefix}_{sanitize_filename(metric)}.png"
    plt.savefig(os.path.join(plot_dir_path, filename))
    plt.close()


# --- Task-Specific Plotting Functions ---

def plot_all_metrics_by_config(df, plot_dir_path):
    """
    Generates plots for ALL collected metrics against N, grouped by the 
    combined configuration (Mode_Order). This covers the general request.
    """
    print("\n--- Generating ALL Metrics Plots: Grouped by Mode & Order ---")
    
    # Create a combined 'Config' column for unified grouping
    df['Config'] = df['Mode'] + '_' + df['Order']
    
    # Identify all columns that appear to be metrics (start with Gprof or Perf)
    metric_columns = [col for col in df.columns if col.startswith('Gprof_') or col.startswith('Perf_')]
    
    if not metric_columns:
        print("Skipping All Metrics plots: No columns identified as metrics (starting with Gprof_ or Perf_).")
        return

    for metric in metric_columns:
        create_scaling_plot(
            df_filtered=df.copy(), # Pass a fresh copy to avoid persistent data manipulation
            metric=metric,
            plot_dir_path=plot_dir_path,
            x_column='N',
            group_by_column='Config',
            title_prefix="All Configs - ",
        )
    # Drop the temporary column after use
    df.drop(columns=['Config'], inplace=True, errors='ignore')


def plot_cpu_vs_io(df, plot_dir_path):
    """Generates plots for Task 1/2: Comparing CPU vs I/O mode performance."""
    print("\n--- Generating Task 1 & 2 Plots: CPU vs I/O Mode (ijk only) ---")
    
    # Filter: Only use 'ijk' loop order for both modes for a clean comparison
    df_filtered = df[df['Order'] == 'ijk'].copy()
    
    if df_filtered.empty:
        print("Skipping Task 1/2 plots: No 'ijk' data found for mode comparison.")
        return

    for metric in TASK1_2_METRICS:
        create_scaling_plot(
            df_filtered=df_filtered,
            metric=metric,
            plot_dir_path=plot_dir_path,
            x_column='N',
            group_by_column='Mode',
            title_prefix="Task 1_2 (ijk) - Mode Comparison - ",
        )
        

def plot_loop_order_effects(df, plot_dir_path):
    """Generates plots for Task 3: Comparing ijk, ikj, jik loop orders in CPU mode."""
    print("\n--- Generating Task 3 Plots: Loop Order Comparison (CPU Mode) ---")
    
    # Filter: Only look at CPU mode runs
    df_cpu = df[df['Mode'] == 'cpu'].copy()
    
    if df_cpu.empty:
        print("Skipping Task 3 plots: No 'cpu' data found for loop order comparison.")
        return

    for metric in TASK3_METRICS:
        create_scaling_plot(
            df_filtered=df_cpu,
            metric=metric,
            plot_dir_path=plot_dir_path,
            x_column='N',
            group_by_column='Order',
            title_prefix="Task 3 (CPU) - Loop Order - ",
        )


# --- Main Execution ---

def main():
    # 1. Find the latest results directory
    latest_results_dir = find_latest_results_dir()
    if not latest_results_dir:
        print("Error: Could not find any results directory. Please run the profile_runner.py script first.")
        return

    # 2. Find the mean metrics CSV
    csv_path = find_latest_csv(latest_results_dir, CSV_FILENAME)
    if not csv_path:
        print(f"Error: Could not find '{CSV_FILENAME}' in the latest results directory: {latest_results_dir}")
        return

    # 3. Load Data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV file at {csv_path}. Error: {e}")
        return

    # Filter for only the mean runs and drop any rows where 'N' is missing or not numeric
    df_mean = df[df['Run_Type'] == 'Mean'].copy()
    df_mean['N'] = pd.to_numeric(df_mean['N'], errors='coerce')
    df_mean.dropna(subset=['N'], inplace=True)
    
    if df_mean.empty:
        print("Error: Filtered data for Mean runs is empty or 'N' column is invalid.")
        return

    # Setup plot directory
    plot_dir_path = os.path.join(latest_results_dir, PLOT_DIR)
    os.makedirs(plot_dir_path, exist_ok=True)
    print(f"\nResults loaded from: {csv_path}")
    print(f"Saving plots to: {plot_dir_path}")

    # 4. Generate Plots
    
    # 4a. General Plots: All metrics grouped by Mode_Order combination (New)
    plot_all_metrics_by_config(df_mean.copy(), plot_dir_path)

    # 4b. Task 1 & 2 Plots: CPU vs I/O Mode (ijk only)
    plot_cpu_vs_io(df_mean.copy(), plot_dir_path)
    
    # 4c. Task 3 Plots: Loop Order Effects (CPU only)
    plot_loop_order_effects(df_mean.copy(), plot_dir_path)
    
    print("\n--- Plot Generation Complete ---")
    print("Check the plots directory for the PNG files.")


if __name__ == '__main__':
    main()
