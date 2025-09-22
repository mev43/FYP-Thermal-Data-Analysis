import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for better compatibility
import matplotlib.pyplot as plt
import os

def load_thermal_data(filename):
    """Load thermal data from CSV file."""
    try:
        df = pd.read_csv(filename)
        print(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def analyze_thermal_data(df):
    """Analyze thermal data by calculating weekly averages."""
    # Get the cycle numbers (first column)
    cycles = df.iloc[:, 0].values
    
    # Dictionary to store results for each test condition
    results = {}
    
    # The CSV structure has test condition indicators followed by data columns
    # We need to identify the sections based on the structure
    condition_starts = []
    condition_names = []
    
    # Find columns that contain test condition names
    for i, col in enumerate(df.columns):
        if any(test in str(col) for test in ['95A 1st', '95A 2nd', '90A 1st', '90A 2nd', '87A 1st', '87A 2nd']):
            condition_starts.append(i)
            condition_names.append(col)
    
    print(f"Found test conditions at columns: {list(zip(condition_starts, condition_names))}")
    
    # Process each test condition section
    for i, (start_col, condition) in enumerate(zip(condition_starts, condition_names)):
        print(f"\nProcessing {condition} starting at column {start_col}...")
        
        # Each condition should have 9 data columns (3 SVF levels × 3 weeks)
        # Find the next condition start or end of dataframe
        if i + 1 < len(condition_starts):
            end_col = condition_starts[i + 1]
        else:
            end_col = len(df.columns)
        
        # Get the data columns for this condition (skip the condition name column and empty columns)
        data_start = start_col + 1
        data_cols = []
        
        for col_idx in range(data_start, end_col):
            if col_idx < len(df.columns):
                col_name = df.columns[col_idx]
                # Skip unnamed/empty columns
                if not col_name.startswith('Unnamed') and 'Wk' in col_name:
                    data_cols.append(col_idx)
        
        print(f"Data columns: {[df.columns[idx] for idx in data_cols]}")
        
        # Group by SVF (50%, 35%, 20%)
        svf_levels = ['50%', '35%', '20%']
        condition_results = {}

        # Assume 3 columns per SVF (3 weeks each)
        for svf_idx, svf in enumerate(svf_levels):
            # Get 3 columns for this SVF
            start_idx = svf_idx * 3
            end_idx = start_idx + 3
            
            if start_idx < len(data_cols) and end_idx <= len(data_cols):
                svf_cols = data_cols[start_idx:end_idx]
                svf_col_names = [df.columns[idx] for idx in svf_cols]

                print(f"  {svf} columns: {svf_col_names}")

                # Extract data and convert to numeric
                svf_data = df.iloc[:, svf_cols].copy()
                
                # Convert to numeric, replacing empty strings and non-numeric values with NaN
                for col_idx in svf_cols:
                    svf_data.iloc[:, svf_cols.index(col_idx)] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')

                # Calculate average across weeks (excluding NaN values)
                averages = svf_data.mean(axis=1, skipna=True)
                
                # Store individual week data as well
                week_data = {}
                for week_idx, col_idx in enumerate(svf_cols):
                    week_num = week_idx + 1
                    week_values = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
                    valid_week_mask = ~week_values.isna()
                    if valid_week_mask.any():
                        week_data[f'week{week_num}'] = {
                            'cycles': cycles[valid_week_mask],
                            'values': week_values[valid_week_mask].values
                        }

                # Only keep rows where we have at least one valid measurement
                valid_mask = ~averages.isna()
                
                if valid_mask.any():
                    condition_results[svf] = {
                        'cycles': cycles[valid_mask],
                        'averages': averages[valid_mask].values,
                        'weeks': week_data
                    }
                    print(f"    {svf}: {len(averages[valid_mask])} valid data points")
                else:
                    print(f"    {svf}: No valid data points")

        results[condition] = condition_results
    
    return results

def plot_thermal_results(results):
    """Create plots for thermal analysis results."""
    # Create subplots for each test condition
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Thermal Data Analysis - Weekly Averages by Test Condition', fontsize=16)
    
    axes = axes.flatten()
    colors = {'50%': '#FF4444', '35%': '#4444FF', '20%': '#44AA44'}
    
    for idx, (condition, data) in enumerate(results.items()):
        ax = axes[idx]

        for SVF_level, svf_data in data.items():
            if svf_data['averages'].size > 0:  # Only plot if we have data
                ax.plot(svf_data['cycles'], svf_data['averages'], 
                       marker='o', label=f'{SVF_level}', color=colors[SVF_level], 
                       linewidth=2, markersize=4)
        
        ax.set_title(condition)
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        if any(data[svf]['averages'].size > 0 for svf in data.keys()):
            all_temps = np.concatenate([data[svf]['averages'] 
                                      for svf in data.keys() 
                                      if data[svf]['averages'].size > 0])
            if len(all_temps) > 0:
                y_min, y_max = np.min(all_temps), np.max(all_temps)
                margin = (y_max - y_min) * 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
    
    plt.tight_layout()
    return fig

def create_summary_plot(results):
    """Create a summary plot comparing all conditions."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Thermal Data Summary - Comparison Across Test Conditions', fontsize=16)

    svf_levels = ['50%', '35%', '20%']
    colors = ['#FF0000', '#FF6600', '#FF9900', '#0066FF', '#0099FF', '#00CCFF', '#00AA00', '#66CC00', '#99FF00']

    for svf_idx, svf in enumerate(svf_levels):
        ax = axes[svf_idx]

        color_idx = 0
        for condition, data in results.items():
            if svf in data and data[svf]['averages'].size > 0:
                ax.plot(data[svf]['cycles'], data[svf]['averages'], 
                       marker='o', label=condition, color=colors[color_idx % len(colors)], 
                       linewidth=2, markersize=3)
                color_idx += 1

        ax.set_title(f'SVF Level: {svf}')
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_statistics(results):
    """Print statistical summary of the data."""
    print("\n" + "="*60)
    print("STATISTICAL SUMMARY")
    print("="*60)
    
    for condition, data in results.items():
        print(f"\n{condition}:")
        for SVF_level, svf_data in data.items():
            if svf_data['averages'].size > 0:
                temps = svf_data['averages']
                print(f"  {SVF_level}:")
                print(f"    Count: {len(temps)}")
                print(f"    Mean: {np.mean(temps):.2f}°C")
                print(f"    Std: {np.std(temps):.2f}°C")
                print(f"    Min: {np.min(temps):.2f}°C")
                print(f"    Max: {np.max(temps):.2f}°C")

def plot_weekly_comparisons(results):
    """Create plots comparing 95A vs 90A vs 87A for each week."""
    # Extract temperature levels from condition names
    temperature_mapping = {}
    for condition in results.keys():
        if '95A' in condition:
            temp_level = '95A'
        elif '90A' in condition:
            temp_level = '90A'
        elif '87A' in condition:
            temp_level = '87A'
        else:
            continue

        if temp_level not in temperature_mapping:
            temperature_mapping[temp_level] = []
        temperature_mapping[temp_level].append(condition)

    # Create plots for each week (1, 2, 3)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Weekly Comparison: 95A vs 90A vs 87A', fontsize=16)
    
    weeks = ['week1', 'week2', 'week3']
    week_labels = ['Week 1', 'Week 2', 'Week 3']
    svf_levels = ['50%', '35%', '20%']
    
    # Define unique colors for each temperature and SVF combination
    colors_combo = {
        ('95A', '50%'): '#FF0000',  # Red
        ('95A', '35%'): '#FF6600',  # Orange-Red
        ('95A', '20%'): '#FF9900',  # Orange
        ('90A', '50%'): '#0066FF',  # Blue
        ('90A', '35%'): '#0099FF',  # Light Blue
        ('90A', '20%'): '#00CCFF',  # Cyan
        ('87A', '50%'): '#00AA00',  # Green
        ('87A', '35%'): '#66CC00',  # Yellow-Green
        ('87A', '20%'): '#99FF00',  # Lime Green
    }
    
    for week_idx, (week, week_label) in enumerate(zip(weeks, week_labels)):
        ax = axes[week_idx]
        
        for temp_level in ['95A', '90A', '87A']:
            if temp_level in temperature_mapping:
                # Average data from both 1st and 2nd tests for this temperature
                for svf in svf_levels:
                    # Collect data from all conditions (1st and 2nd) for this temperature
                    cycle_data = {}  # Dictionary to store values by cycle

                    for condition in temperature_mapping[temp_level]:
                        if svf in results[condition] and 'weeks' in results[condition][svf]:
                            week_data = results[condition][svf]['weeks']
                            if week in week_data:
                                cycles = week_data[week]['cycles']
                                values = week_data[week]['values']
                                
                                # Group values by cycle number
                                for cycle, value in zip(cycles, values):
                                    if cycle not in cycle_data:
                                        cycle_data[cycle] = []
                                    cycle_data[cycle].append(value)
                    
                    # Calculate average for each cycle and prepare for plotting
                    if cycle_data:
                        avg_cycles = []
                        avg_values = []
                        
                        for cycle in sorted(cycle_data.keys()):
                            avg_cycles.append(cycle)
                            avg_values.append(np.mean(cycle_data[cycle]))  # Average across tests
                        
                        color = colors_combo.get((temp_level, svf), '#000000')  # Default to black if not found
                        ax.plot(avg_cycles, avg_values, 
                               color=color,
                               marker='o', markersize=3,
                               label=f'{temp_level} - {svf}',
                               linewidth=2)
        
        ax.set_title(week_label)
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Temperature (°C)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_week_differences(results):
    """Create plots showing differences between weeks."""
    # Extract temperature levels from condition names
    temperature_mapping = {}
    for condition in results.keys():
        if '95A' in condition:
            temp_level = '95A'
        elif '90A' in condition:
            temp_level = '90A'
        elif '87A' in condition:
            temp_level = '87A'
        else:
            continue

        if temp_level not in temperature_mapping:
            temperature_mapping[temp_level] = []
        temperature_mapping[temp_level].append(condition)

    # Create plots for differences (Week2-Week1, Week3-Week1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Temperature Differences Between Weeks', fontsize=16)
    
    svf_levels = ['50%', '35%', '20%']
    
    # Define unique colors for each temperature and SVF combination
    colors_combo = {
        ('95A', '50%'): '#FF0000',  # Red
        ('95A', '35%'): '#FF6600',  # Orange-Red
        ('95A', '20%'): '#FF9900',  # Orange
        ('90A', '50%'): '#0066FF',  # Blue
        ('90A', '35%'): '#0099FF',  # Light Blue
        ('90A', '20%'): '#00CCFF',  # Cyan
        ('87A', '50%'): '#00AA00',  # Green
        ('87A', '35%'): '#66CC00',  # Yellow-Green
        ('87A', '20%'): '#99FF00',  # Lime Green
    }
    
    diff_configs = [
        ('Week 2 - Week 1', 'week2', 'week1'),
        ('Week 3 - Week 1', 'week3', 'week1')
    ]
    
    for diff_idx, (diff_label, week_later, week_base) in enumerate(diff_configs):
        ax = axes[diff_idx]
        for temp_level in ['95A', '90A', '87A']:
            if temp_level in temperature_mapping:
                for svf in svf_levels:
                    # Collect data from all conditions (1st and 2nd) for this temperature
                    later_cycle_data = {}  # Dictionary to store values by cycle for later week
                    base_cycle_data = {}   # Dictionary to store values by cycle for base week

                    for condition in temperature_mapping[temp_level]:
                        if svf in results[condition] and 'weeks' in results[condition][svf]:
                            week_data = results[condition][svf]['weeks']
                            
                            if week_later in week_data and week_base in week_data:
                                # Collect later week data
                                later_cycles = week_data[week_later]['cycles']
                                later_values = week_data[week_later]['values']
                                for cycle, value in zip(later_cycles, later_values):
                                    if cycle not in later_cycle_data:
                                        later_cycle_data[cycle] = []
                                    later_cycle_data[cycle].append(value)
                                
                                # Collect base week data
                                base_cycles = week_data[week_base]['cycles']
                                base_values = week_data[week_base]['values']
                                for cycle, value in zip(base_cycles, base_values):
                                    if cycle not in base_cycle_data:
                                        base_cycle_data[cycle] = []
                                    base_cycle_data[cycle].append(value)
                    
                    # Calculate differences for common cycles
                    common_cycles = set(later_cycle_data.keys()) & set(base_cycle_data.keys())
                    if common_cycles:
                        diff_cycles = []
                        diff_values = []
                        
                        for cycle in sorted(common_cycles):
                            later_avg = np.mean(later_cycle_data[cycle])
                            base_avg = np.mean(base_cycle_data[cycle])
                            diff = later_avg - base_avg
                            diff_cycles.append(cycle)
                            diff_values.append(diff)
                        
                        color = colors_combo.get((temp_level, svf), '#000000')  # Default to black if not found
                        ax.plot(diff_cycles, diff_values, 
                               color=color,
                               marker='o', markersize=3,
                               label=f'{temp_level} - {svf}',
                               linewidth=2)
        
        ax.set_title(diff_label)
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Temperature Difference (°C)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    return fig

def main():
    """Main analysis function."""
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, 'Thermal Data.csv')
    
    print("Thermal Data Analysis")
    print("="*40)
    
    # Load data
    df = load_thermal_data(csv_file)
    if df is None:
        return
    
    # Analyze data
    results = analyze_thermal_data(df)
    
    # Print statistics
    print_statistics(results)
    
    # Create plots
    print("\nCreating plots...")
    
    # Individual condition plots
    fig1 = plot_thermal_results(results)
    
    # Summary comparison plots
    fig2 = create_summary_plot(results)
    
    # Weekly comparison plots (95A vs 90A vs 87A for each week)
    fig3 = plot_weekly_comparisons(results)
    
    # Week difference plots (Week2-Week1, Week3-Week1)
    fig4 = plot_week_differences(results)
    
    # Save plots
    plot1_path = os.path.join(script_dir, 'thermal_analysis_individual.png')
    plot2_path = os.path.join(script_dir, 'thermal_analysis_summary.png')
    plot3_path = os.path.join(script_dir, 'thermal_weekly_comparison.png')
    plot4_path = os.path.join(script_dir, 'thermal_week_differences.png')
    
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    fig3.savefig(plot3_path, dpi=300, bbox_inches='tight')
    fig4.savefig(plot4_path, dpi=300, bbox_inches='tight')
    
    print(f"Plots saved as:")
    print(f"  - {plot1_path}")
    print(f"  - {plot2_path}")
    print(f"  - {plot3_path}")
    print(f"  - {plot4_path}")
    
    # Show plots
    plt.show()

if __name__ == "__main__":
    main()