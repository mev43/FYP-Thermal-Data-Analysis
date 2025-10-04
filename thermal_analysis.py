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
    # Get cycle numbers from the first column
    cycles = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    
    # Find valid data rows (rows with actual cycle numbers)
    valid_data_mask = ~cycles.isna()
    valid_data_rows = df.index[valid_data_mask].tolist()
    cycles = cycles[valid_data_mask].values
    
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
        
        # Group by SVF (50%, 35%, 20%) and dynamically determine weeks per SVF
        svf_levels = ['50%', '35%', '20%']
        condition_results = {}

        # Parse the column structure to identify SVF sections and their weeks
        svf_sections = {'50%': [], '35%': [], '20%': []}
        
        for col_idx in data_cols:
            col_name = df.columns[col_idx]
            for svf in svf_levels:
                if svf in col_name:
                    svf_sections[svf].append(col_idx)
                    break
        
        for svf in svf_levels:
            svf_cols = svf_sections[svf]
            if not svf_cols:
                continue
                
            svf_col_names = [df.columns[idx] for idx in svf_cols]
            print(f"  {svf} columns: {svf_col_names}")

            # Extract data and convert to numeric, but only from valid data rows
            svf_data_list = []
            for row_idx in valid_data_rows:
                row_data = []
                for col_idx in svf_cols:
                    val = pd.to_numeric(df.iloc[row_idx, col_idx], errors='coerce')
                    row_data.append(val)
                svf_data_list.append(row_data)
            
            svf_data = pd.DataFrame(svf_data_list, columns=svf_col_names)

            # Calculate average across weeks (excluding NaN values)
            averages = svf_data.mean(axis=1, skipna=True)
            
            # Store individual week data as well
            week_data = {}
            for week_idx, col_idx in enumerate(svf_cols):
                week_num = week_idx + 1
                week_values = svf_data.iloc[:, week_idx]
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

def create_output_directories(base_dir):
    """Create organized output directories for plots."""
    directories = {
        'material_comparisons': os.path.join(base_dir, 'plots', 'material_comparisons'),
        'svf_comparisons': os.path.join(base_dir, 'plots', 'svf_comparisons'),
        'all_weeks_raw': os.path.join(base_dir, 'plots', 'all_weeks_raw'),
        'week_differences': os.path.join(base_dir, 'plots', 'week_differences')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories

def plot_material_svf_comparison(results, output_dir):
    """For each material condition, create separate plots for each week comparing SVF levels."""
    colors = {'50%': '#FF4444', '35%': '#4444FF', '20%': '#44AA44'}
    
    for condition, data in results.items():
        if not data:  # Skip if no data
            continue
            
        # Find all available weeks across all SVF levels
        all_weeks = set()
        for svf in ['50%', '35%', '20%']:
            if svf in data and 'weeks' in data[svf]:
                all_weeks.update(data[svf]['weeks'].keys())
        
        week_numbers = sorted([int(w.replace('week', '')) for w in all_weeks if w.startswith('week')])
        
        if not week_numbers:
            continue
            
        # Create separate plot for each week
        for week_num in week_numbers:
            week_key = f'week{week_num}'
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            # Extract material number from condition (e.g., "95A" from "95A 1st")
            material_num = condition.split()[0]
            fig.suptitle(f'SVF Comparison - Filament TPU{material_num} (Test {week_num} Temperature Histories)', fontsize=16)
            
            has_data = False
            for svf in ['50%', '35%', '20%']:
                if svf in data and 'weeks' in data[svf] and week_key in data[svf]['weeks']:
                    week_data = data[svf]['weeks'][week_key]
                    ax.plot(week_data['cycles'], week_data['values'], 
                           color=colors[svf], marker='o', markersize=4, linewidth=2,
                           label=f'{svf}')
                    has_data = True
            
            if has_data:
                ax.set_xlabel('Cycle Number')
                ax.set_ylabel('Temperature (°C)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                filename = f'{condition.replace(" ", "_")}_week{week_num}_svf_comparison.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {filename}")
            else:
                plt.close()

def plot_svf_material_comparison(results, output_dir):
    """For each SVF level, create separate plots for each week comparing materials, separating 1st and 2nd sets."""
    # Extract material and set information
    materials = {'95A': {'1st': [], '2nd': []}, '90A': {'1st': [], '2nd': []}, '87A': {'1st': [], '2nd': []}}
    
    for condition in results.keys():
        material = None
        set_type = None
        
        if '95A' in condition:
            material = '95A'
        elif '90A' in condition:
            material = '90A'
        elif '87A' in condition:
            material = '87A'
        
        if '1st' in condition:
            set_type = '1st'
        elif '2nd' in condition:
            set_type = '2nd'
        
        if material and set_type:
            materials[material][set_type].append(condition)
    
    # Colors for different materials and sets
    material_colors = {
        '95A': {'1st': '#FF0000', '2nd': '#FF6666'},  # Red shades
        '90A': {'1st': '#0066FF', '2nd': '#6699FF'},  # Blue shades
        '87A': {'1st': '#00AA00', '2nd': '#66CC66'}   # Green shades
    }
    
    # Create plots for each SVF level and week
    for svf in ['50%', '35%', '20%']:
        # Find all available weeks for this SVF
        all_weeks = set()
        for condition, data in results.items():
            if svf in data and 'weeks' in data[svf]:
                all_weeks.update(data[svf]['weeks'].keys())
        
        week_numbers = sorted([int(w.replace('week', '')) for w in all_weeks if w.startswith('week')])
        
        if not week_numbers:
            continue
        
        # Create separate plots for 1st and 2nd sets, and for each week
        for set_type in ['1st', '2nd']:
            for week_num in week_numbers:
                week_key = f'week{week_num}'
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                fig.suptitle(f'Material Comparison - SVF {svf} (Test {week_num} Temperature Histories)', fontsize=16)
                
                has_data = False
                for material in ['95A', '90A', '87A']:
                    for condition in materials[material][set_type]:
                        if (condition in results and svf in results[condition] and 
                            'weeks' in results[condition][svf] and 
                            week_key in results[condition][svf]['weeks']):
                            
                            week_data = results[condition][svf]['weeks'][week_key]
                            color = material_colors[material][set_type]
                            
                            ax.plot(week_data['cycles'], week_data['values'], 
                                   color=color, marker='o', markersize=4, linewidth=2,
                                   label=f'Filament TPU{material}')
                            has_data = True
                
                if has_data:
                    ax.set_xlabel('Cycle Number')
                    ax.set_ylabel('Temperature (°C)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Save plot
                    filename = f'{svf.replace("%", "percent")}_SVF_{set_type}_set_week{week_num}_material_comparison.png'
                    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved: {filename}")
                else:
                    plt.close()

def plot_all_weeks_raw_data(results, output_dir):
    """Plot all weeks of raw data together for each material/SVF combination."""
    # Define colors for different weeks
    week_colors = ['#FF4444', '#4444FF', '#44AA44', '#FFA500', '#800080', '#00CED1', '#A52A2A', '#FFD700']
    
    for condition, data in results.items():
        for svf in ['50%', '35%', '20%']:
            if svf in data and 'weeks' in data[svf]:
                weeks = data[svf]['weeks']
                if weeks:  # If there are any weeks
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    # Extract material and condition info for title
                    material_num = condition.split()[0]  # e.g., "95A" from "95A 1st"
                    fig.suptitle(f'All Tests - Filament TPU{material_num} SVF {svf} Temperature Histories', fontsize=16)
                    
                    # Sort week keys by week number
                    week_keys = sorted(weeks.keys(), key=lambda x: int(x.replace('week', '')))
                    
                    has_data = False
                    for i, week_key in enumerate(week_keys):
                        week_data = weeks[week_key]
                        color = week_colors[i % len(week_colors)]
                        week_num = week_key.replace('week', '')
                        
                        ax.plot(week_data['cycles'], week_data['values'], 
                               color=color, marker='o', markersize=3, linewidth=2,
                               label=f'Test {week_num}', alpha=0.8)
                        has_data = True
                    
                    if has_data:
                        ax.set_xlabel('Cycle Number')
                        ax.set_ylabel('Temperature (°C)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        # Save plot
                        filename = f'{condition.replace(" ", "_")}_{svf.replace("%", "percent")}_all_weeks_raw.png'
                        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved: {filename}")
                    else:
                        plt.close()

def plot_week_differences_from_first(results, output_dir):
    """Plot differences of each week compared to the first week for each material/SVF combination."""
    # Define colors for different weeks
    week_colors = ['#FF4444', '#4444FF', '#44AA44', '#FFA500', '#800080', '#00CED1', '#A52A2A', '#FFD700']
    
    for condition, data in results.items():
        for svf in ['50%', '35%', '20%']:
            if svf in data and 'weeks' in data[svf]:
                weeks = data[svf]['weeks']
                if len(weeks) > 1:  # Need at least 2 weeks for differences
                    # Sort week keys by week number
                    week_keys = sorted(weeks.keys(), key=lambda x: int(x.replace('week', '')))
                    first_week_key = week_keys[0]
                    first_week = weeks[first_week_key]
                    
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                    # Extract material and condition info for title
                    material_num = condition.split()[0]  # e.g., "95A" from "95A 1st"
                    first_test_num = first_week_key.replace("week", "")
                    fig.suptitle(f'Filament TPU{material_num} SVF {svf} - Test Temperature History Differences from Test {first_test_num}', fontsize=16)
                    
                    has_data = False
                    for i, week_key in enumerate(week_keys[1:], 1):  # Skip first week
                        current_week = weeks[week_key]
                        
                        # Find common cycles
                        first_cycles = set(first_week['cycles'])
                        current_cycles = set(current_week['cycles'])
                        common_cycles = sorted(first_cycles & current_cycles)
                        
                        if common_cycles:
                            differences = []
                            for cycle in common_cycles:
                                first_idx = list(first_week['cycles']).index(cycle)
                                current_idx = list(current_week['cycles']).index(cycle)
                                diff = current_week['values'][current_idx] - first_week['values'][first_idx]
                                differences.append(diff)
                            
                            color = week_colors[i % len(week_colors)]
                            week_num = week_key.replace('week', '')
                            first_week_num = first_week_key.replace('week', '')
                            
                            ax.plot(common_cycles, differences, 
                                   color=color, marker='o', markersize=3, linewidth=2,
                                   label=f'Test {week_num} - Test {first_week_num}', alpha=0.8)
                            has_data = True
                    
                    if has_data:
                        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                        ax.set_xlabel('Cycle Number')
                        ax.set_ylabel('Temperature Difference (°C)')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        # Save plot
                        filename = f'{condition.replace(" ", "_")}_{svf.replace("%", "percent")}_week_differences.png'
                        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"Saved: {filename}")
                    else:
                        plt.close()

def calculate_csv_statistics(results, csv_file):
    """Calculate statistics and update the CSV file sections (corrected method).

    Changes vs previous version:
    - Peak-to-initial differences are computed PER WEEK PER SET (1st/2nd) instead of a
      single set-wide difference using the very first reading across all weeks.
    - Week 1 metrics use only week1 data; total metrics aggregate all available weeks.
    - Relative standard deviation uses sample std (ddof=1) when >=2 values.
    - Results remain written to a copy of the CSV: *original file untouched*.
    """
    print("\nCalculating statistics for CSV file (per-week, per-set baselines)...")

    df = pd.read_csv(csv_file)

    combinations = [
        ('95A - 50%', '95A 1st', '50%', '95A 2nd', '50%'),
        ('95A - 35%', '95A 1st', '35%', '95A 2nd', '35%'),
        ('95A - 20%', '95A 1st', '20%', '95A 2nd', '20%'),
        ('90A - 50%', '90A 1st', '50%', '90A 2nd', '50%'),
        ('90A - 35%', '90A 1st', '35%', '90A 2nd', '35%'),
        ('90A - 20%', '90A 1st', '20%', '90A 2nd', '20%'),
        ('87A - 50%', '87A 1st', '50%', '87A 2nd', '50%'),
        ('87A - 35%', '87A 1st', '35%', '87A 2nd', '35%'),
        ('87A - 20%', '87A 1st', '20%', '87A 2nd', '20%')
    ]

    stats_data = {}

    # Configuration for RSD calculation to mirror Excel if desired
    # rsd_week_filter: list of week numbers to include (e.g., [1,2]) or None for all
    rsd_week_filter = None  # Set to [1,2] if you want only first two weeks like the Excel example
    use_population_rsd = True  # Excel used STDEV.P, so population std

    for combo_name, cond1, svf1, cond2, svf2 in combinations:
        per_week_diffs = []   # all weeks both sets
        per_week_peaks = []   # peak per week per set
        wk1_diffs = []        # week1 only per set
        wk1_peaks = []

        for condition, svf in [(cond1, svf1), (cond2, svf2)]:
            if condition not in results or svf not in results[condition]:
                continue
            svf_obj = results[condition][svf]
            if 'weeks' not in svf_obj:
                continue
            for week_key, week_data in svf_obj['weeks'].items():
                values = week_data['values']
                if len(values) == 0:
                    continue
                arr = np.asarray(values, dtype=float)
                initial = arr[0]
                peak = np.nanmax(arr)
                diff = peak - initial
                per_week_diffs.append(diff)
                per_week_peaks.append(peak)
                if week_key == 'week1':
                    wk1_diffs.append(diff)
                    wk1_peaks.append(peak)

        # Optionally filter weeks for RSD calculation (to emulate Excel references to specific weeks)
        if rsd_week_filter is not None:
            filtered_diffs = []
            # Need to re-derive diffs in the same order; recompute with filter
            filtered_diffs = []
            for condition, svf in [(cond1, svf1), (cond2, svf2)]:
                if condition not in results or svf not in results[condition]:
                    continue
                svf_obj = results[condition][svf]
                if 'weeks' not in svf_obj:
                    continue
                for week_key, week_data in svf_obj['weeks'].items():
                    try:
                        wk_num = int(week_key.replace('week',''))
                    except ValueError:
                        continue
                    if wk_num not in rsd_week_filter:
                        continue
                    vals = week_data['values']
                    if len(vals) == 0:
                        continue
                    a = np.asarray(vals, dtype=float)
                    filtered_diffs.append(np.nanmax(a) - a[0])
            rsd_diffs = filtered_diffs if filtered_diffs else per_week_diffs
        else:
            rsd_diffs = per_week_diffs

        # Week 1 stats
        wk1_mean_change = float(np.mean(wk1_diffs)) if wk1_diffs else np.nan
        wk1_peak_temp = float(np.mean(wk1_peaks)) if wk1_peaks else np.nan
        wk1_max_temp_change = wk1_mean_change  # retained for legacy field

        # Total stats (all weeks)
        if per_week_diffs:
            total_mean_change = float(np.mean(per_week_diffs))
            # Sample variation for information
            if len(rsd_diffs) > 1:
                sample_std = float(np.std(rsd_diffs, ddof=1))
                population_std = float(np.std(rsd_diffs, ddof=0))
            else:
                sample_std = np.nan
                population_std = np.nan
            if (not np.isnan(total_mean_change) and total_mean_change != 0):
                total_relative_std_sample = (sample_std / total_mean_change) * 100 if not np.isnan(sample_std) else np.nan
                total_relative_std_population = (population_std / total_mean_change) * 100 if not np.isnan(population_std) else np.nan
            else:
                total_relative_std_sample = np.nan
                total_relative_std_population = np.nan
            total_relative_std = (total_relative_std_population if use_population_rsd else total_relative_std_sample)
        else:
            total_mean_change = np.nan
            total_relative_std = np.nan
            total_relative_std_sample = np.nan
            total_relative_std_population = np.nan

        total_peak_temp = float(np.mean(per_week_peaks)) if per_week_peaks else np.nan

        stats_data[combo_name] = {
            'wk1_mean_change': wk1_mean_change,
            'wk1_peak_temp': wk1_peak_temp,
            'wk1_max_temp_change': wk1_max_temp_change,
            'total_mean_change': total_mean_change,
            'total_peak_temp': total_peak_temp,
            'total_relative_std': total_relative_std,
            'total_relative_std_sample': total_relative_std_sample if 'total_relative_std_sample' in locals() else np.nan,
            'total_relative_std_population': total_relative_std_population if 'total_relative_std_population' in locals() else np.nan,
            'per_week_diffs_used_for_rsd': rsd_diffs  # for debugging / traceability
        }

    # ---- Write updated stats back to copy of CSV ----
    print("Updating CSV file with per-week statistics...")

    wk1_section_start = None
    total_section_start = None
    for idx, row in df.iterrows():
        first = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ''
        if 'Wk1' in first:
            wk1_section_start = idx
        elif first.startswith('Total'):
            total_section_start = idx

    def fmt(val):
        return f"{val:.2f}" if (val is not None and not np.isnan(val)) else ''

    if wk1_section_start is not None:
        for combo_name, *_ in combinations:
            for idx in range(wk1_section_start + 1, len(df)):
                cell = df.iloc[idx, 0]
                if pd.notna(cell) and combo_name in str(cell):
                    df.iloc[idx, 1] = fmt(stats_data[combo_name]['wk1_mean_change'])
                    df.iloc[idx, 2] = fmt(stats_data[combo_name]['wk1_peak_temp'])
                    break

    if total_section_start is not None:
        for combo_name, *_ in combinations:
            for idx in range(total_section_start + 1, len(df)):
                cell = df.iloc[idx, 0]
                if pd.notna(cell) and combo_name in str(cell):
                    df.iloc[idx, 1] = fmt(stats_data[combo_name]['total_mean_change'])
                    df.iloc[idx, 2] = fmt(stats_data[combo_name]['total_peak_temp'])
                    df.iloc[idx, 3] = fmt(stats_data[combo_name]['total_relative_std'])
                    break

    updated_file = csv_file.replace('.csv', '_with_peak_statistics.csv')
    try:
        df.to_csv(updated_file, index=False)
        print(f"CSV file with per-week peak-based statistics saved as: {updated_file}")
    except PermissionError as e:
        # Likely the file is open in Excel; create a timestamped alternative
        import datetime
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        alt_file = csv_file.replace('.csv', f'_with_peak_statistics_{ts}.csv')
        df.to_csv(alt_file, index=False)
        print("PermissionError writing primary output (possibly open in Excel). Saved instead as:", alt_file)
    except OSError as e:
        print("OS error while saving updated CSV:", e)
    print("Original file left unchanged.")

    return stats_data

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
    
    # Calculate and update CSV statistics
    csv_stats = calculate_csv_statistics(results, csv_file)
    
    # Create organized output directories
    output_dirs = create_output_directories(script_dir)
    
    # Create all the requested plots
    print("\nCreating plots...")
    
    print("\n1. Material SVF Comparisons (for each material, comparing SVF levels across weeks):")
    plot_material_svf_comparison(results, output_dirs['svf_comparisons'])
    
    print("\n2. SVF Material Comparisons (for each SVF level, comparing materials across weeks):")
    plot_svf_material_comparison(results, output_dirs['material_comparisons'])
    
    print("\n3. All Weeks Raw Data (all weeks plotted together for each set):")
    plot_all_weeks_raw_data(results, output_dirs['all_weeks_raw'])

    print("\n4. Week Differences from First Week (differences from first week):")
    plot_week_differences_from_first(results, output_dirs['week_differences'])
    
    print(f"\nAll plots saved in organized subfolders under: {os.path.join(script_dir, 'plots')}")
    print("Subfolders:")
    for name, path in output_dirs.items():
        print(f"  - {name}: {os.path.relpath(path, script_dir)}")

if __name__ == "__main__":
    main()