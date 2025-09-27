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
        'material_week_differences': os.path.join(base_dir, 'plots', 'material_week_differences'),
        'svf_week_differences': os.path.join(base_dir, 'plots', 'svf_week_differences'),
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
            fig.suptitle(f'{condition} - Week {week_num} SVF Comparison', fontsize=16)
            
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
                    fig.suptitle(f'{condition} {svf} - All Weeks Raw Data', fontsize=16)
                    
                    # Sort week keys by week number
                    week_keys = sorted(weeks.keys(), key=lambda x: int(x.replace('week', '')))
                    
                    has_data = False
                    for i, week_key in enumerate(week_keys):
                        week_data = weeks[week_key]
                        color = week_colors[i % len(week_colors)]
                        week_num = week_key.replace('week', '')
                        
                        ax.plot(week_data['cycles'], week_data['values'], 
                               color=color, marker='o', markersize=3, linewidth=2,
                               label=f'Week {week_num}', alpha=0.8)
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
                    fig.suptitle(f'{condition} {svf} - Week Differences from Week {first_week_key.replace("week", "")}', fontsize=16)
                    
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
                                   label=f'Week {week_num} - Week {first_week_num}', alpha=0.8)
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
                fig.suptitle(f'{svf} SVF - {set_type} Set Materials - Week {week_num}', fontsize=16)
                
                has_data = False
                for material in ['95A', '90A', '87A']:
                        if (condition in results and svf in results[condition] and 
                            'weeks' in results[condition][svf] and 
                            week_key in results[condition][svf]['weeks']):
                            
                            week_data = results[condition][svf]['weeks'][week_key]
                            
                            ax.plot(week_data['cycles'], week_data['values'], 
                                   color=color, marker='o', markersize=4, linewidth=2,
                                   label=f'{material} {set_type}')
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

def plot_material_week_differences(results, output_dir):
    """For each material condition, create separate plots for each week difference comparison."""
    colors = {'50%': '#FF4444', '35%': '#4444FF', '20%': '#44AA44'}
    
    for condition, data in results.items():
        if not data:
            continue
        
        # Determine available weeks across all SVF levels
        all_weeks = set()
        for svf in ['50%', '35%', '20%']:
            if svf in data and 'weeks' in data[svf]:
                all_weeks.update(data[svf]['weeks'].keys())
        
        week_numbers = sorted([int(w.replace('week', '')) for w in all_weeks if w.startswith('week')])
        if len(week_numbers) < 2:  # Need at least 2 weeks for differences
            continue
        
        # Create separate plots for each week difference (Week N - Week 1)
        diff_weeks = week_numbers[1:]  # All weeks except the first
        
        for week_num in diff_weeks:
            week_key = f'week{week_num}'
            base_week_key = 'week1'
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            fig.suptitle(f'{condition} - Week {week_num} vs Week 1 Differences', fontsize=16)
            
            has_data = False
            for svf in ['50%', '35%', '20%']:
                if (svf in data and 'weeks' in data[svf] and 
                    week_key in data[svf]['weeks'] and base_week_key in data[svf]['weeks']):
                    
                    current_week = data[svf]['weeks'][week_key]
                    base_week = data[svf]['weeks'][base_week_key]
                    
                    # Find common cycles
                    current_cycles = set(current_week['cycles'])
                    base_cycles = set(base_week['cycles'])
                    common_cycles = sorted(current_cycles & base_cycles)
                    
                    if common_cycles:
                        differences = []
                        for cycle in common_cycles:
                            current_idx = list(current_week['cycles']).index(cycle)
                            base_idx = list(base_week['cycles']).index(cycle)
                            diff = current_week['values'][current_idx] - base_week['values'][base_idx]
                            differences.append(diff)
                        
                        ax.plot(common_cycles, differences, 
                               color=colors[svf], marker='o', markersize=4, linewidth=2,
                               label=f'{svf}')
                        has_data = True
            
            if has_data:
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax.set_xlabel('Cycle Number')
                ax.set_ylabel('Temperature Difference (°C)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                filename = f'{condition.replace(" ", "_")}_week{week_num}_vs_week1_differences.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved: {filename}")
            else:
                plt.close()

def plot_svf_week_differences(results, output_dir):
    """For each SVF level, create separate plots for each week difference comparison, separating 1st and 2nd sets."""
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

    def plot_week0_vs_weeks(results, output_dir):
        """Plot raw Week 0 vs Week N for each material/SVF combination."""
        colors = ['#FF4444', '#4444FF']
        for condition, data in results.items():
            for svf in ['50%', '35%', '20%']:
                if svf in data and 'weeks' in data[svf]:
                    weeks = data[svf]['weeks']
                    if 'week0' in weeks:
                        week0 = weeks['week0']
                        week_keys = [wk for wk in weeks.keys() if wk != 'week0' and wk.startswith('week')]
                        for wk in sorted(week_keys, key=lambda x: int(x.replace('week',''))):
                            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                            fig.suptitle(f'{condition} {svf} - Week 0 vs {wk} (Raw)', fontsize=15)
                            ax.plot(week0['cycles'], week0['values'], color=colors[0], marker='o', markersize=4, linewidth=2, label='Week 0')
                            weekN = weeks[wk]
                            ax.plot(weekN['cycles'], weekN['values'], color=colors[1], marker='o', markersize=4, linewidth=2, label=wk)
                            ax.set_xlabel('Cycle Number')
                            ax.set_ylabel('Temperature (°C)')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            filename = f'{condition.replace(" ", "_")}_{svf.replace("%", "percent")}_week0_vs_{wk}_raw.png'
                            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                            plt.close()
                            print(f"Saved: {filename}")

    def plot_week0_vs_weeks_difference(results, output_dir):
        """Plot Week N - Week 0 differences for each material/SVF combination."""
        color = '#44AA44'
        for condition, data in results.items():
            for svf in ['50%', '35%', '20%']:
                if svf in data and 'weeks' in data[svf]:
                    weeks = data[svf]['weeks']
                    if 'week0' in weeks:
                        week0 = weeks['week0']
                        week_keys = [wk for wk in weeks.keys() if wk != 'week0' and wk.startswith('week')]
                        for wk in sorted(week_keys, key=lambda x: int(x.replace('week',''))):
                            weekN = weeks[wk]
                            cycles0 = set(week0['cycles'])
                            cyclesN = set(weekN['cycles'])
                            common_cycles = sorted(cycles0 & cyclesN)
                            if common_cycles:
                                differences = []
                                for cycle in common_cycles:
                                    idx0 = list(week0['cycles']).index(cycle)
                                    idxN = list(weekN['cycles']).index(cycle)
                                    diff = weekN['values'][idxN] - week0['values'][idx0]
                                    differences.append(diff)
                                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                                fig.suptitle(f'{condition} {svf} - {wk} minus Week 0', fontsize=15)
                                ax.plot(common_cycles, differences, color=color, marker='o', markersize=4, linewidth=2, label=f'{wk} - Week 0')
                                ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                                ax.set_xlabel('Cycle Number')
                                ax.set_ylabel('Temperature Difference (°C)')
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                plt.tight_layout()
                                filename = f'{condition.replace(" ", "_")}_{svf.replace("%", "percent")}_{wk}_minus_week0_difference.png'
                                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                                plt.close()
                                print(f"Saved: {filename}")
    
    for svf in ['50%', '35%', '20%']:
        # Determine available weeks for this SVF
        all_weeks = set()
        for condition, data in results.items():
            if svf in data and 'weeks' in data[svf]:
                all_weeks.update(data[svf]['weeks'].keys())
        
        week_numbers = sorted([int(w.replace('week', '')) for w in all_weeks if w.startswith('week')])
        if len(week_numbers) < 2:
            continue
        
        # Create separate plots for each week difference and each set
        diff_weeks = week_numbers[1:]
        
        for set_type in ['1st', '2nd']:
            for week_num in diff_weeks:
                week_key = f'week{week_num}'
                base_week_key = 'week1'
                
                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                fig.suptitle(f'{svf} SVF - {set_type} Set - Week {week_num} vs Week 1 Differences', fontsize=16)
                
                has_data = False
                for material in ['95A', '90A', '87A']:
                    for condition in materials[material][set_type]:
                        if (condition in results and svf in results[condition] and 
                            'weeks' in results[condition][svf] and 
                            week_key in results[condition][svf]['weeks'] and 
                            base_week_key in results[condition][svf]['weeks']):
                            
                            current_week = results[condition][svf]['weeks'][week_key]
                            base_week = results[condition][svf]['weeks'][base_week_key]
                            
                            # Find common cycles
                            current_cycles = set(current_week['cycles'])
                            base_cycles = set(base_week['cycles'])
                            common_cycles = sorted(current_cycles & base_cycles)
                            
                            if common_cycles:
                                differences = []
                                for cycle in common_cycles:
                                    current_idx = list(current_week['cycles']).index(cycle)
                                    base_idx = list(base_week['cycles']).index(cycle)
                                    diff = current_week['values'][current_idx] - base_week['values'][base_idx]
                                    differences.append(diff)
                                
                                color = material_colors[material][set_type]
                                ax.plot(common_cycles, differences, 
                                       color=color, marker='o', markersize=4, linewidth=2,
                                       label=f'{material} {set_type}')
                                has_data = True
                
                if has_data:
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    ax.set_xlabel('Cycle Number')
                    ax.set_ylabel('Temperature Difference (°C)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    # Save plot
                    filename = f'{svf.replace("%", "percent")}_SVF_{set_type}_set_week{week_num}_vs_week1_differences.png'
                    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved: {filename}")
                else:
                    plt.close()

def calculate_csv_statistics(results, csv_file):
    """Calculate statistics and update the CSV file sections."""
    print("\nCalculating statistics for CSV file...")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Define material-SVF combinations
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
    
    # Calculate statistics for each combination
    stats_data = {}
    
    for combo_name, cond1, svf1, cond2, svf2 in combinations:
        # Collect all temperature data for this combination
        all_temps = []
        wk1_temps = []
        peak_temps = []
        
        # Collect Week 1 peak temperatures for each set separately
        wk1_set_peak_temps = []
        wk1_peak_to_initial_diffs = []
        
        # Process both 1st and 2nd conditions
        for condition, svf in [(cond1, svf1), (cond2, svf2)]:
            if condition in results and svf in results[condition]:
                # Get week 1 temperatures for Wk1 statistics
                if 'weeks' in results[condition][svf] and 'week1' in results[condition][svf]['weeks']:
                    wk1_data = results[condition][svf]['weeks']['week1']['values']
                    wk1_temps.extend(wk1_data)
                    
                    # Calculate peak temperature and peak-to-initial difference for this set
                    if len(wk1_data) > 0:
                        set_wk1_temps = np.array(wk1_data)
                        set_initial_temp = set_wk1_temps[0]
                        set_peak_temp = np.max(set_wk1_temps)
                        wk1_set_peak_temps.append(set_peak_temp)
                        wk1_peak_to_initial_diffs.append(set_peak_temp - set_initial_temp)
                
                # Get all temperature data for total statistics
                if 'weeks' in results[condition][svf]:
                    for week_key, week_data in results[condition][svf]['weeks'].items():
                        all_temps.extend(week_data['values'])
        
        if wk1_set_peak_temps:
            # Mean of peak temperatures across sets for Week 1
            wk1_peak_temp = np.mean(wk1_set_peak_temps)
            # Mean of peak-to-initial differences across sets for Week 1
            wk1_mean_change = np.mean(wk1_peak_to_initial_diffs)
            wk1_max_temp_change = wk1_mean_change  # Same as mean change
        else:
            wk1_mean_change = np.nan
            wk1_peak_temp = np.nan
            wk1_max_temp_change = np.nan
        
        if all_temps:
            # Calculate peak-to-initial temperature differences for each set and collect peak temperatures
            peak_to_initial_differences = []
            weekly_peak_temps = []
            
            # Process both 1st and 2nd conditions (sets)
            for condition, svf in [(cond1, svf1), (cond2, svf2)]:
                if condition in results and svf in results[condition]:
                    if 'weeks' in results[condition][svf]:
                        # Collect all temperatures for this set across all weeks
                        set_temps = []
                        for week_key, week_data in results[condition][svf]['weeks'].items():
                            week_temps = week_data['values']
                            set_temps.extend(week_temps)
                            # Collect peak temperature for each week
                            if len(week_temps) > 0:
                                weekly_peak_temps.append(np.max(week_temps))
                        
                        if len(set_temps) > 0:
                            set_temps = np.array(set_temps)
                            # Use first measurement as initial temperature for this set
                            initial_temp = set_temps[0]
                            # Find peak temperature for this set
                            peak_temp = np.max(set_temps)
                            # Calculate peak-to-initial difference for this set
                            peak_to_initial_diff = peak_temp - initial_temp
                            peak_to_initial_differences.append(peak_to_initial_diff)
            
            if peak_to_initial_differences:
                peak_to_initial_differences = np.array(peak_to_initial_differences)
                
                # Mean temp change: average of peak-to-initial differences across sets
                total_mean_change = np.mean(peak_to_initial_differences)
                
                # Temperature change variation: standard deviation of peak-to-initial differences
                temp_change_variation = np.std(peak_to_initial_differences)
                
                # Relative standard deviation
                total_relative_std = ((temp_change_variation / total_mean_change) * 100) if total_mean_change != 0 else np.nan
            else:
                total_mean_change = np.nan
                temp_change_variation = np.nan
                total_relative_std = np.nan
                
            # Peak temperature: mean of peak temperatures across all weeks
            if weekly_peak_temps:
                total_peak_temp = np.mean(weekly_peak_temps)
            else:
                total_peak_temp = np.nan
        else:
            total_mean_change = np.nan
            total_peak_temp = np.nan
            temp_change_variation = np.nan
            total_relative_std = np.nan
        
        stats_data[combo_name] = {
            'wk1_mean_change': wk1_mean_change,
            'wk1_peak_temp': wk1_peak_temp,
            'wk1_max_temp_change': wk1_max_temp_change,
            'total_mean_change': total_mean_change,
            'total_peak_temp': total_peak_temp,
            'total_relative_std': total_relative_std
        }
    
    # Update the CSV file
    print("Updating CSV file with statistics...")
    
    # Find the statistics section rows
    wk1_section_start = None
    total_section_start = None
    
    for idx, row in df.iterrows():
        if pd.notna(row.iloc[0]) and 'Wk1' in str(row.iloc[0]):
            wk1_section_start = idx
        elif pd.notna(row.iloc[0]) and 'Total' in str(row.iloc[0]):
            total_section_start = idx
    
    if wk1_section_start is not None:
        # Fill Wk1 statistics
        for combo_name, _, _, _, _ in combinations:
            for idx, row in df.iterrows():
                if idx > wk1_section_start and pd.notna(row.iloc[0]) and combo_name in str(row.iloc[0]):
                    if not np.isnan(stats_data[combo_name]['wk1_mean_change']):
                        df.iloc[idx, 1] = f"{stats_data[combo_name]['wk1_mean_change']:.2f}"
                    if not np.isnan(stats_data[combo_name]['wk1_peak_temp']):
                        df.iloc[idx, 2] = f"{stats_data[combo_name]['wk1_peak_temp']:.2f}"
                    break
    
    if total_section_start is not None:
        # Fill Total statistics
        for combo_name, _, _, _, _ in combinations:
            for idx, row in df.iterrows():
                if idx > total_section_start and pd.notna(row.iloc[0]) and combo_name in str(row.iloc[0]):
                    if not np.isnan(stats_data[combo_name]['total_mean_change']):
                        df.iloc[idx, 1] = f"{stats_data[combo_name]['total_mean_change']:.2f}"
                    if not np.isnan(stats_data[combo_name]['total_peak_temp']):
                        df.iloc[idx, 2] = f"{stats_data[combo_name]['total_peak_temp']:.2f}"
                    if not np.isnan(stats_data[combo_name]['total_relative_std']):
                        df.iloc[idx, 3] = f"{stats_data[combo_name]['total_relative_std']:.2f}"
                    break
    
    # Save the updated CSV
    updated_file = csv_file.replace('.csv', '_with_peak_statistics.csv')
    df.to_csv(updated_file, index=False)
    print(f"CSV file with peak-based statistics saved as: {updated_file}")
    print("Original file left unchanged due to permission restrictions.")
    
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
    plot_material_svf_comparison(results, output_dirs['material_comparisons'])
    
    print("\n2. SVF Material Comparisons (for each SVF level, comparing materials across weeks):")
    plot_svf_material_comparison(results, output_dirs['svf_comparisons'])
    
    print("\n3. Material Week Differences (for each material, week differences vs Week 1):")
    plot_material_week_differences(results, output_dirs['material_week_differences'])
    
    print("\n4. SVF Week Differences (for each SVF level, week differences by material vs Week 1):")
    plot_svf_week_differences(results, output_dirs['svf_week_differences'])

    print("\n5. All Weeks Raw Data (all weeks plotted together for each set):")
    plot_all_weeks_raw_data(results, output_dirs['all_weeks_raw'])

    print("\n6. Week Differences from First Week (differences from first week):")
    plot_week_differences_from_first(results, output_dirs['week_differences'])
    
    print(f"\nAll plots saved in organized subfolders under: {os.path.join(script_dir, 'plots')}")
    print("Subfolders:")
    for name, path in output_dirs.items():
        print(f"  - {name}: {os.path.relpath(path, script_dir)}")

if __name__ == "__main__":
    main()