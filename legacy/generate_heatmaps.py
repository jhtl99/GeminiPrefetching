#!/usr/bin/env python3
"""
Generate heatmap visualizations for experiment runs.
Creates red-to-green heatmaps where green=best, red=worst.
Note: For time metric, shorter time is better (green=fast, red=slow).

Usage:
    python generate_heatmaps.py --run 12
    python generate_heatmaps.py --run 15 --save_only  # Don't display, just save
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.colors import LinearSegmentedColormap

GLOBAL_METRICS_FILE = "all_experiments_metrics.csv"

def create_red_green_colormap():
    """Create a custom red-to-green colormap."""
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    return LinearSegmentedColormap.from_list("red_green", colors)

def load_run_data(run_number):
    """Load data for a specific run."""
    if not os.path.exists(GLOBAL_METRICS_FILE):
        print(f"Error: Metrics file {GLOBAL_METRICS_FILE} not found!")
        return None
    
    df = pd.read_csv(GLOBAL_METRICS_FILE)
    run_data = df[df['run_number'] == run_number]
    
    if len(run_data) == 0:
        print(f"Error: No data found for run {run_number}")
        available_runs = sorted(df['run_number'].unique())
        print(f"Available runs: {available_runs}")
        return None
    
    print(f"Loaded {len(run_data)} experiments for run {run_number}")
    print(f"Experiment type: {run_data['experiment_type'].iloc[0]}")
    print(f"Dataset: {run_data['dataset'].iloc[0]}")
    
    return run_data

def create_heatmap_data(run_data, metric):
    """Create pivot table for heatmap visualization."""
    experiment_type = run_data['experiment_type'].iloc[0]
    
    if experiment_type == "hyperparameter_sweep":
        # Use batch_size vs output_size
        if 'batch_size' in run_data.columns and 'output_size' in run_data.columns:
            pivot = run_data.pivot(index='output_size', columns='batch_size', values=metric)
            return pivot, 'Batch Size', 'Output Size'
        else:
            print(f"Warning: batch_size/output_size columns not found for hyperparameter sweep")
            return None, None, None
    
    elif experiment_type == "prompt_comparison":
        # Create a 1D visualization as a horizontal bar
        prompt_data = run_data.set_index('prompt_type')[metric]
        # Convert to a 1-row DataFrame for heatmap
        pivot = pd.DataFrame([prompt_data.values], columns=prompt_data.index, index=['Performance'])
        return pivot, 'Prompt Type', ''
    
    elif experiment_type == "dataset_comparison":
        # Create a 1D visualization as a horizontal bar
        dataset_data = run_data.set_index('dataset')[metric]
        # Convert to a 1-row DataFrame for heatmap
        pivot = pd.DataFrame([dataset_data.values], columns=dataset_data.index, index=['Performance'])
        return pivot, 'Dataset', ''
    
    else:
        print(f"Warning: Unknown experiment type: {experiment_type}")
        return None, None, None

def normalize_metric_for_color(data, metric):
    """Normalize metric values to 0-1 range for coloring (0=red/worst, 1=green/best)."""
    if data is None or data.empty:
        return data
    
    # For time, lower is better, so we need to invert
    if metric == 'avg_time':
        # Invert: fastest time gets value 1 (green), slowest gets value 0 (red)
        min_val = data.min().min()
        max_val = data.max().max()
        if max_val == min_val:
            return data * 0 + 0.5  # All same value, use middle color
        normalized = 1 - (data - min_val) / (max_val - min_val)
    else:
        # For accuracy/precision/recall, higher is better
        min_val = data.min().min()
        max_val = data.max().max()
        if max_val == min_val:
            return data * 0 + 0.5  # All same value, use middle color
        normalized = (data - min_val) / (max_val - min_val)
    
    return normalized

def create_single_heatmap(ax, run_data, metric, metric_name, run_number):
    """Create a single heatmap for a specific metric."""
    # Get the data for heatmap
    pivot_data, x_label, y_label = create_heatmap_data(run_data, metric)
    
    if pivot_data is None:
        ax.text(0.5, 0.5, f'No data available\nfor {metric_name}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
        return
    
    # Normalize for coloring (0=red/worst, 1=green/best)
    color_data = normalize_metric_for_color(pivot_data, metric)
    
    # Create custom colormap (red to green)
    cmap = create_red_green_colormap()
    
    # Create heatmap with custom colors but original values as annotations
    im = ax.imshow(color_data.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns, rotation=45)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index)
    
    # Add value annotations (show original values, not normalized)
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            original_val = pivot_data.iloc[i, j]
            if pd.notna(original_val):
                # Choose text color based on background
                color_val = color_data.iloc[i, j]
                text_color = 'white' if color_val < 0.5 else 'black'
                
                # Format the value nicely
                if metric == 'avg_time':
                    text = f'{original_val:.2f}s'
                else:
                    text = f'{original_val:.3f}'
                
                ax.text(j, i, text, ha='center', va='center', 
                       fontweight='bold', color=text_color, fontsize=10)
    
    # Labels and title
    ax.set_xlabel(x_label, fontweight='bold')
    if y_label:
        ax.set_ylabel(y_label, fontweight='bold')
    
    # Add performance indicator to title
    if metric == 'avg_time':
        title = f'{metric_name}\n(Green=Faster, Red=Slower)'
    else:
        title = f'{metric_name}\n(Green=Better, Red=Worse)'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    if metric == 'avg_time':
        cbar.set_label('Performance (Green=Fast, Red=Slow)', rotation=270, labelpad=20)
    else:
        cbar.set_label('Performance (Green=High, Red=Low)', rotation=270, labelpad=20)

def generate_run_heatmaps(run_number, save_only=False):
    """Generate heatmaps for all metrics in a specific run."""
    # Load data
    run_data = load_run_data(run_number)
    if run_data is None:
        return
    
    # Define metrics and their display names
    metrics = [
        ('avg_accuracy', 'Accuracy'),
        ('avg_precision', 'Precision'), 
        ('avg_recall', 'Recall'),
        ('avg_time', 'Time')
    ]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Create each heatmap
    for i, (metric, name) in enumerate(metrics):
        create_single_heatmap(axes[i], run_data, metric, name, run_number)
    
    # Get run info for title
    experiment_type = run_data['experiment_type'].iloc[0]
    dataset = run_data['dataset'].iloc[0]
    
    # Add overall title
    title = f'Performance Heatmaps - Run {run_number}\n'
    title += f'Experiment: {experiment_type.replace("_", " ").title()}, Dataset: {dataset}'
    if experiment_type == "prompt_comparison":
        batch_size = run_data['batch_size'].iloc[0]
        output_size = run_data['output_size'].iloc[0]
        title += f', Batch Size: {batch_size}, Output Size: {output_size}'
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    filename = f'heatmaps_run{run_number}.png'
    plt.savefig("unified_charts/" + filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved heatmaps to {filename}")
    
    # Show the figure unless save_only is True
    if not save_only:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate heatmap visualizations for experiment runs')
    parser.add_argument('--run', type=int, required=True, help='Run number to visualize')
    parser.add_argument('--save_only', action='store_true', help='Save only, do not display')
    
    args = parser.parse_args()
    
    print(f"Generating heatmaps for run {args.run}")
    print("Color scheme: Green=Best performance, Red=Worst performance")
    print("Note: For time metric, Green=Faster (better), Red=Slower (worse)")
    print("="*60)
    
    generate_run_heatmaps(args.run, args.save_only)

if __name__ == "__main__":
    main()