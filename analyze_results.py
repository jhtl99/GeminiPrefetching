#!/usr/bin/env python3
"""
Analysis script for unified experiment results.
Loads data from the global metrics file and creates visualizations.

Usage:
    python analyze_results.py --run 3                    # Analyze specific run
    python analyze_results.py --experiment prompt_comparison  # Analyze experiment type
    python analyze_results.py --compare_datasets         # Compare datasets
    python analyze_results.py --summary                  # Show overall summary
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

GLOBAL_METRICS_FILE = "all_experiments_metrics.csv"
CHARTS_DIR = "analysis_charts"

# Create charts directory if it doesn't exist
os.makedirs(CHARTS_DIR, exist_ok=True)

def load_results():
    """Load the global metrics file."""
    if not os.path.exists(GLOBAL_METRICS_FILE):
        print(f"No results file found: {GLOBAL_METRICS_FILE}")
        return None
    
    df = pd.read_csv(GLOBAL_METRICS_FILE)
    print(f"Loaded {len(df)} experiments from {GLOBAL_METRICS_FILE}")
    return df

def analyze_run(df, run_number):
    """Analyze results from a specific run."""
    run_data = df[df['run_number'] == run_number]
    if len(run_data) == 0:
        print(f"No data found for run {run_number}")
        return
    
    print(f"\nRUN {run_number} ANALYSIS")
    print("="*60)
    print(f"Experiment type: {run_data['experiment_type'].iloc[0]}")
    print(f"Dataset(s): {run_data['dataset'].unique()}")
    print(f"Number of experiments: {len(run_data)}")
    print(f"Date range: {run_data['timestamp'].min()} to {run_data['timestamp'].max()}")
    
    # Show best results
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall']
    for metric in metrics:
        best_idx = run_data[metric].idxmax()
        best = run_data.loc[best_idx]
        print(f"\nBest {metric}: {best[metric]:.4f}")
        print(f"  Config: batch_size={best['batch_size']}, output_size={best['output_size']}, "
              f"prompt_type={best['prompt_type']}, dataset={best['dataset']}")
    
    # Create visualization
    experiment_type = run_data['experiment_type'].iloc[0]
    if experiment_type == "hyperparameter_sweep":
        create_hyperparameter_heatmaps(run_data, run_number)
    elif experiment_type == "prompt_comparison":
        create_prompt_comparison_plots(run_data, run_number)
    elif experiment_type == "dataset_comparison":
        create_dataset_comparison_plots(run_data, run_number)

def analyze_experiment_type(df, experiment_type):
    """Analyze all runs of a specific experiment type."""
    exp_data = df[df['experiment_type'] == experiment_type]
    if len(exp_data) == 0:
        print(f"No data found for experiment type: {experiment_type}")
        return
    
    print(f"\n{experiment_type.upper()} ANALYSIS")
    print("="*60)
    print(f"Runs: {sorted(exp_data['run_number'].unique())}")
    print(f"Datasets: {exp_data['dataset'].unique()}")
    print(f"Total experiments: {len(exp_data)}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    for metric in ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_time']:
        print(f"  {metric}: μ={exp_data[metric].mean():.4f}, σ={exp_data[metric].std():.4f}")
    
    # Best overall
    best_idx = exp_data['avg_accuracy'].idxmax()
    best = exp_data.loc[best_idx]
    print(f"\nBest accuracy across all runs: {best['avg_accuracy']:.4f}")
    print(f"  Run {best['run_number']}: batch_size={best['batch_size']}, output_size={best['output_size']}, "
          f"prompt_type={best['prompt_type']}, dataset={best['dataset']}")

def compare_datasets(df):
    """Compare performance across different datasets."""
    print(f"\nDATASET COMPARISON")
    print("="*60)
    
    # Group by dataset and show statistics
    dataset_stats = df.groupby('dataset').agg({
        'avg_accuracy': ['mean', 'std', 'count'],
        'avg_precision': ['mean', 'std'], 
        'avg_recall': ['mean', 'std'],
        'avg_time': ['mean', 'std']
    }).round(4)
    
    print("Dataset Performance Summary:")
    print(dataset_stats)
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_time']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Time (s)']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        
        dataset_means = df.groupby('dataset')[metric].mean()
        dataset_stds = df.groupby('dataset')[metric].std()
        
        bars = plt.bar(dataset_means.index, dataset_means.values, 
                      yerr=dataset_stds.values, capsize=5, alpha=0.7)
        
        plt.title(f'{name} by Dataset')
        plt.ylabel(name)
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, mean in zip(bars, dataset_means.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved dataset_comparison.png to {CHARTS_DIR}/")

def create_hyperparameter_heatmaps(run_data, run_number):
    """Create heatmaps for hyperparameter sweep results."""
    plt.figure(figsize=(15, 12))
    
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_time']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Time (s)']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        
        # Create pivot table
        pivot = run_data.pivot(index='output_size', columns='batch_size', values=metric)
        
        # Create heatmap
        cmap = 'rocket_r' if metric == 'avg_time' else 'rocket'
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, cbar_kws={'label': name})
        
        plt.title(f'{name} - Run {run_number}')
        plt.xlabel('Batch Size')
        plt.ylabel('Output Size')
    
    dataset = run_data['dataset'].iloc[0]
    prompt_type = run_data['prompt_type'].iloc[0]
    plt.suptitle(f'Hyperparameter Sweep - Run {run_number}\nDataset: {dataset}, Prompt: {prompt_type}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    filename = f'hyperparameter_sweep_run{run_number}.png'
    plt.savefig(os.path.join(CHARTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {filename} to {CHARTS_DIR}/")

def create_prompt_comparison_plots(run_data, run_number):
    """Create bar charts for prompt comparison results."""
    plt.figure(figsize=(15, 10))
    
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_time']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Time (s)']
    colors = ['skyblue', 'lightgreen', 'salmon', 'orange']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        
        bars = plt.bar(run_data['prompt_type'], run_data[metric], 
                      color=colors[:len(run_data)], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'{name} by Prompt Strategy')
        plt.ylabel(name)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
    
    dataset = run_data['dataset'].iloc[0]
    batch_size = run_data['batch_size'].iloc[0]
    output_size = run_data['output_size'].iloc[0]
    
    plt.suptitle(f'Prompt Strategy Comparison - Run {run_number}\n'
                f'Dataset: {dataset}, Batch Size: {batch_size}, Output Size: {output_size}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    filename = f'prompt_comparison_run{run_number}.png'
    plt.savefig(os.path.join(CHARTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {filename} to {CHARTS_DIR}/")

def create_dataset_comparison_plots(run_data, run_number):
    """Create bar charts for dataset comparison results."""
    plt.figure(figsize=(12, 8))
    
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_time']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'Time (s)']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        
        bars = plt.bar(run_data['dataset'], run_data[metric], alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title(f'{name} by Dataset')
        plt.ylabel(name)
        plt.grid(axis='y', alpha=0.3)
    
    batch_size = run_data['batch_size'].iloc[0]
    output_size = run_data['output_size'].iloc[0]
    prompt_type = run_data['prompt_type'].iloc[0]
    
    plt.suptitle(f'Dataset Comparison - Run {run_number}\n'
                f'Batch Size: {batch_size}, Output Size: {output_size}, Prompt: {prompt_type}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    filename = f'dataset_comparison_run{run_number}.png'
    plt.savefig(os.path.join(CHARTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved {filename} to {CHARTS_DIR}/")

def show_summary(df):
    """Show overall summary of all experiments."""
    print(f"\nOVERALL SUMMARY")
    print("="*60)
    print(f"Total experiments: {len(df)}")
    print(f"Runs: {sorted(df['run_number'].unique())}")
    print(f"Experiment types: {df['experiment_type'].unique()}")
    print(f"Datasets: {df['dataset'].unique()}")
    print(f"Prompt types: {df['prompt_type'].unique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    print(f"\nOverall Performance:")
    for metric in ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_time']:
        print(f"  {metric}: μ={df[metric].mean():.4f}, σ={df[metric].std():.4f}, "
              f"min={df[metric].min():.4f}, max={df[metric].max():.4f}")
    
    # Best configurations
    best_idx = df['avg_accuracy'].idxmax()
    best = df.loc[best_idx]
    print(f"\nBest accuracy overall: {best['avg_accuracy']:.4f}")
    print(f"  Run {best['run_number']}: {best['experiment_type']}")
    print(f"  Config: dataset={best['dataset']}, batch_size={best['batch_size']}, "
          f"output_size={best['output_size']}, prompt_type={best['prompt_type']}")

def main():
    parser = argparse.ArgumentParser(description='Analyze unified experiment results')
    parser.add_argument('--run', type=int, help='Analyze specific run number')
    parser.add_argument('--experiment', type=str, help='Analyze specific experiment type')
    parser.add_argument('--compare_datasets', action='store_true', help='Compare dataset performance')
    parser.add_argument('--summary', action='store_true', help='Show overall summary')
    
    args = parser.parse_args()
    
    df = load_results()
    if df is None:
        return
    
    if args.run is not None:
        analyze_run(df, args.run)
    elif args.experiment:
        analyze_experiment_type(df, args.experiment)
    elif args.compare_datasets:
        compare_datasets(df)
    elif args.summary:
        show_summary(df)
    else:
        # Default: show summary
        show_summary(df)

if __name__ == "__main__":
    main()