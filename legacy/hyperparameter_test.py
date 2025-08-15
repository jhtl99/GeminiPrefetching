#!/usr/bin/env python3
"""
Prompt strategy comparison script for prefetching experiment.
Tests different prompting strategies with fixed optimal hyperparameters.

Usage:
    python hyperparameter_test.py --run 3
    python hyperparameter_test.py --run 3 --analyze-only

Current test setup:
- Fixed hyperparameters: batch_size=150, output_size=50 (optimal from previous runs)
- Prompt strategies: ["minimal", "contextual", "expert"]
  * minimal: Only delta values, no PC pairs, no context
  * contextual: Full explanation of PC/delta meaning with context  
  * expert: Expert role + performance importance emphasis
- 5 API calls per strategy (15 total API calls per run)
- Results saved to results/run{X}/ directory with bar chart visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import time
import os
import argparse
from prefetch_experiment import run_experiment

# Test parameters - Fixed optimal hyperparameters, testing prompt strategies
BATCH_SIZE = 150  # Fixed optimal batch size
OUTPUT_SIZE = 50  # Fixed optimal output size
PROMPT_TYPES = ["minimal", "contextual", "expert"]  # Different prompting strategies

def run_prompt_comparison(run_number):
    """Run the prompt comparison experiment with fixed optimal hyperparameters."""
    results = []
    total_combinations = len(PROMPT_TYPES)
    current_combination = 0
    
    # Create results directory structure
    results_dir = f"results/run{run_number}"
    os.makedirs(results_dir, exist_ok=True)
    metrics_file = f"{results_dir}/metrics.txt"
    
    print(f"Starting prompt comparison with {total_combinations} strategies...")
    print(f"Fixed hyperparameters: batch_size={BATCH_SIZE}, output_size={OUTPUT_SIZE}")
    print(f"Prompt types: {PROMPT_TYPES}")
    print(f"Results will be saved to: {results_dir}")
    print("="*60)
    
    # Initialize metrics file
    with open(metrics_file, 'w') as f:
        f.write("prompt_type,batch_size,output_size,avg_accuracy,avg_precision,avg_recall,avg_time\n")
    
    for prompt_type in PROMPT_TYPES:
        current_combination += 1
        print(f"Running prompt strategy {current_combination}/{total_combinations}: "
              f"prompt_type={prompt_type}")
        
        start_time = time.time()
        
        try:
            # Run the experiment
            result = run_experiment(BATCH_SIZE, OUTPUT_SIZE, prompt_type)
            results.append(result)
            
            # Append to metrics file
            with open(metrics_file, 'a') as f:
                f.write(f"{prompt_type},{BATCH_SIZE},{OUTPUT_SIZE},{result['avg_accuracy']:.6f},"
                       f"{result['avg_precision']:.6f},{result['avg_recall']:.6f},"
                       f"{result['avg_time']:.6f}\n")
            
            elapsed = time.time() - start_time
            print(f"  ✓ Completed in {elapsed:.2f}s - "
                  f"Accuracy: {result['avg_accuracy']:.4f}, "
                  f"Time: {result['avg_time']:.4f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Still append a row with NaN values to maintain structure
            with open(metrics_file, 'a') as f:
                f.write(f"{prompt_type},{BATCH_SIZE},{OUTPUT_SIZE},NaN,NaN,NaN,NaN\n")
        
        print("-" * 40)
    
    print(f"Prompt comparison completed! Results saved to {metrics_file}")
    return results, results_dir

def load_results(metrics_file):
    """Load results from the metrics file."""
    try:
        df = pd.read_csv(metrics_file)
        return df
    except FileNotFoundError:
        print(f"Metrics file {metrics_file} not found. Please run the sweep first.")
        return None

def create_comparison_plots(df, results_dir, run_number):
    """Create bar chart visualizations comparing prompt strategies."""
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_time']
    metric_names = ['Average Accuracy', 'Average Precision', 'Average Recall', 'Average Time (s)']
    
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # Create bar plot
        colors = ['skyblue', 'lightgreen', 'salmon']
        bars = ax.bar(df['prompt_type'], df[metric], color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(f'{name} by Prompt Strategy\n(Batch Size={BATCH_SIZE}, Output Size={OUTPUT_SIZE})', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Prompt Strategy', fontweight='bold')
        ax.set_ylabel(name, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Prompt Strategy Comparison - Run {run_number}\nPrefetching Performance with Fixed Hyperparameters', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    combined_plot_path = f'{results_dir}/prompt_comparison.png'
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved combined comparison plot to {combined_plot_path}")
    
    # Create individual detailed plots
    for metric, name in zip(metrics, metric_names):
        plt.figure(figsize=(10, 8))
        
        colors = ['skyblue', 'lightgreen', 'salmon']
        bars = plt.bar(df['prompt_type'], df[metric], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.title(f'{name} by Prompt Strategy\nBatch Size={BATCH_SIZE}, Output Size={OUTPUT_SIZE}, Run {run_number}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Prompt Strategy', fontweight='bold', fontsize=12)
        plt.ylabel(name, fontweight='bold', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add description of strategies
        strategy_desc = {
            'minimal': 'Delta values only',
            'contextual': 'Full context explanation', 
            'expert': 'Expert + importance emphasis'
        }
        
        # Add text box with strategy descriptions
        desc_text = '\n'.join([f'{k}: {v}' for k, v in strategy_desc.items()])
        plt.text(0.02, 0.98, desc_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save individual plot
        filename = f'{results_dir}/plot_{metric}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved {filename}")

def analyze_results(df, run_number):
    """Analyze and print summary statistics for prompt comparison."""
    print("\n" + "="*60)
    print(f"PROMPT STRATEGY ANALYSIS - RUN {run_number}")
    print("="*60)
    print(f"Fixed Parameters: Batch Size={BATCH_SIZE}, Output Size={OUTPUT_SIZE}")
    print("-"*60)
    
    # Find best performing prompt strategies for each metric
    metrics = ['avg_accuracy', 'avg_precision', 'avg_recall']
    
    for metric in metrics:
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx]
        print(f"\nBest {metric}:")
        print(f"  Prompt Strategy: {best_row['prompt_type']}")
        print(f"  Value: {best_row[metric]:.4f}")
        
        # Show ranking of all strategies for this metric
        sorted_df = df.sort_values(metric, ascending=False)
        print(f"  Ranking: ", end="")
        rankings = [f"{row['prompt_type']}({row[metric]:.4f})" for _, row in sorted_df.iterrows()]
        print(" > ".join(rankings))
    
    # Find fastest strategy
    fastest_idx = df['avg_time'].idxmin()
    fastest_row = df.loc[fastest_idx]
    print(f"\nFastest execution:")
    print(f"  Prompt Strategy: {fastest_row['prompt_type']}")
    print(f"  Time: {fastest_row['avg_time']:.4f}s")
    
    # Show all results in a table
    print(f"\nComplete Results Table:")
    print(f"{'Strategy':<12} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'Time(s)':<8}")
    print("-" * 55)
    for _, row in df.iterrows():
        print(f"{row['prompt_type']:<12} {row['avg_accuracy']:<10.4f} {row['avg_precision']:<11.4f} "
              f"{row['avg_recall']:<8.4f} {row['avg_time']:<8.4f}")
    
    # Strategy descriptions
    print(f"\nStrategy Descriptions:")
    print(f"  minimal    : Only delta values, no PC pairs, no context")
    print(f"  contextual : Full explanation of PC/delta meaning with context")
    print(f"  expert     : Expert role + performance importance emphasis")

def main():
    """Main function to run prompt strategy comparison."""
    parser = argparse.ArgumentParser(description='Run prompt strategy comparison for prefetching experiment')
    parser.add_argument('--run', type=int, required=True, help='Run number (e.g., 1, 2, 3...)')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze existing results, do not run new experiments')
    
    args = parser.parse_args()
    run_number = args.run
    
    print("Prefetching Prompt Strategy Comparison")
    print("="*60)
    print(f"Run: {run_number}")
    print(f"Fixed hyperparameters: batch_size={BATCH_SIZE}, output_size={OUTPUT_SIZE}")
    print(f"Testing prompt strategies: {PROMPT_TYPES}")
    
    results_dir = f"results/run{run_number}"
    metrics_file = f"{results_dir}/metrics.txt"
    
    # Check if we should run the sweep or just analyze existing results
    if args.analyze_only:
        print("Analyzing existing results only...")
        df = load_results(metrics_file)
        if df is not None:
            create_comparison_plots(df, results_dir, run_number)
            analyze_results(df, run_number)
        return
    
    if os.path.exists(metrics_file):
        response = input(f"{metrics_file} already exists. "
                        "Do you want to (r)un new comparison, (a)nalyze existing, or (q)uit? [r/a/q]: ")
        if response.lower() == 'q':
            return
        elif response.lower() == 'a':
            df = load_results(metrics_file)
            if df is not None:
                create_comparison_plots(df, results_dir, run_number)
                analyze_results(df, run_number)
            return
    
    # Run the prompt comparison
    print("Running prompt strategy comparison...")
    results, results_dir = run_prompt_comparison(run_number)
    
    # Load and analyze results
    df = load_results(metrics_file)
    if df is not None:
        create_comparison_plots(df, results_dir, run_number)
        analyze_results(df, run_number)

if __name__ == "__main__":
    main()