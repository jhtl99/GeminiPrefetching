#!/usr/bin/env python3
"""
Unified prefetching experiment system with tunable hyperparameters.
All experiments save to a single global metrics file with full context.

Usage:
    python unified_experiment.py --run 3 --experiment hyperparameter_sweep
    python unified_experiment.py --run 4 --experiment prompt_comparison
    python unified_experiment.py --run 5 --experiment single --batch_size 200 --output_size 60
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import argparse
import sys
import random
from google import genai
from google.genai import types

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get API key from environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.0-flash"

#================================================================================================
# TUNABLE HYPERPARAMETERS - Edit these to configure experiments
#================================================================================================

# Dataset options
AVAILABLE_DATASETS = {
    "cactu_25": "mini-project/cactu/25_filtered_delta.csv",
    "cactu_50": "mini-project/cactu/50_filtered_delta.csv",
    "lrange_25": "mini-project/lrange/25_filtered_delta.csv",
    "lrange_50": "mini-project/lrange/50_filtered_delta.csv",
    "mcf_25": "mini-project/mcf/25_filtered_delta.csv",
    "mcf_50": "mini-project/mcf/50_filtered_delta.csv",
    "numpy_25": "mini-project/numpy/25_filtered_delta.csv",
    "numpy_50": "mini-project/numpy/50_filtered_delta.csv",
    "omnetpp_25": "mini-project/omnetpp/25_filtered_delta.csv",
    "omnetpp_50": "mini-project/omnetpp/50_filtered_delta.csv",
    "pr_25": "mini-project/pr/25_filtered_delta.csv",
    "pr_50": "mini-project/pr/50_filtered_delta.csv",
    "ycsb-c_25": "mini-project/ycsb-c/25_filtered_delta.csv",
    "ycsb-c_50": "mini-project/ycsb-c/50_filtered_delta.csv",
}

# Hyperparameter ranges for sweeps
HYPERPARAMETER_RANGES = {
    "batch_sizes": [100, 125, 150, 175, 200],
    "output_sizes": [10, 20, 30, 40, 50, 60, 70],
    "prompt_types": ["original", "minimal", "contextual", "expert"]
}

# Default values for single experiments
DEFAULT_PARAMS = {
    "batch_size": 150,
    "output_size": 50,
    "prompt_type": "original",
    "dataset": "mcf_25",
    "num_api_calls": 10
}

# Randomness parameters
RANDOMNESS_PARAMS = {
    "random_start_min": 100,   # Minimum random jump after each batch
    "random_start_max": 1000,  # Maximum random jump after each batch
    "use_random_seed": False,   # Set to False for completely random behavior
    "random_seed": 42          # Fixed seed for reproducible experiments
}

# Global metrics file
GLOBAL_METRICS_FILE = "all_experiments_metrics.csv"

#================================================================================================
# PROMPT GENERATION FUNCTIONS
#================================================================================================

def format_prompt(df_batch: pd.DataFrame, batch_size: int, lookahead: int, prompt_type: str = "original") -> str:
    """Generate prompts based on the specified strategy."""
    pc_col = df_batch["pc"].tolist()
    delta_col = df_batch["delta_out"].tolist()
    
    if prompt_type == "minimal":
        history = " ".join(str(d) for d in delta_col)
        return f"""
{history}
Predict the next {lookahead} values in this sequence. Return ONLY the {lookahead} numbers separated by spaces. Do not include any other text, words, explanations, or confirmations like "okay" - just the numbers."""
    
    elif prompt_type == "contextual":
        history = "\n".join(f"{p} {d}" for p, d in zip(pc_col, delta_col))
        pair_len = 2 * lookahead
        return f"""
You are analyzing memory access patterns from a computer program. The data shows:
- PC (Program Counter): The memory address of the instruction being executed (in hexadecimal)
- Delta: The difference between consecutive memory addresses accessed by the program (in decimal)

These patterns help predict which memory pages the program will access next, enabling efficient prefetching.

Here are the most recent {batch_size} PC and delta pairs in sequential execution order:
{history}

Based on these memory access patterns, predict the NEXT {lookahead} PC-delta pairs.
Return ONLY the {pair_len} values in format: pc1 delta1 pc2 delta2 ... (exactly {pair_len} values total)
Do not include any other text, words, explanations, or confirmations like "okay" - just the numbers.
Focus on identifying patterns in both the program counter progression and memory access deltas."""
    
    elif prompt_type == "expert":
        history = "\n".join(f"{p} {d}" for p, d in zip(pc_col, delta_col))
        pair_len = 2 * lookahead
        return f"""
You are an expert computer systems engineer specializing in memory prefetching optimization. Your predictions are critical for system performance.

CONTEXT: You are analyzing memory access patterns from a high-performance computing application. The data shows:
- PC (Program Counter): Instruction memory addresses (hexadecimal) 
- Delta: Memory access stride patterns (decimal) - differences between consecutive memory addresses

MISSION: Your accurate predictions enable the prefetcher to load the right data into cache before it's needed, preventing costly memory stalls that can slow down the entire system by 10-100x.

Here are the most recent {batch_size} PC and delta pairs in sequential execution order:
{history}

TASK: Analyze these patterns and predict the NEXT {lookahead} PC-delta pairs with maximum accuracy.
- Look for recurring patterns in PC progression
- Identify memory access stride patterns in deltas  
- Consider both linear and complex access patterns
- Your predictions directly impact system performance

CRITICAL: Return ONLY the {pair_len} numbers in format: pc1 delta1 pc2 delta2 ... (exactly {pair_len} values total)
Do not include any other text, words, explanations, or confirmations like "okay" - just the numbers.
BE PRECISE - these predictions are performance-critical."""
    
    else:  # original
        pair_len = 2 * lookahead
        history = "\n".join(f"{p} {d}" for p, d in zip(pc_col, delta_col))
        return f"""
Here are the most recent {batch_size} program counter, delta_out pairs in sequential order (pc in hex, delta_out in decimal):
{history}
Predict the NEXT {lookahead} pairs. Return ONLY the {pair_len} numbers in format:
pc1 delta1 pc2 delta2 ... – no other text. Do not include any words, explanations, or confirmations like "okay" - just the numbers.
Ensure that you don't make too few predictions; there should be {pair_len} total values."""

def parse_predictions(raw: str, lookahead: int, prompt_type: str = "original") -> tuple[np.ndarray, np.ndarray]:
    """Parse Gemini's response into numeric arrays with smart prefix removal."""
    if raw == None:
        return None, None
    
    # Clean the response and remove common unwanted prefixes
    cleaned = raw.strip().replace("\n", " ")
    
    # Find the first word that looks like a number (hex or decimal)
    words = cleaned.lower().split()
    start_idx = 0
    for i, word in enumerate(words):
        try:
            if word.startswith(('0x', '-', '+')) or word.isdigit() or all(c in '0123456789abcdefABCDEF-+' for c in word):
                start_idx = i
                break
        except:
            continue
    
    # Rejoin from the first numeric word onwards
    original_words = cleaned.split()
    if start_idx < len(original_words):
        cleaned = " ".join(original_words[start_idx:])
    
    fields = cleaned.split()
    
    if prompt_type == "minimal":
        # Only delta values expected
        if len(fields) >= lookahead:
            fields = fields[:lookahead]
        else:
            fields.extend(["0"] * (lookahead - len(fields)))
        
        pcs_int = np.arange(lookahead, dtype=np.int64)
        deltas_int = []
        for i, x in enumerate(fields):
            try:
                deltas_int.append(int(x))
            except ValueError:
                print(f"Warning: Could not parse delta value '{x}' at position {i}, using 0")
                deltas_int.append(0)
        
        return pcs_int, np.array(deltas_int, dtype=np.int64)
    
    else:
        # PC-delta pairs expected
        pair_len = 2 * lookahead
        if len(fields) >= pair_len:
            fields = fields[:pair_len]
        else:
            fields.extend(["0"] * (pair_len - len(fields)))

        pcs_hex = fields[0::2]
        deltas_str = fields[1::2]

        pcs_int = []
        for i, x in enumerate(pcs_hex):
            try:
                pcs_int.append(int(x, 16))
            except ValueError:
                print(f"Warning: Could not parse PC value '{x}' at position {i}, using 0")
                pcs_int.append(0)
        
        deltas_int = []
        for i, x in enumerate(deltas_str):
            try:
                deltas_int.append(int(x))
            except ValueError:
                print(f"Warning: Could not parse delta value '{x}' at position {i}, using 0")
                deltas_int.append(0)
        
        return np.array(pcs_int), np.array(deltas_int, dtype=np.int64)

#================================================================================================
# CORE EXPERIMENT FUNCTIONS
#================================================================================================

def get_dataset_size(csv_path):
    """Get the total number of lines in the dataset (excluding header)."""
    try:
        # Count lines efficiently
        with open(csv_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract 1 for header
        return total_lines
    except:
        # Fallback: assume large dataset
        return 1000000

def run_single_experiment(batch_size: int, output_size: int, prompt_type: str, dataset: str, num_api_calls: int = 10) -> dict:
    """Run a single experiment configuration with randomized cursor positions."""
    csv_path = AVAILABLE_DATASETS[dataset]
    lookahead = output_size
    COLS = ["pc", "delta_out"]

    # Set random seed for reproducible experiments if specified
    if RANDOMNESS_PARAMS["use_random_seed"]:
        random.seed(RANDOMNESS_PARAMS["random_seed"])
        np.random.seed(RANDOMNESS_PARAMS["random_seed"])

    # Get dataset size to ensure we don't go out of bounds
    dataset_size = get_dataset_size(csv_path)
    min_required_lines = batch_size + lookahead
    
    print(f"  Running {num_api_calls} API calls with {dataset} dataset (randomized positions)...")
    print(f"  Dataset size: {dataset_size} lines, Required per batch: {min_required_lines} lines")

    batch_hits = []
    batch_recall = []
    batch_precision = []
    batch_accuracy = []
    batch_times = []
    cursor_positions = []  # Track where each batch started for debugging

    # Initialize cursor position
    line_cursor = 0

    for batch_num in range(num_api_calls):
        batch_start_time = time.time()

        # Ensure we have enough data remaining
        if line_cursor + min_required_lines >= dataset_size:
            print(f"  Warning: Not enough data remaining at position {line_cursor}, wrapping to beginning")
            line_cursor = 0

        cursor_positions.append(line_cursor)
        
        try:
            # Read input batch
            src_chunk = pd.read_csv(
                csv_path, usecols=COLS,
                skiprows=1 + line_cursor, nrows=batch_size,
                header=None, names=COLS
            )
            
            # Check if we got the expected amount of data
            if len(src_chunk) < batch_size:
                print(f"  Warning: Only got {len(src_chunk)} lines instead of {batch_size} at position {line_cursor}")
            
            # Generate prompt and get prediction
            prompt = format_prompt(src_chunk, batch_size, lookahead, prompt_type)
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            _, pred_delta = parse_predictions(response.text, lookahead, prompt_type)

            # Read ground truth
            gt = pd.read_csv(
                csv_path, usecols=COLS,
                skiprows=1 + line_cursor + batch_size,
                nrows=lookahead,
                header=None, names=COLS
            )
            
            if len(gt) < lookahead:
                print(f"  Warning: Only got {len(gt)} ground truth lines instead of {lookahead}")
            
            true_delta = gt["delta_out"].to_numpy()

            # Calculate metrics
            gt_pages = np.cumsum(true_delta, dtype=int)
            pred_pages = np.cumsum(pred_delta, dtype=int)

            gt_set = set(gt_pages)
            pred_set = set(pred_pages)

            intersection = gt_set & pred_set
            hit_rate = len(intersection) / lookahead
            recall = len(intersection) / len(gt_set) if len(gt_set) > 0 else 0
            precision = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0

            union = gt_set | pred_set
            accuracy = len(intersection) / len(union) if len(union) > 0 else 0

            batch_hits.append(hit_rate)
            batch_recall.append(recall)
            batch_precision.append(precision)
            batch_accuracy.append(accuracy)

            batch_times.append(time.time() - batch_start_time)
            
            # Move to next position: advance by batch_size + lookahead + random jump
            base_advance = batch_size + lookahead
            random_jump = random.randint(RANDOMNESS_PARAMS["random_start_min"], 
                                       RANDOMNESS_PARAMS["random_start_max"])
            line_cursor += base_advance + random_jump
            
            print(f"    Batch {batch_num+1}: position {cursor_positions[-1]}, "
                  f"advance {base_advance} + random {random_jump} = {base_advance + random_jump}")
            
        except Exception as e:
            print(f"  Error in batch {batch_num+1} at position {line_cursor}: {e}")
            # Skip this batch and try to continue
            line_cursor += min_required_lines + random.randint(100, 1000)
            continue

    if len(batch_hits) == 0:
        print(f"  Warning: No successful batches completed!")
        return {
            'avg_accuracy': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'avg_time': 0.0,
            'batch_size': batch_size,
            'output_size': output_size,
            'prompt_type': prompt_type,
            'dataset': dataset,
            'num_api_calls': num_api_calls,
            'successful_batches': 0,
            'cursor_positions': cursor_positions
        }

    print(f"  Completed {len(batch_hits)}/{num_api_calls} batches successfully")
    print(f"  Cursor positions used: {cursor_positions}")

    return {
        'avg_accuracy': np.mean(batch_accuracy),
        'avg_precision': np.mean(batch_precision),
        'avg_recall': np.mean(batch_recall),
        'avg_time': np.mean(batch_times),
        'batch_size': batch_size,
        'output_size': output_size,
        'prompt_type': prompt_type,
        'dataset': dataset,
        'num_api_calls': num_api_calls,
        'successful_batches': len(batch_hits),
        'cursor_positions': cursor_positions
    }

def save_result_to_global_file(result: dict, run_number: int, experiment_type: str):
    """Save a single result to the global metrics file."""
    # Create header if file doesn't exist
    if not os.path.exists(GLOBAL_METRICS_FILE):
        with open(GLOBAL_METRICS_FILE, 'w') as f:
            f.write("run_number,experiment_type,dataset,batch_size,output_size,prompt_type,num_api_calls,avg_accuracy,avg_precision,avg_recall,avg_time,timestamp\n")
    
    # Append result
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(GLOBAL_METRICS_FILE, 'a') as f:
        f.write(f"{run_number},{experiment_type},{result['dataset']},{result['batch_size']},{result['output_size']},"
               f"{result['prompt_type']},{result['num_api_calls']},{result['avg_accuracy']:.6f},"
               f"{result['avg_precision']:.6f},{result['avg_recall']:.6f},{result['avg_time']:.6f},{timestamp}\n")

#================================================================================================
# EXPERIMENT TYPES
#================================================================================================

def run_hyperparameter_sweep(run_number: int, dataset: str = "mcf_25"):
    """Run a full hyperparameter sweep."""
    batch_sizes = HYPERPARAMETER_RANGES["batch_sizes"]
    output_sizes = HYPERPARAMETER_RANGES["output_sizes"]
    prompt_type = DEFAULT_PARAMS["prompt_type"]
    
    total_combinations = len(batch_sizes) * len(output_sizes)
    print(f"Running hyperparameter sweep (Run {run_number})")
    print(f"Dataset: {dataset}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Output sizes: {output_sizes}")
    print(f"Prompt type: {prompt_type}")
    print(f"Total combinations: {total_combinations}")
    print("="*60)
    
    current = 0
    for batch_size, output_size in product(batch_sizes, output_sizes):
        current += 1
        print(f"[{current}/{total_combinations}] batch_size={batch_size}, output_size={output_size}")
        
        try:
            result = run_single_experiment(batch_size, output_size, prompt_type, dataset, num_api_calls=10)
            save_result_to_global_file(result, run_number, "hyperparameter_sweep")
            print(f"  ✓ Accuracy: {result['avg_accuracy']:.4f}, Time: {result['avg_time']:.2f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print("-" * 40)

def run_prompt_comparison(run_number: int, dataset: str = "mcf_25"):
    """Run a prompt strategy comparison."""
    prompt_types = HYPERPARAMETER_RANGES["prompt_types"]
    batch_size = DEFAULT_PARAMS["batch_size"]
    output_size = DEFAULT_PARAMS["output_size"]
    
    print(f"Running prompt comparison (Run {run_number})")
    print(f"Dataset: {dataset}")
    print(f"Fixed: batch_size={batch_size}, output_size={output_size}")
    print(f"Prompt types: {prompt_types}")
    print("="*60)
    
    for i, prompt_type in enumerate(prompt_types):
        print(f"[{i+1}/{len(prompt_types)}] prompt_type={prompt_type}")
        
        try:
            result = run_single_experiment(batch_size, output_size, prompt_type, dataset, num_api_calls=10)
            save_result_to_global_file(result, run_number, "prompt_comparison")
            print(f"  ✓ Accuracy: {result['avg_accuracy']:.4f}, Time: {result['avg_time']:.2f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print("-" * 40)

def run_dataset_comparison(run_number: int):
    """Compare performance across different datasets."""
    datasets = list(AVAILABLE_DATASETS.keys())
    batch_size = DEFAULT_PARAMS["batch_size"]
    output_size = DEFAULT_PARAMS["output_size"]
    prompt_type = DEFAULT_PARAMS["prompt_type"]
    
    print(f"Running dataset comparison (Run {run_number})")
    print(f"Datasets: {datasets}")
    print(f"Fixed: batch_size={batch_size}, output_size={output_size}, prompt_type={prompt_type}")
    print("="*60)
    
    for i, dataset in enumerate(datasets):
        print(f"[{i+1}/{len(datasets)}] dataset={dataset}")
        
        try:
            result = run_single_experiment(batch_size, output_size, prompt_type, dataset)
            save_result_to_global_file(result, run_number, "dataset_comparison")
            print(f"  ✓ Accuracy: {result['avg_accuracy']:.4f}, Time: {result['avg_time']:.2f}s")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print("-" * 40)

#================================================================================================
# MAIN FUNCTION
#================================================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified prefetching experiment system')
    parser.add_argument('--run', type=int, required=True, help='Starting run number')
    parser.add_argument('--experiment', type=str, required=True, 
                       choices=['single', 'hyperparameter_sweep', 'prompt_comparison', 'dataset_comparison'],
                       help='Type of experiment to run')
    
    # Optional parameters for single experiments
    parser.add_argument('--batch_size', type=int, default=DEFAULT_PARAMS["batch_size"])
    parser.add_argument('--output_size', type=int, default=DEFAULT_PARAMS["output_size"])
    parser.add_argument('--prompt_type', type=str, default=DEFAULT_PARAMS["prompt_type"],
                       choices=HYPERPARAMETER_RANGES["prompt_types"])
    parser.add_argument('--dataset', type=str, default=DEFAULT_PARAMS["dataset"],
                       choices=list(AVAILABLE_DATASETS.keys()),
                       help='Single dataset (ignored if --datasets is used)')
    parser.add_argument('--num_api_calls', type=int, default=DEFAULT_PARAMS["num_api_calls"],
                       help='Number of API calls per experiment (default: 5)')
    
    # Multiple dataset support
    parser.add_argument('--datasets', type=str, nargs='+', 
                       choices=list(AVAILABLE_DATASETS.keys()),
                       help='Multiple datasets to run experiments on (auto-increments run numbers)')
    
    # Randomness control
    parser.add_argument('--random_min', type=int, default=RANDOMNESS_PARAMS["random_start_min"],
                       help='Minimum random jump between batches (default: 100)')
    parser.add_argument('--random_max', type=int, default=RANDOMNESS_PARAMS["random_start_max"],
                       help='Maximum random jump between batches (default: 1000)')
    parser.add_argument('--no_random_seed', action='store_true',
                       help='Disable fixed random seed for truly random behavior')
    parser.add_argument('--random_seed', type=int, default=RANDOMNESS_PARAMS["random_seed"],
                       help='Random seed for reproducible experiments (default: 42)')
    
    args = parser.parse_args()
    
    # Update randomness parameters based on command line arguments
    RANDOMNESS_PARAMS["random_start_min"] = args.random_min
    RANDOMNESS_PARAMS["random_start_max"] = args.random_max
    RANDOMNESS_PARAMS["use_random_seed"] = not args.no_random_seed
    RANDOMNESS_PARAMS["random_seed"] = args.random_seed
    
    # Determine which datasets to use
    if args.datasets:
        datasets_to_run = args.datasets
        print(f"Multiple datasets mode: {datasets_to_run}")
    else:
        datasets_to_run = [args.dataset]
        print(f"Single dataset mode: {args.dataset}")
    
    print(f"Unified Prefetching Experiment System")
    print(f"Starting run: {args.run}, Experiment: {args.experiment}")
    print(f"Global metrics file: {GLOBAL_METRICS_FILE}")
    print(f"Datasets: {datasets_to_run}")
    print(f"Randomness: {args.random_min}-{args.random_max} lines between batches")
    if RANDOMNESS_PARAMS["use_random_seed"]:
        print(f"Random seed: {RANDOMNESS_PARAMS['random_seed']} (reproducible)")
    else:
        print("Random seed: None (truly random)")
    print("="*60)
    
    # Run experiments for each dataset with auto-incrementing run numbers
    for i, dataset in enumerate(datasets_to_run):
        current_run = args.run + i
        print(f"\n{'='*60}")
        print(f"DATASET {i+1}/{len(datasets_to_run)}: {dataset} (Run {current_run})")
        print(f"{'='*60}")
        
        if args.experiment == "single":
            print(f"Single experiment: batch_size={args.batch_size}, output_size={args.output_size}")
            print(f"prompt_type={args.prompt_type}, dataset={dataset}")
            result = run_single_experiment(args.batch_size, args.output_size, args.prompt_type, dataset, args.num_api_calls)
            save_result_to_global_file(result, current_run, "single")
            print(f"Result: Accuracy={result['avg_accuracy']:.4f}, Time={result['avg_time']:.2f}s")
            
        elif args.experiment == "hyperparameter_sweep":
            run_hyperparameter_sweep(current_run, dataset)
            
        elif args.experiment == "prompt_comparison":
            run_prompt_comparison(current_run, dataset)
            
        elif args.experiment == "dataset_comparison":
            print("Note: dataset_comparison experiment type ignores individual dataset selection")
            run_dataset_comparison(current_run)
            break  # Only run once for dataset comparison
    
    if len(datasets_to_run) > 1:
        print(f"\n{'='*60}")
        print(f"ALL EXPERIMENTS COMPLETED!")
        print(f"Runs {args.run} to {args.run + len(datasets_to_run) - 1} saved to {GLOBAL_METRICS_FILE}")
        print(f"Used datasets: {datasets_to_run}")
    else:
        print(f"\nExperiment completed! Results saved to {GLOBAL_METRICS_FILE}")

if __name__ == "__main__":
    main()