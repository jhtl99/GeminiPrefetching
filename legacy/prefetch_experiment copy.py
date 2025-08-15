import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import islice
from google import genai
from google.genai import types
import argparse
import sys

# Load environment variables (if using .env file)
from dotenv import load_dotenv
load_dotenv()  # Remove this line if not using .env file

# Get API key from environment
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID="gemini-2.0-flash"

#------------------------------------------------------------------------------------------------
# Test to see if the API is working
# prompt = """
#     Sort the animals from biggest to smallest.
#     Question: Sort Tiger, Bear, Dog
#     Answer: Bear > Tiger > Dog}
#     Question: Sort Cat, Elephant, Zebra
#     Answer: Elephant > Zebra > Cat}
#     Question: Sort Whale, Goldfish, Monkey
#     Answer:
# """

# response=client.models.generate_content(
#     model=MODEL_ID,
#     contents=prompt,
# )
# print(response.text)

#------------------------------------------------------------------------------------------------
CSV_PATH = "mcf_25_delta.csv"

def format_prompt(df_batch: pd.DataFrame, batch_size: int, lookahead: int, prompt_type: str = "original") -> str:
    """
    Build the prompt from the batch_size rows with different formatting strategies.
    
    Args:
        df_batch: DataFrame with pc and delta_out columns
        batch_size: Number of input lines
        lookahead: Number of predictions to request
        prompt_type: "original", "minimal", "contextual", or "expert"
    """
    pc_col = df_batch["pc"].tolist()
    delta_col = df_batch["delta_out"].tolist()
    
    if prompt_type == "minimal":
        # Just delta values, no PC, no context
        history = " ".join(str(d) for d in delta_col)
        return f"""
{history}
Predict the next {lookahead} values in this sequence. Return ONLY the {lookahead} numbers separated by spaces. Do not include any other text, words, explanations, or confirmations like "okay" - just the numbers."""
    
    elif prompt_type == "contextual":
        # Full context about what PC and delta values represent
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
        # Expert context + importance emphasis
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
# Be sure to look for patterns between the pc and the delta_out in the pairs I've given you.
# """

def parse_predictions(raw: str, lookahead: int, prompt_type: str = "original") -> tuple[np.ndarray, np.ndarray]:
    """
    Turn Gemini's free-form answer into two numeric arrays
    of length lookahead each, padding / truncating as required.
    """
    if raw == None:
        return None, None
    
    # Clean the response and remove common unwanted prefixes
    cleaned = raw.strip().replace("\n", " ")
    
    # Remove common response prefixes that might interfere with parsing
    prefixes_to_remove = ["okay,", "okay", "sure,", "sure", "here", "here:", "the", "predictions", "are:"]
    words = cleaned.lower().split()
    
    # Find the first word that looks like a number (hex or decimal)
    start_idx = 0
    for i, word in enumerate(words):
        # Check if word looks like a hex number or decimal number
        try:
            if word.startswith(('0x', '-', '+')) or word.isdigit() or all(c in '0123456789abcdefABCDEF-+' for c in word):
                start_idx = i
                break
        except:
            continue
    
    # Rejoin from the first numeric word onwards, keeping original case
    original_words = cleaned.split()
    if start_idx < len(original_words):
        cleaned = " ".join(original_words[start_idx:])
    
    fields = cleaned.split()
    
    if prompt_type == "minimal":
        # Only delta values expected, no PC values
        if len(fields) >= lookahead:
            fields = fields[:lookahead]
        else:
            fields.extend(["0"] * (lookahead - len(fields)))
        
        # Generate dummy PC values (just incrementing)
        pcs_int = np.arange(lookahead, dtype=np.int64)
        
        # Parse delta values with error handling
        deltas_int = []
        for i, x in enumerate(fields):
            try:
                deltas_int.append(int(x))
            except ValueError:
                print(f"Warning: Could not parse delta value '{x}' at position {i}, using 0")
                deltas_int.append(0)
        
        deltas_int = np.array(deltas_int, dtype=np.int64)
        
        return pcs_int, deltas_int
    
    else:
        # PC-delta pairs expected
        pair_len = 2 * lookahead
        
        if len(fields) >= pair_len:
            fields = fields[:pair_len]
        else:
            fields.extend(["0"] * (pair_len - len(fields)))

        # even index ⇒ pc (hex) | odd index ⇒ delta_out (dec)
        pcs_hex = fields[0::2]
        deltas_str = fields[1::2]

        # convert – invalid values silently → 0, with error reporting
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


def run_experiment(batch_size: int, output_size: int, prompt_type: str = "original") -> dict:
    """
    Run the prefetching experiment with given parameters.
    
    Args:
        batch_size: Number of input lines to provide as context to the model
        output_size: Number of predictions to make (lookahead)
        prompt_type: Type of prompt to use ("original", "minimal", "contextual", "expert")
        
    Returns:
        Dictionary containing average metrics
    """
    lookahead = output_size
    num_api_calls = 5  # Always run exactly 5 batches (5 API calls)
    line_cursor = 0
    COLS = ["pc", "delta_out"]

    batch_hits = []
    batch_recall = []
    batch_precision = []
    batch_accuracy = []
    batch_times = []

    for batch_num in range(num_api_calls):
        batch_start_time = time.time()

        # Read batch_size lines as input context
        src_chunk = pd.read_csv(
            CSV_PATH, usecols=COLS,
            skiprows=1 + line_cursor, nrows=batch_size,
            header=None, names=COLS
        )
        
        # Generate prompt and get prediction
        prompt = format_prompt(src_chunk, batch_size, lookahead, prompt_type)
        response = client.models.generate_content(model=MODEL_ID,
                                                 contents=prompt)
        _, pred_delta = parse_predictions(response.text, lookahead, prompt_type)

        # Read ground truth (the next lookahead lines after the input batch)
        gt = pd.read_csv(
            CSV_PATH, usecols=COLS,
            skiprows=1 + line_cursor + batch_size,
            nrows=lookahead,
            header=None, names=COLS
        )
        true_delta = gt["delta_out"].to_numpy()

        # Calculate cumulative page numbers
        gt_pages = np.cumsum(true_delta, dtype=int)
        pred_pages = np.cumsum(pred_delta, dtype=int)

        # Calculate metrics
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

        batch_end_time = time.time()
        batch_times.append(batch_end_time - batch_start_time)

        # Move to next batch: skip the current batch + lookahead window
        line_cursor += batch_size + lookahead

    return {
        'avg_accuracy': np.mean(batch_accuracy),
        'avg_precision': np.mean(batch_precision),
        'avg_recall': np.mean(batch_recall),
        'avg_time': np.mean(batch_times),
        'batch_size': batch_size,
        'output_size': output_size,
        'prompt_type': prompt_type
    }


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Run prefetching experiment')
    parser.add_argument('--batch_size', type=int, default=150, help='Number of input lines to provide as context')
    parser.add_argument('--output_size', type=int, default=50, help='Number of predictions to make (lookahead)')
    parser.add_argument('--prompt_type', type=str, default='original', 
                       choices=['original', 'minimal', 'contextual', 'expert'],
                       help='Type of prompt to use')
    
    args = parser.parse_args()
    
    results = run_experiment(args.batch_size, args.output_size, args.prompt_type)
    
    print(f"Results for batch_size={args.batch_size}, output_size={args.output_size}, prompt_type={args.prompt_type}")
    print(f"Average accuracy: {results['avg_accuracy']:.4f}")
    print(f"Average precision: {results['avg_precision']:.4f}")
    print(f"Average recall: {results['avg_recall']:.4f}")
    print(f"Average time (s): {results['avg_time']:.4f}")
    print(f"Note: This experiment ran 5 API calls total")


if __name__ == "__main__":
    main()



