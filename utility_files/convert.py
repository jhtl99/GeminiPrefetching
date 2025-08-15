#!/usr/bin/env python3
"""
csv_to_jsonl.py
---------------------------------
Convert a large CSV that has at least the columns `pc` and `delta_out`
into a JSON-Lines file in which each record looks like:

{"contents": [
    {"role": "user", "parts": [{"text": "Here are the most recent 150 program counter..."}]},
    {"role": "model", "parts": [{"text": "delta1 delta2 ... delta50"}]}
]}

* Uses 150 rows of history to predict next 50 pairs
* Writes non-overlapping windows
* Keeps minimal rows in memory for large files
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import deque

def format_prompt(pc_col, delta_col, chunk_size=150, lookahead=50):
    """
    Build the prompt from the last chunk_size rows.
    Returns the formatted prompt string.
    """
    pair_len = 2 * lookahead
    
    # space-separated "pc delta_out" pairs
    history = "\n".join(f"{p} {d}" for p, d in zip(pc_col, delta_col))

    return f"""Here are the most recent {chunk_size} program counter, delta_out pairs in sequential order (pc in hex, delta_out in decimal):
{history}
Predict the NEXT {lookahead} pairs – strictly as:
pc1 delta1 pc2 delta2 ... – no other text. Ensure that you don't make too few predictions; there should be {pair_len} total values.
Be sure to look for patterns between the pc and the delta_out in the pairs I've given you."""

def convert(in_path: Path, out_path: Path,
            chunk_size: int = 150, lookahead: int = 50, n_examples: int = 50) -> None:
    """Stream the CSV and emit `n_examples` JSONL lines."""
    
    # Use deque to efficiently maintain a sliding window
    window = deque(maxlen=chunk_size + lookahead)
    written = 0

    with in_path.open(newline='') as csv_file, out_path.open('w') as jsonl_file:
        reader = csv.DictReader(csv_file)

        if not {'pc', 'delta_out'}.issubset(reader.fieldnames or []):
            sys.exit(f"CSV must contain 'pc' and 'delta_out' columns. "
                     f"Found: {reader.fieldnames}")

        for row in reader:
            # Stop early once we have all the examples we need
            if written >= n_examples:
                break

            pc, delta = row['pc'].strip(), row['delta_out'].strip()
            # Skip blank / malformed rows defensively
            if not pc or not delta:
                continue

            window.append((pc, delta))

            # When we have enough data (history + prediction target)
            if len(window) == chunk_size + lookahead:
                # Split into history and target
                history_data = list(window)[:chunk_size]
                target_data = list(window)[chunk_size:]
                
                # Extract columns for history
                history_pcs = [item[0] for item in history_data]
                history_deltas = [item[1] for item in history_data]
                
                # Extract only delta_out values for target (the model's response)
                target_deltas = [item[1] for item in target_data]
                
                # Format the prompt
                prompt_text = format_prompt(history_pcs, history_deltas, chunk_size, lookahead)
                
                # Create the new format
                record = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": prompt_text}]
                        },
                        {
                            "role": "model", 
                            "parts": [{"text": " ".join(target_deltas)}]
                        }
                    ]
                }
                
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                written += 1
                
                # Reset window to start from the prediction target for next iteration
                # This ensures non-overlapping windows
                new_window = deque(maxlen=chunk_size + lookahead)
                for item in target_data:
                    new_window.append(item)
                window = new_window

    print(f"Finished: wrote {written} JSONL lines to {out_path}")

def main() -> None:
    p = argparse.ArgumentParser(description="CSV → JSONL converter")
    p.add_argument("csv_file", type=Path, help="Input CSV path")
    p.add_argument("jsonl_file", type=Path, help="Output JSONL path")
    p.add_argument("--chunk-size", type=int, default=150,
                   help="History size (default: 150)")
    p.add_argument("--lookahead", type=int, default=50,
                   help="Number of pairs to predict (default: 50)")
    p.add_argument("--examples", type=int, default=50,
                   help="Number of JSONL lines to write (default: 50)")
    args = p.parse_args()
    convert(args.csv_file, args.jsonl_file,
            chunk_size=args.chunk_size, lookahead=args.lookahead, 
            n_examples=args.examples)

if __name__ == "__main__":
    main()