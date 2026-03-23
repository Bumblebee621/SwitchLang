"""
Test script: Run DictaBERT-char-spacefix on the first 1000 lines of
the Hebrew corpus to evaluate whether missing spaces can be restored.

Model: https://huggingface.co/dicta-il/dictabert-char-spacefix

Usage:
    pip install transformers torch
    python scripts/test_spacefix.py

Outputs:
    - data/he_corpus_spacefix_sample.txt   (corrected 1000 lines)
    - data/he_corpus_spacefix_diff.txt     (only lines that changed)
    - Summary statistics printed to stdout
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CORPUS_PATH = os.path.join(PROJECT_DIR, "data", "he_corpus.txt")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "data", "he_corpus_spacefix_sample.txt")
DIFF_PATH   = os.path.join(PROJECT_DIR, "data", "he_corpus_spacefix_diff.txt")

NUM_LINES = 1000
MODEL_NAME = "dicta-il/dictabert-char-spacefix"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- 1. Load the model ------------------------------------------------
    print(f"Loading model: {MODEL_NAME} ...")
    try:
        from transformers import pipeline
    except ImportError:
        print("ERROR: 'transformers' package not found.")
        print("Install with:  pip install transformers torch")
        sys.exit(1)

    oracle = pipeline("token-classification", model=MODEL_NAME)
    print("Model loaded.\n")

    # --- 2. Read corpus lines ---------------------------------------------
    if not os.path.isfile(CORPUS_PATH):
        print(f"ERROR: Corpus file not found at {CORPUS_PATH}")
        sys.exit(1)

    print(f"Reading first {NUM_LINES} lines from: {CORPUS_PATH}")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        lines = []
        for i, line in enumerate(f):
            if i >= NUM_LINES:
                break
            lines.append(line.rstrip("\r\n"))

    print(f"Read {len(lines)} lines.\n")

    # --- 3. Process each line ---------------------------------------------
    fixed_lines = []
    changed_count = 0
    total_spaces_added = 0
    diff_entries = []

    t0 = time.time()
    for idx, original in enumerate(lines):
        # Skip empty / whitespace-only lines
        if not original.strip():
            fixed_lines.append(original)
            continue

        # Run the model
        try:
            raw_output = oracle(original)
        except Exception as e:
            print(f"  WARNING: model error on line {idx+1}: {e}")
            fixed_lines.append(original)
            continue

        # Reconstruct text – LABEL_1 means "insert space before this char"
        fixed = "".join(
            (" " if tok["entity"] == "LABEL_1" else "") + tok["word"]
            for tok in raw_output
        )

        fixed_lines.append(fixed)

        if fixed != original:
            changed_count += 1
            spaces_added = len(fixed) - len(original)
            total_spaces_added += spaces_added
            diff_entries.append(
                f"--- Line {idx+1} ({spaces_added:+d} chars) ---\n"
                f"  ORIG:  {original}\n"
                f"  FIXED: {fixed}\n"
            )

        # Progress
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {idx+1}/{len(lines)} lines  ({elapsed:.1f}s)")

    elapsed = time.time() - t0

    # --- 4. Write outputs -------------------------------------------------
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in fixed_lines:
            f.write(line + "\n")
    print(f"\nCorrected lines written to: {OUTPUT_PATH}")

    with open(DIFF_PATH, "w", encoding="utf-8") as f:
        f.write(f"DictaBERT-char-spacefix diff report\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Lines processed: {len(lines)}\n")
        f.write(f"Lines changed:   {changed_count}\n")
        f.write(f"Total spaces added: {total_spaces_added}\n")
        f.write(f"{'='*60}\n\n")
        for entry in diff_entries:
            f.write(entry + "\n")
    print(f"Diff report written to:     {DIFF_PATH}")

    # --- 5. Summary -------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Lines processed:    {len(lines)}")
    print(f"  Lines changed:      {changed_count}  ({100*changed_count/len(lines):.1f}%)")
    print(f"  Total spaces added: {total_spaces_added}")
    print(f"  Time elapsed:       {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
