#!/usr/bin/env python3
"""summarize_datasets.py — Print average prompt/output lengths for each benchmark.

Reads dataset JSONLs for prompt lengths (chars and approx tokens)
and results JSONs for actual generated token counts.

Usage:
    python3 research/scripts/summarize_datasets.py
    python3 research/scripts/summarize_datasets.py --results-dir research/results/qwen3-8b
    python3 research/scripts/summarize_datasets.py --model qwen3-8b
"""

import argparse
import json
import os
import statistics

DATA_DIR    = "research/data"
RESULTS_DIR = "research/results"

# Dataset registry: name → (jsonl_path, prompt_field, completion_field)
# For flat text (wikitext2), jsonl_path=None and we report fixed chunk size.
DATASETS = [
    ("WikiText2",      None,                                  None,         None),
    ("GSM8K",          "gsm8k_test.jsonl",                    "prompt",     "completion"),
    ("LongBench",      "LongBench/qasper.jsonl",              None,         None),   # context+input
    ("NIAH-4K",        "niah_4096.jsonl",                     "prompt",     "completion"),
    ("NIAH-16K",       "niah_16k.jsonl",                      "prompt",     "completion"),
    ("NIAH-32K",       "niah_32k.jsonl",                      "prompt",     "completion"),
    ("NIAH-128K",      "niah_128k.jsonl",                     "prompt",     "completion"),
    ("HumanEval",      "humaneval.jsonl",                     "prompt",     "completion"),
]

# Results file name → dataset name (for gen_len lookup)
RESULTS_FILES = {
    "WikiText2":   "wikitext2.json",
    "GSM8K":       "gsm8k.json",
    "LongBench":   "longbench_qasper.json",
    "NIAH-4K":     "niah_4k.json",
    "NIAH-16K":    "niah_16k.json",
    "NIAH-32K":    "niah_32k.json",
    "NIAH-128K":   "niah_128k.json",
    "HumanEval":   "humaneval.json",
}


def approx_tokens(text):
    """Rough token count: chars / 4 (typical for English/code)."""
    return len(text) / 4.0


def load_jsonl(path, max_examples=500):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prompt_lengths(records, prompt_field):
    lengths_chars  = []
    lengths_tokens = []
    for rec in records:
        if prompt_field and prompt_field in rec:
            text = rec[prompt_field]
        elif "context" in rec and "input" in rec:
            # LongBench format
            text = rec["context"] + "\n\n" + rec["input"]
        else:
            continue
        lengths_chars.append(len(text))
        lengths_tokens.append(approx_tokens(text))
    return lengths_chars, lengths_tokens


def completion_lengths(records, completion_field):
    lengths = []
    for rec in records:
        if completion_field and completion_field in rec:
            text = rec[completion_field]
        elif "answers" in rec and rec["answers"]:
            text = rec["answers"][0] if isinstance(rec["answers"], list) else rec["answers"]
        else:
            continue
        lengths.append(approx_tokens(text))
    return lengths


def load_gen_len(results_path):
    """Return {quant: mean_gen_tokens} from results JSON."""
    if not os.path.exists(results_path):
        return {}
    with open(results_path) as f:
        data = json.load(f)
    return {q: v.get("mean_gen_tokens") for q, v in data.items()
            if isinstance(v, dict) and v.get("mean_gen_tokens") is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None,
                    help="Results directory to read gen_len from (default: research/results)")
    ap.add_argument("--model", default=None,
                    help="Model subfolder under research/results/ (e.g. qwen3-8b)")
    ap.add_argument("--max-examples", type=int, default=500,
                    help="Max examples to read per dataset for length stats")
    args = ap.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = os.path.join(RESULTS_DIR, args.model) if args.model else RESULTS_DIR

    print(f"\nDataset length summary")
    print(f"  Data dir:    {DATA_DIR}")
    print(f"  Results dir: {results_dir}")
    print()

    hdr = (f"  {'Dataset':<14}  {'n':>5}  {'prompt chars':>13}  "
           f"{'prompt ~tok':>12}  {'gold ~tok':>10}  {'gen tokens (fp16)':>18}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name, jsonl_rel, prompt_field, completion_field in DATASETS:
        # ── Prompt / completion lengths from JSONL ────────────────────────────
        if jsonl_rel is None:
            # WikiText2: fixed sliding window
            print(f"  {'WikiText2':<14}  {'—':>5}  {'—':>13}  "
                  f"{'~1024 (fixed)':>12}  {'n/a':>10}  {'n/a':>18}")
            continue

        jsonl_path = os.path.join(DATA_DIR, jsonl_rel)
        if not os.path.exists(jsonl_path):
            print(f"  {name:<14}  (dataset not found: {jsonl_path})")
            continue

        records = load_jsonl(jsonl_path, max_examples=args.max_examples)
        p_chars, p_toks = prompt_lengths(records, prompt_field)
        c_toks          = completion_lengths(records, completion_field)

        avg_p_chars = statistics.mean(p_chars) if p_chars else 0
        avg_p_toks  = statistics.mean(p_toks)  if p_toks  else 0
        avg_c_toks  = statistics.mean(c_toks)  if c_toks  else 0

        # ── Gen tokens from results JSON ──────────────────────────────────────
        results_path = os.path.join(results_dir, RESULTS_FILES.get(name, ""))
        gen_lens = load_gen_len(results_path)
        fp16_gen = gen_lens.get("fp16")
        gen_str  = f"{fp16_gen:.1f}" if fp16_gen is not None else "—"

        print(f"  {name:<14}  {len(records):>5}  {avg_p_chars:>13,.0f}  "
              f"{avg_p_toks:>12,.0f}  {avg_c_toks:>10,.1f}  {gen_str:>18}")

    # ── Per-quant gen_len breakdown ───────────────────────────────────────────
    print()
    print("  Generated tokens per quant (mean):")
    print(f"  {'Dataset':<14}", end="")

    # Collect all quants across all datasets
    all_quants = []
    all_gen    = {}
    for name, _, _, _ in DATASETS:
        results_path = os.path.join(results_dir, RESULTS_FILES.get(name, ""))
        gen_lens = load_gen_len(results_path)
        all_gen[name] = gen_lens
        for q in gen_lens:
            if q not in all_quants:
                all_quants.append(q)

    for q in all_quants:
        print(f"  {q:>10}", end="")
    print()
    print("  " + "-" * (14 + 12 * len(all_quants)))

    for name, jsonl_rel, _, _ in DATASETS:
        if jsonl_rel is None or name not in all_gen or not all_gen[name]:
            continue
        print(f"  {name:<14}", end="")
        for q in all_quants:
            v = all_gen[name].get(q)
            print(f"  {v:>10.1f}" if v is not None else f"  {'—':>10}", end="")
        print()


if __name__ == "__main__":
    main()
