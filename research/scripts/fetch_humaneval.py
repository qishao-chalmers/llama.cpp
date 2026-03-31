#!/usr/bin/env python3
"""
fetch_humaneval.py — Download HumanEval and format as JSONL for run_sweep structured mode.

HumanEval is a Python code generation benchmark with 164 problems.
Each problem provides a function signature + docstring as prompt;
the canonical_solution is the reference completion.

Evaluation notes:
  - PPL/KL: measure how likely the canonical solution is — works out of the box.
  - Accuracy (pass@1): requires executing the generated code against test cases.
    Use --eval-accuracy with --answer-regex '(?s)(def .+)' to capture the full
    function, then run test_cases externally. For quick smoke tests, just use PPL.

Each output record:
  {"prompt": "<function signature + docstring>",
   "completion": "<canonical_solution>",
   "dataset": "humaneval", "id": <int>, "task_id": "<str>"}

Usage:
  python3 research/scripts/fetch_humaneval.py --out research/data/humaneval.jsonl
  python3 research/scripts/fetch_humaneval.py --out research/data/humaneval.jsonl --n-examples 50
"""

import argparse
import json
import os

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("pip install datasets")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",        default="research/data/humaneval.jsonl")
    parser.add_argument("--n-examples", type=int, default=0,
                        help="Max examples to export (0 = all 164)")
    args = parser.parse_args()

    print("Downloading HumanEval...", flush=True)
    ds = load_dataset("openai_humaneval", split="test")

    records = list(ds)
    if args.n_examples > 0:
        records = records[:args.n_examples]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    prompt_toks = []
    comp_toks   = []
    with open(args.out, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            prompt     = rec["prompt"]             # function sig + docstring
            completion = rec["canonical_solution"] # reference implementation

            prompt_toks.append(len(prompt.split()))
            comp_toks.append(len(completion.split()))

            out = {
                "prompt":      prompt,
                "completion":  completion,
                "test":        rec["test"],        # test harness for eval_code.py
                "entry_point": rec["entry_point"], # function name called by tests
                "dataset":     "humaneval",
                "id":          i,
                "task_id":     rec["task_id"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    n = len(records)
    print(f"Wrote {n} examples to {args.out}")
    print(f"  Prompt length  : avg ~{sum(prompt_toks)//n} words  "
          f"(max {max(prompt_toks)})")
    print(f"  Completion len : avg ~{sum(comp_toks)//n} words  "
          f"(max {max(comp_toks)})")
    print(f"  Recommended flags: --n-ctx 1024 --quants fp16 int8_ch int4_ch int3_ch int2_ch")
    print(f"  Evaluate pass@1 after the sweep:")
    print(f"    python3 research/scripts/eval_code.py \\")
    print(f"        research/results/humaneval_per_example.json \\")
    print(f"        --jsonl {args.out}")


if __name__ == "__main__":
    main()
