#!/usr/bin/env python3
"""eval_code.py — Compute pass@1 for HumanEval predictions from run_sweep.py.

run_sweep.py --eval-metric exact scores all code predictions as 0 (generated code
never literally matches canonical_solution). This script re-scores by executing
each prediction against the original test harness.

Execution model:
  For each example, combine:
    prompt + prediction + "\\n\\n" + test + "\\ncheck(" + entry_point + ")"
  Run in a subprocess with a timeout. Pass = exit code 0.

Usage:
    python3 research/scripts/eval_code.py \\
        research/results/humaneval_per_example.json \\
        --jsonl research/data/humaneval.jsonl

    python3 research/scripts/eval_code.py per_ex.json --jsonl humaneval.jsonl \\
        --quants fp16 int4_ch int2_ch --timeout 10 --out humaneval_pass1.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
import os


def execute_code(code: str, timeout: float) -> bool:
    """Run code string in a subprocess. Return True if exit code == 0."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


def _clean_pred(pred: str) -> str:
    """Strip stop-string remnants that make the code syntactically invalid.

    run_sweep.py stop strings ('\n\ndef ', '\n\nif __name__') are included at
    the end of the prediction. Truncate at the first occurrence of these so
    the assembled code doesn't have an incomplete statement.
    """
    import re
    # Cut at any top-level def or if __name__ that follows a blank line
    pred = re.sub(r'\n\ndef .*', '', pred, flags=re.DOTALL)
    pred = re.sub(r'\n\nif __name__.*', '', pred, flags=re.DOTALL)
    return pred.rstrip()


def score_prediction(prompt: str, pred: str, test: str, entry_point: str,
                     timeout: float) -> bool:
    """Assemble and execute: prompt + pred + test harness + check(entry_point)."""
    pred = _clean_pred(pred)
    code = prompt + pred + "\n\n" + test + f"\ncheck({entry_point})\n"
    return execute_code(code, timeout)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("per_ex_json",
                    help="--save-per-example JSON from run_sweep.py")
    ap.add_argument("--jsonl", required=True,
                    help="Original HumanEval JSONL (must have test + entry_point fields)")
    ap.add_argument("--quants", nargs="+", default=None,
                    help="Quants to evaluate (default: all)")
    ap.add_argument("--timeout", type=float, default=10.0,
                    help="Execution timeout per example in seconds (default: 10)")
    ap.add_argument("--out", default=None,
                    help="Save pass@1 results JSON (default: print only)")
    args = ap.parse_args()

    # Load per_example predictions
    with open(args.per_ex_json) as f:
        per_ex = json.load(f)

    # Load original JSONL (test harnesses indexed by position)
    originals = []
    with open(args.jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                originals.append(json.loads(line))

    quants = args.quants or list(per_ex.keys())
    results = {}

    for quant in quants:
        if quant not in per_ex:
            print(f"WARNING: '{quant}' not in per_ex JSON, skipping", file=sys.stderr)
            continue

        examples = per_ex[quant]
        if len(examples) > len(originals):
            print(f"WARNING: more predictions ({len(examples)}) than originals "
                  f"({len(originals)}) for {quant}", file=sys.stderr)

        n_pass = 0
        n_total = 0
        detail = []

        for i, ex in enumerate(examples):
            if i >= len(originals):
                break
            orig = originals[i]

            if "test" not in orig or "entry_point" not in orig:
                print(f"ERROR: JSONL entry {i} missing 'test'/'entry_point' — "
                      f"re-run fetch_humaneval.py", file=sys.stderr)
                sys.exit(1)

            pred = ex.get("pred", "")
            passed = score_prediction(
                prompt=orig["prompt"],
                pred=pred,
                test=orig["test"],
                entry_point=orig["entry_point"],
                timeout=args.timeout,
            )

            n_pass += int(passed)
            n_total += 1
            detail.append({
                "task_id":    orig.get("task_id", f"example/{i}"),
                "passed":     passed,
                "pred":       pred,
            })

            status = "PASS" if passed else "FAIL"
            print(f"  [{quant}] {orig.get('task_id', i):30s}  {status}")

        pass_at_1 = n_pass / n_total if n_total > 0 else float("nan")
        print(f"\n{quant}: pass@1 = {pass_at_1:.3f}  ({n_pass}/{n_total})\n")
        results[quant] = {
            "pass_at_1": pass_at_1,
            "n_pass":    n_pass,
            "n_total":   n_total,
            "examples":  detail,
        }

    # Summary table
    print("\n" + "=" * 45)
    print(f"  {'quant':<18}  {'pass@1':>7}  {'n_pass':>7}  {'n_total':>7}")
    print("  " + "-" * 41)
    for q, r in results.items():
        p = r["pass_at_1"]
        print(f"  {q:<18}  {p:>7.3f}  {r['n_pass']:>7}  {r['n_total']:>7}")

    if args.out:
        # Strip per-example detail for the summary file to keep it small
        summary = {q: {k: v for k, v in r.items() if k != "examples"}
                   for q, r in results.items()}
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary → {args.out}")


if __name__ == "__main__":
    main()
