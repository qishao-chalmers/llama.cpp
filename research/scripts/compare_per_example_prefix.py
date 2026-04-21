#!/usr/bin/env python3
"""Compare accuracy across incomplete sweep runs on the same task order.

run_sweep.py writes ``<out_stem>_per_example.json`` with one list per quant key;
each element is one *completed* example in order (``aime#0``, ``aime#1``, ...).

If run A finished 25 examples and run B finished 20, this script uses the first
``min(25, 20) = 20`` positions and reports accuracy on that common prefix so
runs are comparable.

Inputs:
  * Paths to ``*_per_example.json``, and/or
  * Paths to ``*.log`` files whose first line contains ``--out .../foo.json``
    (resolves ``foo_per_example.json`` next to ``foo.json``, with basename
    fallback if the absolute ``--out`` path does not exist locally).

Optional:
  * ``--max-prefix N`` — cap the prefix length (after taking min across runs).
  * ``--strict-labels`` — require matching ``label`` strings for compared indices.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Optional


def _out_path_from_log(log_path: str) -> Optional[str]:
    with open(log_path, encoding="utf-8", errors="replace") as f:
        line = f.readline()
    m = re.search(r"--out\s+(\S+)", line)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def _per_example_json_from_out(out_path: str, log_path: str) -> str:
    out_path = os.path.expanduser(out_path)
    if os.path.isfile(out_path):
        base = out_path[:-5] if out_path.endswith(".json") else out_path
        return base + "_per_example.json"
    base_name = os.path.basename(out_path)
    if base_name.endswith(".json"):
        stem = base_name[:-5]
    else:
        stem = base_name
    sibling = os.path.join(os.path.dirname(os.path.abspath(log_path)), stem + "_per_example.json")
    return sibling


def _per_example_from_log(log_path: str) -> str:
    out = _out_path_from_log(log_path)
    if not out:
        raise ValueError(f"No --out in first line of {log_path!r}")
    pe = _per_example_json_from_out(out, log_path)
    if not os.path.isfile(pe):
        raise FileNotFoundError(
            f"Per-example JSON not found: {pe!r} (from log {log_path!r}, --out was {out!r})"
        )
    return pe


def load_per_example(path: str) -> tuple[str, list[float], list[str], list[str]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object at top level: {path}")

    quant_key = ""
    rows: list[dict[str, Any]] = []
    for k, v in data.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "score" in v[0]:
            quant_key = str(k)
            rows = v
            break
    if not rows:
        raise ValueError(f"No per-example list found in {path!r}")

    scores = [float(r["score"]) for r in rows]
    labels = [str(r.get("label", "")) for r in rows]
    golds = [str(r.get("gold", "")) for r in rows]
    return quant_key, scores, labels, golds


def _acc_prefix(scores: list[float], n: int) -> tuple[float, int]:
    pref = scores[:n]
    # Treat score >= 0.5 as correct for soft scores
    correct = sum(1 for s in pref if s >= 0.5)
    return correct / max(1, n), correct


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="append", default=[], metavar="PATH", help="*_per_example.json (repeatable)")
    ap.add_argument("--log", action="append", default=[], metavar="PATH", help="run_sweep .log (repeatable)")
    ap.add_argument(
        "--max-prefix",
        type=int,
        default=0,
        metavar="N",
        help="After min-length across runs, use at most N examples (0 = no extra cap).",
    )
    ap.add_argument(
        "--strict-labels",
        action="store_true",
        help="Fail if labels differ across runs for any compared index.",
    )
    args = ap.parse_args()

    paths: list[str] = []
    for p in args.json:
        paths.append(os.path.abspath(os.path.expanduser(p)))
    for lp in args.log:
        paths.append(os.path.abspath(_per_example_from_log(os.path.expanduser(lp))))

    paths = list(dict.fromkeys(paths))
    if len(paths) < 1:
        print("Provide --json and/or --log.", file=sys.stderr)
        sys.exit(2)

    runs: list[dict[str, Any]] = []
    for p in paths:
        q, scores, labels, golds = load_per_example(p)
        runs.append(
            {
                "path": p,
                "rel": os.path.relpath(p, os.getcwd()),
                "quant": q,
                "scores": scores,
                "labels": labels,
                "golds": golds,
                "n_done": len(scores),
            }
        )

    n_min = min(r["n_done"] for r in runs)
    if args.max_prefix and args.max_prefix > 0:
        n_min = min(n_min, int(args.max_prefix))
    if n_min <= 0:
        print("No overlapping prefix (empty results).", file=sys.stderr)
        sys.exit(2)

    # Label consistency
    if args.strict_labels:
        ref_labels = runs[0]["labels"][:n_min]
        for r in runs[1:]:
            for i in range(n_min):
                if r["labels"][i] != ref_labels[i]:
                    print(
                        f"[error] label mismatch at index {i}: {runs[0]['rel']!r} has {ref_labels[i]!r}, "
                        f"{r['rel']!r} has {r['labels'][i]!r}",
                        file=sys.stderr,
                    )
                    sys.exit(3)
    else:
        # Gold consistency (sanity)
        ref_gold = runs[0]["golds"][:n_min]
        for r in runs[1:]:
            for i in range(n_min):
                if r["golds"][i] != ref_gold[i]:
                    print(
                        f"[warn] gold mismatch at index {i}: {runs[0]['rel']!r} vs {r['rel']!r} "
                        f"({ref_gold[i]!r} vs {r['golds'][i]!r})",
                        file=sys.stderr,
                    )

    print(f"Common prefix length: {n_min}  (min completed across {len(runs)} runs)")
    print()
    hdr = f"{'run':50s} {'quant':10s} {'n_done':>6s} {'acc@n':>8s} {'correct':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        acc, cor = _acc_prefix(r["scores"], n_min)
        print(
            f"{r['rel'][:50]:50s} {r['quant']:10s} {r['n_done']:6d} {acc:8.4f} {cor:8d}"
        )

    # Pairwise agreement on prefix (first run vs others)
    if len(runs) >= 2:
        print()
        print("Pairwise disagreement on common prefix (first run vs each other):")
        base_scores = runs[0]["scores"][:n_min]
        for r in runs[1:]:
            b_ok = [s >= 0.5 for s in base_scores]
            r_ok = [s >= 0.5 for s in r["scores"][:n_min]]
            both = sum(1 for i in range(n_min) if b_ok[i] and r_ok[i])
            only_a = sum(1 for i in range(n_min) if b_ok[i] and not r_ok[i])
            only_b = sum(1 for i in range(n_min) if not b_ok[i] and r_ok[i])
            neither = sum(1 for i in range(n_min) if not b_ok[i] and not r_ok[i])
            print(
                f"  {runs[0]['rel'][:40]!r} vs {r['rel'][:40]!r}: "
                f"both_right={both} only_first={only_a} only_second={only_b} both_wrong={neither}"
            )


if __name__ == "__main__":
    main()
