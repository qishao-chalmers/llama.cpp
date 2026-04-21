#!/usr/bin/env python3
"""Compare accuracy across incomplete sweep runs on the same task order.

Aligns on a **contiguous** prefix ``aime#0 .. aime#(n-1)`` present in every run.

Primary data: ``*_per_example.json`` from run_sweep (if present next to ``--out``).

Fallback: parse ``acc ex ...`` lines from ``*.log`` (same summary run_sweep prints).
Use this when per-example JSON was never written (crashed job, different machine path).

If run A has scores through ``aime#24`` and B through ``aime#19``, we compare ``n=20``
only when both have ``aime#0``..``aime#19`` with no gap (``n = min`` contiguous depth).

Optional:
  * ``--max-prefix N`` — cap after taking the common contiguous length.
  * ``--strict-labels`` — reserved for JSON path (log path always uses labels from lines).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Optional

# run_sweep.py: print(f"  acc ex {ei+1}/{len(examples)}: {label} | "
#                     f"gold={gold_disp}  pred={pred_disp}  {mark}  gen_len={gen_len}...")
# Lines may have arbitrary text before ``acc ex`` (streaming output).
_ACC_EX = re.compile(
    r"acc ex (\d+)/(\d+):\s*([^\s|]+)\s*\|\s*"
    r"gold=(\S+)\s+pred=(\S+)\s+(\S+(?:\([^)]*\))?)\s+gen_len=(\d+)"
)


def _out_path_from_log(log_path: str) -> Optional[str]:
    with open(log_path, encoding="utf-8", errors="replace") as f:
        line = f.readline()
    m = re.search(r"--out\s+(\S+)", line)
    if not m:
        return None
    return m.group(1).strip().strip("'\"")


def _quant_from_log_command(log_path: str) -> str:
    with open(log_path, encoding="utf-8", errors="replace") as f:
        line = f.readline()
    m = re.search(r"--quants\s+(\S+)", line)
    return m.group(1).strip() if m else ""


def _per_example_json_from_out(out_path: str, log_path: str) -> str:
    out_path = os.path.expanduser(out_path)
    if os.path.isfile(out_path):
        base = out_path[:-5] if out_path.endswith(".json") else out_path
        return base + "_per_example.json"
    base_name = os.path.basename(out_path)
    stem = base_name[:-5] if base_name.endswith(".json") else base_name
    return os.path.join(os.path.dirname(os.path.abspath(log_path)), stem + "_per_example.json")


def load_scores_from_log(log_path: str) -> tuple[str, dict[str, float], dict[str, str]]:
    """label -> score (0/1), label -> gold; quant from first-line --quants if present."""

    quant = _quant_from_log_command(log_path)
    by_label: dict[str, float] = {}
    golds: dict[str, str] = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _ACC_EX.search(line)
            if not m:
                continue
            _i, _ntot, label, gold, _pred, mark, _gl = m.groups()
            label = str(label).strip()
            correct = str(mark).startswith("✓")
            by_label[label] = 1.0 if correct else 0.0
            golds[label] = str(gold)
    return quant, by_label, golds


def load_per_example_json(path: str) -> tuple[str, dict[str, float], dict[str, str]]:
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

    by_label: dict[str, float] = {}
    golds: dict[str, str] = {}
    for r in rows:
        lab = str(r.get("label", "")).strip()
        by_label[lab] = float(r["score"])
        golds[lab] = str(r.get("gold", ""))
    return quant_key, by_label, golds


def contiguous_aime_depth(by_label: dict[str, float]) -> int:
    """Largest n such that aime#0 .. aime#(n-1) are all present."""

    n = 0
    while f"aime#{n}" in by_label:
        n += 1
    return n


def score_list_for_prefix(by_label: dict[str, float], n: int) -> list[float]:
    return [float(by_label[f"aime#{i}"]) for i in range(n)]


def _acc_prefix(scores: list[float], n: int) -> tuple[float, int]:
    pref = scores[:n]
    correct = sum(1 for s in pref if s >= 0.5)
    return correct / max(1, n), correct


def resolve_run_from_log(log_path: str) -> dict[str, Any]:
    """Load per-example JSON if present; else parse log lines."""

    log_path = os.path.abspath(os.path.expanduser(log_path))
    out = _out_path_from_log(log_path)
    if not out:
        raise ValueError(f"No --out in first line of {log_path!r}")

    pe = _per_example_json_from_out(out, log_path)
    rel_log = os.path.relpath(log_path, os.getcwd())

    if os.path.isfile(pe):
        q, by_l, golds = load_per_example_json(pe)
        return {
            "path": pe,
            "rel": os.path.relpath(pe, os.getcwd()),
            "quant": q,
            "by_label": by_l,
            "golds": golds,
            "source": "per_example_json",
            "n_done": contiguous_aime_depth(by_l),
        }

    q, by_l, golds = load_scores_from_log(log_path)
    print(
        f"[info] {rel_log}: no {os.path.basename(pe)!r}; using acc ex lines from log "
        f"({len(by_l)} examples parsed).",
        file=sys.stderr,
    )
    return {
        "path": log_path,
        "rel": rel_log,
        "quant": q or "?",
        "by_label": by_l,
        "golds": golds,
        "source": "log_parse",
        "n_done": contiguous_aime_depth(by_l),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="append", default=[], metavar="PATH", help="*_per_example.json (repeatable)")
    ap.add_argument("--log", action="append", default=[], metavar="PATH", help="run_sweep .log (repeatable)")
    ap.add_argument(
        "--max-prefix",
        type=int,
        default=0,
        metavar="N",
        help="After common contiguous depth, use at most N examples (0 = no extra cap).",
    )
    ap.add_argument(
        "--strict-labels",
        action="store_true",
        help="Check gold strings match on compared indices (warn unless strict error desired).",
    )
    args = ap.parse_args()

    runs: list[dict[str, Any]] = []
    for p in args.json:
        p = os.path.abspath(os.path.expanduser(p))
        q, by_l, golds = load_per_example_json(p)
        runs.append(
            {
                "path": p,
                "rel": os.path.relpath(p, os.getcwd()),
                "quant": q,
                "by_label": by_l,
                "golds": golds,
                "source": "per_example_json",
                "n_done": contiguous_aime_depth(by_l),
            }
        )
    for lp in args.log:
        runs.append(resolve_run_from_log(lp))

    runs = list({r["path"]: r for r in runs}.values())
    if len(runs) < 1:
        print("Provide --json and/or --log.", file=sys.stderr)
        sys.exit(2)

    n_min = min(r["n_done"] for r in runs)
    if args.max_prefix and args.max_prefix > 0:
        n_min = min(n_min, int(args.max_prefix))
    if n_min <= 0:
        print("No overlapping contiguous aime#0.. prefix across runs.", file=sys.stderr)
        sys.exit(2)

    for r in runs:
        r["scores"] = score_list_for_prefix(r["by_label"], n_min)
        r["labels"] = [f"aime#{i}" for i in range(n_min)]
        r["golds_list"] = [r["golds"].get(f"aime#{i}", "") for i in range(n_min)]

    if args.strict_labels:
        ref_g = runs[0]["golds_list"]
        for r in runs[1:]:
            for i in range(n_min):
                if r["golds_list"][i] != ref_g[i]:
                    print(
                        f"[error] gold mismatch at {i}: {runs[0]['rel']!r} vs {r['rel']!r}",
                        file=sys.stderr,
                    )
                    sys.exit(3)
    else:
        ref_g = runs[0]["golds_list"]
        for r in runs[1:]:
            for i in range(n_min):
                if r["golds_list"][i] != ref_g[i]:
                    print(
                        f"[warn] gold mismatch at aime#{i}: {runs[0]['rel']!r} vs {r['rel']!r}",
                        file=sys.stderr,
                    )

    print(
        f"Common contiguous prefix: aime#0 .. aime#{n_min - 1}  (n={n_min}, {len(runs)} runs)"
    )
    print()
    hdr = f"{'run':42s} {'src':16s} {'quant':10s} {'depth':>5s} {'acc@n':>8s} {'ok':>5s}"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        acc, cor = _acc_prefix(r["scores"], n_min)
        print(
            f"{r['rel'][:42]:42s} {r['source'][:16]:16s} {str(r['quant'])[:10]:10s} "
            f"{r['n_done']:5d} {acc:8.4f} {cor:5d}"
        )

    if len(runs) >= 2:
        print()
        print("Pairwise disagreement on common prefix (first run vs each other):")
        base_ok = [s >= 0.5 for s in runs[0]["scores"]]
        for r in runs[1:]:
            rok = [s >= 0.5 for s in r["scores"]]
            both = sum(1 for i in range(n_min) if base_ok[i] and rok[i])
            only_a = sum(1 for i in range(n_min) if base_ok[i] and not rok[i])
            only_b = sum(1 for i in range(n_min) if not base_ok[i] and rok[i])
            neither = sum(1 for i in range(n_min) if not base_ok[i] and not rok[i])
            print(
                f"  {runs[0]['rel'][:36]!r} vs {r['rel'][:36]!r}: "
                f"both_ok={both} only_1st={only_a} only_2nd={only_b} both_bad={neither}"
            )


if __name__ == "__main__":
    main()
