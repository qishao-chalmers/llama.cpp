#!/usr/bin/env python3
"""analyze_alarms.py — Evaluate per-step generation metrics as early-warning signals.

Reads a --save-per-example JSON produced by run_sweep.py (must have been run
with --save-per-example so gen_diags are present) and answers two questions:

  1. Which metric (H, p_max, rep_rate, self_surp) best separates correct from
     wrong generations across quant types?

  2. At a given threshold, how early does the alarm fire relative to generation
     length?  (Lead time — how many steps before the end of generation.)

Metrics tracked per step:
  H         — entropy (nats).  High = model confused.   Alarm when H > threshold.
  p_max     — top-1 probability.  Low = uncertain.      Alarm when p_max < threshold.
  rep_rate  — 3-gram repetition rate in last 20 tokens. Alarm when rep_rate > threshold.
  self_surp — log-prob of prev top-1 under current dist. Low = incoherent.
              Alarm when self_surp < threshold (skip NaN first step).

For each metric and quant:
  TPR  — fraction of WRONG examples where alarm fires at least once
  FPR  — fraction of CORRECT examples where alarm fires at least once (false alarm rate)
  E[pos] — mean step at which alarm first fires in wrong examples (lead time = gen_len - E[pos])

Usage:
    python3 analyze_alarms.py research/results/niah_per_ex.json
    python3 analyze_alarms.py research/results/gsm8k_per_ex.json --quants fp16 int2_ch
    python3 analyze_alarms.py per_ex.json --metric H --sweep     # sweep threshold grid
"""

import argparse
import json
import math
import sys

import numpy as np


# ── Default thresholds (one sensible starting point per metric) ───────────────
DEFAULT_THRESHOLDS = {
    "H":         5.0,    # nats; ~32k vocab model has max H≈10.4; fp16 typical ~2-4
    "p_max":     0.10,   # below 10% confidence → alarm
    "rep_rate":  0.40,   # >40% repeated 3-grams in last 20 tokens
    "self_surp": -5.0,   # log-prob < -5 nats → current dist finds prev top-1 very unlikely
}

# Direction: "high" = alarm when value > threshold; "low" = alarm when value < threshold
ALARM_DIRECTION = {
    "H":         "high",
    "p_max":     "low",
    "rep_rate":  "high",
    "self_surp": "low",
}


def _fires(values, threshold, direction):
    """Return index of first alarm step, or None if no alarm fires."""
    for i, v in enumerate(values):
        if v != v:          # NaN (e.g. self_surp step 0)
            continue
        if direction == "high" and v > threshold:
            return i
        if direction == "low"  and v < threshold:
            return i
    return None


def _max_val(values):
    vals = [v for v in values if v == v]
    return max(vals) if vals else float("nan")


def _min_val(values):
    vals = [v for v in values if v == v]
    return min(vals) if vals else float("nan")


def analyze(per_ex_by_quant, quants, metric, threshold):
    """For each quant, compute TPR/FPR/mean_alarm_step at given threshold."""
    direction = ALARM_DIRECTION[metric]
    rows = []
    for q in quants:
        examples = per_ex_by_quant.get(q, [])
        wrong    = [e for e in examples if e["score"] < 0.5 and "gen_diags" in e]
        correct  = [e for e in examples if e["score"] >= 0.5 and "gen_diags" in e]
        if not examples:
            continue

        tp = fp = 0
        alarm_steps = []
        for e in wrong:
            vals = e["gen_diags"].get(metric, [])
            pos  = _fires(vals, threshold, direction)
            if pos is not None:
                tp += 1
                alarm_steps.append(pos)
        for e in correct:
            vals = e["gen_diags"].get(metric, [])
            if _fires(vals, threshold, direction) is not None:
                fp += 1

        tpr = tp / len(wrong)   if wrong   else float("nan")
        fpr = fp / len(correct) if correct else float("nan")
        mean_alarm  = sum(alarm_steps) / len(alarm_steps) if alarm_steps else float("nan")
        mean_gen    = sum(e["gen_len"] for e in wrong) / len(wrong) if wrong else float("nan")
        mean_lead   = mean_gen - mean_alarm if mean_alarm == mean_alarm else float("nan")

        rows.append({
            "quant":      q,
            "n_wrong":    len(wrong),
            "n_correct":  len(correct),
            "tpr":        tpr,
            "fpr":        fpr,
            "mean_alarm_step": mean_alarm,
            "mean_gen_len":    mean_gen,
            "mean_lead_time":  mean_lead,
        })
    return rows


def sweep_thresholds(per_ex_by_quant, quants, metric, n_points=20):
    """Sweep thresholds and return (thresholds, tpr_per_quant, fpr_per_quant)."""
    direction = ALARM_DIRECTION[metric]
    # Collect all values across all quants to determine range
    all_vals = []
    for q in quants:
        for e in per_ex_by_quant.get(q, []):
            if "gen_diags" in e:
                all_vals.extend(v for v in e["gen_diags"].get(metric, []) if v == v)
    if not all_vals:
        return [], {}, {}

    lo, hi = np.percentile(all_vals, 2), np.percentile(all_vals, 98)
    thresholds = np.linspace(lo, hi, n_points).tolist()

    tpr_per_quant = {q: [] for q in quants}
    fpr_per_quant = {q: [] for q in quants}
    for thr in thresholds:
        for row in analyze(per_ex_by_quant, quants, metric, thr):
            q = row["quant"]
            tpr_per_quant[q].append(row["tpr"])
            fpr_per_quant[q].append(row["fpr"])

    return thresholds, tpr_per_quant, fpr_per_quant


def print_table(rows, metric, threshold):
    direction = ALARM_DIRECTION[metric]
    cmp = ">" if direction == "high" else "<"
    print(f"\nMetric: {metric}  threshold: {cmp} {threshold}")
    print(f"  {'quant':<18}  {'n_wrong':>7}  {'n_correct':>9}  "
          f"{'TPR':>5}  {'FPR':>5}  "
          f"{'alarm@step':>10}  {'gen_len':>7}  {'lead_time':>9}")
    print("  " + "-" * 80)
    for r in rows:
        tpr = f"{r['tpr']:.2f}" if r['tpr'] == r['tpr'] else "  — "
        fpr = f"{r['fpr']:.2f}" if r['fpr'] == r['fpr'] else "  — "
        al  = f"{r['mean_alarm_step']:.1f}" if r['mean_alarm_step'] == r['mean_alarm_step'] else "  —"
        gl  = f"{r['mean_gen_len']:.1f}"   if r['mean_gen_len']    == r['mean_gen_len']    else "  —"
        lt  = f"{r['mean_lead_time']:.1f}" if r['mean_lead_time']  == r['mean_lead_time']  else "  —"
        print(f"  {r['quant']:<18}  {r['n_wrong']:>7}  {r['n_correct']:>9}  "
              f"{tpr:>5}  {fpr:>5}  {al:>10}  {gl:>7}  {lt:>9}")


def print_sweep(thresholds, tpr_per_quant, fpr_per_quant, metric, quants):
    direction = ALARM_DIRECTION[metric]
    cmp = ">" if direction == "high" else "<"
    print(f"\nThreshold sweep — metric: {metric}")
    header = f"  {'thr':>8}  " + "  ".join(f"{q[:12]:>12}" for q in quants)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, thr in enumerate(thresholds):
        vals = "  ".join(
            f"  {tpr_per_quant[q][i]:.2f}/{fpr_per_quant[q][i]:.2f}"
            if i < len(tpr_per_quant[q]) else "          — "
            for q in quants
        )
        print(f"  {cmp}{thr:>7.3f}  {vals}")
    print("  (TPR/FPR per quant)")


def print_value_summary(per_ex_by_quant, quants):
    """Print mean/max/min of each metric per quant, split by correct/wrong."""
    metrics = list(DEFAULT_THRESHOLDS.keys())
    print("\nMetric value summary (mean ± std | max) — correct vs wrong examples:")
    for metric in metrics:
        print(f"\n  {metric}:")
        for q in quants:
            examples = per_ex_by_quant.get(q, [])
            for label, exs in [("correct", [e for e in examples if e["score"] >= 0.5]),
                                ("wrong",   [e for e in examples if e["score"] <  0.5])]:
                vals = []
                for e in exs:
                    if "gen_diags" in e:
                        vals.extend(v for v in e["gen_diags"].get(metric, []) if v == v)
                if not vals:
                    continue
                arr = np.array(vals)
                print(f"    {q:<18} {label:<8}  "
                      f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
                      f"max={arr.max():.3f}  p95={np.percentile(arr, 95):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_ex_json", help="--save-per-example JSON from run_sweep.py")
    ap.add_argument("--quants",    nargs="+", default=None,
                    help="Quant names to include (default: all in file)")
    _valid_metrics = list(DEFAULT_THRESHOLDS.keys())
    ap.add_argument("--metric",    nargs="*", default=None,
                    metavar="METRIC",
                    help=f"Metrics to analyse (default: all). Choices: {_valid_metrics}")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override default threshold (only used when exactly one --metric given)")
    ap.add_argument("--sweep",     action="store_true",
                    help="Sweep thresholds and print TPR/FPR grid (loops over each --metric)")
    ap.add_argument("--summary",   action="store_true",
                    help="Print value distribution summary instead of alarm table")
    args = ap.parse_args()

    # Validate metric names
    _valid_metrics = list(DEFAULT_THRESHOLDS.keys())
    if args.metric:
        bad = [m for m in args.metric if m not in _valid_metrics]
        if bad:
            ap.error(f"invalid metric(s): {bad}; choose from {_valid_metrics}")

    with open(args.per_ex_json) as f:
        data = json.load(f)

    # data is {quant: [per_ex_dicts]}
    quants = args.quants or list(data.keys())
    per_ex_by_quant = {q: data[q] for q in quants if q in data}

    # Check gen_diags are present
    n_with_diags = sum(
        1 for q in quants for e in per_ex_by_quant.get(q, []) if "gen_diags" in e
    )
    if n_with_diags == 0:
        print("ERROR: No gen_diags found in per_ex JSON.\n"
              "Re-run with --save-per-example; gen_diags are collected automatically "
              "when --save-per-example is set.", file=sys.stderr)
        sys.exit(1)

    n_total = sum(len(per_ex_by_quant.get(q, [])) for q in quants)
    print(f"Loaded {n_total} examples across {len(quants)} quants: {quants}")
    print(f"  Examples with gen_diags: {n_with_diags}/{n_total}")

    if args.summary:
        print_value_summary(per_ex_by_quant, quants)
        return

    metrics = args.metric if args.metric else list(DEFAULT_THRESHOLDS.keys())

    if args.sweep:
        if not args.metric:
            print("--sweep requires --metric", file=sys.stderr)
            sys.exit(1)
        for metric in metrics:
            thresholds, tpr_per_quant, fpr_per_quant = sweep_thresholds(
                per_ex_by_quant, quants, metric)
            print_sweep(thresholds, tpr_per_quant, fpr_per_quant, metric, quants)
        return

    for metric in metrics:
        threshold = args.threshold if (args.threshold is not None and len(metrics) == 1) \
                    else DEFAULT_THRESHOLDS[metric]
        rows = analyze(per_ex_by_quant, quants, metric, threshold)
        print_table(rows, metric, threshold)


if __name__ == "__main__":
    main()
