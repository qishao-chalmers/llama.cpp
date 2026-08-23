#!/usr/bin/env python3
"""analyze_quant_results.py — Print detailed KV-quant benchmark results across models.

Usage:
    python3 research/scripts/analyze_quant_results.py
    python3 research/scripts/analyze_quant_results.py --results-dir research/results/server_results
    python3 research/scripts/analyze_quant_results.py --models qwen3-8b llama4-scout-17b
    python3 research/scripts/analyze_quant_results.py --benchmarks wikitext2 gsm8k
    python3 research/scripts/analyze_quant_results.py --no-delta   # hide Δ vs fp16
"""

import argparse
import json
import os
import sys

# ── Config ─────────────────────────────────────────────────────────────────────

QUANTS = [
    "fp16",
    "int8_ch",
    "int4_ch",
    "int3_ch",
    "int3_half_1357_ch",
    "int3_half_1357_ch:int3_half_1357_tok",
    "int2_ch",
]
QLABELS = {
    "fp16":                                  "fp16",
    "int8_ch":                               "int8_ch",
    "int4_ch":                               "int4_ch",
    "int3_ch":                               "int3_ch",
    "int3_half_1357_ch":                     "i3h_ch",
    "int3_half_1357_ch:int3_half_1357_tok":  "i3h_ch:tok",
    "int2_ch":                               "int2_ch",
}

ALL_MODELS = ["qwen3-8b", "qwen3-14b", "llama4-scout-17b", "gpt-oss-20b"]
MLABELS = {
    "qwen3-8b":          "Qwen3-8B",
    "qwen3-14b":         "Qwen3-14B",
    "llama4-scout-17b":  "Llama4-Scout",
    "gpt-oss-20b":       "GPT-OSS-20B",
}

ALL_BENCHMARKS = ["wikitext2", "gsm8k", "longbench", "niah_4k", "humaneval", "aime"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load(base, model, task):
    """Load a result JSON, trying common path variants."""
    candidates = [
        os.path.join(base, model, f"{task}.json"),
        os.path.join(base, model, "niah", f"{task}.json"),  # qwen3-8b niah sub-dir
        os.path.join(base, f"{task}.json"),                 # legacy flat layout
    ]
    for p in candidates:
        if os.path.exists(p):
            return json.load(open(p))
    return {}


def get(d, quant, key):
    return (d.get(quant) or {}).get(key)


def fv(v, decimals=3):
    """Format a value, or return '—' if None."""
    if v is None:
        return "—"
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(v)


def fd(v, ref, decimals=3):
    """Format value with delta vs ref in parentheses."""
    if v is None:
        return "—"
    s = fv(v, decimals)
    if ref is not None and v != ref:
        d = v - ref
        sign = "+" if d >= 0 else ""
        ds = f"{{:.{decimals}f}}".format(d)
        s += f"({sign}{ds})"
    return s


def header(title):
    print()
    print("=" * 110)
    print(f"  {title}")
    print("=" * 110)


def col_header(models):
    labels = [f"{MLABELS.get(m, m):>22}" for m in models]
    print(f"  {'quant':<14}" + "".join(labels))
    print("  " + "-" * (14 + 22 * len(models)))


def row(qlabel, cells):
    print(f"  {qlabel:<14}" + "".join(f"{c:>22}" for c in cells))


# ── Benchmark printers ─────────────────────────────────────────────────────────

def print_wikitext2(base, models, show_delta):
    header("WikiText2 PPL  (lower = better)")
    for window in [512, 1024, 2048]:
        key = f"ppl_{window}"
        print(f"\n  [PPL @ {window} tokens]")
        col_header(models)
        data = {m: load(base, m, "wikitext2") for m in models}
        refs = {m: get(data[m], "fp16", key) for m in models}
        for q in QUANTS:
            cells = []
            for m in models:
                v = get(data[m], q, key)
                if not show_delta or q == "fp16":
                    cells.append(fv(v, 2))
                else:
                    cells.append(fd(v, refs[m], 2))
            row(QLABELS[q], cells)

    # KL divergence (mean over all positions)
    print(f"\n  [Mean KL divergence vs fp16]")
    col_header(models)
    data = {m: load(base, m, "wikitext2") for m in models}
    for q in QUANTS:
        cells = []
        for m in models:
            v = get(data[m], q, "mean_kl")
            cells.append(fv(v, 4) if v is not None else "—")
        row(QLABELS[q], cells)


def print_accuracy_bench(base, models, show_delta, task, display_name, metric_key):
    header(f"{display_name}  (higher = better)")
    data = {m: load(base, m, task) for m in models}
    refs = {m: get(data[m], "fp16", metric_key) for m in models}

    # Main metric
    col_header(models)
    for q in QUANTS:
        cells = []
        for m in models:
            v = get(data[m], q, metric_key)
            if not show_delta or q == "fp16":
                cells.append(fv(v, 3))
            else:
                cells.append(fd(v, refs[m], 3))
        row(QLABELS[q], cells)

    # Detail: n_correct / n_total, n_truncated, mean_gen_tokens
    print(f"\n  [n_correct / n_total  |  n_truncated  |  mean_gen_tokens]")
    col_header(models)
    for q in QUANTS:
        cells = []
        for m in models:
            d = (data[m].get(q) or {})
            nc = d.get("n_correct")
            nt = d.get("n_total")
            ntr = d.get("n_truncated")
            mgt = d.get("mean_gen_tokens")
            if nc is None and nt is None:
                cells.append("—")
            else:
                nc_s  = str(nc)  if nc  is not None else "?"
                nt_s  = str(nt)  if nt  is not None else "?"
                ntr_s = str(ntr) if ntr is not None else "?"
                mgt_s = f"{mgt:.0f}" if mgt is not None else "?"
                cells.append(f"{nc_s}/{nt_s} tr={ntr_s} g={mgt_s}")
        row(QLABELS[q], cells)


def print_humaneval(base, models, show_delta):
    header("HumanEval pass@1  (higher = better)")
    data = {m: load(base, m, "humaneval_pass1") for m in models}
    refs = {m: (data[m].get("fp16") or {}).get("pass_at_1") for m in models}

    col_header(models)
    for q in QUANTS:
        cells = []
        for m in models:
            v = (data[m].get(q) or {}).get("pass_at_1")
            if not show_delta or q == "fp16":
                cells.append(fv(v, 3))
            else:
                cells.append(fd(v, refs[m], 3))
        row(QLABELS[q], cells)

    print(f"\n  [n_pass / n_total]")
    col_header(models)
    for q in QUANTS:
        cells = []
        for m in models:
            d = data[m].get(q) or {}
            np_ = d.get("n_pass")
            nt  = d.get("n_total")
            if np_ is None:
                cells.append("—")
            else:
                cells.append(f"{np_}/{nt}")
        row(QLABELS[q], cells)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="research/results/server_results",
                   help="Root directory containing per-model result sub-folders")
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS, metavar="MODEL",
                   help="Models to include (default: all)")
    p.add_argument("--benchmarks", nargs="+", default=ALL_BENCHMARKS,
                   choices=ALL_BENCHMARKS, metavar="BENCH",
                   help="Benchmarks to print (default: all)")
    p.add_argument("--no-delta", action="store_true",
                   help="Show raw values only, no Δ vs fp16")
    args = p.parse_args()

    base       = args.results_dir
    models     = args.models
    benchmarks = set(args.benchmarks)
    show_delta = not args.no_delta

    print(f"Results dir : {base}")
    print(f"Models      : {', '.join(MLABELS.get(m, m) for m in models)}")
    print(f"Benchmarks  : {', '.join(benchmarks)}")

    if "wikitext2" in benchmarks:
        print_wikitext2(base, models, show_delta)

    if "gsm8k" in benchmarks:
        print_accuracy_bench(base, models, show_delta,
                             "gsm8k", "GSM8K accuracy", "accuracy")

    if "longbench" in benchmarks:
        print_accuracy_bench(base, models, show_delta,
                             "longbench_qasper", "LongBench Qasper F1", "f1")

    if "niah_4k" in benchmarks:
        print_accuracy_bench(base, models, show_delta,
                             "niah_4k", "NIAH-4k accuracy", "accuracy")

    if "humaneval" in benchmarks:
        print_humaneval(base, models, show_delta)

    if "aime" in benchmarks:
        print_accuracy_bench(base, models, show_delta,
                             "aime", "AIME accuracy", "accuracy")

    print()


if __name__ == "__main__":
    main()
