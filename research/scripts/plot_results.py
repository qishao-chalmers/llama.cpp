#!/usr/bin/env python3
"""
plot_results.py — Visualize KV cache quantization perplexity results.

Usage:
    python3 plot_results.py results.json --out figure.png
    python3 plot_results.py --inline       # use hardcoded preliminary data
"""

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data ──────────────────────────────────────────────────────────────────────

# Preliminary results: 100-token chunk, Strategy B, Qwen3-8B-Q8_0
PRELIMINARY = {
    "fp16":     {"ppl": 11.2435, "n_tokens": 99},
    "bf16":     {"ppl": 11.1751, "n_tokens": 99},
    "fp8_e5m2": {"ppl": 11.0068, "n_tokens": 99},
    "fp8_e4m3": {"ppl": 11.5279, "n_tokens": 99},
    "int8":     {"ppl": 11.3053, "n_tokens": 99},
    "int8_ch":  {"ppl": None,    "n_tokens": 0},   # placeholder
    "int4":     {"ppl": 78.8554, "n_tokens": 99},
    "int4_ch":  {"ppl": None,    "n_tokens": 0},   # placeholder
    "nf4":      {"ppl": 24.9863, "n_tokens": 99},
    "int2":     {"ppl": 3808.6,  "n_tokens": 99},
}

# ── Plotting ──────────────────────────────────────────────────────────────────

# Order and display names
FORMATS = [
    ("fp16",     "FP16\n(baseline)"),
    ("bf16",     "BF16"),
    ("fp8_e5m2", "FP8\nE5M2"),
    ("fp8_e4m3", "FP8\nE4M3"),
    ("int8",     "INT8\nper-tensor"),
    ("int8_ch",  "INT8\nper-chan"),
    ("int4",     "INT4\nper-tensor"),
    ("int4_ch",  "INT4\nper-chan"),
    ("nf4",      "NF4"),
    ("int2",     "INT2"),
]

# Color palette by precision tier
COLORS = {
    "fp16":     "#2196F3",   # blue  — baseline
    "bf16":     "#4CAF50",   # green — safe
    "fp8_e5m2": "#4CAF50",
    "fp8_e4m3": "#8BC34A",
    "int8":     "#8BC34A",
    "int8_ch":  "#8BC34A",
    "int4":     "#FF9800",   # orange — degraded
    "int4_ch":  "#FF9800",
    "nf4":      "#FF5722",   # deep orange
    "int2":     "#F44336",   # red — catastrophic
}


def make_figure(data: dict, out_path: str, title_suffix: str = ""):
    baseline_ppl = data.get("fp16", {}).get("ppl")
    if baseline_ppl is None:
        raise ValueError("Need fp16 baseline in data")

    keys      = [k for k, _ in FORMATS]
    labels    = [l for _, l in FORMATS]
    ppls      = [data.get(k, {}).get("ppl") for k in keys]
    available = [p is not None for p in ppls]

    # ── Figure layout: two panels ────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"KV Cache Simulated Quantization — Perplexity Impact\n"
                 f"Qwen3-8B-Q8_0 · WikiText-2 · Strategy B (token-by-token){title_suffix}",
                 fontsize=12, fontweight="bold", y=1.01)

    x = np.arange(len(keys))
    bar_w = 0.65
    colors = [COLORS[k] for k in keys]

    # ── Panel 1: absolute perplexity (log scale, excluding int2) ─────────────
    ppls_panel1 = [p if (p is not None and k != "int2") else None for k, p in zip(keys, ppls)]
    bars1 = []
    for i, (p, avail, c) in enumerate(zip(ppls_panel1, available, colors)):
        if p is not None:
            b = ax1.bar(i, p, width=bar_w, color=c, edgecolor="white", linewidth=0.5, zorder=3)
            bars1.append(b)
        elif not avail:
            ax1.bar(i, 0, width=bar_w, color="#CCCCCC", edgecolor="white", linewidth=0.5, zorder=3)

    ax1.axhline(baseline_ppl, color="#2196F3", linestyle="--", linewidth=1.2, alpha=0.7, label=f"FP16 baseline ({baseline_ppl:.2f})")
    ax1.set_yscale("log")
    ax1.set_ylabel("Perplexity (log scale)", fontsize=11)
    ax1.set_title("Absolute Perplexity\n(INT2 excluded — off chart)", fontsize=10)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_ylim(bottom=8)

    # Add value labels on bars
    for i, p in enumerate(ppls_panel1):
        if p is not None:
            ax1.text(i, p * 1.05, f"{p:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # ── Panel 2: % degradation relative to FP16 (excluding fp16 itself) ──────
    pct = []
    for k, p in zip(keys, ppls):
        if p is None or k == "fp16":
            pct.append(None)
        else:
            pct.append(100 * (p - baseline_ppl) / baseline_ppl)

    # Cap at 200% for display; use broken axis effect via annotation
    cap = 200.0
    pct_capped = [min(v, cap) if v is not None else None for v in pct]
    overflow   = {k: v for k, v, p in zip(keys, pct, pct_capped) if v is not None and v > cap}

    for i, (p, c, k) in enumerate(zip(pct_capped, colors, keys)):
        if p is not None and k != "fp16":
            ax2.bar(i, p, width=bar_w, color=c, edgecolor="white", linewidth=0.5, zorder=3)

    ax2.axhline(0, color="#2196F3", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Perplexity change vs FP16 (%)", fontsize=11)
    ax2.set_title(f"Relative Degradation (capped at {cap:.0f}%)", fontsize=10)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_ylim(-20, cap + 30)

    # Add value labels
    for i, (v_raw, v_cap, k) in enumerate(zip(pct, pct_capped, keys)):
        if v_raw is None or k == "fp16":
            continue
        label = f"+{v_raw:.0f}%" if v_raw <= cap else f"+{v_raw:.0f}%⬆"
        color = "black" if v_raw <= cap else "red"
        y_pos = v_cap + 4
        ax2.text(i, y_pos, label, ha="center", va="bottom", fontsize=7.5,
                 fontweight="bold", color=color)

    # ── Legend patches ────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#4CAF50", label="Safe (< 3% degradation)"),
        mpatches.Patch(color="#8BC34A", label="Acceptable (< 10%)"),
        mpatches.Patch(color="#FF9800", label="Degraded (10–100%)"),
        mpatches.Patch(color="#F44336", label="Catastrophic (> 100%)"),
    ]
    ax2.legend(handles=legend_items, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", help="JSON results file")
    parser.add_argument("--out", default="research/poc/kv_quant_figure.png")
    parser.add_argument("--inline", action="store_true", help="Use built-in preliminary data")
    args = parser.parse_args()

    if args.inline or args.results is None:
        data = PRELIMINARY
        suffix = " [preliminary — 100 tokens]"
    else:
        with open(args.results) as f:
            data = json.load(f)
        n = next(iter(data.values())).get("n_tokens", "?")
        suffix = f" [{n} tokens]"

    make_figure(data, args.out, suffix)

    # Print table
    baseline = data.get("fp16", {}).get("ppl")
    print(f"\n{'Format':<12} {'PPL':>8}  {'vs FP16':>10}")
    print("-" * 34)
    for k, label in FORMATS:
        p = data.get(k, {}).get("ppl")
        if p is None:
            print(f"  {k:<10} {'N/A':>8}")
        else:
            pct = 100 * (p - baseline) / baseline if baseline else 0
            sign = "+" if pct >= 0 else ""
            print(f"  {k:<10} {p:>8.2f}   {sign}{pct:>+6.1f}%")


if __name__ == "__main__":
    main()
