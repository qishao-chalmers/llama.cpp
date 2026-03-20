#!/usr/bin/env python3
"""plot_entropy.py — Plot per-token entropy (H), log-prob, and self-surprisal vs position.

Reads the JSON produced by run_sweep.py --save-diags FILE.

Usage:
    python3 plot_entropy.py diags.json \
        --quants fp16 int3_ch int2_ch \
        --metric H \
        --smooth 50 \
        --chunk 0 \
        --out entropy_plot.png

Metrics:
    H         : output entropy in nats (high = confused)
    lp        : log-prob of the correct token (low = wrong)
    p_max     : probability of the top-1 prediction
    self_surp : log-prob of prev step's top-1 under current distribution
    kl        : KL divergence vs fp16 (requires fp16 in data)
    ppl_curve : cumulative PPL up to each position (from lp)

Options:
    --chunk N   : which chunk index to plot (default: 0; use 'all' to average)
    --smooth N  : rolling-window half-width for smoothing (default: 0 = no smoothing)
    --annotate  : mark positions where int3_ch (or first non-fp16 quant) diverges most
    --diff      : plot quant - fp16 instead of raw values (useful for H and lp)
"""

import argparse
import json
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── helpers ──────────────────────────────────────────────────────────────────

def smooth(arr, w):
    """Rolling mean with half-width w (box filter, edge-clipped)."""
    if w <= 0:
        return arr
    k = 2 * w + 1
    padded = np.pad(arr, (w, w), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def kl_vs_fp16(lp_quant, lp_fp16):
    """Approximate token-level KL = lp_fp16 - lp_quant (nats, ≥ 0 on average)."""
    return np.array(lp_fp16) - np.array(lp_quant)


def ppl_curve(lps):
    """Cumulative PPL at each position."""
    lps = np.array(lps)
    cum = np.cumsum(-lps)
    ns  = np.arange(1, len(lps) + 1)
    return np.exp(cum / ns)


def get_metric(cdata, quant, metric, fp16_lps=None):
    """Extract the requested metric array for a given quant from one chunk's data."""
    diags = cdata["diags"].get(quant, {})
    lps   = cdata["log_probs"].get(quant, [])

    if metric == "H":
        return np.array(diags.get("H", []), dtype=float)
    elif metric == "lp":
        return np.array(lps, dtype=float)
    elif metric == "p_max":
        return np.array(diags.get("p_max", []), dtype=float)
    elif metric == "self_surp":
        return np.array(diags.get("self_surp", []), dtype=float)
    elif metric == "kl":
        if fp16_lps is None:
            raise ValueError("kl metric requires fp16 in --quants")
        q = np.array(lps, dtype=float)
        f = np.array(fp16_lps, dtype=float)
        n = min(len(q), len(f))
        return kl_vs_fp16(q[:n], f[:n])
    elif metric == "ppl_curve":
        return ppl_curve(np.array(lps, dtype=float))
    else:
        raise ValueError(f"Unknown metric: {metric}")


METRIC_LABELS = {
    "H":         "Entropy H (nats)",
    "lp":        "Log-prob of correct token",
    "p_max":     "Top-1 probability",
    "self_surp": "Self-surprisal (nats)",
    "kl":        "KL divergence vs fp16 (nats)",
    "ppl_curve": "Cumulative PPL",
}

COLORS = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f"]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("diags_json", help="JSON file from run_sweep.py --save-diags")
    ap.add_argument("--quants",   nargs="*", default=None,
                    help="Quant names to plot (default: all in file)")
    ap.add_argument("--metric",   default="H",
                    choices=["H", "lp", "p_max", "self_surp", "kl", "ppl_curve"],
                    help="Which metric to plot (default: H)")
    ap.add_argument("--metrics",  nargs="*", default=None,
                    help="Plot multiple metrics in separate subplots (overrides --metric)")
    ap.add_argument("--smooth",   type=int, default=50,
                    help="Rolling-average half-width in tokens (default 50; 0 = raw)")
    ap.add_argument("--chunk",    default="0",
                    help="Chunk index to plot (int), or 'all' to average across chunks")
    ap.add_argument("--diff",     action="store_true",
                    help="Plot (quant - fp16) instead of raw values")
    ap.add_argument("--annotate", action="store_true",
                    help="Mark the top-10 highest-divergence positions on the plot")
    ap.add_argument("--out",      default="entropy_plot.png",
                    help="Output image file (default: entropy_plot.png)")
    ap.add_argument("--title",    default=None,
                    help="Plot title override")
    args = ap.parse_args()

    with open(args.diags_json) as f:
        data = json.load(f)

    if not data:
        print("No data in diags file.", file=sys.stderr)
        sys.exit(1)

    # Quant names: use file order if not specified
    sample_cdata = next(iter(data.values()))
    all_quants = list(sample_cdata["diags"].keys())
    quants = args.quants if args.quants else all_quants
    # Ensure fp16 is present if needed for kl
    fp16_available = "fp16" in all_quants

    metrics = args.metrics if args.metrics else [args.metric]

    # ── aggregate data ───────────────────────────────────────────────────────
    # For each quant and metric: list of per-position arrays (one per chunk)
    # Then average across chunks (truncate to min length)

    def collect(quant, metric):
        chunks_data = []
        for chunk_key in sorted(data.keys(), key=int):
            if args.chunk != "all" and chunk_key != str(args.chunk):
                continue
            cdata = data[chunk_key]
            fp16_lps = cdata["log_probs"].get("fp16") if fp16_available else None
            try:
                arr = get_metric(cdata, quant, metric, fp16_lps)
            except ValueError as e:
                print(f"Warning: {e}", file=sys.stderr)
                return None
            if len(arr) == 0:
                continue
            chunks_data.append(arr)

        if not chunks_data:
            return None

        if len(chunks_data) == 1:
            arr = chunks_data[0]
        else:
            n = min(len(a) for a in chunks_data)
            arr = np.mean([a[:n] for a in chunks_data], axis=0)

        return arr

    # ── plot ─────────────────────────────────────────────────────────────────
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics), squeeze=False)

    for mi, metric in enumerate(metrics):
        ax = axes[mi][0]
        fp16_arr = collect("fp16", metric) if fp16_available else None

        for qi, quant in enumerate(quants):
            arr = collect(quant, metric)
            if arr is None or len(arr) == 0:
                print(f"Warning: no data for quant={quant} metric={metric}", file=sys.stderr)
                continue

            if args.diff and fp16_arr is not None and quant != "fp16":
                n = min(len(arr), len(fp16_arr))
                arr = arr[:n] - fp16_arr[:n]

            xs = np.arange(len(arr))
            color = COLORS[qi % len(COLORS)]

            # Raw (thin, transparent)
            if args.smooth > 0:
                ax.plot(xs, arr, color=color, alpha=0.15, linewidth=0.6)
            ax.plot(xs, smooth(arr, args.smooth), color=color,
                    linewidth=1.6, label=quant)

            # Annotate top divergence positions (for first non-fp16 quant)
            if args.annotate and quant != "fp16" and fp16_arr is not None and qi == (1 if fp16_available else 0):
                n = min(len(arr), len(fp16_arr))
                diff = np.abs(arr[:n] - fp16_arr[:n])
                top10 = np.argsort(diff)[-10:]
                for pos in top10:
                    ax.axvline(pos, color="grey", alpha=0.3, linewidth=0.8)

        ylabel = METRIC_LABELS.get(metric, metric)
        if args.diff and fp16_available:
            ylabel = f"Δ {ylabel} (quant − fp16)"
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Decode step (token position)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        if metric in ("H", "lp", "self_surp", "kl") and not args.diff:
            # Flip lp so "worse" is visually up
            pass  # keep natural orientation; user can interpret

    title = args.title or f"{args.diags_json}  |  chunk={args.chunk}  smooth={args.smooth}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
