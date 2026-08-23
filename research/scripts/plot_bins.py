#!/usr/bin/env python3
"""plot_bins.py — Visualize KV cache quantization bin usage distributions.

Reads the JSON produced by run_sweep.py --save-bins and draws a bar chart
per quant type showing what fraction of values land in each bin.

A flat distribution (all bins equal) means the quantization range is well-used.
A spiky distribution (one or two bins dominate) means most values cluster there
and the other bins are wasted — effective precision is lower than nominal.

Usage:
    python3 research/scripts/plot_bins.py research/results/bins.json
    python3 research/scripts/plot_bins.py research/results/bins.json \\
        --quants int2_ch int3_ch int4_ch --out bins.png
    python3 research/scripts/plot_bins.py research/results/bins.json --separate
"""

import argparse, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _load(path):
    with open(path) as f:
        return json.load(f)


def _plot_all(data, quant_specs, out_path, separate=False):
    """Main plotting function.

    data: dict {quant_spec: {inner_name: tracker_dict}}
    Each tracker_dict has keys: name, asym, n_bins, labels, counts.
    """
    # Collect (quant_spec, inner_name, tracker) triples to plot
    entries = []
    for spec in quant_specs:
        if spec not in data:
            print(f"WARNING: '{spec}' not found in bins file, skipping", file=sys.stderr)
            continue
        for inner_name, td in data[spec].items():
            if td["n_bins"] == 0:
                continue  # float quant — no bins
            entries.append((spec, inner_name, td))

    if not entries:
        print("No integer quant entries found to plot.", file=sys.stderr)
        return

    n = len(entries)
    if separate:
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
    else:
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.5 * nrows),
                             squeeze=False)
    fig.suptitle("KV Cache Quantization — Bin Usage Distribution", fontsize=13, y=1.01)

    for idx, (spec, inner_name, td) in enumerate(entries):
        ax = axes[idx // ncols][idx % ncols]
        counts  = np.array(td["counts"], dtype=np.float64)
        labels  = td["labels"]
        n_bins  = td["n_bins"]
        total   = counts.sum()
        fracs   = counts / total if total > 0 else np.zeros(n_bins)

        x = np.arange(n_bins)
        bars = ax.bar(x, fracs, color="steelblue", edgecolor="white", linewidth=0.5)

        # Color the most-used bin differently
        peak = int(fracs.argmax())
        bars[peak].set_color("tomato")

        # Uniform reference line
        ax.axhline(1.0 / n_bins, color="gray", linestyle="--", linewidth=0.8,
                   label=f"uniform (1/{n_bins})")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
        ax.set_ylabel("Fraction of values")
        ax.set_xlabel("Bin")

        title = spec if inner_name == spec else f"{spec}\n({inner_name})"
        if td["asym"]:
            title += " [asym]"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)

        # Annotate entropy: how uniform is the distribution?
        # H_max = log2(n_bins), H_actual = -sum(p*log2(p))
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.sum(fracs * np.log2(np.where(fracs > 0, fracs, 1)))
        h_max = np.log2(n_bins)
        utilization = h / h_max if h_max > 0 else 0
        ax.text(0.98, 0.97, f"utilization={utilization:.1%}\nH={h:.2f}/{h_max:.2f} bits",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Hide unused subplots
    for idx in range(len(entries), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


def _print_table(data, quant_specs):
    """Print a text summary table of bin utilization."""
    print(f"\n{'Quant':<25}  {'n_bins':>6}  {'utilization':>11}  {'peak_bin':>10}  {'peak_frac':>10}  {'total_hits':>12}")
    print("-" * 80)
    for spec in quant_specs:
        if spec not in data:
            continue
        for inner_name, td in data[spec].items():
            if td["n_bins"] == 0:
                continue
            counts  = np.array(td["counts"], dtype=np.float64)
            n_bins  = td["n_bins"]
            labels  = td["labels"]
            total   = counts.sum()
            fracs   = counts / total if total > 0 else np.zeros(n_bins)
            with np.errstate(divide="ignore", invalid="ignore"):
                h     = -np.sum(fracs * np.log2(np.where(fracs > 0, fracs, 1)))
            h_max     = np.log2(n_bins)
            util      = h / h_max if h_max > 0 else 0
            peak      = int(fracs.argmax())
            row_label = spec if inner_name == spec else f"{spec} ({inner_name})"
            if td["asym"]:
                row_label += " [asym]"
            print(f"{row_label:<25}  {n_bins:>6}  {util:>10.1%}  {labels[peak]:>10}  "
                  f"{fracs[peak]:>9.1%}  {int(total):>12,}")


def main():
    ap = argparse.ArgumentParser(description="Plot KV quantization bin usage from --save-bins JSON")
    ap.add_argument("bins_file", help="JSON file produced by run_sweep.py --save-bins")
    ap.add_argument("--quants",   nargs="*", default=None,
                    help="Which quant specs to plot (default: all)")
    ap.add_argument("--out",      default=None, help="Output image file (default: show)")
    ap.add_argument("--separate", action="store_true",
                    help="One figure per quant spec instead of grid")
    ap.add_argument("--no-plot",  action="store_true",
                    help="Print text table only, skip matplotlib")
    args = ap.parse_args()

    data = _load(args.bins_file)
    quant_specs = args.quants if args.quants else list(data.keys())

    _print_table(data, quant_specs)

    if not args.no_plot:
        _plot_all(data, quant_specs, args.out, separate=args.separate)


if __name__ == "__main__":
    main()
