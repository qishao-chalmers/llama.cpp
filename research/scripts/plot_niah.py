#!/usr/bin/env python3
"""plot_niah.py — Plot needle-in-a-haystack accuracy vs needle position per quant.

Reads the JSON from run_sweep.py --save-per-example and the original NIAH JSONL
(for needle_pos metadata), then plots accuracy vs relative position for each quant.

Usage:
    python3 plot_niah.py research/results/niah_per_ex.json \
        --jsonl research/data/niah_4096.jsonl \
        --quants fp16 int8_ch int4_ch int3_ch int2_ch \
        --out niah_plot.png

The plot shows:
  - X axis: needle position (0 = beginning, 1 = end, 0.5 = middle)
  - Y axis: accuracy (fraction correct across replicates at that position)
  - One line per quant; fp16 is the baseline (U-shaped reference curve)
  - Shaded region: ±1 SE across replicates

If --jsonl is omitted and labels contain 'pos_X.XX', position is parsed from the label.
"""

import argparse
import json
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def parse_pos_from_label(label: str):
    """Try to parse needle_pos from label like 'niah_pos_0.50_n0'."""
    m = re.search(r"pos[_\-](\d+\.\d+)", label)
    return float(m.group(1)) if m else None


def load_positions(jsonl_path: str) -> dict:
    """Load {label: needle_pos} mapping from NIAH JSONL."""
    pos_map = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = rec.get("dataset", "niah") + "#" + str(rec.get("id", ""))
            # Also try task_id-based label
            if "needle_pos" in rec:
                pos_map[label] = rec["needle_pos"]
                # Also store by task_id if present
                if "task_id" in rec:
                    pos_map[rec["task_id"]] = rec["needle_pos"]
    return pos_map


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("per_example_json", help="JSON from run_sweep.py --save-per-example")
    ap.add_argument("--jsonl", default=None,
                    help="Original NIAH JSONL (for needle_pos metadata). "
                         "If omitted, position is parsed from example labels.")
    ap.add_argument("--quants", nargs="*", default=None,
                    help="Quant names to plot (default: all)")
    ap.add_argument("--out",    default="niah_plot.png")
    ap.add_argument("--title",  default=None)
    ap.add_argument("--show-gen-len", action="store_true",
                    help="Add a second subplot showing mean gen length vs position")
    args = ap.parse_args()

    with open(args.per_example_json) as f:
        data = json.load(f)

    quants = args.quants or list(data.keys())

    # ── Build position map ───────────────────────────────────────────────────
    jsonl_pos_map = load_positions(args.jsonl) if args.jsonl else {}

    def get_pos(ex: dict):
        label = ex["label"]
        # Try JSONL map first
        for key in (label, label.split("/")[-1]):
            if key in jsonl_pos_map:
                return jsonl_pos_map[key]
        # Fall back to parsing label
        p = parse_pos_from_label(label)
        if p is not None:
            return p
        return None

    # ── Collect (position, score, gen_len) per quant ─────────────────────────
    # Structure: {quant: {pos: [scores]}}
    quant_data = {}
    for quant in quants:
        if quant not in data:
            print(f"Warning: quant '{quant}' not found in per-example JSON", file=sys.stderr)
            continue
        pos_scores = {}
        pos_lens   = {}
        for ex in data[quant]:
            pos = get_pos(ex)
            if pos is None:
                print(f"Warning: could not determine position for label '{ex['label']}'",
                      file=sys.stderr)
                continue
            pos_scores.setdefault(pos, []).append(ex["score"])
            pos_lens.setdefault(pos, []).append(ex.get("gen_len", float("nan")))
        quant_data[quant] = {"scores": pos_scores, "lens": pos_lens}

    if not quant_data:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    # All positions across all quants
    all_positions = sorted({pos for qd in quant_data.values()
                            for pos in qd["scores"].keys()})

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_subplots = 2 if args.show_gen_len else 1
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, 4 * n_subplots), squeeze=False)
    ax_acc = axes[0][0]

    for qi, quant in enumerate(quants):
        if quant not in quant_data:
            continue
        qd = quant_data[quant]
        xs, ys, errs = [], [], []
        for pos in all_positions:
            scores = qd["scores"].get(pos, [])
            if not scores:
                continue
            xs.append(pos)
            ys.append(np.mean(scores))
            errs.append(np.std(scores) / max(len(scores) ** 0.5, 1))

        color = COLORS[qi % len(COLORS)]
        ax_acc.plot(xs, ys, "o-", color=color, label=quant, linewidth=2, markersize=5)
        ax_acc.fill_between(xs,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            alpha=0.15, color=color)

    ax_acc.axvline(0.5, color="grey", linestyle="--", alpha=0.5, linewidth=1,
                   label="middle (0.5)")
    ax_acc.set_xlabel("Needle position (0 = beginning, 1 = end)")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_xlim(-0.02, 1.02)
    ax_acc.legend(loc="lower center", fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_xticks(all_positions)
    ax_acc.set_xticklabels([f"{p:.2f}" for p in all_positions], rotation=45, fontsize=8)

    if args.show_gen_len:
        ax_len = axes[1][0]
        for qi, quant in enumerate(quants):
            if quant not in quant_data:
                continue
            qd = quant_data[quant]
            xs, ys = [], []
            for pos in all_positions:
                lens = [l for l in qd["lens"].get(pos, []) if not np.isnan(l)]
                if not lens:
                    continue
                xs.append(pos)
                ys.append(np.mean(lens))
            color = COLORS[qi % len(COLORS)]
            ax_len.plot(xs, ys, "o-", color=color, label=quant, linewidth=2, markersize=5)
        ax_len.axvline(0.5, color="grey", linestyle="--", alpha=0.5, linewidth=1)
        ax_len.set_xlabel("Needle position")
        ax_len.set_ylabel("Mean generated tokens")
        ax_len.legend(loc="upper center", fontsize=9)
        ax_len.grid(True, alpha=0.3)
        ax_len.set_xticks(all_positions)
        ax_len.set_xticklabels([f"{p:.2f}" for p in all_positions], rotation=45, fontsize=8)

    title = args.title or f"Needle-in-a-Haystack: Accuracy vs Position  [{args.per_example_json}]"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.out}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'pos':>6}", end="")
    for quant in quants:
        if quant in quant_data:
            print(f"  {quant:>12}", end="")
    print()
    for pos in all_positions:
        print(f"{pos:>6.2f}", end="")
        for quant in quants:
            if quant not in quant_data:
                continue
            scores = quant_data[quant]["scores"].get(pos, [])
            if scores:
                print(f"  {np.mean(scores):>12.3f}", end="")
            else:
                print(f"  {'—':>12}", end="")
        print()


if __name__ == "__main__":
    main()
