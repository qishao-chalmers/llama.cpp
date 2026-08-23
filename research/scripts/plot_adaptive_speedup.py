#!/usr/bin/env python3
"""plot_adaptive_speedup.py — Visualise adaptive KV cache speedup landscape.

Three-axis sweep: context length × batch size × acceptance rate.
Uses roofline_layer.py engine for per-layer performance estimates.

Usage:
    python3 research/scripts/plot_adaptive_speedup.py --out research/figures/adaptive_speedup.png
    python3 research/scripts/plot_adaptive_speedup.py --model qwen3-8b --hw a100-80g
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(__file__))
from roofline_layer import (
    HARDWARE_PRESETS, MODEL_PRESETS, QUANT_CONFIGS, ops_for_layer,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _layer_time(stage, n_q, n_ctx, model, hw, kv_quant, batch_size, W=32):
    cp = hw["compute_tflops"] * 1e12
    bw = hw["memory_bw_gbps"] * 1e9
    kw = dict(model=model, compute_eff=0.70, mem_eff=0.85, attn_eff=0.85,
              compute_peak=cp, weight_bw=bw, attn_bw=bw, act_bw=bw,
              flash_attn=True, kv_group_size=64, padding_eff=1.0,
              weight_bits=16, kv_quant=kv_quant, batch_size=batch_size)
    ops = ops_for_layer(stage, n_q, n_ctx, **kw)
    nl = model["n_layers"]
    return sum(op.time_s * nl for op in ops)


def adaptive_speedup(model, hw, ctx, B, accept_rate,
                     draft_q="int3_half_1357_ch", verify_q="int4_ch", W=32):
    """Return speedup of adaptive scheme over plain int4, given acceptance rate."""
    t_int4  = _layer_time("decode", B, ctx, model, hw, verify_q, B)
    t_draft = _layer_time("decode", B, ctx, model, hw, draft_q, B)
    t_ver   = _layer_time("verify", B * W, ctx + W, model, hw, verify_q, B)

    n_draft = int(accept_rate * W)
    n_fall  = W - n_draft

    t_adaptive = t_draft * n_draft + t_int4 * n_fall + t_ver
    t_baseline = t_int4 * W

    return t_baseline / t_adaptive if t_adaptive > 0 else 1.0


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3-8b", choices=list(MODEL_PRESETS))
    p.add_argument("--hw",    default="a100-80g",  choices=list(HARDWARE_PRESETS))
    p.add_argument("--out",   default="research/figures/adaptive_speedup.png")
    args = p.parse_args()

    model = dict(MODEL_PRESETS[args.model])
    hw    = HARDWARE_PRESETS[args.hw]

    ctx_list    = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    batch_list  = [1, 4, 8, 16, 32, 64, 128]
    accept_list = [1.0, 0.7, 0.5, 0.3]

    ctx_arr = np.array(ctx_list)
    B_arr   = np.array(batch_list)

    # ── Figure 1: heatmaps (ctx × batch) for each acceptance rate ────────
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
    fig.suptitle(
        f"Adaptive KV Cache Speedup over int4  —  {args.model} / {args.hw}\n"
        f"Draft: int3_half_1357 (2-bit)  |  Verify: int4  |  W=32",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, acc in zip(axes, accept_list):
        grid = np.zeros((len(batch_list), len(ctx_list)))
        for i, B in enumerate(batch_list):
            for j, ctx in enumerate(ctx_list):
                grid[i, j] = adaptive_speedup(model, hw, ctx, B, acc)

        vmin = min(0.5, grid.min())
        vmax = max(1.5, grid.max())
        norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

        im = ax.imshow(grid, aspect="auto", origin="lower",
                       cmap="RdYlGn", norm=norm)

        ax.set_xticks(range(len(ctx_list)))
        ax.set_xticklabels([f"{c//1024}K" if c >= 1024 else str(c)
                            for c in ctx_list], rotation=45, fontsize=9)
        ax.set_xlabel("Context length", fontsize=11)

        if ax == axes[0]:
            ax.set_yticks(range(len(batch_list)))
            ax.set_yticklabels([str(b) for b in batch_list], fontsize=9)
            ax.set_ylabel("Batch size", fontsize=11)

        fail_pct = int((1 - acc) * 100)
        ax.set_title(f"Accept {int(acc*100)}%  (fail {fail_pct}%)", fontsize=12)

        for i in range(len(batch_list)):
            for j in range(len(ctx_list)):
                val = grid[i, j]
                color = "white" if abs(val - 1.0) > 0.2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Speedup over plain int4  (>1 = wins)", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"  Saved heatmaps → {args.out}")

    # ── Figure 2: line plots — speedup vs context for key batch sizes ────
    out2 = args.out.replace(".png", "_lines.png")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    fig2.suptitle(
        f"Adaptive Speedup vs Context Length  —  {args.model} / {args.hw}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    colors_B = {1: "#888888", 8: "#2196F3", 32: "#FF9800", 128: "#E91E63"}
    key_batches = [1, 8, 32, 128]
    key_accepts = [1.0, 0.7, 0.5]

    for ax, acc in zip(axes2, key_accepts):
        for B in key_batches:
            spds = [adaptive_speedup(model, hw, ctx, B, acc) for ctx in ctx_list]
            ax.semilogx(ctx_arr, spds, "o-", color=colors_B[B], linewidth=2,
                        markersize=5, label=f"B={B}", base=2)

        ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.fill_between(ctx_arr, 0.5, 1.0, alpha=0.05, color="red")
        ax.fill_between(ctx_arr, 1.0, 2.0, alpha=0.05, color="green")

        ax.set_xlabel("Context length", fontsize=11)
        ax.set_title(f"Accept rate = {int(acc*100)}%", fontsize=12)
        ax.set_ylim(0.5, 1.6)
        ax.set_xlim(ctx_arr[0], ctx_arr[-1])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

        xticks = [512, 2048, 8192, 32768, 131072]
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{c//1024}K" if c >= 1024 else str(c) for c in xticks])

    axes2[0].set_ylabel("Speedup over plain int4", fontsize=11)

    plt.tight_layout()
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    print(f"  Saved line plots → {out2}")

    # ── Figure 3: acceptance rate sensitivity at sweet-spot configs ───────
    out3 = args.out.replace(".png", "_sensitivity.png")
    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    fig3.suptitle(
        f"Speedup vs Acceptance Rate at Key Scenarios  —  {args.model} / {args.hw}",
        fontsize=13, fontweight="bold",
    )

    scenarios = [
        (1,   131072, "#888888", "B=1, 128K"),
        (8,   32768,  "#2196F3", "B=8, 32K"),
        (32,  32768,  "#FF9800", "B=32, 32K"),
        (32,  131072, "#E91E63", "B=32, 128K"),
        (128, 32768,  "#9C27B0", "B=128, 32K"),
        (128, 131072, "#4CAF50", "B=128, 128K"),
    ]

    acc_sweep = np.linspace(0.0, 1.0, 51)
    for B, ctx, color, label in scenarios:
        spds = [adaptive_speedup(model, hw, ctx, B, a) for a in acc_sweep]
        ax3.plot(acc_sweep * 100, spds, linewidth=2.2, color=color, label=label)

    ax3.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax3.fill_between(acc_sweep * 100, 0.5, 1.0, alpha=0.05, color="red")
    ax3.fill_between(acc_sweep * 100, 1.0, 2.0, alpha=0.05, color="green")

    ax3.set_xlabel("Acceptance rate (%)", fontsize=12)
    ax3.set_ylabel("Speedup over plain int4", fontsize=12)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0.5, 1.6)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, loc="upper left")

    plt.tight_layout()
    fig3.savefig(out3, dpi=180, bbox_inches="tight")
    print(f"  Saved sensitivity → {out3}")


if __name__ == "__main__":
    main()
