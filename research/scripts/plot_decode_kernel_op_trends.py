#!/usr/bin/env python3
"""Plot normalized detailed kernel-op trends from decode_sweep_Qwen3-8B.txt.

Parses blocks like:
  ### CUDA PERF DECODE row  npp=...  npl=...
  ...
  op                                       avg_ms      pct    count
  ...

Outputs:
1) op vs npl curves, normalized to npl=1 for each context.
2) op vs context curves, normalized to smallest context for each npl.
3) summary text with scaling ratios.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt


ROW_RE = re.compile(r"^### CUDA PERF DECODE row\s+npp=(\d+)\s+npl=(\d+)\s+ntg=(\d+)\s+model=(.+)$")
OP_RE = re.compile(r"^\s*(.+?)\s+([0-9.]+)\s+([0-9.]+)%\s+(\d+)\s*$")


@dataclass(frozen=True)
class OpPoint:
    npp: int
    npl: int
    op: str
    avg_ms: float
    pct: float
    count: int


def parse_decode_kernel_ops(path: str) -> list[OpPoint]:
    pts: list[OpPoint] = []
    cur_npp: int | None = None
    cur_npl: int | None = None
    in_table = False

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m_row = ROW_RE.match(line.strip())
            if m_row:
                cur_npp = int(m_row.group(1))
                cur_npl = int(m_row.group(2))
                in_table = False
                continue

            if cur_npp is None or cur_npl is None:
                continue

            if line.strip().startswith("op") and "avg_ms" in line and "pct" in line:
                in_table = True
                continue

            if not in_table:
                continue

            s = line.strip()
            if not s:
                continue
            if s.startswith("---"):
                continue
            if s.startswith("==="):
                in_table = False
                continue

            m_op = OP_RE.match(line)
            if not m_op:
                continue
            op = m_op.group(1).strip()
            avg_ms = float(m_op.group(2))
            pct = float(m_op.group(3))
            count = int(m_op.group(4))
            pts.append(OpPoint(cur_npp, cur_npl, op, avg_ms, pct, count))
    if not pts:
        raise ValueError(f"no kernel op rows parsed from {path}")
    return pts


def infer_fused_ffn_npl1(
    pts: list[OpPoint],
    ratio_gate: float,
    ratio_up: float,
    ratio_swiglu: float,
) -> list[OpPoint]:
    """For npl=1 rows where FFN is fused into ffn_gate, infer split components.

    Rule:
      - If (npp, npl=1) has `ffn_gate:MUL_MAT` but missing either
        `ffn_up:MUL_MAT` or `ffn_swiglu:GLU`, treat `ffn_gate:MUL_MAT` as fused total.
      - Replace / inject npl=1 values:
          gate   = fused * rg / (rg + ru + rs)
          up     = fused * ru / (rg + ru + rs)
          swiglu = fused * rs / (rg + ru + rs)
    """

    denom = ratio_gate + ratio_up + ratio_swiglu
    if denom <= 0:
        return pts

    grouped: dict[tuple[int, int], dict[str, OpPoint]] = {}
    for p in pts:
        key = (p.npp, p.npl)
        grouped.setdefault(key, {})[canonical_op_name(p.op)] = p

    out = list(pts)
    # Remove old npl=1 gate/up/swiglu entries only when doing fused replacement.
    to_drop: set[tuple[int, int, str]] = set()
    to_add: list[OpPoint] = []

    for (npp, npl), mp in grouped.items():
        if npl != 1:
            continue
        gate = mp.get("ffn_gate:MUL_MAT")
        up = mp.get("ffn_up:MUL_MAT")
        sw = mp.get("ffn_swiglu:GLU")
        if gate is None:
            continue
        if up is not None and sw is not None:
            continue

        fused = gate.avg_ms
        g_ms = fused * ratio_gate / denom
        u_ms = fused * ratio_up / denom
        s_ms = fused * ratio_swiglu / denom

        # Mark existing entries for replacement.
        for name in ("ffn_gate:MUL_MAT", "ffn_up:MUL_MAT", "ffn_swiglu:GLU"):
            if name in mp:
                to_drop.add((npp, npl, name))

        # Keep pct as NaN-like sentinel (-1), since inferred split is avg_ms-only.
        to_add.extend(
            [
                OpPoint(npp=npp, npl=1, op="ffn_gate:MUL_MAT", avg_ms=g_ms, pct=-1.0, count=gate.count),
                OpPoint(npp=npp, npl=1, op="ffn_up:MUL_MAT", avg_ms=u_ms, pct=-1.0, count=gate.count),
                OpPoint(npp=npp, npl=1, op="ffn_swiglu:GLU", avg_ms=s_ms, pct=-1.0, count=gate.count),
            ]
        )

    if not to_drop and not to_add:
        return out

    out2: list[OpPoint] = []
    for p in out:
        key = (p.npp, p.npl, canonical_op_name(p.op))
        if key in to_drop:
            continue
        out2.append(p)
    out2.extend(to_add)
    return out2


def canonical_op_name(op: str) -> str:
    # Keep categories stable while preserving useful detail.
    return op


def display_op_name(op: str) -> str:
    """Short display label for subplot titles."""
    aliases = {
        "__fattn__:FLASH_ATTN_EXT": "FLASH_ATTN_EXT",
        "norm:RMS_NORM": "RMS_NORM",
        "Qcur:MUL_MAT": "Qcur_MUL_MAT",
        "Kcur:MUL_MAT": "Kcur_MUL_MAT",
        "Vcur:MUL_MAT": "Vcur_MUL_MAT",
        "Qcur:ROPE": "Qcur_ROPE",
        "Kcur:ROPE": "Kcur_ROPE",
        "ffn_gate:MUL_MAT": "ffn_gate_MUL_MAT",
        "ffn_up:MUL_MAT": "ffn_up_MUL_MAT",
        "ffn_out:MUL_MAT": "ffn_out_MUL_MAT",
        "ffn_swiglu:GLU": "ffn_swiglu_GLU",
        "node_*:MUL_MAT": "node_MUL_MAT",
    }
    return aliases.get(op, op)


def collect_ops(pts: list[OpPoint], min_presence: int) -> list[str]:
    seen: dict[str, int] = {}
    for p in pts:
        k = canonical_op_name(p.op)
        seen[k] = seen.get(k, 0) + 1
    ops = [k for k, c in seen.items() if c >= min_presence]
    ops.sort()
    return ops


def build_index(pts: list[OpPoint]) -> dict[tuple[str, int, int], OpPoint]:
    out: dict[tuple[str, int, int], OpPoint] = {}
    for p in pts:
        out[(canonical_op_name(p.op), p.npp, p.npl)] = p
    return out


def _coarse_ylim(ax, op: str, yvals: list[float]) -> None:
    if not yvals:
        return
    y_min = min(yvals)
    y_max = max(yvals)
    if y_max <= 0:
        return
    dev = max(abs(y_min - 1.0), abs(y_max - 1.0))
    if dev <= 0.12:
        low, high = 0.8, 1.2
        ax.set_ylim(low, high)
        ax.set_yticks([0.8, 1.0, 1.2])
    elif dev <= 0.25:
        low, high = 0.7, 1.3
        ax.set_ylim(low, high)
        ax.set_yticks([0.7, 0.9, 1.1, 1.3])
    else:
        top = math.ceil(y_max * 2.0) / 2.0
        bot = min(0.0, math.floor(y_min * 2.0) / 2.0)
        if top - bot < 0.5:
            top = bot + 0.5
        ax.set_ylim(bot, top)
        span = top - bot
        if span <= 4:
            step = 0.5
        elif span <= 10:
            step = 1.0
        elif span <= 20:
            step = 2.0
        elif span <= 40:
            step = 5.0
        else:
            step = 10.0
        ticks = []
        t = bot
        while t <= top + 1e-9:
            ticks.append(round(t, 3))
            t += step
        if len(ticks) >= 2:
            ax.set_yticks(ticks)


def _subplot_shape(n: int) -> tuple[int, int]:
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 9:
        return 3, 3
    if n <= 12:
        return 3, 4
    return 4, 4


def plot_npl_norm(
    idx: dict[tuple[str, int, int], OpPoint],
    ops: list[str],
    ctxs: list[int],
    npls: list[int],
    out_png: str,
) -> None:
    nr, nc = _subplot_shape(len(ops))
    fig, axes = plt.subplots(nr, nc, figsize=(4.5 * nc, 3.3 * nr), constrained_layout=True)
    axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for i, op in enumerate(ops):
        ax = axes_flat[i]
        yvals: list[float] = []
        for ctx in ctxs:
            base = idx.get((op, ctx, 1))
            if base is None or base.avg_ms <= 0:
                continue
            xs, ys = [], []
            for b in npls:
                p = idx.get((op, ctx, b))
                if p is None:
                    continue
                xs.append(b)
                y = p.avg_ms / base.avg_ms
                ys.append(y)
                yvals.append(y)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.4, label=f"ctx={ctx}")
        ax.set_title(f"{display_op_name(op)}\n(base npl=1)", fontsize=10)
        ax.set_xlabel("npl")
        ax.set_ylabel("norm")
        ax.set_xscale("log", base=2)
        _coarse_ylim(ax, op, yvals)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, ncols=2)
    for j in range(len(ops), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Detailed Kernel Ops vs Batch (normalized to npl=1)", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ctx_norm(
    idx: dict[tuple[str, int, int], OpPoint],
    ops: list[str],
    ctxs: list[int],
    npls: list[int],
    out_png: str,
) -> None:
    nr, nc = _subplot_shape(len(ops))
    fig, axes = plt.subplots(nr, nc, figsize=(4.5 * nc, 3.3 * nr), constrained_layout=True)
    axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for i, op in enumerate(ops):
        ax = axes_flat[i]
        yvals: list[float] = []
        for b in npls:
            # base context = smallest available for this op+npl
            cands = [c for c in ctxs if (op, c, b) in idx]
            if not cands:
                continue
            c0 = cands[0]
            base = idx[(op, c0, b)]
            if base.avg_ms <= 0:
                continue
            xs, ys = [], []
            for c in cands:
                y = idx[(op, c, b)].avg_ms / base.avg_ms
                xs.append(c)
                ys.append(y)
                yvals.append(y)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.4, label=f"npl={b}")
        ax.set_title(f"{display_op_name(op)}\n(base ctx=min)", fontsize=10)
        ax.set_xlabel("context (npp)")
        ax.set_ylabel("norm")
        ax.set_xscale("log", base=2)
        _coarse_ylim(ax, op, yvals)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, ncols=2)
    for j in range(len(ops), len(axes_flat)):
        axes_flat[j].axis("off")
    fig.suptitle("Detailed Kernel Ops vs Context (normalized to min ctx)", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_summary(
    idx: dict[tuple[str, int, int], OpPoint],
    ops: list[str],
    ctxs: list[int],
    npls: list[int],
    out_txt: str,
) -> None:
    lines: list[str] = []
    lines.append("# Detailed kernel-op trend summary")
    lines.append(f"contexts={ctxs}")
    lines.append(f"npl={npls}")
    lines.append("")
    lines.append("op | mean(ctx_max/ctx_min across npl) | mean(npl_max/npl_min across ctx)")
    lines.append("---|---:|---:")
    for op in ops:
        ctx_ratios = []
        for b in npls:
            cands = [c for c in ctxs if (op, c, b) in idx]
            if len(cands) < 2:
                continue
            r = idx[(op, cands[-1], b)].avg_ms / idx[(op, cands[0], b)].avg_ms
            ctx_ratios.append(r)
        b_ratios = []
        for c in ctxs:
            cands_b = [b for b in npls if (op, c, b) in idx]
            if len(cands_b) < 2:
                continue
            r = idx[(op, c, cands_b[-1])].avg_ms / idx[(op, c, cands_b[0])].avg_ms
            b_ratios.append(r)
        mu_ctx = sum(ctx_ratios) / len(ctx_ratios) if ctx_ratios else float("nan")
        mu_b = sum(b_ratios) / len(b_ratios) if b_ratios else float("nan")
        lines.append(f"{op} | {mu_ctx:.3f} | {mu_b:.3f}")
    lines.append("")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="decode_sweep_Qwen3-8B.txt")
    ap.add_argument("--out-dir", default="", help="default: same dir as input")
    ap.add_argument("--prefix", default="decode_kernel_ops")
    ap.add_argument(
        "--ops",
        default="",
        help="comma list of ops to plot; default auto-select by min presence",
    )
    ap.add_argument(
        "--min-presence",
        type=int,
        default=6,
        help="auto-select ops appearing in at least this many rows",
    )
    ap.add_argument(
        "--infer-fused-ffn-npl1",
        action="store_true",
        help="Infer npl=1 ffn_gate/ffn_up/ffn_swiglu split when fused into ffn_gate.",
    )
    ap.add_argument(
        "--ffn-fused-ratio",
        default="10,10,1",
        help="Split ratio for inferred fused FFN at npl=1: gate,up,swiglu",
    )
    args = ap.parse_args()

    in_path = os.path.abspath(os.path.expanduser(args.input))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir)) if args.out_dir else os.path.dirname(in_path)
    os.makedirs(out_dir, exist_ok=True)

    pts = parse_decode_kernel_ops(in_path)
    inferred_note = ""
    if args.infer_fused_ffn_npl1:
        parts = [x.strip() for x in args.ffn_fused_ratio.split(",")]
        if len(parts) != 3:
            raise ValueError("--ffn-fused-ratio must have 3 comma-separated numbers")
        rg, ru, rs = float(parts[0]), float(parts[1]), float(parts[2])
        pts = infer_fused_ffn_npl1(pts, rg, ru, rs)
        inferred_note = f"inferred_fused_ffn_npl1 ratio={rg}:{ru}:{rs}"
    idx = build_index(pts)
    ctxs = sorted({p.npp for p in pts})
    npls = sorted({p.npl for p in pts})

    if args.ops.strip():
        ops = [x.strip() for x in args.ops.split(",") if x.strip()]
    else:
        ops = collect_ops(pts, min_presence=int(args.min_presence))
    if not ops:
        raise ValueError("no ops selected")

    png_npl = os.path.join(out_dir, f"{args.prefix}_npl_norm.png")
    png_ctx = os.path.join(out_dir, f"{args.prefix}_ctx_norm.png")
    txt = os.path.join(out_dir, f"{args.prefix}_summary.txt")

    plot_npl_norm(idx, ops, ctxs, npls, png_npl)
    plot_ctx_norm(idx, ops, ctxs, npls, png_ctx)
    write_summary(idx, ops, ctxs, npls, txt)

    if inferred_note:
        with open(txt, "a", encoding="utf-8") as f:
            f.write(f"\n# NOTE: {inferred_note}\n")

    print(png_npl)
    print(png_ctx)
    print(txt)


if __name__ == "__main__":
    main()
