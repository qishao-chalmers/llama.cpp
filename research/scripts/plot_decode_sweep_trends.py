#!/usr/bin/env python3
"""Plot normalized op trends from decode_sweep.txt.

Reads the "DECODE DURATION (ms)" table and creates:
1) op vs npl curves, normalized to npl=1 for each context length.
2) op vs context curves, normalized to smallest context for each npl.
3) a text summary of scaling ratios.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt


OPS = ["QKV+O", "RoPE", "Attn", "FFN", "Norm", "Other"]


@dataclass(frozen=True)
class Row:
    npp: int
    npl: int
    vals: dict[str, float]


def parse_duration_rows(path: str) -> list[Row]:
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    in_duration = False
    rows: list[Row] = []
    for line in lines:
        s = line.strip()
        if "DECODE DURATION (ms)" in s:
            in_duration = True
            continue
        if not in_duration:
            continue
        if not s or s.startswith("=") or s.startswith("-") or s.startswith("npp"):
            continue
        if "DECODE PERCENTAGE" in s:
            break

        parts = s.split()
        if len(parts) != 10:
            continue
        try:
            npp = int(parts[0])
            npl = int(parts[1])
            vals = {
                "QKV+O": float(parts[2]),
                "RoPE": float(parts[3]),
                "Attn": float(parts[4]),
                "FFN": float(parts[5]),
                "Norm": float(parts[6]),
                "Other": float(parts[7]),
                "Wall": float(parts[8]),
                "tok/s": float(parts[9]),
            }
            rows.append(Row(npp=npp, npl=npl, vals=vals))
        except ValueError:
            continue
    if not rows:
        # Fallback: file may be a per-op CUDA PERF listing (decode_sweep_ops_*.txt).
        rows = parse_perf_decode_rows(path)
    if not rows:
        raise ValueError(f"no duration rows parsed from {path}")
    return rows


def parse_perf_decode_rows(path: str) -> list[Row]:
    """Parse '### CUDA PERF DECODE row ...' blocks and aggregate into buckets.

    Expected format:
      ### CUDA PERF DECODE row  npp=...  npl=... ...
        kernel=.. ms  wall=.. ms ...
        op  avg_ms  pct  count
        ...
    """

    import re

    row_re = re.compile(r"^### CUDA PERF DECODE row\s+npp=(\d+)\s+npl=(\d+)\b")
    kw_re = re.compile(r"^\s*kernel=([0-9.]+)\s*ms\s+wall=([0-9.]+)\s*ms\b")
    op_re = re.compile(r"^\s*(.+?)\s+([0-9.]+)\s+([0-9.]+)%\s+(\d+)\s*$")

    def bucket(op_name: str) -> str:
        s = op_name.strip()
        if s.startswith("norm:"):
            return "Norm"
        if s.endswith(":ROPE") or ":ROPE" in s:
            return "RoPE"
        if "FLASH_ATTN" in s or "__fattn__" in s:
            return "Attn"
        if s.startswith("ffn_"):
            return "FFN"
        if "MUL_MAT" in s:
            # Q/K/V projections, output projection, misc matmuls
            return "QKV+O"
        return "Other"

    rows: list[Row] = []
    cur_npp: int | None = None
    cur_npl: int | None = None
    cur_wall: float | None = None
    cur_kernel: float | None = None
    in_table = False
    acc: dict[str, float] = {}

    def flush() -> None:
        nonlocal acc, cur_npp, cur_npl, cur_wall, cur_kernel
        if cur_npp is None or cur_npl is None:
            return
        out = {k: 0.0 for k in OPS}
        for k, v in acc.items():
            if k in out:
                out[k] += float(v)
        wall = float(cur_wall) if cur_wall is not None else float(sum(out.values()))
        if wall <= 0:
            return
        out["Wall"] = wall
        out["tok/s"] = 1000.0 * float(cur_npl) / wall
        rows.append(Row(npp=int(cur_npp), npl=int(cur_npl), vals=out))
        acc = {}
        cur_wall = None
        cur_kernel = None

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = row_re.match(line.strip())
            if m:
                flush()
                cur_npp = int(m.group(1))
                cur_npl = int(m.group(2))
                in_table = False
                acc = {}
                continue

            if cur_npp is None or cur_npl is None:
                continue

            mk = kw_re.match(line.strip())
            if mk:
                cur_kernel = float(mk.group(1))
                cur_wall = float(mk.group(2))
                continue

            if line.strip().startswith("op") and "avg_ms" in line and "pct" in line:
                in_table = True
                continue
            if not in_table:
                continue

            s = line.strip()
            if not s or s.startswith("---"):
                continue
            mo = op_re.match(line)
            if not mo:
                continue
            op_name = mo.group(1).strip()
            avg_ms = float(mo.group(2))
            b = bucket(op_name)
            acc[b] = acc.get(b, 0.0) + avg_ms

    flush()
    return rows


def _index(rows: list[Row]) -> dict[tuple[int, int], Row]:
    return {(r.npp, r.npl): r for r in rows}


def _apply_coarse_ylim(ax: Any, op: str, yvals: list[float]) -> None:
    if not yvals:
        return
    y_min = min(yvals)
    y_max = max(yvals)
    if y_max <= 0:
        return

    if op == "Attn":
        # Keep attention readable but avoid tiny auto-scale steps.
        top = max(1.0, y_max)
        top = math.ceil(top * 2.0) / 2.0
        ax.set_ylim(0.0, top)
        if top <= 4:
            step = 0.5
        elif top <= 10:
            step = 1.0
        elif top <= 20:
            step = 2.0
        elif top <= 40:
            step = 5.0
        else:
            step = 10.0
        ticks = []
        t = 0.0
        while t <= top + 1e-9:
            ticks.append(round(t, 3))
            t += step
        if len(ticks) >= 2:
            ax.set_yticks(ticks)
        return

    # Non-attention ops in this chart are usually close to 1.0.
    # Use a fixed coarse band so tiny deltas (e.g., 0.98~1.06) are not over-emphasized.
    dev = max(abs(y_min - 1.0), abs(y_max - 1.0))
    if dev <= 0.12:
        ax.set_ylim(0.8, 1.2)
        ax.set_yticks([0.8, 1.0, 1.2])
    elif dev <= 0.20:
        ax.set_ylim(0.75, 1.25)
        ax.set_yticks([0.75, 1.0, 1.25])
    else:
        low = min(0.0, math.floor(y_min * 5.0) / 5.0)
        high = math.ceil(y_max * 5.0) / 5.0
        if high - low < 0.4:
            high = low + 0.4
        ax.set_ylim(low, high)
        step = 0.2 if (high - low) <= 1.0 else 0.5
        ticks = []
        t = low
        while t <= high + 1e-9:
            ticks.append(round(t, 3))
            t += step
        if len(ticks) >= 2:
            ax.set_yticks(ticks)


def plot_npl_normalized(rows: list[Row], out_png: str) -> None:
    idx = _index(rows)
    ctxs = sorted({r.npp for r in rows})
    batches = sorted({r.npl for r in rows})

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for i, op in enumerate(OPS):
        ax = axes[i // 3][i % 3]
        plotted_y: list[float] = []
        for ctx in ctxs:
            base_row = idx.get((ctx, 1))
            if base_row is None:
                continue
            base = base_row.vals[op]
            if base <= 0:
                continue
            xs = []
            ys = []
            for b in batches:
                row = idx.get((ctx, b))
                if row is None:
                    continue
                xs.append(b)
                y = row.vals[op] / base
                ys.append(y)
                plotted_y.append(y)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"ctx={ctx}")
        ax.set_title(f"{op} vs npl (base npl=1)")
        ax.set_xlabel("npl (batch size)")
        ax.set_ylabel("normalized ms")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches])
        _apply_coarse_ylim(ax, op, plotted_y)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, ncols=2)
    fig.suptitle("Decode Op Trends Normalized to npl=1", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_ctx_normalized(rows: list[Row], out_png: str) -> None:
    idx = _index(rows)
    ctxs = sorted({r.npp for r in rows})
    batches = sorted({r.npl for r in rows})
    ctx0 = ctxs[0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for i, op in enumerate(OPS):
        ax = axes[i // 3][i % 3]
        plotted_y: list[float] = []
        for b in batches:
            base_row = idx.get((ctx0, b))
            if base_row is None:
                continue
            base = base_row.vals[op]
            if base <= 0:
                continue
            xs = []
            ys = []
            for ctx in ctxs:
                row = idx.get((ctx, b))
                if row is None:
                    continue
                xs.append(ctx)
                y = row.vals[op] / base
                ys.append(y)
                plotted_y.append(y)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"npl={b}")
        ax.set_title(f"{op} vs context (base ctx={ctx0})")
        ax.set_xlabel("context length (npp)")
        ax.set_ylabel("normalized ms")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ctxs)
        ax.set_xticklabels([str(c) for c in ctxs])
        _apply_coarse_ylim(ax, op, plotted_y)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8, ncols=2)
    fig.suptitle("Decode Op Trends Across Context Length", fontsize=14)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_summary(rows: list[Row], out_txt: str) -> None:
    idx = _index(rows)
    ctxs = sorted({r.npp for r in rows})
    batches = sorted({r.npl for r in rows})
    ctx0, ctxn = ctxs[0], ctxs[-1]
    b0, bn = batches[0], batches[-1]

    lines: list[str] = []
    lines.append("# Trend Summary (from decode duration table)")
    lines.append(f"context range: {ctx0} -> {ctxn}")
    lines.append(f"npl range: {b0} -> {bn}")
    lines.append("")

    lines.append("## Context scaling ratio (ctx_max / ctx_min) by npl")
    lines.append("op      npl      ratio    ctx_used")
    for op in OPS:
        for b in batches:
            if (ctx0, b) not in idx:
                continue
            a = idx[(ctx0, b)].vals[op]
            if a <= 0:
                continue
            # use farthest available context for this npl
            cands = [c for c in reversed(ctxs) if (c, b) in idx]
            if not cands:
                continue
            c_used = cands[0]
            z = idx[(c_used, b)].vals[op]
            ratio = z / a
            lines.append(f"{op:7s} {b:4d}  {ratio:8.3f}    {ctx0}->{c_used}")
    lines.append("")

    lines.append("## Batch scaling ratio (npl_max / npl_min) by context")
    lines.append("op      ctx      ratio    npl_used")
    for op in OPS:
        for ctx in ctxs:
            if (ctx, b0) not in idx:
                continue
            a = idx[(ctx, b0)].vals[op]
            if a <= 0:
                continue
            # use largest available npl for this context
            b_cands = [b for b in reversed(batches) if (ctx, b) in idx]
            if not b_cands:
                continue
            b_used = b_cands[0]
            z = idx[(ctx, b_used)].vals[op]
            ratio = z / a
            lines.append(f"{op:7s} {ctx:5d}  {ratio:8.3f}    {b0}->{b_used}")
    lines.append("")

    lines.append("## Mean context sensitivity across npl")
    lines.append("op      mean_ratio   std_ratio")
    for op in OPS:
        vals = []
        for b in batches:
            if (ctx0, b) not in idx:
                continue
            a = idx[(ctx0, b)].vals[op]
            if a <= 0:
                continue
            cands = [c for c in reversed(ctxs) if (c, b) in idx]
            if not cands:
                continue
            z = idx[(cands[0], b)].vals[op]
            vals.append(z / a)
        good = [v for v in vals if not math.isnan(v)]
        if not good:
            lines.append(f"{op:7s} {float('nan'):10.3f} {float('nan'):10.3f}")
            continue
        mu = sum(good) / len(good)
        var = sum((v - mu) ** 2 for v in good) / len(good)
        sd = math.sqrt(var)
        lines.append(f"{op:7s} {mu:10.3f} {sd:10.3f}")
    lines.append("")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to decode_sweep.txt")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output dir (default: same dir as input)",
    )
    ap.add_argument(
        "--prefix",
        default="decode_sweep_trend",
        help="Output file prefix",
    )
    args = ap.parse_args()

    in_path = os.path.abspath(os.path.expanduser(args.input))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir)) if args.out_dir else os.path.dirname(in_path)
    os.makedirs(out_dir, exist_ok=True)

    rows = parse_duration_rows(in_path)

    png1 = os.path.join(out_dir, f"{args.prefix}_npl_norm.png")
    png2 = os.path.join(out_dir, f"{args.prefix}_ctx_norm.png")
    txt = os.path.join(out_dir, f"{args.prefix}_summary.txt")

    plot_npl_normalized(rows, png1)
    plot_ctx_normalized(rows, png2)
    write_summary(rows, txt)

    print(png1)
    print(png2)
    print(txt)


if __name__ == "__main__":
    main()
