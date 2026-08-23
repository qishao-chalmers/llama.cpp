#!/usr/bin/env python3
"""plot_roofline_gap.py — Figure: measured vs roofline tok/s gap across sweeps.

**Inputs**

1. **JSON** from ``benchmark_kv_timing.py`` (``rows[]``: measured_ms, weight_tag,
   kv_type, decode_len, batch_size, model_preset, …). Roofline is recomputed when
   ``roofline_ms`` is null.

2. **Cluster ``.out`` logs** (e.g. ``kv_timing_*.out``): same benchmark prints
   **decode buckets** — per-segment ms/tok and tok/s while KV grows during one long
   run. We parse those lines and set ``mid_ctx`` to each bucket's ``ctx_mid`` so
   the roofline matches **that** context length (not only ``prompt + decode_len//2``).
   Use ``--x decode_pos`` (generated-token position ≈ ``ctx_mid - prompt_len``) or
   ``--x ctx_mid`` to see gap vs context as decoding progresses.

**Y-axis** (``--metric``)

  - ``delta_tok_s`` — measured tok/s − roofline tok/s (gap; negative ⇒ roofline optimistic)
  - ``rel_pct``     — 100 × (measured − roof) / roof  (tok/s space)
  - ``tok_s``       — **both** curves in each subplot: measured tok/s (solid) and roofline tok/s (dashed), same color per weight quant

**Default layout** (``--layout faceted``)

  - Subplots: rows = batch size, columns = KV type; lines = weight quant.

**X-axis** (``--x``)

  - ``decode_len`` / ``batch_size`` — from JSON sweep rows.
  - ``decode_pos`` / ``ctx_mid`` — from ``.out`` decode buckets (recommended for
    fixed total ``decode_len`` in a single run).

Usage::

    python3 research/scripts/plot_roofline_gap.py \\
        research/results/kv_calibration_analysis.json \\
        --hw h100-sxm --out research/figures/roofline_gap.png

    python3 research/scripts/plot_roofline_gap.py \\
        research/results/qwen3-8b/profile/kv_timing_38871926.out \\
        --hw h100-sxm --x decode_pos --prompt-len 2048 \\
        --out research/figures/roofline_gap_decode_buckets.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Iterable

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print("Need matplotlib + numpy:", e, file=sys.stderr)
    sys.exit(1)

from benchmark_kv_timing import roofline_ms  # noqa: E402


def parse_kv_timing_out(path: str) -> list[dict[str, Any]]:
    """Parse ``benchmark_kv_timing`` text log: decode bucket lines → synthetic rows.

    Each row has ``mid_ctx`` = bucket ``ctx_mid`` for roofline, ``decode_pos`` =
    ``ctx_mid - prompt_len``, and ``measured_ms`` / ``tok_per_s`` from the bucket.
    """
    weight_tag = None
    weight_bits = None
    preset = None
    current: dict[str, Any] | None = None
    in_decode = False
    rows: list[dict[str, Any]] = []
    pat_kv = re.compile(r"^\s*kv=(\w+)\s+prompt=\s*(\d+)\s+decode=\s*(\d+)\s+B=\s*(\d+)")
    pat_bucket = re.compile(
        r"\[\s*(\d+)\s*,\s*(\d+)\]\s+ctx_mid=\s*(\d+)\s+([\d.]+)\s+ms/tok\s+\(([\d.]+)\s+tok/s\)"
    )
    pat_wt = re.compile(r"weight_tag=(\S+)\s+weight_bits=([\d.]+)\s+preset=(\S+)")
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat_wt.search(line)
            if m:
                weight_tag, weight_bits, preset = m.group(1), float(m.group(2)), m.group(3)
                continue
            m = pat_kv.match(line)
            if m:
                in_decode = False
                current = {
                    "kv_type": m.group(1),
                    "prompt_len": int(m.group(2)),
                    "decode_len": int(m.group(3)),
                    "batch_size": int(m.group(4)),
                    "weight_tag": weight_tag,
                    "weight_bits": weight_bits,
                    "model_preset": preset,
                }
                continue
            if "decode buckets" in line:
                in_decode = True
                continue
            if "prefill buckets" in line:
                in_decode = False
                continue
            if in_decode and current:
                m = pat_bucket.search(line)
                if m:
                    df, dt, ctx_mid = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    ms_tok, tokps = float(m.group(4)), float(m.group(5))
                    pl = int(current["prompt_len"])
                    rows.append(
                        {
                            **current,
                            "decode_bucket_from": df,
                            "decode_bucket_to": dt,
                            "mid_ctx": ctx_mid,
                            "ctx_mid": ctx_mid,
                            "decode_pos": ctx_mid - pl,
                            "measured_ms": ms_tok,
                            "tok_per_s": tokps,
                            "_source": "out_bucket",
                        }
                    )
    return rows


def _norm_hw(name: str | None) -> str:
    if not name:
        return "h100-sxm"
    h = name.strip().lower()
    if h in ("h100", "h100-sxm", "h100_sxm"):
        return "h100-sxm"
    if h in ("a100", "a100-80g", "a100_80g"):
        return "a100-80g"
    return h


def _load_rows(paths: Iterable[str]) -> tuple[list[dict[str, Any]], str | None]:
    """Merge rows from JSON benchmark outputs and/or ``kv_timing*.out`` text logs."""
    rows: list[dict[str, Any]] = []
    hw_hint: str | None = None
    for path in paths:
        if path.lower().endswith(".out"):
            rows.extend(parse_kv_timing_out(path))
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            chunk = data
            meta: dict[str, Any] = {}
        else:
            meta = data
            chunk = data.get("rows", [])
            if hw_hint is None:
                hw_hint = meta.get("hardware") or meta.get("hw")
        top_decode = meta.get("decode_len") if isinstance(meta, dict) else None
        top_prompt = meta.get("prompt_len") if isinstance(meta, dict) else None
        for r in chunk:
            if not isinstance(r, dict):
                continue
            rr = dict(r)
            if rr.get("decode_len") is None and top_decode is not None:
                rr["decode_len"] = top_decode
            if rr.get("prompt_len") is None and top_prompt is not None:
                rr["prompt_len"] = top_prompt
            rows.append(rr)
    return rows, hw_hint


def _x_val(row: dict[str, Any], x_key: str) -> float:
    """Numeric x for sorting/plotting (supports JSON rows and ``.out`` buckets)."""
    if x_key == "decode_pos":
        if "decode_pos" in row:
            return float(row["decode_pos"])
        return float(row["mid_ctx"]) - float(row.get("prompt_len", 0))
    if x_key in ("ctx_mid", "mid_ctx"):
        return float(row["mid_ctx"])
    return float(row[x_key])


def _x_label(x_key: str) -> str:
    return {
        "decode_len": "decode_len (sweep)",
        "batch_size": "batch_size",
        "decode_pos": "decode position (ctx_mid − prompt_len)",
        "ctx_mid": "ctx_mid (KV length)",
        "mid_ctx": "mid_ctx",
    }.get(x_key, x_key.replace("_", " "))


def _roof_tok_per_s(row: dict[str, Any], hw: str) -> tuple[float, float]:
    """Return (roofline_ms, roof_tok_per_s)."""
    ms = row.get("roofline_ms")
    preset = row.get("model_preset")
    wbits = float(row.get("weight_bits", 16))
    kv = str(row.get("kv_type", "f16"))
    pl = int(row.get("prompt_len", 0))
    dl = int(row.get("decode_len", 0))
    bs = int(row.get("batch_size", 1))
    mid = row.get("mid_ctx")
    if mid is None:
        mid = pl + dl // 2
    mid = int(mid)

    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        ms = roofline_ms(str(preset), hw, wbits, kv, mid, bs)
    else:
        ms = float(ms)

    if ms <= 0 or math.isnan(ms):
        return float("nan"), float("nan")
    return ms, 1000.0 / ms


def _meas_tok_per_s(row: dict[str, Any]) -> float:
    tps = row.get("tok_per_s")
    if tps is not None:
        return float(tps)
    m = float(row.get("measured_ms", 0))
    if m <= 0:
        return float("nan")
    return 1000.0 / m


def _gather_series(
    rows: list[dict[str, Any]],
    hw: str,
    *,
    model_preset: str | None,
    prompt_len: int | None,
    weight_tags: set[str] | None,
    kv_types: set[str] | None,
    batch_sizes: set[int] | None,
) -> list[dict[str, Any]]:
    """Filter rows and attach gap fields."""
    out: list[dict[str, Any]] = []
    for r in rows:
        if model_preset and str(r.get("model_preset")) != model_preset:
            continue
        if prompt_len is not None and int(r.get("prompt_len", -1)) != prompt_len:
            continue
        wt = str(r.get("weight_tag", ""))
        if weight_tags is not None and wt not in weight_tags:
            continue
        kv = str(r.get("kv_type", ""))
        if kv_types is not None and kv not in kv_types:
            continue
        bs = int(r.get("batch_size", 1))
        if batch_sizes is not None and bs not in batch_sizes:
            continue

        roof_ms, roof_tps = _roof_tok_per_s(r, hw)
        meas_tps = _meas_tok_per_s(r)
        if math.isnan(roof_tps) or math.isnan(meas_tps):
            continue

        delta = meas_tps - roof_tps
        rel_pct = 100.0 * (meas_tps - roof_tps) / roof_tps if roof_tps > 0 else float("nan")

        q = dict(r)
        pl = int(q.get("prompt_len", 0))
        dl = int(q.get("decode_len", 0))
        if q.get("mid_ctx") is None:
            q["mid_ctx"] = pl + dl // 2
        q.setdefault("ctx_mid", q["mid_ctx"])
        if "decode_pos" not in q:
            q["decode_pos"] = int(q["mid_ctx"]) - pl

        q["_roof_ms"] = roof_ms
        q["_roof_tps"] = roof_tps
        q["_meas_tps"] = meas_tps
        q["_delta_tok_s"] = delta
        q["_rel_pct"] = rel_pct
        out.append(q)
    return out


def _order_weight_tags(tags: Iterable[str]) -> list[str]:
    """Stable sort: Q2_K, Q3_K*, Q4_K*, Q8_0, … then rest lexicographic."""

    def key(t: str) -> tuple:
        u = t.upper()
        pref = (99, u)
        for i, p in enumerate(("Q2_K", "Q3_K", "Q4_K", "Q8_0", "F16", "FP16")):
            if u.startswith(p) or p in u:
                pref = (i, u)
                break
        return pref

    return sorted(set(tags), key=key)


def plot_faceted(
    series: list[dict[str, Any]],
    *,
    x_key: str,
    metric: str,
    title: str,
    figsize: tuple[float, float],
) -> plt.Figure:
    """Faceted grid: rows = batch_size, cols = kv_type."""
    batch_sizes = sorted({int(r["batch_size"]) for r in series})
    kv_list = sorted({str(r["kv_type"]) for r in series})
    if not batch_sizes or not kv_list:
        raise ValueError("No data after filtering")

    nrows, ncols = len(batch_sizes), len(kv_list)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    by_bs_kv: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in series:
        by_bs_kv[(int(r["batch_size"]), str(r["kv_type"]))].append(r)

    if metric == "tok_s":
        y_label = "tok/s"
    elif metric == "delta_tok_s":
        y_label = "Δ tok/s (meas − roof)"
    else:
        y_label = "% gap on tok/s (meas vs roof)"

    for i, bs in enumerate(batch_sizes):
        for j, kv in enumerate(kv_list):
            ax = axes[i][j]
            cell = by_bs_kv.get((bs, kv), [])
            by_wt: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in cell:
                by_wt[str(r["weight_tag"])].append(r)

            colors = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(by_wt))))
            for wi, wt in enumerate(_order_weight_tags(by_wt.keys())):
                pts = sorted(by_wt[wt], key=lambda r: _x_val(r, x_key))
                if not pts:
                    continue
                xs = [_x_val(p, x_key) for p in pts]
                c = colors[wi % len(colors)]
                if metric == "tok_s":
                    ax.plot(
                        xs,
                        [p["_meas_tps"] for p in pts],
                        marker="o",
                        markersize=3,
                        linestyle="-",
                        linewidth=1.5,
                        label=f"{wt} measured",
                        color=c,
                    )
                    ax.plot(
                        xs,
                        [p["_roof_tps"] for p in pts],
                        marker=".",
                        markersize=2,
                        linestyle="--",
                        linewidth=1.2,
                        label=f"{wt} roofline",
                        color=c,
                        alpha=0.9,
                    )
                else:
                    ys = [p["_delta_tok_s"] if metric == "delta_tok_s" else p["_rel_pct"] for p in pts]
                    ax.plot(xs, ys, marker="o", markersize=3, label=wt, color=c)

            if metric != "tok_s":
                ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
            ax.set_title(f"B={bs}  KV={kv}", fontsize=9)
            if j == 0:
                ax.set_ylabel(y_label, fontsize=8)
            if i == nrows - 1:
                ax.set_xlabel(_x_label(x_key), fontsize=8)
            ax.grid(True, alpha=0.3)

    # Dedupe legend across subplots (each cell may plot a subset of weight tags)
    legend_map: dict[str, Any] = {}
    for ax in axes.flat:
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            if li not in legend_map:
                legend_map[li] = hi
    if legend_map:
        ncol = min(8, len(legend_map)) if metric == "tok_s" else min(6, len(legend_map))
        fig.legend(
            legend_map.values(),
            legend_map.keys(),
            loc="upper center",
            ncol=ncol,
            fontsize=7 if metric == "tok_s" else 8,
            frameon=True,
        )

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_single_panel(
    series: list[dict[str, Any]],
    *,
    x_key: str,
    metric: str,
    title: str,
    figsize: tuple[float, float],
    hue: str,
) -> plt.Figure:
    """One panel: x = x_key, hue = weight_tag or batch_size or kv_type."""
    fig, ax = plt.subplots(figsize=figsize)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in series:
        if hue == "weight_tag":
            k = str(r["weight_tag"])
        elif hue == "batch_size":
            k = f"B={int(r['batch_size'])}"
        elif hue == "kv_type":
            k = str(r["kv_type"])
        else:
            k = str(r.get(hue, "?"))
        groups[k].append(r)

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(10, len(groups))))
    for wi, key in enumerate(sorted(groups.keys())):
        pts = sorted(
            groups[key],
            key=lambda r: (_x_val(r, x_key), str(r.get("weight_tag", ""))),
        )
        xs = [_x_val(p, x_key) for p in pts]
        c = colors[wi % len(colors)]
        if metric == "tok_s":
            ax.plot(
                xs,
                [p["_meas_tps"] for p in pts],
                marker="o",
                markersize=3,
                linestyle="-",
                label=f"{key} measured",
                color=c,
            )
            ax.plot(
                xs,
                [p["_roof_tps"] for p in pts],
                marker=".",
                linestyle="--",
                label=f"{key} roofline",
                color=c,
                alpha=0.9,
            )
        else:
            ys = [p["_delta_tok_s"] if metric == "delta_tok_s" else p["_rel_pct"] for p in pts]
            ax.plot(xs, ys, marker="o", markersize=3, label=key, color=c)

    if metric != "tok_s":
        ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    if metric == "tok_s":
        y_label = "tok/s"
    elif metric == "delta_tok_s":
        y_label = "Δ tok/s (meas − roof)"
    else:
        y_label = "% gap on tok/s (meas vs roof)"
    ax.set_ylabel(y_label)
    ax.set_xlabel(_x_label(x_key))
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "json_paths",
        nargs="+",
        help="benchmark JSON (rows[]) and/or kv_timing *.out decode-bucket logs",
    )
    ap.add_argument("--out", default=None, help="Output PNG path")
    ap.add_argument("--hw", default=None, help="Hardware preset for roofline (default: from JSON or h100-sxm)")
    ap.add_argument("--model-preset", default=None, help="Filter rows to this model_preset")
    ap.add_argument("--prompt-len", type=int, default=None, help="Filter to this prompt length")
    ap.add_argument("--weight-tags", nargs="*", default=None, help="Filter to these weight_tag values")
    ap.add_argument("--kv-types", nargs="*", default=None, help="Filter to these kv_type values")
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=None, help="Filter to these batch sizes")
    ap.add_argument(
        "--metric",
        choices=("delta_tok_s", "rel_pct", "tok_s"),
        default="delta_tok_s",
        help="delta_tok_s = gap; rel_pct = %% gap; tok_s = measured + roofline curves",
    )
    ap.add_argument(
        "--x",
        dest="x_key",
        choices=("decode_len", "batch_size", "decode_pos", "ctx_mid", "mid_ctx"),
        default=None,
        help="X-axis (default: decode_pos for *.out buckets; else decode_len or batch_size)",
    )
    ap.add_argument(
        "--layout",
        choices=("faceted", "single"),
        default="faceted",
        help="faceted: grid batch_size × kv_type; single: one plot, --hue for series",
    )
    ap.add_argument("--hue", default="weight_tag", help="For --layout single: weight_tag | batch_size | kv_type")
    ap.add_argument("--title", default=None, help="Figure title")
    ap.add_argument("--figwidth", type=float, default=14.0)
    ap.add_argument("--figheight", type=float, default=10.0)
    args = ap.parse_args()

    rows, hw_hint = _load_rows(args.json_paths)
    hw = _norm_hw(args.hw or hw_hint)

    wt_set = set(args.weight_tags) if args.weight_tags else None
    kv_set = set(args.kv_types) if args.kv_types else None
    bs_set = set(args.batch_sizes) if args.batch_sizes else None

    series = _gather_series(
        rows,
        hw,
        model_preset=args.model_preset,
        prompt_len=args.prompt_len,
        weight_tags=wt_set,
        kv_types=kv_set,
        batch_sizes=bs_set,
    )
    if not series:
        print("No rows left after filtering (check --model-preset / --prompt-len / inputs).", file=sys.stderr)
        sys.exit(1)

    x_key = args.x_key
    if x_key is None:
        has_bucket = any(r.get("_source") == "out_bucket" for r in series)
        if has_bucket:
            x_key = "decode_pos"
            print(
                "[info] decode-bucket rows from .out: using x=decode_pos "
                "(ctx_mid−prompt; use --x ctx_mid for full KV length)",
                file=sys.stderr,
            )
        else:
            decode_vals = {int(r["decode_len"]) for r in series}
            if len(decode_vals) <= 1:
                bs_vals = {int(r["batch_size"]) for r in series}
                if len(bs_vals) > 1:
                    x_key = "batch_size"
                    print(
                        f"[info] only one decode_len={next(iter(decode_vals))}; "
                        "using x=batch_size for spread",
                        file=sys.stderr,
                    )
                else:
                    x_key = "decode_len"
            else:
                x_key = "decode_len"

    title = args.title or f"Roofline vs measured tok/s gap (hw={hw}, x={x_key}, metric={args.metric})"

    if args.layout == "faceted":
        fig = plot_faceted(
            series,
            x_key=x_key,
            metric=args.metric,
            title=title,
            figsize=(args.figwidth, args.figheight),
        )
    else:
        fig = plot_single_panel(
            series,
            x_key=x_key,
            metric=args.metric,
            title=title,
            figsize=(args.figwidth, args.figheight),
            hue=args.hue,
        )

    out = args.out
    if not out:
        out = os.path.join(_SCRIPT_DIR, "../figures/roofline_gap.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
