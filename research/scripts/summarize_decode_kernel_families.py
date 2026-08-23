#!/usr/bin/env python3
"""summarize_decode_kernel_families.py — Representative kernel time breakdown CSV.

Reads per-report CSVs from parse_nsys_decode_kernels.py (flat stage) and writes
one row per (model, quant, npl) with per-launch mean us / pct for kernel families:

  matmul      mul_mat_vec_q, mul_mat_q (+ stream_k fixups)
  flash_attn  flash_attn_ext_*, combine, mask, stream_k fixup
  quantize    quantize_q8_1, quantize_mmq_q8_1
  rms_norm    rms_norm_f32
  rope        rope_*
  other       everything else

Columns:
  {family}_mean_us  — mean duration of one launch of the family's dominant
                      kernel (largest sum_us), e.g. flash_attn_ext_* not combine
  {family}_pct      — share of total decode GPU kernel time (full family sum)
  total_us          — sum of all flat-stage kernel time (whole decode window)

Compare microbenchmarks to {family}_mean_us (e.g. flash_attn ~15 µs), not total_us.

Usage:
    python3 research/scripts/summarize_decode_kernel_families.py \\
        --in-dir research/results/nsys/decode_kernels \\
        --out research/results/nsys/decode_kernels/kernel_families.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path


FAMILIES = ("matmul", "flash_attn", "quantize", "rms_norm", "rope", "other")

NAME_RE = re.compile(
    r"^(?P<model>qwen3-\d+b|gemma3-\d+b)_(?P<quant>Q2_K|Q3_K_M|Q8_0)"
    r"_npp(?P<npp>\d+)_ntg(?P<ntg>\d+)_npl(?P<npl>\d+)"
)

HDR_RE = re.compile(
    r"n_layers=(?P<n_layers>\d+)\s+n_tg_est=(?P<n_tg>\d+)\s+"
    r"decode_wall_ms=(?P<wall_ms>[\d.]+)\s+decode_tok_s=(?P<tok_s>[\d.]+)"
)


def classify(kernel: str) -> str:
    k = kernel.lower()
    if k.startswith("mul_mat"):
        return "matmul"
    if "flash_attn" in k:
        return "flash_attn"
    if k.startswith("quantize"):
        return "quantize"
    if "rms_norm" in k:
        return "rms_norm"
    if k.startswith("rope"):
        return "rope"
    return "other"


def load_flat_families(path: Path) -> tuple[dict, dict[str, dict]]:
    """Return meta and per-family stats from flat-stage rows.

    mean_us is taken from the family's dominant kernel (largest sum_us),
    so flash_attn_mean_us ≈ flash_attn_ext_* mean (not diluted by combine).
    pct still uses the full family sum.
    """
    text = path.read_text()
    lines = text.splitlines()
    meta = {"report": path.stem.replace("_decode_kernels", "")}
    if lines and lines[0].startswith("#"):
        hm = HDR_RE.search(lines[0])
        if hm:
            meta.update(hm.groupdict())
    nm = NAME_RE.search(path.name)
    if nm:
        meta.update(nm.groupdict())

    body = "\n".join(ln for ln in lines if not ln.startswith("#"))
    fam: dict[str, dict] = {
        f: {"sum_us": 0.0, "dom_sum": -1.0, "dom_mean": 0.0, "dom_kernel": ""}
        for f in FAMILIES
    }
    for row in csv.DictReader(io.StringIO(body)):
        if row.get("stage") != "flat":
            continue
        f = classify(row["kernel"])
        sum_us = float(row["sum_us"])
        mean_us = float(row["mean_us"])
        fam[f]["sum_us"] += sum_us
        if sum_us > fam[f]["dom_sum"]:
            fam[f]["dom_sum"] = sum_us
            fam[f]["dom_mean"] = mean_us
            fam[f]["dom_kernel"] = row["kernel"]
    return meta, fam


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-dir",
        type=Path,
        default=Path("research/results/nsys/decode_kernels"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: <in-dir>/kernel_families.csv)",
    )
    args = ap.parse_args()
    out = args.out or (args.in_dir / "kernel_families.csv")

    rows = []
    for path in sorted(args.in_dir.glob("*_decode_kernels.csv")):
        meta, fam = load_flat_families(path)
        if "model" not in meta:
            continue
        total = sum(fam[f]["sum_us"] for f in FAMILIES) or 1.0
        row = {
            "report": meta["report"],
            "model": meta["model"],
            "quant": meta["quant"],
            "npp": meta.get("npp", ""),
            "ntg": meta.get("ntg", ""),
            "npl": int(meta["npl"]),
            "n_layers": meta.get("n_layers", ""),
            "n_tg": meta.get("n_tg", ""),
            "decode_tok_s": meta.get("tok_s", ""),
            "decode_wall_ms": meta.get("wall_ms", ""),
            "total_us": round(total, 3),
        }
        for f in FAMILIES:
            us = fam[f]["sum_us"]
            row[f"{f}_mean_us"] = round(fam[f]["dom_mean"], 3)
            row[f"{f}_pct"] = round(100.0 * us / total, 3)
        rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["quant"], r["npl"]))

    fields = [
        "report", "model", "quant", "npp", "ntg", "npl",
        "n_layers", "n_tg", "decode_tok_s", "decode_wall_ms",
        "total_us",
    ]
    for f in FAMILIES:
        fields += [f"{f}_mean_us", f"{f}_pct"]

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {out} rows={len(rows)}")
    print(
        f"{'model':12s} {'quant':7s} {'npl':>3s} "
        f"{'attn_mean':>10s} {'matmul%':>8s} {'attn%':>7s} {'total_ms':>9s}"
    )
    for r in rows:
        print(
            f"{r['model']:12s} {r['quant']:7s} {r['npl']:3d} "
            f"{r['flash_attn_mean_us']:9.2f}u "
            f"{r['matmul_pct']:7.1f}% {r['flash_attn_pct']:6.1f}% "
            f"{r['total_us']/1e3:9.1f}"
        )


if __name__ == "__main__":
    main()
