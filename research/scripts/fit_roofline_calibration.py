#!/usr/bin/env python3
"""fit_roofline_calibration.py — Fit two-parameter linear calibration to benchmark_kv_timing rows.

Model (same as research/context/roofline_calibration.md):

  measured_ms_per_tok ≈ t_floor_ms / batch_size + scale × roofline_ms_per_tok

where roofline_ms_per_tok is the analytic value from benchmark_kv_timing.roofline_ms()
at mid_ctx = prompt_len + decode_len // 2.

Fits (t_floor_ms, scale) with least squares on selected rows, writes JSON for use with
benchmark_kv_timing.py --roofline-calibration-json.

Example::

    python3 research/scripts/fit_roofline_calibration.py \\
        research/results/qwen3-8b/profile/kv_timing_h100.json \\
        --model-preset qwen3-8b --hw h100-sxm \\
        --weight-tag Q8_0 --kv-type f16 \\
        -o research/results/roofline_calibration_parameters.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np

from benchmark_kv_timing import roofline_ms


def _load_rows(path: str) -> tuple[list[dict], dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    return data.get("rows", []), data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_paths", nargs="+", help="benchmark_kv_timing JSON with rows[]")
    ap.add_argument("--model-preset", required=True, help="MODEL_PRESETS key, e.g. qwen3-8b")
    ap.add_argument("--hw", required=True, help="HARDWARE_PRESETS key, e.g. h100-sxm")
    ap.add_argument("--weight-tag", default=None, help="Keep only this weight_tag (e.g. Q8_0)")
    ap.add_argument("--kv-type", default=None, help="Keep only this kv_type (e.g. f16)")
    ap.add_argument("--prompt-len", type=int, default=None, help="Filter prompt_len")
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--out", "-o", required=True, help="Output calibration JSON")
    args = ap.parse_args()

    rows: list[dict] = []
    for p in args.json_paths:
        chunk, _ = _load_rows(p)
        rows.extend(chunk)

    xs_inv_b: list[float] = []
    xs_roof: list[float] = []
    ys: list[float] = []
    meta: list[dict] = []

    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("model_preset")) != args.model_preset:
            continue
        if args.weight_tag and str(r.get("weight_tag")) != args.weight_tag:
            continue
        if args.kv_type and str(r.get("kv_type", "")).lower() != args.kv_type.lower():
            continue
        if args.prompt_len is not None and int(r.get("prompt_len", -1)) != args.prompt_len:
            continue
        B = int(r.get("batch_size", 0))
        if B < args.min_batch:
            continue
        meas = float(r.get("measured_ms", 0))
        if meas <= 0:
            continue
        wbits = float(r.get("weight_bits", 16))
        kv = str(r.get("kv_type", "f16"))
        pl = int(r.get("prompt_len", 0))
        dl = int(r.get("decode_len", 0))
        mid = int(r.get("mid_ctx", pl + dl // 2))
        roof = roofline_ms(args.model_preset, args.hw, wbits, kv, mid, B)
        if math.isnan(roof) or roof <= 0:
            continue
        xs_inv_b.append(1.0 / B)
        xs_roof.append(roof)
        ys.append(meas)
        meta.append(
            {
                "batch_size": B,
                "measured_ms": meas,
                "roofline_ms": roof,
                "prompt_len": pl,
                "decode_len": dl,
                "weight_tag": r.get("weight_tag"),
                "kv_type": kv,
            }
        )

    n = len(ys)
    if n < 2:
        print(
            f"Need at least 2 valid rows; got {n}. Relax filters or add JSON files.",
            file=sys.stderr,
        )
        sys.exit(1)

    X = np.column_stack([xs_inv_b, xs_roof])
    y = np.array(ys, dtype=float)
    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    t_floor = float(coef[0])
    scale = float(coef[1])
    yhat = X @ coef
    rmse = float(math.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))

    out = {
        "_comment": (
            "Linear decode calibration: measured_ms ≈ t_floor_ms / batch_size + scale * roofline_ms. "
            "Use with benchmark_kv_timing.py --roofline-calibration-json."
        ),
        "model_preset": args.model_preset,
        "hw": args.hw,
        "weight_tag": args.weight_tag,
        "kv_type": args.kv_type,
        "prompt_len_filter": args.prompt_len,
        "t_floor_ms": round(t_floor, 6),
        "scale": round(scale, 6),
        "rmse_ms": round(rmse, 6),
        "mae_ms": round(mae, 6),
        "n_fit_points": n,
        "sources": list(args.json_paths),
        "fit_rows": meta,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Fitted t_floor_ms={t_floor:.4f}  scale={scale:.4f}  RMSE={rmse:.4f} ms  (n={n})")
    print(f"Wrote {args.out}")
    print("\nPer-row (measured vs predicted):")
    for m, pred in zip(meta, yhat):
        print(
            f"  B={m['batch_size']:3d}  meas={m['measured_ms']:.3f}  "
            f"pred={pred:.3f}  roof={m['roofline_ms']:.3f}"
        )


if __name__ == "__main__":
    main()
