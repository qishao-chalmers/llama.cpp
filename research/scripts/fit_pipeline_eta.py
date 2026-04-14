#!/usr/bin/env python3
"""Fit whole-step pipeline η (eta_wt, eta_kv, t_floor_ms) to measured kv_timing JSON rows.

Model:
    step_ms = T_floor + wt_bytes / eff_BW_wt + kv_bytes / eff_BW_kv
    ms/tok  = step_ms / B

where wt_bytes and kv_bytes come from compute_step_bytes() (GGUF-exact if --gguf is set)
and eff_BW_{wt,kv} = peak_BW × hw_efficiency × η_{wt,kv}.

Physical motivation: the GPU's HBM bus services weight reads and KV reads in a continuous
stream while SMs compute; the serial per-op roofline overestimates by treating each op as
an independent memory bottleneck. Two η parameters capture the different BW efficiency
of weight GEMV/GEMM vs attention KV kernels.

Example::

    python3 fit_pipeline_eta.py \\
        ../results/qwen3-8b/profile/kv_timing_h100.json \\
        --hw h100-sxm --preset qwen3-8b \\
        --gguf ../../models/Qwen3-8B-Q8_0.gguf \\
        -o ../results/qwen3-8b/profile/pipeline_eta_h100.json

    # Without GGUF (synthetic bytes from model structure):
    python3 fit_pipeline_eta.py \\
        ../results/qwen3-8b/profile/kv_timing_h100.json \\
        --hw h100-sxm --preset qwen3-8b \\
        -o ../results/qwen3-8b/profile/pipeline_eta_h100.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from layerwise_roofline_sim import (
    HARDWARE_PRESETS,
    load_structure_catalog,
    predict_ms_per_token_pipeline,
    resolve_kv_quant_key,
    resolve_weight_bits,
)
from gguf_layerwise_weights import load_gguf_tensor_n_bytes

try:
    from scipy.optimize import minimize
    _HAS_SCIPY = True
except ImportError:
    minimize = None  # type: ignore
    _HAS_SCIPY = False


def load_rows(
    paths: list[str],
    catalog: dict[str, Any],
    preset_filter: Optional[str],
) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("rows", []):
            mp = r.get("model_preset")
            if not mp or mp not in catalog["presets"]:
                continue
            if preset_filter and str(mp) != preset_filter:
                continue
            meas = float(r.get("measured_ms", 0))
            if meas <= 0:
                continue
            pl = int(r.get("prompt_len", 0))
            dl = int(r.get("decode_len", 0))
            mid = int(r.get("mid_ctx", pl + dl // 2))
            rows.append({
                "model":       dict(catalog["presets"][mp]),
                "preset":      mp,
                "B":           int(r["batch_size"]),
                "ctx":         mid,
                "weight_bits": float(r.get("weight_bits", 16)),
                "weight_tag":  r.get("weight_tag"),
                "kv_type":     str(r.get("kv_type", "f16")),
                "measured_ms": meas,
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_paths", nargs="+", help="kv_timing*.json files")
    ap.add_argument(
        "--catalog",
        default=os.path.join(_SCRIPT_DIR, "model_structures.json"),
    )
    ap.add_argument("--hw", default="h100-sxm", help="Hardware preset key")
    ap.add_argument("--preset", default=None, help="Filter to this model_preset")
    ap.add_argument("--gguf", default=None, metavar="PATH",
                    help="GGUF file for exact per-tensor weight bytes (recommended)")
    ap.add_argument("--norm-weight-bits", type=float, default=16.0)
    ap.add_argument("--eta-min", type=float, default=0.05)
    ap.add_argument("--eta-max", type=float, default=1.0)
    ap.add_argument("--t-floor-min", type=float, default=-5.0,
                    help="Lower bound for T_floor_ms (ms per step)")
    ap.add_argument("--t-floor-max", type=float, default=10.0)
    ap.add_argument("-o", "--out", required=True, help="Output JSON path")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cat = load_structure_catalog(args.catalog)
    rows = load_rows(args.json_paths, cat, preset_filter=args.preset)
    if len(rows) < 3:
        print(f"Need at least 3 rows; got {len(rows)}.", file=sys.stderr)
        sys.exit(1)

    gguf_bytes: Optional[dict[str, int]] = None
    if args.gguf:
        gguf_bytes = load_gguf_tensor_n_bytes(args.gguf)
        print(f"Loaded GGUF: {args.gguf} ({len(gguf_bytes)} tensors)")

    hw_name = args.hw

    def residuals(params: np.ndarray) -> np.ndarray:
        eta_wt, eta_kv, t_floor = float(params[0]), float(params[1]), float(params[2])
        out = []
        for row in rows:
            m = row["model"]
            wb, _ = resolve_weight_bits(m, row.get("weight_tag"), row["weight_bits"])
            kv_key = resolve_kv_quant_key(row["kv_type"])
            pred = predict_ms_per_token_pipeline(
                m,
                batch_size=row["B"],
                ctx_len=row["ctx"],
                hw_name=hw_name,
                eta_wt=eta_wt,
                eta_kv=eta_kv,
                t_floor_ms=t_floor,
                weight_bits=wb,
                norm_weight_bits=args.norm_weight_bits,
                kv_quant_key=kv_key,
                gguf_tensor_bytes=gguf_bytes,
            )
            out.append(pred - row["measured_ms"])
        return np.array(out, dtype=float)

    def cost(params: np.ndarray) -> float:
        r = residuals(params)
        return float(np.sum(r ** 2))

    # Initial guess: moderate efficiency, no floor
    x0 = np.array([0.40, 0.35, 0.5])
    bounds = [
        (args.eta_min, args.eta_max),
        (args.eta_min, args.eta_max),
        (args.t_floor_min, args.t_floor_max),
    ]

    if _HAS_SCIPY and minimize is not None:
        result = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                          options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10,
                                   "maxfun": 10000})
        x_fit = result.x
        opt_meta = {"success": bool(result.success), "message": str(result.message),
                    "nfev": int(result.nfev), "optimizer": "scipy.minimize.L-BFGS-B"}
    else:
        # Coordinate grid search fallback
        x_fit = x0.copy()
        best_c = cost(x_fit)
        n_rounds, n_grid = 60, 32
        for _ in range(n_rounds):
            for i, (lo, hi) in enumerate(bounds):
                best_t, best_ci = float(x_fit[i]), best_c
                for trial in np.linspace(lo, hi, n_grid):
                    x_fit[i] = trial
                    c = cost(x_fit)
                    if c < best_ci:
                        best_ci, best_t = c, trial
                x_fit[i] = best_t
                best_c = best_ci
        opt_meta = {"optimizer": "coordinate_grid", "n_rounds": n_rounds, "n_grid": n_grid}

    eta_wt, eta_kv, t_floor = float(x_fit[0]), float(x_fit[1]), float(x_fit[2])

    errs = residuals(x_fit)
    rmse = float(math.sqrt(np.mean(errs ** 2)))
    mae  = float(np.mean(np.abs(errs)))

    print(f"\nFitted pipeline η (hw='{hw_name}'):")
    print(f"  eta_wt    = {eta_wt:.6f}  (weight GEMV/GEMM BW efficiency)")
    print(f"  eta_kv    = {eta_kv:.6f}  (KV cache read BW efficiency)")
    print(f"  t_floor   = {t_floor:.4f} ms  (per-step fixed overhead)")
    print(f"  RMSE      = {rmse:.4f} ms/tok")
    print(f"  MAE       = {mae:.4f} ms/tok")
    print(f"  n rows    = {len(rows)}")

    if args.verbose:
        print("\nPer-row residuals:")
        print(f"  {'B':>3}  {'kv':8s}  {'meas':>8}  {'pred':>8}  {'err':>8}  {'ratio':>6}")
        for row, e in zip(rows, errs):
            pred = row["measured_ms"] + e
            print(f"  {row['B']:3d}  {row['kv_type']:8s}  {row['measured_ms']:8.4f}  "
                  f"{pred:8.4f}  {e:+8.4f}  {pred/row['measured_ms']:6.3f}")

    # Compute per-row predictions for output JSON
    fit_rows = []
    for row, e in zip(rows, errs):
        m = row["model"]
        wb, _ = resolve_weight_bits(m, row.get("weight_tag"), row["weight_bits"])
        kv_key = resolve_kv_quant_key(row["kv_type"])
        from layerwise_roofline_sim import compute_step_bytes, resolve_weight_bits as _rwb
        wt_b, kv_b = compute_step_bytes(
            m, row["B"], row["ctx"], kv_key, wb/8.0, args.norm_weight_bits/8.0,
            gguf_tensor_bytes=gguf_bytes,
        )
        fit_rows.append({
            "preset":      row["preset"],
            "batch_size":  row["B"],
            "kv_type":     row["kv_type"],
            "weight_tag":  row.get("weight_tag"),
            "measured_ms": row["measured_ms"],
            "pred_ms":     round(row["measured_ms"] + float(e), 6),
            "wt_bytes":    int(wt_b),
            "kv_bytes":    int(kv_b),
        })

    out_data = {
        "_comment": (
            "Whole-step pipeline η fit: ms/tok = (T_floor + wt_bytes/eff_BW_wt + kv_bytes/eff_BW_kv) / B. "
            "Use with layerwise_roofline_sim.predict_ms_per_token_pipeline()."
        ),
        "hw": hw_name,
        "eta_wt":    round(eta_wt, 8),
        "eta_kv":    round(eta_kv, 8),
        "t_floor_ms": round(t_floor, 6),
        "rmse_ms":   round(rmse, 6),
        "mae_ms":    round(mae, 6),
        "n_fit_rows": len(rows),
        "gguf_path": args.gguf,
        "sources":   list(args.json_paths),
        "optimizer": opt_meta,
        "fit_rows":  fit_rows,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
