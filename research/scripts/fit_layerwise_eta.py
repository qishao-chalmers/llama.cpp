#!/usr/bin/env python3
"""Fit layerwise_roofline_sim η (per-op-family efficiencies) to measured kv_timing JSON rows.

Minimizes sum of squared errors between predicted ms/tok and measured_ms.
Uses ``scipy.optimize.least_squares`` when available; otherwise a bounded
coordinate grid search (install scipy for best results).

Example::

    python3 fit_layerwise_eta.py \\
        ../results/qwen3-8b/profile/kv_timing_h100.json \\
        --hw h100-sxm -o ../results/qwen3-8b/profile/layerwise_eta_h100.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from layerwise_roofline_sim import (  # noqa: E402
    ETA_FIELD_NAMES,
    Eta,
    load_structure_catalog,
    predict_ms_per_token,
    resolve_kv_quant_key,
    resolve_sim_physics,
    resolve_weight_bits,
    eta_to_dict,
)

try:
    from scipy.optimize import least_squares

    _HAS_SCIPY = True
except ImportError:
    least_squares = None  # type: ignore
    _HAS_SCIPY = False


def fit_eta_coordinate_descent(
    residuals,
    x0: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    n_rounds: int = 40,
    n_grid: int = 24,
) -> np.ndarray:
    """Bounded coordinate search (no scipy). Slower but deterministic."""

    x = np.clip(x0.astype(float).copy(), lo, hi)

    def cost(xv: np.ndarray) -> float:
        return float(np.sum(residuals(xv) ** 2))

    best = cost(x)
    for _ in range(n_rounds):
        for i in range(len(x)):
            best_t = float(x[i])
            best_c = best
            for trial in np.linspace(lo[i], hi[i], n_grid):
                x[i] = trial
                c = cost(x)
                if c < best_c:
                    best_c = c
                    best_t = trial
            x[i] = best_t
            best = best_c
    return x


def mid_ctx(r: dict) -> int:
    pl = int(r.get("prompt_len", 0))
    dl = int(r.get("decode_len", 0))
    return int(r.get("mid_ctx", pl + dl // 2))


def pack_eta(e: Eta) -> np.ndarray:
    return np.array([getattr(e, k) for k in ETA_FIELD_NAMES], dtype=float)


def unpack_eta(x: np.ndarray) -> Eta:
    return Eta(**{k: float(x[i]) for i, k in enumerate(ETA_FIELD_NAMES)})


def load_rows(
    paths: list[str],
    catalog: dict[str, Any],
    *,
    preset_filter: Optional[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            rows.append(
                {
                    "model": dict(catalog["presets"][mp]),
                    "preset": mp,
                    "B": int(r["batch_size"]),
                    "ctx": mid_ctx(r),
                    "weight_bits": float(r.get("weight_bits", 16)),
                    "weight_tag": r.get("weight_tag"),
                    "kv_type": str(r.get("kv_type", "f16")),
                    "measured_ms": meas,
                    "source": os.path.relpath(path, _SCRIPT_DIR),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_paths", nargs="+", help="kv_timing*.json files")
    ap.add_argument(
        "--catalog",
        default=os.path.join(_SCRIPT_DIR, "model_structures.json"),
        help="model_structures.json",
    )
    ap.add_argument("--hw", default="h100-sxm", help="Hardware key (must match layerwise sim)")
    ap.add_argument("--preset", default=None, help="Keep only rows with this model_preset")
    ap.add_argument("--norm-weight-bits", type=float, default=16.0)
    ap.add_argument("--eta-min", type=float, default=0.05, help="Lower bound for each η")
    ap.add_argument("--eta-max", type=float, default=1.0, help="Upper bound for each η")
    ap.add_argument("-o", "--out", required=True, help="Output JSON path")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--sim-physics-json",
        default=None,
        help="Optional sim physics (kv_attn_byte_mode, attn_time_scale, …)",
    )
    ap.add_argument(
        "--kv-attn-byte-mode",
        choices=["fp16_equiv_dequant", "storage"],
        default=None,
    )
    ap.add_argument("--attn-time-scale", type=float, default=None)
    ap.add_argument("--attn-time-scale-inv-batch", type=float, default=None)
    args = ap.parse_args()

    cat = load_structure_catalog(args.catalog)
    rows = load_rows(args.json_paths, cat, preset_filter=args.preset)
    if len(rows) < len(ETA_FIELD_NAMES):
        print(
            f"Need at least {len(ETA_FIELD_NAMES)} valid rows; got {len(rows)}. "
            "Add more JSON files or relax --preset.",
            file=sys.stderr,
        )
        sys.exit(1)

    hw = args.hw
    phys = resolve_sim_physics(
        args.sim_physics_json,
        kv_attn_byte_mode=args.kv_attn_byte_mode,
        attn_time_scale=args.attn_time_scale,
        attn_time_scale_inv_batch=args.attn_time_scale_inv_batch,
    )

    def residuals(x: np.ndarray) -> np.ndarray:
        eta = unpack_eta(x)
        out = []
        for row in rows:
            m = row["model"]
            wb, _ = resolve_weight_bits(m, str(row["weight_tag"]) if row["weight_tag"] else None, row["weight_bits"])
            kv_key = resolve_kv_quant_key(row["kv_type"])
            pred = predict_ms_per_token(
                m,
                batch_size=row["B"],
                ctx_len=row["ctx"],
                hw_name=hw,
                eta=eta,
                weight_bits=wb,
                norm_weight_bits=args.norm_weight_bits,
                kv_quant_key=kv_key,
                kv_attn_byte_mode=str(phys["kv_attn_byte_mode"]),
                attn_time_scale=float(phys["attn_time_scale"]),
                attn_time_scale_inv_batch=float(phys["attn_time_scale_inv_batch"]),
                attn_scale_by_batch=phys.get("attn_scale_by_batch"),
                attn_scale_by_batch_and_kv=phys.get("attn_scale_by_batch_and_kv"),
            )
            out.append(pred - row["measured_ms"])
        return np.array(out, dtype=float)

    x0 = pack_eta(Eta())
    lo = np.full(len(ETA_FIELD_NAMES), args.eta_min)
    hi = np.full(len(ETA_FIELD_NAMES), args.eta_max)

    ls_meta: dict[str, Any] = {}
    if _HAS_SCIPY and least_squares is not None:
        result = least_squares(
            residuals, x0, bounds=(lo, hi), verbose=2 if args.verbose else 0
        )
        x_fit = result.x
        ls_meta = {
            "cost": float(result.cost),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "optimizer": "scipy.least_squares",
        }
    else:
        x_fit = fit_eta_coordinate_descent(residuals, x0, lo, hi)
        ls_meta = {
            "optimizer": "coordinate_descent_grid",
            "note": "Install scipy for faster/better fit (scipy.optimize.least_squares).",
        }

    eta_fit = unpack_eta(x_fit)
    res = residuals(x_fit)
    rmse = float(math.sqrt(np.mean(res**2)))
    mae = float(np.mean(np.abs(res)))

    # Baseline (default η)
    res0 = residuals(x0)
    rmse0 = float(math.sqrt(np.mean(res0**2)))

    out_data: dict[str, Any] = {
        "_comment": (
            "Fitted η for layerwise_roofline_sim.py. Use: "
            "python3 layerwise_roofline_sim.py ... --eta-json THIS_FILE"
        ),
        "schema_version": 1,
        "hw": hw,
        "norm_weight_bits": args.norm_weight_bits,
        "eta_min": args.eta_min,
        "eta_max": args.eta_max,
        "n_fit_rows": len(rows),
        "sources": list(dict.fromkeys(r["source"] for r in rows)),
        "preset_filter": args.preset,
        "sim_physics": phys,
        "eta": eta_to_dict(eta_fit),
        "rmse_ms": round(rmse, 6),
        "mae_ms": round(mae, 6),
        "rmse_ms_baseline_default_eta": round(rmse0, 6),
        "fit_optimizer": ls_meta,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"Fitted η on {len(rows)} rows  (hw={hw!r})")
    print(f"  RMSE: {rmse:.4f} ms/tok  (baseline default η RMSE: {rmse0:.4f})")
    print(f"  MAE:  {mae:.4f} ms/tok")
    print("  η:")
    for k in ETA_FIELD_NAMES:
        print(f"    {k:12s} = {getattr(eta_fit, k):.6f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
