#!/usr/bin/env python3
"""Fit decode-only stream_perf_model calibration JSON from kv_timing*.json rows.

Fits:
  fixed_overhead_ms_by_batch[B]  (5 values for B in rows)
  eta_weight_bw, eta_compute
  eta_kv_bw_by_kv[kv_quant_key] for kv keys seen
  tail_ms (scalar)
  alpha_wm, alpha_mk (partial overlap knobs; see stream_perf_model.py)

Objective: minimize sum (pred_ms_per_tok - measured_ms)^2

Requires GGUF paths on rows (model_path) unless --allow-no-gguf (uniform bpw fallback).

Many fitted parameters can trade off (fixed overhead, tail, etas, alphas); the JSON
includes fit.n_params and fit.rows_per_param for a quick identifiability sanity check.
Prefer holdouts by batch/KV/context before interpreting etas as physical efficiencies.
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

import layerwise_roofline_sim as sim  # noqa: E402
import stream_perf_model as spm  # noqa: E402
from gguf_layerwise_weights import load_gguf_tensor_n_bytes  # noqa: E402

try:
    from scipy.optimize import least_squares  # type: ignore

    _HAS_SCIPY = True
except Exception:
    least_squares = None  # type: ignore
    _HAS_SCIPY = False


def _load_rows(path: str) -> list[dict[str, Any]]:
    data = json.load(open(path, encoding="utf-8"))
    return list(data.get("rows", []))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kv_timing_json", help="kv_timing*.json")
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--hw", default="h100-sxm", choices=list(sim.HARDWARE_PRESETS.keys()))
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument(
        "--gguf-dir",
        action="append",
        default=[],
        help="Directory to search for GGUF by basename (repeatable). Default includes /home/qshao/Project/Fun/models",
    )
    ap.add_argument("--kv-attn-byte-mode", choices=["fp16_equiv_dequant", "storage"], default="fp16_equiv_dequant")
    ap.add_argument("--attn-impl", default="simple", choices=["simple", "flash"])
    ap.add_argument("--fa-bc", type=int, default=128)
    ap.add_argument("--attn-naive-spill", action="store_true")
    ap.add_argument("--allow-no-gguf", action="store_true", help="Allow missing/unreadable GGUF (uniform bpw fallback)")
    args = ap.parse_args()

    rows = _load_rows(args.kv_timing_json)
    cat = sim.load_structure_catalog(args.catalog)

    # Discover batch sizes + kv keys in this file
    batches = sorted({int(r["batch_size"]) for r in rows if int(r.get("batch_size", 0) or 0) > 0})
    if not batches:
        raise ValueError("No batch sizes found")

    gguf_cache: dict[str, dict[str, int]] = {}

    def gguf_tb_for_row(r: dict[str, Any]) -> Optional[dict[str, int]]:
        mp = r.get("model_path")
        if not mp:
            return None
        base = os.path.basename(str(mp))
        if not base.endswith(".gguf"):
            return None

        candidates: list[str] = []
        p0 = os.path.abspath(os.path.expanduser(str(mp)))
        candidates.append(p0)
        search_dirs = list(args.gguf_dir)
        search_dirs.append("/home/qshao/Project/Fun/models")
        for d in search_dirs:
            if not d:
                continue
            candidates.append(os.path.join(os.path.abspath(os.path.expanduser(d)), base))

        p = next((c for c in candidates if os.path.isfile(c)), None)
        if p is None:
            return None
        if p not in gguf_cache:
            gguf_cache[p] = load_gguf_tensor_n_bytes(p)
        return gguf_cache[p]

    points: list[dict[str, Any]] = []
    for r in rows:
        preset = str(r.get("model_preset", "")).strip()
        if not preset or preset not in cat["presets"]:
            continue
        meas = float(r.get("measured_ms", 0.0))
        if meas <= 0:
            continue
        B = int(r.get("batch_size", 0) or 0)
        if B <= 0:
            continue
        kv_key = sim.resolve_kv_quant_key(str(r.get("kv_type", "f16")))
        model = dict(cat["presets"][preset])
        wtag = str(r.get("weight_tag", "") or "").strip()
        wb, _ = sim.resolve_weight_bits(model, wtag if wtag else None, float(r.get("weight_bits", 16.0)))
        tb = gguf_tb_for_row(r)
        if tb is None and not args.allow_no_gguf:
            continue

        feats = spm.extract_decode_features(
            model,
            batch_size=B,
            ctx_len=spm.mid_ctx(r),
            hw_name=str(args.hw),
            kv_quant_key=str(kv_key),
            weight_bpe=float(wb) / 8.0,
            norm_bpe=16.0 / 8.0,
            gguf_tensor_bytes=tb,
            kv_group_size=None,
            kv_asym=False,
            attn_impl=str(args.attn_impl),
            fa_bc=int(args.fa_bc),
            attn_naive_spill=bool(args.attn_naive_spill),
            kv_attn_byte_mode=str(args.kv_attn_byte_mode),
        )
        points.append(
            {
                "preset": preset,
                "B": B,
                "kv": str(kv_key),
                "meas": meas,
                "feats": feats,
            }
        )

    if len(points) < 3:
        raise ValueError(
            f"Need >=3 usable rows; got {len(points)}. "
            f"Check GGUF paths on rows or pass --allow-no-gguf."
        )

    # Parameter packing:
    # x = [fixed_ms[B0..], eta_w_logit?, use direct bounded instead]
    # We'll use scipy with bounds if available; else naive grid on etas + linear least squares for fixed.

    b_index = {b: i for i, b in enumerate(batches)}

    def unpack_x(x: np.ndarray) -> dict[str, Any]:
        fixed = {int(b): float(x[b_index[b]]) for b in batches}
        eta_w = float(x[len(batches)])
        eta_c = float(x[len(batches) + 1])
        tail_ms = float(x[len(batches) + 2])
        alpha_wm = float(x[len(batches) + 3])
        alpha_mk = float(x[len(batches) + 4])
        # remaining: etas for kv keys in stable order
        kv_keys = sorted({p["kv"] for p in points})
        eta_kv = {}
        base = len(batches) + 5
        for i, k in enumerate(kv_keys):
            eta_kv[k] = float(x[base + i])
        cal = {
            "schema_version": 1,
            "model": "stream_decode_v1",
            "hw": str(args.hw),
            "kv_attn_byte_mode": str(args.kv_attn_byte_mode),
            "fixed_overhead_ms_by_batch": {str(k): float(v) for k, v in fixed.items()},
            "tail_ms": float(tail_ms),
            "eta_weight_bw": float(eta_w),
            "eta_compute": float(eta_c),
            "eta_kv_bw_by_kv": {k: float(v) for k, v in eta_kv.items()},
            "alpha_wm": float(alpha_wm),
            "alpha_mk": float(alpha_mk),
        }
        return cal

    kv_keys = sorted({p["kv"] for p in points})
    n = len(batches) + 5 + len(kv_keys)

    def residuals(x: np.ndarray) -> np.ndarray:
        cal = unpack_x(x)
        out = []
        for p in points:
            pred = spm.predict_decode_ms_per_tok(
                p["feats"],
                batch_size=int(p["B"]),
                hw_name=str(args.hw),
                kv_quant_key=str(p["kv"]),
                cal=cal,
            )
            out.append(pred - float(p["meas"]))
        return np.array(out, dtype=float)

    # Initial guess
    x0 = np.zeros((n,), dtype=float)
    # fixed overhead ~0.6ms for all B initially
    for b in batches:
        x0[b_index[b]] = 0.6
    x0[len(batches)] = 0.40
    x0[len(batches) + 1] = 0.35
    x0[len(batches) + 2] = 0.0
    x0[len(batches) + 3] = 0.0
    x0[len(batches) + 4] = 0.0
    base = len(batches) + 5
    for i in range(len(kv_keys)):
        x0[base + i] = 0.30

    lo = np.zeros((n,), dtype=float)
    hi = np.zeros((n,), dtype=float)
    for b in batches:
        lo[b_index[b]] = 0.0
        hi[b_index[b]] = 4.0
    lo[len(batches)] = 0.05
    hi[len(batches)] = 1.0
    lo[len(batches) + 1] = 0.05
    hi[len(batches) + 1] = 1.0
    lo[len(batches) + 2] = 0.0
    hi[len(batches) + 2] = 2.0
    lo[len(batches) + 3] = 0.0
    hi[len(batches) + 3] = 2.0
    lo[len(batches) + 4] = 0.0
    hi[len(batches) + 4] = 2.0
    for i in range(len(kv_keys)):
        lo[base + i] = 0.05
        hi[base + i] = 1.0

    if _HAS_SCIPY and least_squares is not None:
        res = least_squares(residuals, x0, bounds=(lo, hi), max_nfev=400)
        x_fit = res.x
        opt_meta = {"optimizer": "scipy.least_squares", "success": bool(res.success), "message": str(res.message)}
    else:
        # Fallback: crude coordinate descent on x0
        x_fit = np.clip(x0, lo, hi)
        best = float(np.sum(residuals(x_fit) ** 2))
        for _ in range(30):
            cost_before = best
            for i in range(len(x_fit)):
                best_local = float(x_fit[i])
                best_cost = best
                for t in np.linspace(lo[i], hi[i], 18):
                    x_fit[i] = t
                    c = float(np.sum(residuals(x_fit) ** 2))
                    if c < best_cost:
                        best_cost = c
                        best_local = t
                x_fit[i] = best_local
                best = best_cost
            if best >= cost_before - 1e-18:
                break
        opt_meta = {"optimizer": "coordinate_descent_grid", "note": "Install scipy for better fits."}

    cal_out = unpack_x(x_fit)
    resv = residuals(x_fit)
    rmse = float(math.sqrt(float(np.mean(resv**2))))
    mae = float(np.mean(np.abs(resv)))

    cal_out["fit"] = {
        "sources": [os.path.relpath(args.kv_timing_json, os.getcwd())],
        "n_rows": int(len(points)),
        "n_params": int(n),
        "rows_per_param": round(float(len(points)) / float(max(1, n)), 6),
        "rmse_ms_per_tok": round(rmse, 6),
        "mae_ms_per_tok": round(mae, 6),
        "optimizer": opt_meta,
        "attn_impl": str(args.attn_impl),
        "fa_bc": int(args.fa_bc),
        "attn_naive_spill": bool(args.attn_naive_spill),
        "gguf_cache_n": int(len(gguf_cache)),
        "allow_no_gguf": bool(args.allow_no_gguf),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cal_out, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.out}")
    print(f"fit rows={len(points)}  RMSE={rmse:.4f} ms/tok  MAE={mae:.4f} ms/tok")


if __name__ == "__main__":
    main()
