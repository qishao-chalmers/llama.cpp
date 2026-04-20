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

Holdout / generalization (optional):
  --holdout-batch 32          exclude B=32 from fit; report test metrics on held-in file
  --holdout-kv q4_0          exclude that KV type (resolved via resolve_kv_quant_key)
  --test-json other.json      after fit on kv_timing_json, evaluate on all rows of other.json

For leave-one-batch-out, fitted fixed_overhead keys exist only for training batches.
Evaluation uses --lobo-fixed-fallback (default): missing batch keys are filled with the
mean of fitted fixed overheads so the test is not trivially broken by zero overhead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Callable, Optional

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
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("rows", []))


def _metrics_ms_per_tok(meas: list[float], pred: list[float]) -> dict[str, float]:
    if not meas:
        return {"n": 0.0, "mae": float("nan"), "rmse": float("nan"), "mean_pred_over_meas": float("nan")}
    e = [p - m for p, m in zip(pred, meas)]
    mae = float(sum(abs(x) for x in e) / len(e))
    rmse = float(math.sqrt(sum(x * x for x in e) / len(e)))
    ratios = [p / m for p, m in zip(pred, meas) if m > 0]
    mean_r = float(sum(ratios) / len(ratios)) if ratios else float("nan")
    return {"n": float(len(meas)), "mae": mae, "rmse": rmse, "mean_pred_over_meas": mean_r}


def collect_points(
    rows: list[dict[str, Any]],
    cat: dict[str, Any],
    gguf_tb_for_row: Callable[[dict[str, Any]], Optional[dict[str, int]]],
    *,
    allow_no_gguf: bool,
    hw: str,
    kv_attn_byte_mode: str,
    attn_impl: str,
    fa_bc: int,
    attn_naive_spill: bool,
) -> list[dict[str, Any]]:
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
        if tb is None and not allow_no_gguf:
            continue

        feats = spm.extract_decode_features(
            model,
            batch_size=B,
            ctx_len=spm.mid_ctx(r),
            hw_name=str(hw),
            kv_quant_key=str(kv_key),
            weight_bpe=float(wb) / 8.0,
            norm_bpe=16.0 / 8.0,
            gguf_tensor_bytes=tb,
            kv_group_size=None,
            kv_asym=False,
            attn_impl=str(attn_impl),
            fa_bc=int(fa_bc),
            attn_naive_spill=bool(attn_naive_spill),
            kv_attn_byte_mode=str(kv_attn_byte_mode),
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
    return points


def cal_with_fixed_batch_fallback(cal: dict[str, Any], eval_points: list[dict[str, Any]]) -> dict[str, Any]:
    """Fill missing fixed_overhead_ms_by_batch keys using mean of present values (LOBO eval)."""

    fix = dict(cal.get("fixed_overhead_ms_by_batch") or {})
    vals = [float(v) for v in fix.values()]
    mean_fix = float(sum(vals) / len(vals)) if vals else 0.0
    need_bs = {int(p["B"]) for p in eval_points}
    out_fix = dict(fix)
    for b in sorted(need_bs):
        sk = str(int(b))
        if sk not in out_fix and int(b) not in out_fix:
            out_fix[sk] = mean_fix
    c = dict(cal)
    c["fixed_overhead_ms_by_batch"] = {str(k): float(v) for k, v in out_fix.items()}
    return c


def predict_points(cal: dict[str, Any], points: list[dict[str, Any]], hw: str) -> list[float]:
    out: list[float] = []
    for p in points:
        out.append(
            float(
                spm.predict_decode_ms_per_tok(
                    p["feats"],
                    batch_size=int(p["B"]),
                    hw_name=str(hw),
                    kv_quant_key=str(p["kv"]),
                    cal=cal,
                )
            )
        )
    return out


def fit_stream_calib(
    train_points: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    gguf_cache_n: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (calibration dict, fit_meta with optimizer info)."""

    if len(train_points) < 3:
        raise ValueError(
            f"Need >=3 training rows; got {len(train_points)}. Relax holdout or add data."
        )

    batches = sorted({int(p["B"]) for p in train_points})
    b_index = {b: i for i, b in enumerate(batches)}

    def unpack_x(x: np.ndarray, tp: list[dict[str, Any]]) -> dict[str, Any]:
        fixed = {int(b): float(x[b_index[b]]) for b in batches}
        eta_w = float(x[len(batches)])
        eta_c = float(x[len(batches) + 1])
        tail_ms = float(x[len(batches) + 2])
        alpha_wm = float(x[len(batches) + 3])
        alpha_mk = float(x[len(batches) + 4])
        kv_keys = sorted({p["kv"] for p in tp})
        eta_kv = {}
        base = len(batches) + 5
        for i, k in enumerate(kv_keys):
            eta_kv[k] = float(x[base + i])
        return {
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

    kv_keys = sorted({p["kv"] for p in train_points})
    n = len(batches) + 5 + len(kv_keys)

    def residuals(x: np.ndarray) -> np.ndarray:
        cal = unpack_x(x, train_points)
        out = []
        for p in train_points:
            pred = spm.predict_decode_ms_per_tok(
                p["feats"],
                batch_size=int(p["B"]),
                hw_name=str(args.hw),
                kv_quant_key=str(p["kv"]),
                cal=cal,
            )
            out.append(pred - float(p["meas"]))
        return np.array(out, dtype=float)

    x0 = np.zeros((n,), dtype=float)
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

    cal_out = unpack_x(x_fit, train_points)
    resv = residuals(x_fit)
    rmse = float(math.sqrt(float(np.mean(resv**2))))
    mae = float(np.mean(np.abs(resv)))

    cal_out["fit"] = {
        "sources": [os.path.relpath(args.kv_timing_json, os.getcwd())],
        "n_rows": int(len(train_points)),
        "n_rows_train": int(len(train_points)),
        "n_params": int(n),
        "rows_per_param": round(float(len(train_points)) / float(max(1, n)), 6),
        "rmse_ms_per_tok": round(rmse, 6),
        "mae_ms_per_tok": round(mae, 6),
        "rmse_ms_per_tok_train": round(rmse, 6),
        "mae_ms_per_tok_train": round(mae, 6),
        "optimizer": opt_meta,
        "attn_impl": str(args.attn_impl),
        "fa_bc": int(args.fa_bc),
        "attn_naive_spill": bool(args.attn_naive_spill),
        "gguf_cache_n": int(gguf_cache_n),
        "allow_no_gguf": bool(args.allow_no_gguf),
    }
    fit_meta = {"rmse": rmse, "mae": mae, "opt_meta": opt_meta, "n": n, "x_fit": x_fit}
    return cal_out, fit_meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kv_timing_json", help="kv_timing*.json (training source)")
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--hw", default="h100-sxm", choices=list(sim.HARDWARE_PRESETS.keys()))
    ap.add_argument("-o", "--out", required=True, help="Write fitted calibration JSON here")
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
    ap.add_argument(
        "--holdout-batch",
        action="append",
        type=int,
        default=[],
        help="Batch size(s) to exclude from training (repeatable). Evaluated as test if still in file.",
    )
    ap.add_argument(
        "--holdout-kv",
        action="append",
        default=[],
        help="KV type(s) to exclude from training, e.g. f16 q8_0 q4_0 (repeatable). Resolved via resolve_kv_quant_key.",
    )
    ap.add_argument(
        "--test-json",
        action="append",
        default=[],
        help="Additional kv_timing JSON: evaluate fitted cal on all its rows (cross-profile test). Repeatable.",
    )
    ap.add_argument(
        "--lobo-fixed-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For held-out batch sizes, fill missing fixed_overhead with mean of fitted batch overheads (default: on).",
    )
    args = ap.parse_args()

    rows = _load_rows(args.kv_timing_json)
    cat = sim.load_structure_catalog(args.catalog)

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

    all_points = collect_points(
        rows,
        cat,
        gguf_tb_for_row,
        allow_no_gguf=bool(args.allow_no_gguf),
        hw=str(args.hw),
        kv_attn_byte_mode=str(args.kv_attn_byte_mode),
        attn_impl=str(args.attn_impl),
        fa_bc=int(args.fa_bc),
        attn_naive_spill=bool(args.attn_naive_spill),
    )
    if not all_points:
        raise ValueError("No usable rows in training file.")

    holdout_bs = {int(b) for b in (args.holdout_batch or [])}
    holdout_kv_canon = {sim.resolve_kv_quant_key(str(x)) for x in (args.holdout_kv or [])}

    def is_train(p: dict[str, Any]) -> bool:
        if int(p["B"]) in holdout_bs:
            return False
        if holdout_kv_canon and str(p["kv"]) in holdout_kv_canon:
            return False
        return True

    train_points = [p for p in all_points if is_train(p)]
    test_in_file = [p for p in all_points if not is_train(p)]

    cal_out, _ = fit_stream_calib(train_points, args, gguf_cache_n=len(gguf_cache))

    holdout_desc: dict[str, Any] = {
        "holdout_batches": sorted(holdout_bs),
        "holdout_kv_resolved": sorted(holdout_kv_canon),
        "n_train": len(train_points),
        "n_test_same_file": len(test_in_file),
    }

    # Train metrics (in-sample on training subset)
    pred_tr = predict_points(cal_out, train_points, str(args.hw))
    meas_tr = [float(p["meas"]) for p in train_points]
    m_tr = _metrics_ms_per_tok(meas_tr, pred_tr)

    # Same-file holdout
    m_hold: Optional[dict[str, float]] = None
    if test_in_file:
        cal_eval = (
            cal_with_fixed_batch_fallback(cal_out, test_in_file)
            if args.lobo_fixed_fallback
            else cal_out
        )
        pred_h = predict_points(cal_eval, test_in_file, str(args.hw))
        meas_h = [float(p["meas"]) for p in test_in_file]
        m_hold = _metrics_ms_per_tok(meas_h, pred_h)

    # Cross-profile test JSONs
    test_reports: list[dict[str, Any]] = []
    for tpath in args.test_json:
        trows = _load_rows(tpath)
        tpoints = collect_points(
            trows,
            cat,
            gguf_tb_for_row,
            allow_no_gguf=bool(args.allow_no_gguf),
            hw=str(args.hw),
            kv_attn_byte_mode=str(args.kv_attn_byte_mode),
            attn_impl=str(args.attn_impl),
            fa_bc=int(args.fa_bc),
            attn_naive_spill=bool(args.attn_naive_spill),
        )
        cal_eval = (
            cal_with_fixed_batch_fallback(cal_out, tpoints) if args.lobo_fixed_fallback else cal_out
        )
        pred_t = predict_points(cal_eval, tpoints, str(args.hw))
        meas_t = [float(p["meas"]) for p in tpoints]
        mt = _metrics_ms_per_tok(meas_t, pred_t)
        test_reports.append({"path": tpath, **mt})
        print(
            f"[test] {tpath}: n={int(mt['n'])}  MAE={mt['mae']:.4f}  RMSE={mt['rmse']:.4f}  "
            f"mean(pred/meas)={mt['mean_pred_over_meas']:.3f}"
        )

    cal_out["fit"]["holdout"] = holdout_desc
    cal_out["fit"]["validation"] = {
        "train_mae_ms_per_tok": round(m_tr["mae"], 6),
        "train_rmse_ms_per_tok": round(m_tr["rmse"], 6),
        "train_mean_pred_over_meas": round(m_tr["mean_pred_over_meas"], 6),
        "lobo_fixed_fallback": bool(args.lobo_fixed_fallback),
    }
    if m_hold is not None:
        cal_out["fit"]["validation"]["same_file_holdout"] = {
            "n": int(m_hold["n"]),
            "mae_ms_per_tok": round(m_hold["mae"], 6),
            "rmse_ms_per_tok": round(m_hold["rmse"], 6),
            "mean_pred_over_meas": round(m_hold["mean_pred_over_meas"], 6),
        }
    if test_reports:
        cal_out["fit"]["validation"]["test_json"] = [
            {
                "path": tr["path"],
                "n": int(tr["n"]),
                "mae_ms_per_tok": round(tr["mae"], 6),
                "rmse_ms_per_tok": round(tr["rmse"], 6),
                "mean_pred_over_meas": round(tr["mean_pred_over_meas"], 6),
            }
            for tr in test_reports
        ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cal_out, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.out}")
    print(
        f"[train] n={int(m_tr['n'])}  MAE={m_tr['mae']:.4f}  RMSE={m_tr['rmse']:.4f}  "
        f"mean(pred/meas)={m_tr['mean_pred_over_meas']:.3f}"
    )
    if m_hold is not None:
        print(
            f"[holdout same file] n={int(m_hold['n'])}  MAE={m_hold['mae']:.4f}  "
            f"RMSE={m_hold['rmse']:.4f}  mean(pred/meas)={m_hold['mean_pred_over_meas']:.3f}"
        )
    elif holdout_bs or holdout_kv_canon:
        print("[holdout same file] n=0 (no rows matched holdout filter)")


if __name__ == "__main__":
    main()
