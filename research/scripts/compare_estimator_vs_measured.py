#!/usr/bin/env python3
"""Compare analytic roofline, 2-param and 3-param calibration vs measured decode ms/tok.

Reads benchmark_kv_timing JSON files, recomputes roofline_ms, and fits two models:

  2-param (per slice):  ms = T_floor/B + scale*(wt_ms + kv_ms)
  3-param (per slice):  ms = T_floor/B + alpha*wt_ms + beta*kv_ms

Both models use the same per-(model, weight_tag, kv_type) slices.  Within a single
slice, wt_ms ∝ 1/B (collinear with T_floor term), so 2-param and 3-param give the
same per-slice fit.  The 3-param only shows improvement in the global pooled fit
where different weight/KV quants give genuinely different wt_ms and kv_ms values.

Global pooled fit (one per model, all weight×kv combinations):
  alpha: weight-BW efficiency correction (hardware property, not quant-specific)
  beta:  KV-BW efficiency correction     (hardware property, not quant-specific)
  kv_ms uses fp16-equivalent effective bytes (KV_DEQUANT_OVERHEAD) because native
  GPU quants (q8_0, q4_0) are ~8% slower than fp16 despite fewer storage bytes.

Trust metrics:
  - Raw roofline: |meas - roof|, ratio meas/roof
  - Calibrated (in-sample LS): residuals on the same points used to fit
  - Per-slice LOOCV when n >= 3/4 (within-slice generalisation)
  - Cross-quant CV (global): leave-one-(weight,kv)-slice-out; tests generalisation

Example::

    python3 compare_estimator_vs_measured.py \\
        ../results/qwen3-8b/profile/kv_timing_h100.json \\
        ../results/qwen3-14b/profile/kv_timing_h100.json

    # All available profiles:
    python3 compare_estimator_vs_measured.py --glob-results ../results
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np

from benchmark_kv_timing import roofline_ms, roofline_decompose_ms

DEFAULT_HW = "h100-sxm"


def load_rows(path: str) -> tuple[list[dict], dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, {}
    return data.get("rows", []), data


def mid_ctx(r: dict) -> int:
    pl = int(r.get("prompt_len", 0))
    dl = int(r.get("decode_len", 0))
    return int(r.get("mid_ctx", pl + dl // 2))


def fit_two_param(inv_b: np.ndarray, roof: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([inv_b, roof])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def predict(t_floor: float, scale: float, inv_b: float, roof: float) -> float:
    return t_floor * inv_b + scale * roof


def loocv_rmse(inv_b: np.ndarray, roof: np.ndarray, y: np.ndarray) -> float | None:
    n = len(y)
    if n < 3:
        return None
    errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        tf, sc = fit_two_param(inv_b[mask], roof[mask], y[mask])
        pred = predict(tf, sc, inv_b[i], roof[i])
        errs.append(pred - y[i])
    return float(math.sqrt(np.mean(np.array(errs) ** 2)))


# ── Three-parameter model ─────────────────────────────────────────────────────
#
# ms_per_tok = T_floor/B  +  alpha * wt_ms_per_tok  +  beta * kv_ms_per_tok
#
# where wt_ms_per_tok = w_bytes/(bw_eff*B)*1000  (decreases with B — weight BW amortised)
#   and kv_ms_per_tok = kv_bpt*ctx/bw_eff*1000   (constant in B — KV not amortised)
#
# alpha corrects weight-BW efficiency (improves with B due to GEMV→GEMM tiling).
# beta  corrects KV-BW efficiency  (roughly constant across B — flash-attn bound).
# Separating them removes the U-shaped residual of the single-scale 2-param fit.

def fit_three_param(
    inv_b: np.ndarray, x_wt: np.ndarray, x_kv: np.ndarray, y: np.ndarray
) -> tuple[float, float, float]:
    X = np.column_stack([inv_b, x_wt, x_kv])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def predict_3p(
    t_floor: float, alpha: float, beta: float,
    inv_b: float, x_wt: float, x_kv: float,
) -> float:
    return t_floor * inv_b + alpha * x_wt + beta * x_kv


def loocv_rmse_3p(
    inv_b: np.ndarray, x_wt: np.ndarray, x_kv: np.ndarray, y: np.ndarray
) -> float | None:
    """Leave-one-out CV for 3-param model; needs n>=4 (3 params + 1 dof)."""
    n = len(y)
    if n < 4:
        return None
    errs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        tf, al, be = fit_three_param(inv_b[mask], x_wt[mask], x_kv[mask], y[mask])
        pred = predict_3p(tf, al, be, inv_b[i], x_wt[i], x_kv[i])
        errs.append(pred - y[i])
    return float(math.sqrt(np.mean(np.array(errs) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths",
        nargs="*",
        help="benchmark JSON files (default: all **/kv_timing*.json under research/results)",
    )
    ap.add_argument("--hw", default=DEFAULT_HW, help="Hardware key for roofline_ms()")
    ap.add_argument(
        "--glob-results",
        default=None,
        metavar="DIR",
        help="If no paths given, glob DIR/**/kv_timing*.json",
    )
    args = ap.parse_args()

    if args.paths:
        paths = sorted(args.paths)
    else:
        base = args.glob_results or os.path.join(_SCRIPT_DIR, "../results")
        paths = sorted(glob.glob(os.path.join(base, "**", "kv_timing*.json"), recursive=True))

    if not paths:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    # Collect all rows with valid roofline
    all_points: list[dict] = []
    for path in paths:
        rows, _ = load_rows(path)
        for r in rows:
            if not isinstance(r, dict):
                continue
            mp = r.get("model_preset")
            if not mp:
                continue
            meas = float(r.get("measured_ms", 0))
            if meas <= 0:
                continue
            wbits = float(r.get("weight_bits", 16))
            kv = str(r.get("kv_type", "f16"))
            B = int(r.get("batch_size", 0))
            if B <= 0:
                continue
            roof = roofline_ms(str(mp), args.hw, wbits, kv, mid_ctx(r), B)
            if math.isnan(roof) or roof <= 0:
                continue
            wt_ms, kv_ms = roofline_decompose_ms(str(mp), args.hw, wbits, kv, mid_ctx(r), B)
            all_points.append(
                {
                    "file": os.path.relpath(path, _SCRIPT_DIR),
                    "model_preset": str(mp),
                    "weight_tag": str(r.get("weight_tag", "")),
                    "kv_type": kv,
                    "batch_size": B,
                    "prompt_len": int(r.get("prompt_len", 0)),
                    "decode_len": int(r.get("decode_len", 0)),
                    "measured_ms": meas,
                    "roofline_ms": roof,
                    "wt_ms": wt_ms,          # weight component (decreases with B): W_bytes/(bw*B)
                    "wt_ms_base": wt_ms * B, # weight time at B=1 (W_bytes/bw); for delta search
                    "kv_ms": kv_ms,          # KV component (constant per token)
                }
            )

    # Group by (model_preset, weight_tag, kv_type) — same prompt/decode in these files
    from collections import defaultdict

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for p in all_points:
        key = (p["model_preset"], p["weight_tag"], p["kv_type"])
        groups[key].append(p)

    print("=" * 100)
    print("ESTIMATOR vs MEASURED — detailed comparison")
    print(f"roofline_ms(..., hw={args.hw!r})  |  calibration: meas ≈ t_floor_ms/B + scale × roofline_ms")
    print(f"Total valid points (roofline OK): {len(all_points)}  |  JSON files: {len(paths)}")
    print("=" * 100)

    raw_mae_all: list[float] = []
    raw_rmse_all: list[float] = []
    cal_mae_all: list[float] = []
    cal_rmse_all: list[float] = []
    loocv_all: list[float] = []

    loocv_3p_all: list[float] = []
    cal_mae_3p_all: list[float] = []
    cal_rmse_3p_all: list[float] = []

    for key in sorted(groups.keys()):
        mp, wt, kv = key
        pts = groups[key]
        n = len(pts)
        inv_b = np.array([1.0 / p["batch_size"] for p in pts], dtype=float)
        roof  = np.array([p["roofline_ms"] for p in pts], dtype=float)
        x_wt  = np.array([p["wt_ms"]       for p in pts], dtype=float)
        x_kv  = np.array([p["kv_ms"]       for p in pts], dtype=float)
        y     = np.array([p["measured_ms"]  for p in pts], dtype=float)

        # Raw roofline errors
        raw_err = y - roof
        raw_mae = float(np.mean(np.abs(raw_err)))
        raw_rmse = float(math.sqrt(np.mean(raw_err**2)))
        raw_ratio_mean = float(np.mean(y / roof))

        # KV fraction at each B (for context)
        kv_frac_mean = float(np.mean(x_kv / (x_wt + x_kv + 1e-12)))

        print(f"\n--- Slice: model={mp}  weight={wt}  kv={kv}  (n={n}) ---")
        print(
            f"  Raw roofline:  MAE={raw_mae:.4f} ms/tok  RMSE={raw_rmse:.4f}  "
            f"mean(meas/roof)={raw_ratio_mean:.3f}  mean_kv_frac={kv_frac_mean:.2f}"
        )
        raw_mae_all.append(raw_mae)
        raw_rmse_all.append(raw_rmse)

        if n < 2:
            print("  Calibrated:   need n>=2 to fit; skip.")
            continue

        # ── 2-parameter fit: ms = T_floor/B + scale*(wt+kv) ─────────────────
        t_floor, scale = fit_two_param(inv_b, roof, y)
        pred_2p = t_floor * inv_b + scale * roof
        cal_err_2p = pred_2p - y
        cal_mae_2p = float(np.mean(np.abs(cal_err_2p)))
        cal_rmse_2p = float(math.sqrt(np.mean(cal_err_2p**2)))
        loocv_2p = loocv_rmse(inv_b, roof, y)

        print(
            f"  2-param:       t_floor_ms={t_floor:+.4f}  scale={scale:.4f}  "
            f"MAE={cal_mae_2p:.4f}  RMSE={cal_rmse_2p:.4f}"
            + (f"  LOOCV={loocv_2p:.4f}" if loocv_2p is not None else "  LOOCV=n<3")
        )
        if loocv_2p is not None:
            loocv_all.append(loocv_2p)
        cal_mae_all.append(cal_mae_2p)
        cal_rmse_all.append(cal_rmse_2p)

        # ── 3-parameter fit: ms = T_floor/B + alpha*wt + beta*kv ────────────
        if n >= 3:
            t_floor_3p, alpha, beta = fit_three_param(inv_b, x_wt, x_kv, y)
            pred_3p = np.array([
                predict_3p(t_floor_3p, alpha, beta, inv_b[i], x_wt[i], x_kv[i])
                for i in range(n)], dtype=float)
            cal_err_3p = pred_3p - y
            cal_mae_3p = float(np.mean(np.abs(cal_err_3p)))
            cal_rmse_3p = float(math.sqrt(np.mean(cal_err_3p**2)))
            loocv_3p = loocv_rmse_3p(inv_b, x_wt, x_kv, y)
            # Implied effective BW for each component (reference bw_eff = hw_peak * 0.70)
            bw_note = f"alpha={alpha:.3f}×  beta={beta:.3f}×"
            print(
                f"  3-param:       t_floor_ms={t_floor_3p:+.4f}  {bw_note}  "
                f"MAE={cal_mae_3p:.4f}  RMSE={cal_rmse_3p:.4f}"
                + (f"  LOOCV={loocv_3p:.4f}" if loocv_3p is not None else "  LOOCV=n<4")
            )
            if loocv_3p is not None:
                loocv_3p_all.append(loocv_3p)
                loocv_gain = loocv_2p - loocv_3p if loocv_2p is not None else None
                if loocv_gain is not None:
                    tag = f"  ({loocv_gain:+.4f} vs 2-param)"
                    print(f"             LOOCV gain:{tag}")
            cal_mae_3p_all.append(cal_mae_3p)
            cal_rmse_3p_all.append(cal_rmse_3p)
        else:
            pred_3p = pred_2p  # fallback for per-row table
            t_floor_3p = alpha = beta = float("nan")
            print("  3-param:       n<3 — skip")

        # Per-row table (sort by batch)
        print(f"\n  {'B':>4} {'wt_ms':>7} {'kv_ms':>7} {'meas':>8} {'roof':>8}"
              f" {'2p_pred':>8} {'2p_err':>7} {'3p_pred':>8} {'3p_err':>7}")
        order = np.argsort([p["batch_size"] for p in pts])
        for i in order:
            p   = pts[int(i)]
            m   = float(p["measured_ms"])
            ro  = float(p["roofline_ms"])
            p2  = float(pred_2p[int(i)])
            p3  = float(pred_3p[int(i)])
            wt  = float(p["wt_ms"])
            kv_ = float(p["kv_ms"])
            print(
                f"  {p['batch_size']:4d} {wt:7.4f} {kv_:7.4f} {m:8.4f} {ro:8.4f}"
                f" {p2:8.4f} {p2/m:7.4f} {p3:8.4f} {p3/m:7.4f}"
            )

    print("\n" + "=" * 100)
    print("SUMMARY (per-slice averages — each slice weighted equally, not per-row)")
    if raw_mae_all:
        print(
            f"  Raw roofline:  mean MAE={float(np.mean(raw_mae_all)):.4f}  "
            f"mean RMSE={float(np.mean(raw_rmse_all)):.4f} ms/tok  ({len(raw_mae_all)} slices)"
        )
    if cal_mae_all:
        print(
            f"  2-param cal:   mean MAE={float(np.mean(cal_mae_all)):.4f}  "
            f"mean RMSE={float(np.mean(cal_rmse_all)):.4f} ms/tok  (in-sample LS)"
        )
    if loocv_all:
        print(
            f"  2-param LOOCV: mean RMSE={float(np.mean(loocv_all)):.4f} ms/tok  "
            f"({len(loocv_all)} slices with n>=3)"
        )
    if cal_mae_3p_all:
        print(
            f"  3-param cal:   mean MAE={float(np.mean(cal_mae_3p_all)):.4f}  "
            f"mean RMSE={float(np.mean(cal_rmse_3p_all)):.4f} ms/tok  (in-sample LS)"
        )
    if loocv_3p_all:
        gain = (float(np.mean(loocv_all)) - float(np.mean(loocv_3p_all))) if loocv_all else 0.0
        print(
            f"  3-param LOOCV: mean RMSE={float(np.mean(loocv_3p_all)):.4f} ms/tok  "
            f"({len(loocv_3p_all)} slices with n>=4)  gain vs 2-param: {gain:+.4f} ms"
        )

    # ── Global pooled fit with B-dependent weight efficiency (delta search) ─────
    #
    # The 3-param model has a systematic U-shaped residual: overpredicts at large B,
    # underpredicts at medium B.  Root cause: weight reads in GEMM mode (large B) achieve
    # higher HBM bandwidth than GEMV (B=1) even without becoming compute-bound — better
    # warp utilisation and memory coalescing improve effective BW.
    #
    # Fix: parameterise weight efficiency as alpha / B^delta (delta=0 → original model):
    #
    #   ms = T_floor/B  +  alpha × wt_ms_base / B^(1+delta)  +  beta × kv_ms
    #
    # where wt_ms_base = W_bytes / bw_eff × 1000  (weight time at B=1, no 1/B factor).
    # wt_ms_base / B^(1+delta) is NOT collinear with 1/B for delta≠0, so the extra
    # parameter is identifiable per-slice (not just in the global fit).
    #
    # We grid-search delta ∈ [0, 0.6] and pick the value minimising cross-quant CV RMSE.
    # For each candidate delta the inner fit (T_floor, alpha, beta) is a plain least-squares.

    print("\n" + "═" * 100)
    print("  GLOBAL POOLED FIT WITH B-DEPENDENT WEIGHT EFFICIENCY (delta search)")
    print("  ms_per_tok = T_floor/B  +  alpha × wt_ms_base/B^(1+delta)  +  beta × kv_ms")
    print("  delta=0: original 3-param  |  delta>0: weight reads more efficient at large B")
    print("─" * 100)

    from collections import defaultdict
    by_model: dict[str, list[dict]] = defaultdict(list)
    for p in all_points:
        by_model[p["model_preset"]].append(p)

    DELTA_GRID = [round(d * 0.05, 2) for d in range(0, 13)]   # 0.00 … 0.60

    def _fit_delta(pts: list[dict], delta: float) -> tuple:
        """Fit 3-param model for given delta; return (T_floor, alpha, beta, pred_array)."""
        Bs    = np.array([p["batch_size"]  for p in pts], dtype=float)
        inv_b = 1.0 / Bs
        x_wt  = np.array([p["wt_ms_base"] for p in pts], dtype=float) / Bs ** (1.0 + delta)
        x_kv  = np.array([p["kv_ms"]      for p in pts], dtype=float)
        y     = np.array([p["measured_ms"] for p in pts], dtype=float)
        tf, al, be = fit_three_param(inv_b, x_wt, x_kv, y)
        pred = tf * inv_b + al * x_wt + be * x_kv
        return tf, al, be, pred, x_wt, x_kv, inv_b, y

    def _cross_quant_cv(pts: list[dict], delta: float) -> float | None:
        """Leave-one-(weight,kv)-slice-out CV RMSE for given delta."""
        slice_keys = list({(p["weight_tag"], p["kv_type"]) for p in pts})
        if len(slice_keys) < 2:
            return None
        errs = []
        for sk in slice_keys:
            mask_bool = [(p["weight_tag"], p["kv_type"]) != sk for p in pts]
            train = [p for p, m in zip(pts, mask_bool) if m]
            test  = [p for p, m in zip(pts, mask_bool) if not m]
            if len(train) < 3:
                continue
            tf, al, be, _, _, _, _, _ = _fit_delta(train, delta)
            for p in test:
                B = float(p["batch_size"])
                x_wt_p = p["wt_ms_base"] / B ** (1.0 + delta)
                pred_p = tf / B + al * x_wt_p + be * p["kv_ms"]
                errs.append(pred_p - p["measured_ms"])
        return math.sqrt(float(np.mean(np.array(errs) ** 2))) if errs else None

    for model_key in sorted(by_model.keys()):
        pts = by_model[model_key]
        n   = len(pts)
        if n < 4:
            print(f"\n  {model_key}: n={n} < 4, skip")
            continue

        weight_tags = sorted({p["weight_tag"] for p in pts})
        kv_types    = sorted({p["kv_type"]    for p in pts})
        print(f"\n  {model_key}  (n={n}, "
              f"weights=[{','.join(weight_tags)}], kv=[{','.join(kv_types)}])")

        # Grid search
        best_delta  = 0.0
        best_cv     = float("inf")
        delta_table = []
        for delta in DELTA_GRID:
            tf, al, be, pred, x_wt, x_kv, inv_b, y = _fit_delta(pts, delta)
            rmse_in = float(math.sqrt(np.mean((pred - y) ** 2)))
            cv_rmse = _cross_quant_cv(pts, delta)
            delta_table.append((delta, tf, al, be, rmse_in, cv_rmse))
            if cv_rmse is not None and cv_rmse < best_cv:
                best_cv    = cv_rmse
                best_delta = delta

        # Print delta sweep table
        print(f"    {'delta':>6} {'T_floor':>9} {'alpha':>7} {'beta':>7}"
              f" {'in-RMSE':>9} {'cv-RMSE':>9}  {'<best' if best_delta == 0 else ''}")
        for delta, tf, al, be, rmse_in, cv in delta_table:
            marker = " ◄ best" if delta == best_delta else ""
            cv_s = f"{cv:.4f}" if cv is not None else "  n/a "
            print(f"    {delta:>6.2f} {tf:>9.4f} {al:>7.4f} {be:>7.4f}"
                  f" {rmse_in:>9.4f} {cv_s:>9}{marker}")

        # Best-delta predictions
        tf_b, al_b, be_b, pred_b, x_wt_b, x_kv_b, inv_b_b, y_b = _fit_delta(pts, best_delta)
        print(f"\n    Best delta={best_delta}:  T_floor={tf_b:+.4f} ms  "
              f"alpha={al_b:.4f}×  beta={be_b:.4f}×  cv-RMSE={best_cv:.4f} ms")

        # Per-row table with delta=0 vs best-delta comparison
        Bs_b = np.array([p["batch_size"] for p in pts], dtype=float)
        x_wt0 = np.array([p["wt_ms"] for p in pts], dtype=float)   # delta=0 weight term
        tf0, al0, be0 = fit_three_param(1.0/Bs_b, x_wt0, x_kv_b, y_b)
        pred0 = tf0/Bs_b + al0*x_wt0 + be0*x_kv_b

        print(f"\n    {'weight':>7} {'kv':>6} {'B':>3} {'meas':>8}"
              f" {'3p(d=0)':>9} {'d=0 err':>8} {'best-d':>9} {'bd err':>8}")
        order_g = np.argsort([p["batch_size"] for p in pts])
        prev_wk = None
        for idx in order_g:
            p = pts[int(idx)]
            wk = (p["weight_tag"], p["kv_type"])
            if wk != prev_wk:
                if prev_wk is not None:
                    print()
                prev_wk = wk
            m   = float(p["measured_ms"])
            p0  = float(pred0[int(idx)])
            pb  = float(pred_b[int(idx)])
            print(f"    {p['weight_tag']:>7} {p['kv_type']:>6} {p['batch_size']:>3} {m:>8.4f}"
                  f" {p0:>9.4f} {p0/m:>8.3f}x {pb:>9.4f} {pb/m:>8.3f}x")

    print("\n" + "=" * 100)
    print("\nInterpretation:")
    print("  • delta=0: original 3-param model  (alpha constant, wt_ms = W_bytes/(bw*B))")
    print("  • delta>0: weight reads become more efficient as B grows beyond GEMV regime.")
    print("             alpha(B) = alpha_0 / B^delta, so high-B weight time drops faster than 1/B.")
    print("  • Best delta: minimises cross-quant CV RMSE (predict held-out (weight,kv) slice).")
    print("  • Negative T_floor in delta=0 is a model-form artefact; delta>0 gives physical values.")
    print("  • With only Q8_0 weights in profiles, wt_ms_base is the same across rows at each B,")
    print("    so alpha and T_floor are confounded; alpha is identified via cross-B shape of 1/B^(1+d).")
    print("  • Adding Q2_K / Q4_K_M weight profiles would make alpha fully identifiable from T_floor.")
    print("=" * 100)


if __name__ == "__main__":
    main()
