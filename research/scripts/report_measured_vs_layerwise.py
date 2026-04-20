#!/usr/bin/env python3
"""
report_measured_vs_layerwise.py — Table of measured vs layerwise_roofline_sim predictions.

This script does NOT modify layerwise_roofline_sim.py. It imports the simulator and
evaluates the same model for every measured row in a benchmark JSON (e.g. kv_timing_h100.json),
then prints a table of discrepancies.

Typical usage:

  python3 research/scripts/report_measured_vs_layerwise.py \
    --measured-json research/results/qwen3-8b/profile/kv_timing_h100.json \
    --hw h100-sxm \
    --eta-json research/results/qwen3-8b/profile/layerwise_eta_h100.json \
    --sim-physics-json research/results/qwen3-8b/profile/sim_physics_h100_with_weight_scale.json

Optional:
  --calibration-json research/results/qwen3-8b/profile/layerwise_calibration_h100.json
  --only-preset qwen3-8b
  --only-kv q8_0
  --only-weight-tag Q8_0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Optional

# Import the simulator as a library (same directory)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import layerwise_roofline_sim as sim  # noqa: E402


def _load_rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return [r for r in data["rows"] if isinstance(r, dict)]
    raise ValueError(f"Unsupported measured-json format: {path!r}")


def _safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _mid_ctx(r: dict[str, Any]) -> int:
    mid = int(r.get("mid_ctx", 0) or 0)
    if mid > 0:
        return mid
    pl = int(r.get("prompt_len", 0) or 0)
    dl = int(r.get("decode_len", 0) or 0)
    return pl + dl // 2


def _fmt(x: float, width: int = 9) -> str:
    if math.isnan(x) or math.isinf(x):
        return " " * (width - 3) + "nan"
    return f"{x:{width}.4f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--measured-json",
        action="append",
        default=[],
        help="benchmark_kv_timing JSON (kv_timing*.json). Can be passed multiple times.",
    )
    ap.add_argument(
        "--glob-measured",
        default=None,
        metavar="GLOB",
        help="Glob for kv_timing*.json (e.g. 'research/results/**/kv_timing_h100.json').",
    )
    ap.add_argument("--hw", default="h100-sxm", choices=list(sim.HARDWARE_PRESETS.keys()))
    ap.add_argument("--catalog", default=sim._DEFAULT_STRUCT, help="Path to model_structures.json")
    ap.add_argument("--structure-json", default=None, help="Optional structure override JSON")
    ap.add_argument("--eta-json", default=None, help="Eta JSON (fit_layerwise_eta.py output)")
    ap.add_argument(
        "--pipeline-eta-json",
        default=None,
        help="Pipeline eta JSON from fit_pipeline_eta.py (eta_wt, eta_kv, t_floor_ms). "
        "If set, uses predict_ms_per_token_pipeline (separate weight vs KV efficiencies) instead of layerwise simulate.",
    )
    ap.add_argument("--sim-physics-json", default=None, help="sim_physics JSON (weight scaling, etc.)")
    ap.add_argument("--calibration-json", default=None, help="Optional calibration JSON (t_floor_ms, scale)")
    ap.add_argument("--kv-group-size", type=int, default=None)
    ap.add_argument("--kv-asymmetric", action="store_true")
    ap.add_argument("--attn-impl", default="simple", choices=["simple", "flash"])
    ap.add_argument("--fa-bc", type=int, default=128)
    ap.add_argument("--attn-naive-spill", action="store_true")
    ap.add_argument("--kv-attn-byte-mode", default=None, choices=[None, "fp16_equiv_dequant", "storage"])
    ap.add_argument("--attn-time-scale", type=float, default=None)
    ap.add_argument("--attn-time-scale-inv-batch", type=float, default=None)

    ap.add_argument("--only-preset", default=None, help="Filter: model_preset")
    ap.add_argument("--only-kv", default=None, help="Filter: kv_type (e.g. f16, q8_0, q4_0)")
    ap.add_argument("--only-weight-tag", default=None, help="Filter: weight_tag (e.g. Q8_0)")
    ap.add_argument("--only-batch", type=int, default=0, help="Filter: batch_size (0=all)")
    ap.add_argument("--max-rows", type=int, default=0, help="Limit rows printed (0=all)")
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only per-file summary stats (no per-row table).",
    )
    ap.add_argument(
        "--auto-eta",
        action="store_true",
        help="Auto-pick per-file eta-json from the same directory as each measured JSON "
        "(layerwise_eta_<hw>.json for the selected --hw). Overrides global --eta-json when found.",
    )
    ap.add_argument(
        "--skip-missing-eta",
        action="store_true",
        help="When --auto-eta and the per-file eta is missing, skip that measured file instead of "
        "using the global --eta-json fallback.",
    )

    args = ap.parse_args()

    paths: list[str] = []
    for p in args.measured_json:
        if p:
            paths.append(os.path.abspath(os.path.expanduser(p)))
    if args.glob_measured:
        import glob
        paths.extend(sorted(glob.glob(args.glob_measured, recursive=True)))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No measured JSONs provided. Use --measured-json or --glob-measured.")

    global_eta: Optional[sim.Eta] = None
    if args.eta_json:
        eta_base = sim.load_eta_json(args.eta_json)
        global_eta = sim.Eta(
            gemm_comp=eta_base.gemm_comp,
            gemm_bw=eta_base.gemm_bw,
            attn_comp=eta_base.attn_comp,
            attn_bw=eta_base.attn_bw,
            elem_comp=eta_base.elem_comp,
            elem_bw=eta_base.elem_bw,
            kv_bw=eta_base.kv_bw,
        )

    global_pipeline: Optional[dict[str, float]] = None
    if args.pipeline_eta_json:
        ppath = os.path.abspath(os.path.expanduser(args.pipeline_eta_json))
        with open(ppath, encoding="utf-8") as f:
            pj = json.load(f)
        global_pipeline = {
            "eta_wt": float(pj["eta_wt"]),
            "eta_kv": float(pj["eta_kv"]),
            "t_floor_ms": float(pj["t_floor_ms"]),
        }
    phys = sim.resolve_sim_physics(
        args.sim_physics_json,
        kv_attn_byte_mode=args.kv_attn_byte_mode,
        attn_time_scale=args.attn_time_scale,
        attn_time_scale_inv_batch=args.attn_time_scale_inv_batch,
    )
    cal = sim.load_calibration_json(args.calibration_json) if args.calibration_json else None

    model_cache: dict[str, dict[str, Any]] = {}

    def get_model(preset: str) -> dict[str, Any]:
        if preset in model_cache:
            return model_cache[preset]
        pid, m = sim.resolve_model(
            preset=preset,
            structure_json=args.structure_json,
            catalog_path=args.catalog,
        )
        sim.validate_structure(m)
        model_cache[preset] = dict(m)
        return model_cache[preset]

    hdr = f"{'preset':10s} {'wtag':8s} {'kv':6s} {'B':>3s} {'mid':>5s} {'meas':>9s} {'pred':>9s} {'err':>9s} {'pred/meas':>9s}"
    all_errs: list[float] = []
    all_ratios: list[float] = []

    def _eta_for_file(measured_path: str) -> tuple[Optional[sim.Eta], Optional[str]]:
        """Return (eta, eta_path_used)."""
        if not args.auto_eta:
            return global_eta, args.eta_json
        d = os.path.dirname(os.path.abspath(measured_path))
        # Common naming in this repo: layerwise_eta_h100.json (not h100_sxm).
        hw_l = str(args.hw).lower()
        candidates = []
        if "h100" in hw_l:
            candidates.append(os.path.join(d, "layerwise_eta_h100.json"))
        if "a100" in hw_l:
            candidates.append(os.path.join(d, "layerwise_eta_a100_80g.json"))
        candidates.append(os.path.join(d, f"layerwise_eta_{args.hw.replace('-', '_')}.json"))
        # Fallback: any layerwise_eta*.json in the same folder
        try:
            import glob
            candidates.extend(sorted(glob.glob(os.path.join(d, "layerwise_eta*.json"))))
        except Exception:
            pass
        eta_path = next((p for p in candidates if os.path.isfile(p)), None)
        if eta_path:
            e0 = sim.load_eta_json(eta_path)
            e = sim.Eta(
                gemm_comp=e0.gemm_comp,
                gemm_bw=e0.gemm_bw,
                attn_comp=e0.attn_comp,
                attn_bw=e0.attn_bw,
                elem_comp=e0.elem_comp,
                elem_bw=e0.elem_bw,
                kv_bw=e0.kv_bw,
            )
            return e, eta_path
        return global_eta, args.eta_json

    for measured_path in paths:
        eta, eta_used = _eta_for_file(measured_path)
        use_pipeline = global_pipeline is not None
        if not use_pipeline and eta is None:
            print(f"[skip] no eta available for {measured_path!r}. Provide --eta-json/--auto-eta or set --pipeline-eta-json.", file=sys.stderr)
            continue
        if args.auto_eta and eta_used is None:
            # Missing per-file eta, using global fallback (or none)
            d = os.path.dirname(os.path.abspath(measured_path))
            eta_name = "layerwise_eta_h100.json" if "h100" in str(args.hw).lower() else f"layerwise_eta_{args.hw.replace('-', '_')}.json"
            eta_path = os.path.join(d, eta_name)
            cmd = (
                f"python3 research/scripts/fit_layerwise_eta.py {os.path.relpath(measured_path, os.getcwd())} "
                f"--hw {args.hw} -o {os.path.relpath(eta_path, os.getcwd())}"
            )
            if args.skip_missing_eta:
                print(f"[skip] missing per-file eta: {eta_path!r}  (generate with: {cmd})", file=sys.stderr)
                continue
            else:
                print(f"[warn] missing per-file eta: {eta_path!r}  (generate with: {cmd})", file=sys.stderr)
        rows = _load_rows(measured_path)
        if not rows:
            print(f"[skip] no rows: {measured_path!r}", file=sys.stderr)
            continue

        errs: list[float] = []
        ratios: list[float] = []
        by_slice_errs: dict[tuple[str, str, str], list[float]] = defaultdict(list)

        if not args.summary_only:
            print("=" * 110)
            print("MEASURED vs PREDICTED (layerwise_roofline_sim)", flush=True)
            print(f"measured_json={measured_path!r}", flush=True)
            if use_pipeline:
                print(f"hw={args.hw!r}  pipeline_eta_json={args.pipeline_eta_json!r}  sim_physics_json={args.sim_physics_json!r}  "
                      f"calibration_json={args.calibration_json!r}", flush=True)
            else:
                print(f"hw={args.hw!r}  eta_json={eta_used!r}  sim_physics_json={args.sim_physics_json!r}  "
                      f"calibration_json={args.calibration_json!r}", flush=True)
            print("=" * 110)
            print(hdr)
            print("-" * len(hdr))

        shown = 0
        for r in rows:
            mp = str(r.get("model_preset", "")).strip()
            if not mp:
                continue
            if args.only_preset and mp != args.only_preset:
                continue

            wtag = str(r.get("weight_tag", "")).strip()
            if args.only_weight_tag and wtag != args.only_weight_tag:
                continue

            kv_cli = str(r.get("kv_type", "")).strip()
            if not kv_cli:
                continue
            if args.only_kv and kv_cli != args.only_kv:
                continue

            B = int(r.get("batch_size", 0) or 0)
            if B <= 0:
                continue
            if args.only_batch and B != int(args.only_batch):
                continue

            mid = _mid_ctx(r)
            if mid <= 0:
                continue

            meas = _safe_float(r.get("measured_ms", float("nan")))
            if math.isnan(meas) or meas <= 0:
                continue

            w_bits = _safe_float(r.get("weight_bits", float("nan")))
            if math.isnan(w_bits) or w_bits <= 0:
                w_bits = sim.tag_to_weight_bits(wtag) if wtag else 16.0
            wbpe = w_bits / 8.0

            kv_key = sim.resolve_kv_quant_key(kv_cli)
            m = get_model(mp)
            if use_pipeline:
                pred_raw = sim.predict_ms_per_token_pipeline(
                    m,
                    batch_size=B,
                    ctx_len=mid,
                    hw_name=str(args.hw),
                    eta_wt=float(global_pipeline["eta_wt"]),
                    eta_kv=float(global_pipeline["eta_kv"]),
                    t_floor_ms=float(global_pipeline["t_floor_ms"]),
                    weight_bits=w_bits,
                    norm_weight_bits=16.0,
                    kv_quant_key=kv_key,
                    kv_group_size=args.kv_group_size,
                    kv_asym=bool(args.kv_asymmetric),
                    gguf_tensor_bytes=None,
                )
                pred = float(pred_raw)
            else:
                total_s, _events = sim.simulate_decode_step(
                    m,
                    batch_size=B,
                    ctx_len=mid,
                    hw=dict(sim.HARDWARE_PRESETS[args.hw]),
                    eta=eta,
                    weight_bpe=wbpe,
                    norm_bpe=16.0 / 8.0,
                    kv_quant_key=kv_key,
                    kv_group_size=args.kv_group_size,
                    kv_asym=bool(args.kv_asymmetric),
                    gguf_tensor_bytes=None,
                    attn_impl=args.attn_impl,
                    fa_bc=int(args.fa_bc),
                    attn_naive_spill=bool(args.attn_naive_spill),
                    kv_attn_byte_mode=str(phys["kv_attn_byte_mode"]),
                    attn_time_scale=float(phys["attn_time_scale"]),
                    attn_time_scale_inv_batch=float(phys["attn_time_scale_inv_batch"]),
                    attn_scale_by_batch=phys.get("attn_scale_by_batch"),
                    weight_tag=wtag,
                    weight_time_scale_by_tag=phys.get("weight_time_scale_by_tag"),
                )
                pred_raw = (total_s * 1000.0) / float(B)
                pred = pred_raw
            if cal is not None:
                pred = sim.calibrated_ms_per_token(
                    float(pred_raw),
                    B,
                    float(cal["t_floor_ms"]),
                    float(cal["scale"]),
                )

            err = pred - meas
            ratio = pred / meas
            if not args.summary_only:
                print(f"{mp:10s} {wtag:8s} {kv_cli:6s} {B:3d} {mid:5d} "
                      f"{_fmt(meas)} {_fmt(pred)} {_fmt(err)} {ratio:9.3f}")

            errs.append(err)
            ratios.append(ratio)
            by_slice_errs[(mp, wtag, kv_cli)].append(err)
            shown += 1
            if args.max_rows and shown >= int(args.max_rows):
                break

        if not errs:
            print(f"[skip] no comparable rows after filtering: {measured_path!r}", file=sys.stderr)
            continue

        mae = sum(abs(e) for e in errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        mean_ratio = sum(ratios) / len(ratios)
        all_errs.extend(errs)
        all_ratios.extend(ratios)

        base = os.path.relpath(measured_path, os.getcwd())
        if not args.summary_only:
            print("-" * len(hdr))
        print(f"[summary] {base}: n={len(errs)}  MAE={mae:.4f}  RMSE={rmse:.4f}  mean(pred/meas)={mean_ratio:.3f}")
        if not args.summary_only:
            print("  Per-slice summary (mean err, rmse, n):")
            for key in sorted(by_slice_errs.keys()):
                e = by_slice_errs[key]
                m_err = sum(e) / len(e)
                r_err = math.sqrt(sum(x * x for x in e) / len(e))
                mp, wtag, kv = key
                print(f"    {mp:10s} {wtag:8s} {kv:6s}  mean_err={m_err:+.4f}  rmse={r_err:.4f}  n={len(e)}")

    if all_errs:
        mae = sum(abs(e) for e in all_errs) / len(all_errs)
        rmse = math.sqrt(sum(e * e for e in all_errs) / len(all_errs))
        mean_ratio = sum(all_ratios) / len(all_ratios)
        print(f"\n[overall] files={len(paths)}  n={len(all_errs)}  MAE={mae:.4f}  RMSE={rmse:.4f}  mean(pred/meas)={mean_ratio:.3f}")
    else:
        print("No comparable rows across all files.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

