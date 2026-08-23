#!/usr/bin/env python3
"""Compare roofline_layer analytic decode ms/tok to real cluster timings in kv_timing JSON.

Loads benchmark_kv_timing profiles (research/results/.../profile/kv_timing_h100.json), runs the
uncorrected roofline model, and records:

  roofline_ms_per_token     — analytic (this repo's formulas, at row's mid_ctx)
  measured_ms_per_token     — cluster (row mean or per-bucket match_ctx)
  calibration_scale         — measured / roofline

Output:
  JSON  — research/results/roofline_real_calibration_report.json
  Table — printed to stdout: per-row breakdown, per-group stats, per-(model,B) median

Usage:
  python3 research/scripts/build_roofline_calibration_report.py \\
      --scan-root research/results --glob '**/profile/kv_timing_h100.json'

  # With per-bucket comparison (uses decode_buckets from each row):
  python3 research/scripts/build_roofline_calibration_report.py --bucket-policy match_ctx

  # Filter to one model:
  python3 research/scripts/build_roofline_calibration_report.py --filter-model qwen3-8b
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
import types
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _SCRIPT_DIR)

import roofline_layer as rl  # noqa: E402

NATIVE_KV_TO_ROOFLINE = {
    "f16":  "fp16",
    "q8_0": "int8_ch",
    "q4_0": "int4_ch",
}


# ── Args builder ─────────────────────────────────────────────────────────────

def _make_args(row: dict, hw_name: str, bucket_policy: str) -> Tuple[dict, Any]:
    """Build model dict and args namespace matching roofline_layer defaults for this row."""
    model = dict(rl.MODEL_PRESETS[row["model_preset"]])
    args = types.SimpleNamespace()
    args.compute_eff          = 0.7
    args.mem_eff              = 0.85
    args.attn_eff             = 0.85
    args.weight_bw_fraction   = 1.0
    args.act_bw_fraction      = 1.0
    args.attn_bw_fraction     = 1.0
    args.attn_bw              = None
    args.flash_attn           = True
    args.kv_group_size        = 64
    args.padding_efficiency   = 1.0
    args.weight_bpw           = None
    args.weight_bpw_profile   = None
    args.main_gguf_quant      = row["weight_tag"]
    args.draft_gguf_quant     = None
    args.draft_weight_bpw     = None
    args.draft_weight_bpw_profile = None
    args.calibration_bucket_policy      = bucket_policy
    args.calibration_relax_decode_len   = False
    args.n_prompt    = int(row["prompt_len"])
    args.n_decode    = int(row["decode_len"])
    args.batch_size  = int(row["batch_size"])
    args.model       = row["model_preset"]
    args.hw          = hw_name
    kv_native = str(row.get("kv_type", "f16")).lower()
    args.kv_quant = NATIVE_KV_TO_ROOFLINE.get(kv_native, "fp16")
    rl._apply_gguf_weight_quant_args(args)
    return model, args


# ── Per-bucket comparison ─────────────────────────────────────────────────────

def _bucket_rows_for_profile_row(
        row: dict, model: dict, hw: dict, args: Any,
) -> List[Dict[str, Any]]:
    """For each decode_bucket in the row, compute roofline at bucket's ctx_mid."""
    buckets = row.get("decode_buckets")
    if not isinstance(buckets, list) or not buckets:
        return []
    out = []
    for b in buckets:
        ctx = int(b.get("ctx_mid", 0))
        if ctx <= 0:
            continue
        meas = float(b.get("ms_per_token", 0.0))
        if meas <= 0:
            continue
        roof = rl.roofline_decode_ms_at_ctx(model, hw, args, ctx)
        scale = (meas / roof) if roof > 0 else None
        out.append({
            "model_preset":  row.get("model_preset"),
            "weight_tag":    row.get("weight_tag"),
            "kv_type":       row.get("kv_type"),
            "batch_size":    row.get("batch_size"),
            "prompt_len":    row.get("prompt_len"),
            "decode_len":    row.get("decode_len"),
            "ctx_mid":       ctx,
            "decode_from":   b.get("decode_from"),
            "decode_to":     b.get("decode_to"),
            "measured_ms_per_token":     round(meas, 6),
            "roofline_ms_per_token":     round(roof, 6) if roof else None,
            "calibration_scale":         round(scale, 6) if scale else None,
        })
    return out


# ── Terminal table helpers ────────────────────────────────────────────────────

def _fmt(v, fmt=".3f", na="  n/a"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return na
    return format(v, fmt)


def _print_row_table(rows_out: List[Dict], title: str = "Per-row calibration"):
    W = 120
    print(f"\n{'═'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")
    hdr = (f"  {'model':>12} {'weight':>7} {'kv':>6} {'B':>3}"
           f" {'prompt':>7} {'decode':>7} {'mid_ctx':>8}"
           f" {'measured':>10} {'roofline':>10} {'scale':>7}  note")
    print(hdr)
    print(f"{'─'*W}")
    prev_group = None
    for r in rows_out:
        group = (r.get("model_preset"), r.get("batch_size"))
        if prev_group is not None and group != prev_group:
            print()
        prev_group = group
        sc = r.get("calibration_scale_measured_over_roofline")
        sc_s = _fmt(sc, ".3f") + "×" if sc else "  n/a"
        roof = r.get("roofline_ms_per_token")
        note = r.get("hardware_warning") or r.get("measured_note", "")
        print(f"  {r.get('model_preset','?'):>12} {r.get('weight_tag','?'):>7}"
              f" {r.get('kv_type','?'):>6} {r.get('batch_size',0):>3}"
              f" {r.get('prompt_len',0):>7} {r.get('decode_len',0):>7}"
              f" {r.get('mid_ctx', r.get('prompt_len',0) + r.get('decode_len',0)//2):>8}"
              f" {_fmt(r.get('measured_ms_per_token'), '.4f'):>10}"
              f" {_fmt(roof, '.4f'):>10}"
              f" {sc_s:>8}  {str(note)[:35]}")
    print(f"{'═'*W}")


def _print_group_table(rows_out: List[Dict]):
    """Per-(model, weight_tag, kv_type, batch_size) breakdown."""
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows_out:
        sc = r.get("calibration_scale_measured_over_roofline")
        if sc is None:
            continue
        key = (r.get("model_preset","?"), r.get("weight_tag","?"),
                r.get("kv_type","?"), r.get("batch_size",0))
        groups[key].append(sc)

    W = 100
    print(f"\n{'═'*W}")
    print(f"  Calibration scale breakdown  (measured / roofline)")
    print(f"{'─'*W}")
    print(f"  {'model':>12} {'weight':>7} {'kv':>6} {'B':>3}"
          f"  {'n':>3} {'mean':>7} {'median':>8} {'min':>7} {'max':>7}  assessment")
    print(f"{'─'*W}")
    prev_model = None
    for key in sorted(groups.keys()):
        model_p, weight, kv, B = key
        if prev_model is not None and model_p != prev_model:
            print()
        prev_model = model_p
        vals = groups[key]
        n = len(vals)
        mean = statistics.mean(vals)
        med  = statistics.median(vals)
        mn   = min(vals)
        mx   = max(vals)
        # Assessment: is the roofline tracking physics correctly?
        spread = (mx - mn) / mean if mean > 0 else 0
        if spread < 0.15:
            assess = "consistent (scalar calib OK)"
        elif spread < 0.35:
            assess = "moderate spread"
        else:
            assess = "high spread — model mismatch"
        print(f"  {model_p:>12} {weight:>7} {kv:>6} {B:>3}"
              f"  {n:>3} {mean:>7.3f}× {med:>7.3f}× {mn:>7.3f}× {mx:>7.3f}×  {assess}")
    print(f"{'═'*W}")


def _print_model_batch_summary(rows_out: List[Dict]):
    """Per-(model, batch_size) median scale — the recommended calibration factor."""
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows_out:
        sc = r.get("calibration_scale_measured_over_roofline")
        if sc is None:
            continue
        key = (r.get("model_preset","?"), r.get("batch_size",0))
        groups[key].append(sc)

    W = 80
    print(f"\n{'─'*W}")
    print(f"  Recommended calibration scale  (per model × batch_size)")
    print(f"  Apply this to roofline decode ms/tok to match cluster timing.")
    print(f"{'─'*W}")
    print(f"  {'model':>14} {'B':>3}  {'n':>3} {'median':>8} {'mean':>8}"
          f" {'stdev':>7}  interpretation")
    print(f"{'─'*W}")
    prev = None
    for key in sorted(groups.keys()):
        model_p, B = key
        if prev and model_p != prev:
            print()
        prev = model_p
        vals = groups[key]
        n    = len(vals)
        med  = statistics.median(vals)
        mean = statistics.mean(vals)
        std  = statistics.pstdev(vals)
        if B == 1:
            note = "weight+overhead bound; fixed-cost dominated"
        elif B <= 4:
            note = "transition regime; weight ≈ KV"
        else:
            note = "KV-bound; roofline more predictive"
        print(f"  {model_p:>14} {B:>3}  {n:>3} {med:>8.3f}× {mean:>8.3f}×"
              f" {std:>7.3f}×  {note}")
    print(f"{'─'*W}")


def _print_bucket_summary(bucket_rows: List[Dict]):
    """Per-(model, batch_size) bucket-level scale stats — shows KV-trend tracking."""
    if not bucket_rows:
        return
    groups: Dict[Tuple, List] = defaultdict(list)
    for b in bucket_rows:
        sc = b.get("calibration_scale")
        if sc is None:
            continue
        key = (b.get("model_preset","?"), b.get("weight_tag","?"),
                b.get("kv_type","?"), b.get("batch_size",0))
        groups[key].append((b["ctx_mid"], sc))

    W = 100
    print(f"\n{'─'*W}")
    print(f"  Per-bucket calibration scale  (each 64-token decode slice, roofline at bucket ctx_mid)")
    print(f"  Low stdev → roofline correctly tracks KV growth trend.")
    print(f"{'─'*W}")
    print(f"  {'model':>12} {'weight':>7} {'kv':>6} {'B':>3}"
          f"  {'n_bkt':>6} {'mean':>7} {'min_sc':>8} {'max_sc':>8}"
          f" {'ctx_range':>14}  KV trend")
    print(f"{'─'*W}")
    prev = None
    for key in sorted(groups.keys()):
        model_p, weight, kv, B = key
        if prev and model_p != prev:
            print()
        prev = model_p
        data = groups[key]
        scales = [s for _, s in data]
        ctxs   = [c for c, _ in data]
        n      = len(scales)
        mean   = statistics.mean(scales)
        mn, mx = min(scales), max(scales)
        ctx_lo, ctx_hi = min(ctxs), max(ctxs)
        # Does scale grow/shrink with ctx? (linear fit slope)
        try:
            slope = (scales[-1] - scales[0]) / (ctxs[-1] - ctxs[0]) * 1000
        except ZeroDivisionError:
            slope = 0.0
        trend = f"{slope:+.4f}×/ktok" if abs(slope) > 0.001 else "flat"
        print(f"  {model_p:>12} {weight:>7} {kv:>6} {B:>3}"
              f"  {n:>6} {mean:>7.3f}× {mn:>7.3f}× {mx:>7.3f}×"
              f" {ctx_lo:>5}–{ctx_hi:<7}  {trend}")
    print(f"{'─'*W}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan-root", default=os.path.join(_REPO_ROOT, "research", "results"))
    ap.add_argument("--glob", default="**/profile/kv_timing_h100.json")
    ap.add_argument("--profiles", nargs="*", metavar="PATH",
                    help="Explicit kv_timing JSON paths (skip glob if set)")
    ap.add_argument("--hw", default="h100-sxm",
                    help="Hardware preset (must match cluster, default h100-sxm)")
    ap.add_argument(
        "--bucket-policy", choices=("mean", "match_ctx"), default="mean",
        help="How to pick measured ms/tok per row: mean=full-decode average, "
             "match_ctx=decode_bucket closest to roofline avg_ctx (default: mean)")
    ap.add_argument("-o", "--output",
                    default=os.path.join(_REPO_ROOT, "research", "results",
                                         "roofline_real_calibration_report.json"))
    ap.add_argument("--filter-model", default=None,
                    help="Only rows whose model_preset equals this")
    ap.add_argument("--no-buckets", action="store_true",
                    help="Skip per-bucket comparison (faster when profiles are large)")
    args_cli = ap.parse_args()

    # Collect profile paths
    if args_cli.profiles:
        paths = [os.path.expanduser(p) for p in args_cli.profiles]
    else:
        paths = sorted(glob.glob(
            os.path.join(args_cli.scan_root, args_cli.glob), recursive=True))
    if not paths:
        print("No profiles matched.", file=sys.stderr)
        sys.exit(1)

    hw = dict(rl.HARDWARE_PRESETS[args_cli.hw])
    rows_out: List[Dict[str, Any]]   = []
    bucket_rows: List[Dict[str, Any]] = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            cal = json.load(f)
        hw_warn = (
            None if rl._calibration_hw_matches(cal.get("hardware"), args_cli.hw)
            else f"profile_hw={cal.get('hardware')!r} vs --hw={args_cli.hw!r}")

        for idx, row in enumerate(cal.get("rows") or []):
            if not isinstance(row, dict):
                continue
            mp = row.get("model_preset")
            if mp not in rl.MODEL_PRESETS:
                continue
            if args_cli.filter_model and mp != args_cli.filter_model:
                continue

            model, args = _make_args(row, args_cli.hw, args_cli.bucket_policy)

            # Row-level roofline (at mid_ctx = n_prompt + n_decode//2)
            roof_ms = rl.roofline_decode_ms_per_token(model, hw, args)

            # Row-level measured (mean or match_ctx)
            meas_ms, meas_note = rl._measured_ms_from_kv_timing_row(row, args)

            scale = (meas_ms / roof_ms) if (roof_ms and roof_ms > 0) else None

            rows_out.append({
                "profile":     os.path.relpath(path, _REPO_ROOT),
                "row_index":   idx,
                "model_preset": mp,
                "weight_tag":  row.get("weight_tag"),
                "kv_type":     row.get("kv_type"),
                "batch_size":  row.get("batch_size"),
                "prompt_len":  row.get("prompt_len"),
                "decode_len":  row.get("decode_len"),
                "mid_ctx":     row.get("mid_ctx",
                                        int(row.get("prompt_len",0)) +
                                        int(row.get("decode_len",0)) // 2),
                "measured_ms_per_token": meas_ms,
                "measured_note": meas_note,
                "roofline_ms_per_token":
                    round(roof_ms, 6) if roof_ms else None,
                "calibration_scale_measured_over_roofline":
                    round(scale, 6) if scale is not None else None,
                "hardware_warning": hw_warn,
            })

            # Per-bucket comparison
            if not args_cli.no_buckets:
                br = _bucket_rows_for_profile_row(row, model, hw, args)
                bucket_rows.extend(br)

    if not rows_out:
        print("No rows processed.", file=sys.stderr)
        sys.exit(1)

    # ── Print tables ─────────────────────────────────────────────────────────
    _print_row_table(rows_out, title=(
        f"Per-row calibration  [hw={args_cli.hw}  policy={args_cli.bucket_policy}]"))
    _print_group_table(rows_out)
    _print_model_batch_summary(rows_out)
    if bucket_rows:
        _print_bucket_summary(bucket_rows)

    # ── Build JSON report ─────────────────────────────────────────────────────
    scales = [r["calibration_scale_measured_over_roofline"] for r in rows_out
              if r["calibration_scale_measured_over_roofline"] is not None]

    # Per-(model, B) median calibration
    per_model_batch: Dict[str, Dict] = {}
    from collections import defaultdict
    mb_groups: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows_out:
        sc = r.get("calibration_scale_measured_over_roofline")
        if sc is None:
            continue
        mb_groups[(r["model_preset"], r["batch_size"])].append(sc)
    for (mp, B), vals in sorted(mb_groups.items()):
        per_model_batch[f"{mp} B{B}"] = {
            "n": len(vals),
            "median_scale": round(statistics.median(vals), 6),
            "mean_scale":   round(statistics.mean(vals), 6),
            "stdev_scale":  round(statistics.pstdev(vals), 6),
        }

    # Per-(model, weight_tag, kv_type, B) exact calibration scale
    # Key: "{model} {weight_tag} {kv_type} B{batch_size}"
    # Value: exact measured/roofline ratio (1 row per combination in current profiles)
    # Also includes bucket_mean_scale when decode_buckets are present.
    per_combination: Dict[str, Dict] = {}
    combo_bucket_scales: Dict[Tuple, List[float]] = defaultdict(list)
    for br in bucket_rows:
        sc = br.get("calibration_scale")
        if sc is None:
            continue
        combo = (br["model_preset"], br["weight_tag"], br["kv_type"], br["batch_size"])
        combo_bucket_scales[combo].append(sc)

    for r in rows_out:
        sc = r.get("calibration_scale_measured_over_roofline")
        combo = (r["model_preset"], r["weight_tag"], r["kv_type"], r["batch_size"])
        key = f"{r['model_preset']} {r['weight_tag']} {r['kv_type']} B{r['batch_size']}"
        bscales = combo_bucket_scales.get(combo, [])
        per_combination[key] = {
            "row_scale": round(sc, 6) if sc is not None else None,
            "bucket_mean_scale": round(statistics.mean(bscales), 6) if bscales else None,
            "bucket_min_scale":  round(min(bscales), 6) if bscales else None,
            "bucket_max_scale":  round(max(bscales), 6) if bscales else None,
            "measured_ms": r["measured_ms_per_token"],
            "roofline_ms": r["roofline_ms_per_token"],
        }

    report = {
        "_comment": (
            "roofline vs real timing. calibration_scale = measured_ms / roofline_ms_per_token. "
            "Use per_combination_calibration for exact per-(model,weight,kv,B) correction; "
            "use per_model_batch_calibration for coarser (model,B) median when weight_tag not in measured set."
        ),
        "hw_preset": args_cli.hw,
        "bucket_policy": args_cli.bucket_policy,
        "n_rows": len(rows_out),
        "n_bucket_rows": len(bucket_rows),
        "calibration_scale_mean":  round(statistics.mean(scales),  6) if scales else None,
        "calibration_scale_stdev": round(statistics.pstdev(scales), 6) if len(scales) > 1 else 0.0,
        "per_combination_calibration": per_combination,
        "per_model_batch_calibration": per_model_batch,
        "rows": rows_out,
        "bucket_rows": bucket_rows,
    }

    out_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    print(f"\nWrote {out_path}  ({len(rows_out)} rows, {len(bucket_rows)} bucket rows)")
    if report["calibration_scale_mean"] is not None:
        print(f"  global mean(scale) = {report['calibration_scale_mean']:.4f}  "
              f"stdev = {report['calibration_scale_stdev']:.4f}")
    print(f"\n  Per-(model, B) median calibration scales:")
    for k, v in per_model_batch.items():
        print(f"    {k:<20}  median={v['median_scale']:.4f}×  "
              f"mean={v['mean_scale']:.4f}×  stdev={v['stdev_scale']:.4f}×")


if __name__ == "__main__":
    main()
