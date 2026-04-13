#!/usr/bin/env python3
"""Compare roofline_layer analytic decode ms/tok to real cluster timings in kv_timing JSON.

Loads benchmark_kv_timing profiles (research/results/.../profile/kv_timing_h100.json), runs the
uncorrected roofline model via roofline_decode_ms_per_token(), and records:

  roofline_ms_per_token — analytic (this repo’s formulas)
  measured_ms_per_token — cluster (row mean or decode_buckets / match_ctx)
  calibration_scale     — measured / roofline  (apply to roofline to match hardware)

Writes JSON (default: research/results/roofline_real_calibration_report.json).

Example:
  python3 research/scripts/build_roofline_calibration_report.py \\
      --scan-root research/results --glob '**/profile/kv_timing_h100.json'
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
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _SCRIPT_DIR)

import roofline_layer as rl  # noqa: E402

NATIVE_KV_TO_ROOFLINE = {
    "f16":  "fp16",
    "q8_0": "int8_ch",
    "q4_0": "int4_ch",
}


def _make_args_for_row(
        row: dict,
        hw_name: str,
        bucket_policy: str,
        relax_decode_len: bool,
) -> tuple[dict, Any, str]:
    """Return (model_dict, args_namespace, note)."""
    model = dict(rl.MODEL_PRESETS[row["model_preset"]])
    args = types.SimpleNamespace()
    # Defaults aligned with roofline_layer.py main()
    args.compute_eff = 0.7
    args.mem_eff = 0.85
    args.attn_eff = 0.85
    args.weight_bw_fraction = 1.0
    args.act_bw_fraction = 1.0
    args.attn_bw_fraction = 1.0
    args.attn_bw = None
    args.flash_attn = True
    args.kv_group_size = 64
    args.padding_efficiency = 1.0
    args.weight_bpw = None
    args.weight_bpw_profile = None
    args.main_gguf_quant = row["weight_tag"]
    args.draft_gguf_quant = None
    args.draft_weight_bpw = None
    args.draft_weight_bpw_profile = None
    args.calibration_bucket_policy = bucket_policy
    args.calibration_relax_decode_len = relax_decode_len
    args.n_prompt = int(row["prompt_len"])
    args.n_decode = int(row["decode_len"])
    args.batch_size = int(row["batch_size"])
    args.model = row["model_preset"]
    args.hw = hw_name
    kv_native = str(row.get("kv_type", "f16")).lower()
    args.kv_quant = NATIVE_KV_TO_ROOFLINE.get(kv_native, "fp16")

    rl._apply_gguf_weight_quant_args(args)

    note = ""
    if relax_decode_len and args.n_decode != int(row["decode_len"]):
        note = f"synthetic n_decode={args.n_decode} vs row decode_len={row['decode_len']}"

    return model, args, note


def _measured_ms(row: dict, args_like: Any) -> tuple[float, str]:
    """Reuse roofline bucket policy for measured side."""
    return rl._measured_ms_from_kv_timing_row(row, args_like)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan-root", default=os.path.join(_REPO_ROOT, "research", "results"))
    ap.add_argument("--glob", default="**/profile/kv_timing_h100.json")
    ap.add_argument("--hw", default="h100-sxm", help="Must match profile / cluster hardware preset")
    ap.add_argument(
        "--bucket-policy", choices=("mean", "match_ctx"), default="mean",
        help="How to pick measured ms/tok from each row (default mean = row aggregate)")
    ap.add_argument(
        "--relax-decode-len", action="store_true",
        help="Ignored for report rows (each row uses its own decode_len).")
    ap.add_argument(
        "-o", "--output",
        default=os.path.join(_REPO_ROOT, "research", "results", "roofline_real_calibration_report.json"),
        help="Output JSON path")
    ap.add_argument("--filter-model", default=None)
    args_cli = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args_cli.scan_root, args_cli.glob), recursive=True))
    if not paths:
        print("No profiles matched.", file=sys.stderr)
        sys.exit(1)

    hw = dict(rl.HARDWARE_PRESETS[args_cli.hw])
    rows_out: List[Dict[str, Any]] = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            cal = json.load(f)
        for idx, row in enumerate(cal.get("rows") or []):
            if not isinstance(row, dict):
                continue
            if args_cli.filter_model and row.get("model_preset") != args_cli.filter_model:
                continue
            model, args, _note = _make_args_for_row(
                row, args_cli.hw, args_cli.bucket_policy, args_cli.relax_decode_len)
            if not rl._calibration_hw_matches(cal.get("hardware"), args_cli.hw):
                hw_warn = f"profile_hw={cal.get('hardware')} vs --hw={args_cli.hw}"
            else:
                hw_warn = None

            roof_ms = rl.roofline_decode_ms_per_token(model, hw, args)
            meas_ms, meas_note = _measured_ms(row, args)
            scale = (meas_ms / roof_ms) if roof_ms and roof_ms > 0 else float("nan")

            rows_out.append({
                "profile": os.path.relpath(path, _REPO_ROOT),
                "row_index": idx,
                "model_preset": row.get("model_preset"),
                "weight_tag": row.get("weight_tag"),
                "kv_type": row.get("kv_type"),
                "batch_size": row.get("batch_size"),
                "prompt_len": row.get("prompt_len"),
                "decode_len": row.get("decode_len"),
                "mid_ctx": row.get("mid_ctx"),
                "measured_ms_per_token": meas_ms,
                "measured_note": meas_note,
                "roofline_ms_per_token": round(roof_ms, 6) if roof_ms else None,
                "calibration_scale_measured_over_roofline": round(scale, 6)
                if not math.isnan(scale) else None,
                "hardware_warning": hw_warn,
            })

    scales = [
        r["calibration_scale_measured_over_roofline"]
        for r in rows_out
        if r["calibration_scale_measured_over_roofline"] is not None
    ]
    report = {
        "_comment": (
            "roofline vs real timing: calibration_scale = measured_ms / roofline_ms_per_token. "
            "Multiply roofline decode times by this scale to match cluster (for this hw preset)."
        ),
        "hw_preset": args_cli.hw,
        "bucket_policy": args_cli.bucket_policy,
        "n_rows": len(rows_out),
        "calibration_scale_mean": round(statistics.mean(scales), 6) if scales else None,
        "calibration_scale_stdev": round(statistics.pstdev(scales), 6) if len(scales) > 1 else 0.0,
        "rows": rows_out,
    }

    out_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}  ({len(rows_out)} rows)")
    if report["calibration_scale_mean"] is not None:
        print(f"  mean(calibration_scale) = {report['calibration_scale_mean']}")


if __name__ == "__main__":
    main()
