#!/usr/bin/env python3
"""calibrate_from_kv_timing.py — Recompute roofline + calibration from benchmark_kv_timing JSON.

Cluster runs often have roofline_ms=null (no perf_model on node). This script loads
kv_timing_*.json files, fills roofline using the same formula as benchmark_kv_timing.py,
computes calib_factor = measured_ms / roofline_ms, and writes:

  - A merged analysis JSON (all rows + stats)
  - Per-model preset JSON snippets for roofline_layer.py --calibration-json
    (decode_ms_per_token_fp16_baseline from measured f16 KV, Q8_0 weight, batch 1)

Usage:
    python3 research/scripts/calibrate_from_kv_timing.py
    python3 research/scripts/calibrate_from_kv_timing.py \\
        research/results/qwen3-8b/profile/kv_timing_h100.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

try:
    from perf_model import MODEL_PRESETS, HARDWARE_PRESETS
    from perf_model import kv_bytes_per_token, weight_bytes, model_step_flops
except ImportError as e:
    print("ERROR: perf_model.py required.", e, file=sys.stderr)
    sys.exit(1)

# Match benchmark_kv_timing.KV_EFFECTIVE_BPE + roofline_ms
KV_EFFECTIVE_BPE = {
    "f16":   2.0,
    "q8_0":  1.0 + 2.0 / 32,
    "q5_0":  0.625 + 2.0 / 32,
    "q4_0":  0.5 + 2.0 / 32,
    "q4_1":  0.5 + 4.0 / 32,
}

# benchmark JSON uses "h100"; perf_model uses "h100-sxm"
HW_ALIASES = {
    "h100": "h100-sxm",
    "h100-sxm": "h100-sxm",
    "a100": "a100-80g",
    "gh200": "gh200",
    "mn5-h100": "mn5-h100",
}


def roofline_ms(
    model_name: str,
    hw_name: str,
    weight_bits: float,
    kv_type: str,
    ctx_len: int,
    batch_size: int = 1,
) -> float:
    hw_key = HW_ALIASES.get(hw_name, hw_name)
    model = dict(MODEL_PRESETS.get(model_name, {}))
    if not model:
        return float("nan")
    hw = HARDWARE_PRESETS.get(hw_key)
    if hw is None:
        return float("nan")

    model["weight_bits"] = weight_bits
    bpe = KV_EFFECTIVE_BPE.get(kv_type, 2.0)
    bw_ratio = bpe / 2.0
    fp16_kv_bpt = kv_bytes_per_token("fp16", model)
    kv_bpt = fp16_kv_bpt * bw_ratio
    w_bytes = weight_bytes(model)
    sflops = model_step_flops(model, ctx_len)
    bw_eff = hw["memory_bw_gbps"] * 1e9 * hw["efficiency"]
    comp_eff = hw["compute_tflops"] * 1e12 * hw["efficiency"]
    t_mem_step = (w_bytes + kv_bpt * ctx_len * batch_size) / bw_eff
    t_compute_step = 2.0 * sflops * batch_size / comp_eff
    t_step = max(t_mem_step, t_compute_step)
    return t_step / batch_size * 1000.0


def enrich_rows(rows: list[dict], hw_from_file: str) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        mid = r.get("mid_ctx")
        if mid is None:
            mid = r.get("prompt_len", 0) + r.get("decode_len", 0) // 2
        ms = roofline_ms(
            r["model_preset"],
            hw_from_file,
            float(r["weight_bits"]),
            r["kv_type"],
            int(mid),
            int(r["batch_size"]),
        )
        r["roofline_ms_recomputed"] = None if math.isnan(ms) else ms
        meas = r.get("measured_ms")
        if meas is not None and not math.isnan(ms) and ms > 0:
            r["calib_factor"] = float(meas) / ms
        else:
            r["calib_factor"] = None
        out.append(r)
    return out


def default_input_globs(repo_root: str) -> list[str]:
    pattern = os.path.join(repo_root, "research", "results", "*", "profile", "kv_timing*.json")
    return sorted(glob.glob(pattern))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "inputs",
        nargs="*",
        help="kv_timing JSON paths (default: research/results/*/profile/kv_timing*.json)",
    )
    p.add_argument(
        "--repo-root",
        default=os.path.join(_SCRIPT_DIR, "..", ".."),
        help="Repository root for default glob",
    )
    p.add_argument(
        "--out-analysis",
        default=None,
        help="Write merged analysis JSON (default: research/results/kv_calibration_analysis.json)",
    )
    p.add_argument(
        "--out-calibration-dir",
        default=None,
        help="Write per-preset calibration JSON for roofline_layer (default: research/data/kv_calibration)",
    )
    args = p.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    paths = [os.path.abspath(x) for x in args.inputs] if args.inputs else default_input_globs(repo_root)
    if not paths:
        print("No input JSON files found.", file=sys.stderr)
        sys.exit(1)

    out_analysis = args.out_analysis or os.path.join(
        repo_root, "research", "results", "kv_calibration_analysis.json"
    )
    out_cal_dir = args.out_calibration_dir or os.path.join(
        repo_root, "research", "data", "kv_calibration"
    )

    all_rows: list[dict[str, Any]] = []
    file_meta: list[dict[str, Any]] = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        hw = data.get("hardware", "h100")
        rows = enrich_rows(data.get("rows", []), hw)
        rel = os.path.relpath(path, repo_root)
        for r in rows:
            r["_source_file"] = rel
        all_rows.extend(rows)
        file_meta.append({"path": rel, "hardware": hw, "n_rows": len(rows)})

    # Stats: calib_factor by (model_preset, kv_type, batch_size) for Q8_0 weight
    calib_vals = [
        r["calib_factor"]
        for r in all_rows
        if r.get("calib_factor") is not None
        and r.get("weight_tag") == "Q8_0"
    ]
    calib_mean = sum(calib_vals) / len(calib_vals) if calib_vals else None
    calib_std = (
        math.sqrt(sum((x - calib_mean) ** 2 for x in calib_vals) / len(calib_vals))
        if calib_vals and len(calib_vals) > 1
        else None
    )

    # fp16 baseline decode ms/tok for roofline_layer: f16 KV, Q8_0, B=1
    baselines: dict[str, float] = {}
    for r in all_rows:
        if (
            r.get("kv_type") == "f16"
            and r.get("batch_size") == 1
            and r.get("weight_tag") == "Q8_0"
        ):
            key = r["model_preset"]
            if key not in baselines:
                baselines[key] = float(r["measured_ms"])

    analysis = {
        "_comment": (
            "Merged KV timing + recomputed roofline (perf_model). "
            "calib_factor = measured_ms / roofline_ms_recomputed."
        ),
        "sources": file_meta,
        "calib_factor_stats_q8_weights": {
            "n": len(calib_vals),
            "mean": calib_mean,
            "std": calib_std,
        },
        "fp16_kv_decode_ms_per_token_b1_q8_weights": baselines,
        "rows": all_rows,
    }

    os.makedirs(os.path.dirname(out_analysis) or ".", exist_ok=True)
    with open(out_analysis, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"Wrote analysis → {out_analysis}")

    os.makedirs(out_cal_dir, exist_ok=True)
    hw_tag = "h100"
    for preset, ms in baselines.items():
        safe = preset.replace(".", "_")
        cal_path = os.path.join(out_cal_dir, f"decode_fp16_kv_{hw_tag}_{safe}.json")
        payload = {
            "_comment": (
                f"Use with roofline_layer.py --calibration-json. "
                f"Measured decode ms/token on cluster (KV=f16, Q8_0 weights, batch=1). "
                f"Scenario: prompt/decode from benchmark_kv_timing sweep."
            ),
            "hardware": hw_tag,
            "model_preset": preset,
            "decode_ms_per_token_fp16_baseline": ms,
        }
        # Attach first matching row for scenario IDs
        for r in all_rows:
            if (
                r.get("model_preset") == preset
                and r.get("kv_type") == "f16"
                and r.get("batch_size") == 1
                and r.get("weight_tag") == "Q8_0"
            ):
                payload["prompt_len"] = r.get("prompt_len")
                payload["decode_len"] = r.get("decode_len")
                payload["mid_ctx"] = r.get("mid_ctx")
                r0 = r.get("roofline_ms_recomputed")
                if r0 is not None:
                    payload["roofline_decode_ms_per_token_fp16_recomputed"] = r0
                    payload["decode_scale_measured_over_roofline"] = ms / r0
                break
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote calibration → {cal_path}")

    # Short table to stdout
    print("\n── Summary: measured decode ms/tok (Q8_0 weights) ──")
    keys = sorted({(r["model_preset"], r["kv_type"], r["batch_size"]) for r in all_rows})
    for mp, kv, bs in keys:
        sub = [
            r
            for r in all_rows
            if r["model_preset"] == mp
            and r["kv_type"] == kv
            and r["batch_size"] == bs
            and r.get("weight_tag") == "Q8_0"
        ]
        if not sub:
            continue
        r = sub[0]
        meas = r.get("measured_ms")
        roof = r.get("roofline_ms_recomputed")
        cf = r.get("calib_factor")
        rs = f"{roof:.3f}" if roof is not None else "n/a"
        cs = f"{cf:.3f}" if cf is not None else "n/a"
        print(f"  {mp:14s}  kv={kv:5s}  B={bs}  meas={meas:.3f} ms/tok  roof={rs}  calib={cs}")

    print(f"\nGlobal mean calib_factor (Q8_0 weights, all kv×B): {calib_mean:.4f}" if calib_mean else "")


if __name__ == "__main__":
    main()
