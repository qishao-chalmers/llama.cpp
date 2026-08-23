#!/usr/bin/env python3
"""Fit per-weight-tag GEMM time scale for layerwise_roofline_sim (sim_physics JSON).

Fits a mapping:
  sim_physics["weight_time_scale_by_tag"][weight_tag] = s

where s scales only the GEMM family time in the layerwise decode-step simulation:
  pred_ms/tok ≈ s * gemm_ms + other_ms

This helps capture quant-kernel regime changes that a single global η cannot.

Example:
  python3 research/scripts/fit_sim_physics_weight_scale.py \
    research/results/qwen3-8b-q3km/profile/kv_timing_h100.json \
    --hw h100-sxm \
    --eta-json research/results/qwen3-8b-q3km/profile/layerwise_eta_h100.json \
    -o research/results/qwen3-8b-q3km/profile/sim_physics_h100_with_weight_scale.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from layerwise_roofline_sim import (  # noqa: E402
    Eta,
    HARDWARE_PRESETS,
    load_eta_json,
    load_sim_physics_json,
    resolve_kv_quant_key,
    resolve_sim_physics,
    simulate_decode_step,
    load_structure_catalog,
)


def mid_ctx(r: dict) -> int:
    pl = int(r.get("prompt_len", 0))
    dl = int(r.get("decode_len", 0))
    return int(r.get("mid_ctx", pl + dl // 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kv_timing_json", help="kv_timing*.json (benchmark_kv_timing output)")
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--hw", default="h100-sxm")
    ap.add_argument("--eta-json", default=None, help="η from fit_layerwise_eta.py (recommended)")
    ap.add_argument("--base-sim-physics-json", default=None,
                    help="Base sim_physics JSON to start from (optional).")
    ap.add_argument("-o", "--out", required=True, help="Output sim_physics JSON path")
    ap.add_argument("--min-scale", type=float, default=0.50)
    ap.add_argument("--max-scale", type=float, default=2.00)
    ap.add_argument("--min-rows", type=int, default=2,
                    help="Min rows per weight_tag to fit a scale (else keep 1.0).")
    args = ap.parse_args()

    with open(args.kv_timing_json, encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("rows", []) if isinstance(data, dict) else []
    if not rows:
        raise ValueError(f"No rows found in {args.kv_timing_json!r}")

    cat = load_structure_catalog(args.catalog)
    eta = load_eta_json(args.eta_json) if args.eta_json else Eta()
    base_phys = load_sim_physics_json(args.base_sim_physics_json)
    # Resolve with overrides=None to normalize keys/shape
    phys = resolve_sim_physics(
        args.base_sim_physics_json,
        kv_attn_byte_mode=None,
        attn_time_scale=None,
        attn_time_scale_inv_batch=None,
    )
    # Ensure we keep any other fields from base file
    for k, v in base_phys.items():
        if k not in phys:
            phys[k] = v

    hw = dict(HARDWARE_PRESETS[args.hw])

    # Collect per-tag (gemm_ms, other_ms, meas_ms)
    per_tag = defaultdict(list)
    n_skip = 0
    for r in rows:
        mp = r.get("model_preset")
        if not mp or mp not in cat["presets"]:
            n_skip += 1
            continue
        meas = float(r.get("measured_ms", 0))
        if meas <= 0:
            n_skip += 1
            continue
        wtag = str(r.get("weight_tag", "") or "").strip()
        if not wtag:
            n_skip += 1
            continue
        model = dict(cat["presets"][mp])
        B = int(r.get("batch_size", 0))
        if B <= 0:
            n_skip += 1
            continue
        ctx = mid_ctx(r)
        kv_key = resolve_kv_quant_key(str(r.get("kv_type", "f16")))
        wbits = float(r.get("weight_bits", 16.0))
        total_s, events = simulate_decode_step(
            model,
            batch_size=B,
            ctx_len=ctx,
            hw=hw,
            eta=eta,
            weight_bpe=wbits / 8.0,
            norm_bpe=16.0 / 8.0,
            kv_quant_key=kv_key,
            kv_group_size=None,
            kv_asym=False,
            gguf_tensor_bytes=None,
            attn_impl=str(phys.get("attn_impl", "simple")),
            fa_bc=int(phys.get("fa_bc", 128)),
            attn_naive_spill=bool(phys.get("attn_naive_spill", False)),
            kv_attn_byte_mode=str(phys.get("kv_attn_byte_mode", "fp16_equiv_dequant")),
            attn_time_scale=float(phys.get("attn_time_scale", 1.0)),
            attn_time_scale_inv_batch=float(phys.get("attn_time_scale_inv_batch", 0.0)),
            attn_scale_by_batch=phys.get("attn_scale_by_batch"),
            attn_scale_by_batch_and_kv=phys.get("attn_scale_by_batch_and_kv"),
            weight_tag=wtag,
            weight_time_scale_by_tag=None,  # fit scale on top of raw sim
        )
        ms_step = total_s * 1000.0
        ms_per_tok = ms_step / float(B)
        # Decompose into gemm vs other by summing event families
        gemm_s = sum(e.seconds for e in events if e.family == "gemm")
        other_s = total_s - gemm_s
        gemm_ms = (gemm_s * 1000.0) / float(B)
        other_ms = (other_s * 1000.0) / float(B)
        # Sanity: gemm_ms + other_ms == ms_per_tok (up to FP noise)
        _ = ms_per_tok
        per_tag[wtag].append((gemm_ms, other_ms, meas))

    scales = {}
    for wtag, pts in sorted(per_tag.items()):
        if len(pts) < int(args.min_rows):
            scales[wtag] = 1.0
            continue
        num = 0.0
        den = 0.0
        for gemm_ms, other_ms, meas in pts:
            num += gemm_ms * (meas - other_ms)
            den += gemm_ms * gemm_ms
        s = (num / den) if den > 0 else 1.0
        s = max(float(args.min_scale), min(float(args.max_scale), float(s)))
        scales[wtag] = float(s)

    phys["weight_time_scale_by_tag"] = dict(scales)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(phys, f, indent=2)

    tags = list(scales.keys())
    print(f"Fitted weight_time_scale_by_tag for {len(tags)} tag(s) "
          f"(rows={sum(len(per_tag[t]) for t in per_tag)}; skipped={n_skip}).")
    for t in tags:
        print(f"  {t:10s}: {scales[t]:.4f}  (n={len(per_tag[t])})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

