#!/usr/bin/env python3
"""Fit weight_time_scale_by_tag for layerwise_roofline_sim.

Goal: explain real decode ms/tok gaps for weight-quantized models (Q4_K_M, Q2_K, ...)
that are not captured by the bytes-only roofline. We model this as a multiplier on
GEMM-family time only:

  measured_step_ms  ≈  other_ms + gemm_ms * weight_time_scale_by_tag[weight_tag]

Where (other_ms, gemm_ms) come from layerwise_roofline_sim.simulate_decode_step with
weight_time_scale_by_tag disabled (i.e. scale=1).

This produces a small sim_physics JSON fragment you can pass via --sim-physics-json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from layerwise_roofline_sim import (  # noqa: E402
    Eta,
    HARDWARE_PRESETS,
    aggregate_by_family,
    load_eta_json,
    load_structure_catalog,
    resolve_kv_quant_key,
    resolve_sim_physics,
    resolve_weight_bits,
    simulate_decode_step,
)


def mid_ctx(r: dict[str, Any]) -> int:
    pl = int(r.get("prompt_len", 0))
    dl = int(r.get("decode_len", 0))
    return int(r.get("mid_ctx", pl + dl // 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_paths", nargs="+", help="kv_timing*.json files (different weight_tags)")
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--hw", default="h100-sxm", choices=list(HARDWARE_PRESETS.keys()))
    ap.add_argument(
        "--eta-json",
        default=None,
        help="η JSON (same one used for your layerwise tables)",
    )
    ap.add_argument(
        "--sim-physics-json",
        default=None,
        help="Optional base sim physics (kv_attn_byte_mode, attn_scale_by_batch, ...). "
        "weight_time_scale_by_tag is fitted on top.",
    )
    ap.add_argument(
        "--preset",
        default=None,
        help="Only rows with this model_preset (default: keep all known presets)",
    )
    ap.add_argument(
        "--kv-types",
        nargs="*",
        default=None,
        help="If set, only include these kv_type rows (e.g. f16 q8_0 q4_0)",
    )
    ap.add_argument(
        "--batch-sizes",
        nargs="*",
        type=int,
        default=None,
        help="If set, only include these batch sizes",
    )
    ap.add_argument(
        "--min-scale",
        type=float,
        default=1.0,
        help="Lower bound for fitted weight_time_scale_by_tag (default: 1.0)",
    )
    ap.add_argument("-o", "--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    cat = load_structure_catalog(args.catalog)
    eta = load_eta_json(args.eta_json) if args.eta_json else Eta()
    phys = resolve_sim_physics(args.sim_physics_json)

    kv_keep = set(k.lower() for k in args.kv_types) if args.kv_types else None
    b_keep = set(int(b) for b in args.batch_sizes) if args.batch_sizes else None

    # Collect training rows grouped by weight_tag
    rows_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_sources: list[str] = []

    for path in args.json_paths:
        used_sources.append(os.path.abspath(os.path.expanduser(path)))
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("rows", []):
            mp = r.get("model_preset")
            if not mp or mp not in cat["presets"]:
                continue
            if args.preset and str(mp) != str(args.preset):
                continue
            kv = str(r.get("kv_type", "")).lower()
            if kv_keep is not None and kv not in kv_keep:
                continue
            B = int(r.get("batch_size", 0))
            if b_keep is not None and B not in b_keep:
                continue
            meas_ms = float(r.get("measured_ms", 0))
            if meas_ms <= 0:
                continue
            wtag = r.get("weight_tag")
            if not wtag:
                continue
            rows_by_tag[str(wtag)].append(r)

    if not rows_by_tag:
        print("No usable rows found.", file=sys.stderr)
        sys.exit(1)

    hw = dict(HARDWARE_PRESETS[args.hw])
    scales: dict[str, float] = {}
    fit_meta: dict[str, Any] = {}

    for tag, rows in sorted(rows_by_tag.items()):
        xs: list[float] = []
        ys: list[float] = []
        n_used = 0
        for r in rows:
            mp = str(r["model_preset"])
            model = dict(cat["presets"][mp])
            B = int(r["batch_size"])
            ctx = mid_ctx(r)

            kv_key = resolve_kv_quant_key(str(r.get("kv_type", "f16")))
            wbits_row = float(r.get("weight_bits", model.get("weight_bits", 16)))
            wb, wtag_res = resolve_weight_bits(model, str(tag), wbits_row)
            wbpe = wb / 8.0
            norm_bpe = 16.0 / 8.0

            total_s, events = simulate_decode_step(
                model,
                batch_size=B,
                ctx_len=ctx,
                hw=hw,
                eta=eta,
                weight_bpe=wbpe,
                norm_bpe=norm_bpe,
                kv_quant_key=kv_key,
                attn_impl="simple",
                kv_attn_byte_mode=str(phys["kv_attn_byte_mode"]),
                attn_time_scale=float(phys["attn_time_scale"]),
                attn_time_scale_inv_batch=float(phys["attn_time_scale_inv_batch"]),
                attn_scale_by_batch=phys.get("attn_scale_by_batch"),
                weight_tag=wtag_res or str(tag),
                weight_time_scale_by_tag=None,  # IMPORTANT: baseline split (scale=1)
            )
            by_fam = aggregate_by_family(events)
            gemm_ms = float(by_fam.get("gemm", 0.0)) * 1000.0
            other_ms = (float(total_s) - float(by_fam.get("gemm", 0.0))) * 1000.0
            meas_step_ms = float(r["measured_ms"]) * float(B)

            # Solve: meas_step_ms - other_ms ≈ gemm_ms * scale
            xs.append(gemm_ms)
            ys.append(meas_step_ms - other_ms)
            n_used += 1

        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        denom = float(x @ x)
        scale = float((x @ y) / denom) if denom > 0 else 1.0
        scale = max(float(args.min_scale), scale)
        scales[str(tag)] = float(scale)

        # diagnostics
        yhat = x * scale
        resid = y - yhat
        rmse = float(math.sqrt(float(np.mean(resid**2)))) if n_used else float("nan")
        mae = float(float(np.mean(np.abs(resid)))) if n_used else float("nan")
        fit_meta[str(tag)] = {
            "n_rows": int(n_used),
            "rmse_ms_on_(meas_step-other)_vs_gemm": round(rmse, 6),
            "mae_ms_on_(meas_step-other)_vs_gemm": round(mae, 6),
            "scale": round(scale, 6),
        }

    # Output is a sim_physics JSON that can be passed directly to layerwise_roofline_sim.py
    out: dict[str, Any] = {
        "schema_version": 1,
        "_comment": (
            "Sim physics with fitted weight_time_scale_by_tag for layerwise_roofline_sim. "
            "weight_time_scale_by_tag scales GEMM family only. "
            "Fit objective per row: measured_step_ms - other_ms ≈ gemm_ms * scale(tag)."
        ),
        # ---- sim physics keys (consumed by layerwise_roofline_sim.resolve_sim_physics) ----
        "kv_attn_byte_mode": phys["kv_attn_byte_mode"],
        "attn_time_scale": phys["attn_time_scale"],
        "attn_time_scale_inv_batch": phys["attn_time_scale_inv_batch"],
        "attn_scale_by_batch": phys.get("attn_scale_by_batch", {}),
        "weight_time_scale_by_tag": scales,
        # ---- provenance / diagnostics ----
        "_fit": {
            "hw": args.hw,
            "eta_json": args.eta_json,
            "sources": used_sources,
            "filters": {
                "preset": args.preset,
                "kv_types": args.kv_types,
                "batch_sizes": args.batch_sizes,
                "min_scale": args.min_scale,
            },
            "fit_meta": fit_meta,
        },
    }

    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(out_path)


if __name__ == "__main__":
    main()

