#!/usr/bin/env python3
"""Build a kv_timing-shaped JSON filled with layerwise_roofline_sim predictions.

Reads an existing ``kv_timing*.json`` (cluster benchmark) as a **grid template**:
same ``rows[]`` keys for model_preset, weight_bits, kv_type, prompt_len, decode_len,
batch_size, etc. Replaces ``measured_ms`` / ``tok_per_s`` with **simulated** values and
sets ``measured_ms`` to null. Omits large ``prefill_buckets`` / ``decode_buckets`` blobs.

Example::

    python3 export_layerwise_kv_json.py \\
        --template-json ../results/qwen3-8b/profile/kv_timing_h100.json \\
        --out ../results/qwen3-8b/profile/layerwise_kv_timing_h100.json \\
        --hw h100-sxm \\
        --eta-json ../results/qwen3-8b/profile/layerwise_eta_h100.json \\
        --calibration-json ../results/qwen3-8b/profile/layerwise_calibration_h100.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from benchmark_kv_timing import roofline_ms  # noqa: E402
from gguf_layerwise_weights import load_gguf_tensor_n_bytes  # noqa: E402
from layerwise_roofline_sim import (  # noqa: E402
    Eta,
    HARDWARE_PRESETS,
    aggregate_by_family,
    calibrated_ms_per_token,
    load_calibration_json,
    load_eta_json,
    load_structure_catalog,
    resolve_kv_quant_key,
    resolve_weight_bits,
    simulate_decode_step,
)


def mid_ctx(r: dict[str, Any]) -> int:
    pl = int(r.get("prompt_len", 0))
    dl = int(r.get("decode_len", 0))
    return int(r.get("mid_ctx", pl + dl // 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--template-json",
        required=True,
        help="Existing kv_timing JSON (defines rows grid and metadata)",
    )
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--hw", default="h100-sxm", choices=list(HARDWARE_PRESETS.keys()))
    ap.add_argument("--eta-json", default=None, help="Optional η JSON from fit_layerwise_eta.py")
    ap.add_argument(
        "--calibration-json",
        default=None,
        help="Optional linear calibration JSON from fit_layerwise_calibration.py",
    )
    ap.add_argument(
        "--gguf",
        default=None,
        metavar="PATH",
        help="Optional GGUF for per-tensor weight bytes in the sim",
    )
    ap.add_argument(
        "--include-roofline",
        action="store_true",
        help="Also fill aggregate benchmark_kv_timing.roofline_ms per row",
    )
    ap.add_argument(
        "--attn-impl",
        default="simple",
        choices=["simple", "flash"],
        help="Passed to layerwise_roofline_sim.simulate_decode_step (default: simple)",
    )
    ap.add_argument(
        "--fa-bc",
        type=int,
        default=128,
        metavar="N",
        help="Flash KV tile size along sequence (metadata / simulate_decode_step)",
    )
    ap.add_argument(
        "--attn-naive-spill",
        action="store_true",
        help="With --attn-impl simple: add unfused logits/probs HBM traffic",
    )
    args = ap.parse_args()

    tpl_path = os.path.abspath(os.path.expanduser(args.template_json))
    with open(tpl_path, encoding="utf-8") as f:
        data = json.load(f)

    cat = load_structure_catalog(args.catalog)
    hw = dict(HARDWARE_PRESETS[args.hw])
    eta_base = load_eta_json(args.eta_json) if args.eta_json else Eta()
    cal: Optional[dict[str, Any]] = None
    if args.calibration_json:
        cal = load_calibration_json(args.calibration_json)

    gguf_tb: Optional[dict[str, int]] = None
    if args.gguf:
        gguf_tb = load_gguf_tensor_n_bytes(os.path.expanduser(args.gguf))

    rows_in = data.get("rows", [])
    rows_out: list[dict[str, Any]] = []

    for r in rows_in:
        mp = r.get("model_preset")
        if not mp or mp not in cat["presets"]:
            continue
        model = dict(cat["presets"][mp])
        B = int(r["batch_size"])
        ctx = mid_ctx(r)
        wbits = float(r.get("weight_bits", model.get("weight_bits", 16)))
        wtag = r.get("weight_tag")
        wb, _ = resolve_weight_bits(model, str(wtag) if wtag else None, wbits)
        wbpe = wb / 8.0
        norm_bpe = 16.0 / 8.0
        kv_key = resolve_kv_quant_key(str(r.get("kv_type", "f16")))

        total_s, events = simulate_decode_step(
            model,
            batch_size=B,
            ctx_len=ctx,
            hw=hw,
            eta=eta_base,
            weight_bpe=wbpe,
            norm_bpe=norm_bpe,
            kv_quant_key=kv_key,
            gguf_tensor_bytes=gguf_tb,
            attn_impl=args.attn_impl,
            fa_bc=args.fa_bc,
            attn_naive_spill=args.attn_naive_spill,
        )
        lay_raw = total_s * 1000.0 / float(B)
        lay_cal: Optional[float] = None
        if cal is not None:
            lay_cal = calibrated_ms_per_token(
                lay_raw,
                B,
                float(cal["t_floor_ms"]),
                float(cal["scale"]),
            )

        by_fam = aggregate_by_family(events)

        row_out: dict[str, Any] = {
            "model_path": r.get("model_path"),
            "weight_tag": r.get("weight_tag"),
            "weight_bits": wbits,
            "model_preset": mp,
            "kv_type": r.get("kv_type"),
            "prompt_len": int(r["prompt_len"]),
            "decode_len": int(r["decode_len"]),
            "mid_ctx": ctx,
            "batch_size": B,
            "measured_ms": None,
            "layerwise_ms_per_tok": round(lay_raw, 9),
            "tok_per_s": round(1000.0 / lay_raw, 6) if lay_raw > 0 else 0.0,
            "layerwise_ms_per_tok_calibrated": round(lay_cal, 9) if lay_cal is not None else None,
            "tok_per_s_calibrated": round(1000.0 / lay_cal, 6)
            if lay_cal is not None and lay_cal > 0
            else None,
            "roofline_ms": None,
            "roofline_aggregate_ms_per_tok": None,
            "calib_factor": None,
            "speedup_vs_f16": None,
            "seconds_by_family": {k: round(v, 9) for k, v in by_fam.items()},
        }

        if args.include_roofline:
            roof = roofline_ms(str(mp), args.hw, wbits, str(r["kv_type"]), ctx, batch_size=B)
            row_out["roofline_aggregate_ms_per_tok"] = (
                round(roof, 9) if not math.isnan(float(roof)) else None
            )

        rows_out.append(row_out)

    # speedup_vs_f16 within synthetic rows (per batch_size, prompt, decode)
    f16_map: dict[tuple[Any, ...], float] = {}
    for row in rows_out:
        if str(row.get("kv_type")).lower() == "f16":
            key = (
                row["model_preset"],
                row["weight_tag"],
                row["prompt_len"],
                row["decode_len"],
                row["batch_size"],
            )
            f16_map[key] = float(row["layerwise_ms_per_tok"])

    for row in rows_out:
        key = (
            row["model_preset"],
            row["weight_tag"],
            row["prompt_len"],
            row["decode_len"],
            row["batch_size"],
        )
        if str(row.get("kv_type")).lower() == "f16":
            row["speedup_vs_f16"] = 1.0
            continue
        base = f16_map.get(key)
        if base and base > 0 and row.get("layerwise_ms_per_tok"):
            row["speedup_vs_f16"] = round(base / float(row["layerwise_ms_per_tok"]), 6)

    out: dict[str, Any] = {
        "_comment": (
            "Synthetic predictions from layerwise_roofline_sim.py (not cluster measured). "
            "measured_ms is null; see layerwise_ms_per_tok. "
            f"Template: {os.path.relpath(tpl_path, os.path.dirname(os.path.abspath(args.out)) or '.')}"
        ),
        "hardware": data.get("hardware"),
        "n_warmup": data.get("n_warmup"),
        "decode_lens": data.get("decode_lens"),
        "prompt_lens": data.get("prompt_lens"),
        "batch_sizes": data.get("batch_sizes"),
        "kv_types": data.get("kv_types"),
        "decode_bucket_size": data.get("decode_bucket_size"),
        "prefill_bucket_size": data.get("prefill_bucket_size"),
        "layerwise_meta": {
            "hw": args.hw,
            "catalog": os.path.relpath(args.catalog, _SCRIPT_DIR),
            "eta_json": args.eta_json,
            "calibration_json": args.calibration_json,
            "gguf": args.gguf,
            "attn_impl": args.attn_impl,
            "fa_bc": int(args.fa_bc),
            "attn_naive_spill": bool(args.attn_naive_spill),
            "script": "export_layerwise_kv_json.py",
        },
        "rows": rows_out,
    }

    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
