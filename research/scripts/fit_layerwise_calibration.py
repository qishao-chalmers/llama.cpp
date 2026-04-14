#!/usr/bin/env python3
"""Fit linear calibration on top of layerwise ms/tok (cluster traces).

  measured_ms/tok ≈ t_floor_ms / batch_size + scale × layerwise_ms/tok

Uses the same structure as fit_roofline_calibration / roofline_calibration.md.
Layerwise prediction can use --eta-json from fit_layerwise_eta.py.

Example::

    python3 fit_layerwise_calibration.py \\
        ../results/qwen3-8b/profile/kv_timing_h100.json \\
        --hw h100-sxm --eta-json ../results/qwen3-8b/profile/layerwise_eta_h100.json \\
        -o ../results/qwen3-8b/profile/layerwise_calibration_h100.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from layerwise_roofline_sim import (  # noqa: E402
    Eta,
    load_eta_json,
    load_structure_catalog,
    predict_ms_per_token,
    resolve_kv_quant_key,
    resolve_weight_bits,
)

DEFAULT_HW = "h100-sxm"


def mid_ctx(r: dict) -> int:
    pl = int(r.get("prompt_len", 0))
    dl = int(r.get("decode_len", 0))
    return int(r.get("mid_ctx", pl + dl // 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_paths", nargs="+", help="kv_timing*.json files")
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--hw", default=DEFAULT_HW)
    ap.add_argument("--preset", default=None, help="Only rows with this model_preset")
    ap.add_argument(
        "--eta-json",
        default=None,
        help="η from fit_layerwise_eta.py (recommended)",
    )
    ap.add_argument("-o", "--out", required=True, help="Output calibration JSON")
    args = ap.parse_args()

    cat = load_structure_catalog(args.catalog)
    eta = load_eta_json(args.eta_json) if args.eta_json else Eta()

    xs_inv_b: list[float] = []
    xs_lay: list[float] = []
    ys: list[float] = []
    meta: list[dict[str, Any]] = []

    for path in args.json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for r in data.get("rows", []):
            mp = r.get("model_preset")
            if not mp or mp not in cat["presets"]:
                continue
            if args.preset and str(mp) != args.preset:
                continue
            meas = float(r.get("measured_ms", 0))
            if meas <= 0:
                continue
            model = dict(cat["presets"][mp])
            B = int(r["batch_size"])
            ctx = mid_ctx(r)
            wbits = float(r.get("weight_bits", 16))
            kv_key = resolve_kv_quant_key(str(r.get("kv_type", "f16")))
            lay = predict_ms_per_token(
                model,
                batch_size=B,
                ctx_len=ctx,
                hw_name=args.hw,
                eta=eta,
                weight_bits=wbits,
                norm_weight_bits=16.0,
                kv_quant_key=kv_key,
            )
            xs_inv_b.append(1.0 / B)
            xs_lay.append(lay)
            ys.append(meas)
            meta.append(
                {
                    "preset": mp,
                    "batch_size": B,
                    "measured_ms": meas,
                    "layerwise_ms": lay,
                    "kv_type": r.get("kv_type"),
                    "weight_tag": r.get("weight_tag"),
                }
            )

    n = len(ys)
    if n < 2:
        print(f"Need at least 2 rows; got {n}.", file=sys.stderr)
        sys.exit(1)

    X = np.column_stack([xs_inv_b, xs_lay])
    y = np.array(ys, dtype=float)
    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    t_floor = float(coef[0])
    scale = float(coef[1])
    yhat = X @ coef
    rmse = float(math.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))

    out: dict[str, Any] = {
        "_comment": (
            "Linear calibration on layerwise_roofline_sim: "
            "measured_ms/tok ≈ t_floor_ms / batch_size + scale × layerwise_ms/tok. "
            "Use with layerwise_roofline_sim.py --calibration-json."
        ),
        "schema_version": 1,
        "hw": args.hw,
        "t_floor_ms": round(t_floor, 6),
        "scale": round(scale, 6),
        "rmse_ms": round(rmse, 6),
        "mae_ms": round(mae, 6),
        "n_fit_points": n,
        "sources": list(dict.fromkeys(args.json_paths)),
        "eta_json": args.eta_json,
        "fit_rows": meta,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Fitted t_floor_ms={t_floor:.4f}  scale={scale:.4f}  RMSE={rmse:.4f} ms/tok  MAE={mae:.4f}  (n={n})")
    print(f"Wrote {args.out}")
    print("\nPer-row (meas vs pred):")
    for m, pred in zip(meta, yhat):
        print(
            f"  B={m['batch_size']:3d}  kv={m['kv_type']!s:6s}  "
            f"meas={m['measured_ms']:.3f}  lay={m['layerwise_ms']:.3f}  pred={pred:.3f}"
        )


if __name__ == "__main__":
    main()
