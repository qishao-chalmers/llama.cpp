#!/usr/bin/env python3
"""Fit attention-family time scale(s) for layerwise_roofline_sim (sim_physics JSON).

Goal: reduce batch-regime prediction error by fitting small attention corrections in
sim_physics, without relying on a global linear calibration layer.

We fit either:
  - attn_scale_by_batch:             scale[B] applied to attn_core family only
  - attn_scale_by_batch_and_kv:      scale[kv_quant_key][B] applied to attn_core only

Per measured row, we run simulate_decode_step with *neutral* attention scaling
(no per-batch overrides), decompose:

  meas_ms/tok ≈ s * attn_ms/tok + other_ms/tok

Closed form least-squares:
  s = sum(attn_ms * (meas_ms - other_ms)) / sum(attn_ms^2)

Use ms/tok for fitting (NOT tok/s), to avoid overweighting high batch.
"""

from __future__ import annotations

import argparse
import json
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
    load_structure_catalog,
    resolve_kv_quant_key,
    resolve_sim_physics,
    simulate_decode_step,
)


def mid_ctx(r: dict[str, Any]) -> int:
    pl = int(r.get("prompt_len", 0) or 0)
    dl = int(r.get("decode_len", 0) or 0)
    return int(r.get("mid_ctx", pl + dl // 2))


def _fit_scale(points: list[tuple[float, float, float]], *, min_scale: float, max_scale: float) -> float:
    """points: (attn_ms, other_ms, meas_ms) in ms/tok."""
    num = 0.0
    den = 0.0
    for attn_ms, other_ms, meas_ms in points:
        num += attn_ms * (meas_ms - other_ms)
        den += attn_ms * attn_ms
    s = (num / den) if den > 0 else 1.0
    if s < float(min_scale):
        s = float(min_scale)
    if s > float(max_scale):
        s = float(max_scale)
    return float(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_paths", nargs="+", help="kv_timing*.json files (benchmark_kv_timing output)")
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--hw", default="h100-sxm", choices=list(HARDWARE_PRESETS.keys()))
    ap.add_argument("--eta-json", default=None, help="η from fit_layerwise_eta.py (fallback when not using --auto-eta)")
    ap.add_argument(
        "--auto-eta",
        action="store_true",
        help="Auto-pick per-file eta-json from the same directory as each measured JSON "
             "(layerwise_eta_<hw>.json for the selected --hw). Overrides global --eta-json when found.",
    )
    ap.add_argument(
        "--skip-missing-eta",
        action="store_true",
        help="When --auto-eta and the per-file eta is missing, skip that measured file "
             "instead of using the global --eta-json fallback.",
    )
    ap.add_argument(
        "--base-sim-physics-json",
        default=None,
        help="Base sim_physics JSON to start from (optional).",
    )
    ap.add_argument(
        "--mode",
        choices=["batch", "batch_kv"],
        default="batch_kv",
        help="Fit scale by batch only, or by (kv_quant_key, batch).",
    )
    ap.add_argument("-o", "--out", required=True, help="Output sim_physics JSON path")
    ap.add_argument("--min-scale", type=float, default=0.25)
    ap.add_argument("--max-scale", type=float, default=2.00)
    ap.add_argument("--min-rows", type=int, default=2,
                    help="Min rows per key to fit a scale (else keep 1.0).")
    args = ap.parse_args()

    cat = load_structure_catalog(args.catalog)

    base_phys = load_sim_physics_json(args.base_sim_physics_json)
    phys = resolve_sim_physics(args.base_sim_physics_json)
    for k, v in base_phys.items():
        if k not in phys:
            phys[k] = v

    hw = dict(HARDWARE_PRESETS[args.hw])

    def _eta_for_file(measured_path: str) -> tuple[Eta, Optional[str]]:
        if not args.auto_eta:
            return (load_eta_json(args.eta_json) if args.eta_json else Eta()), args.eta_json
        d = os.path.dirname(os.path.abspath(measured_path))
        hw_l = str(args.hw).lower()
        candidates: list[str] = []
        if "h100" in hw_l:
            candidates.append(os.path.join(d, "layerwise_eta_h100.json"))
        if "a100" in hw_l:
            candidates.append(os.path.join(d, "layerwise_eta_a100_80g.json"))
        candidates.append(os.path.join(d, f"layerwise_eta_{args.hw.replace('-', '_')}.json"))
        try:
            import glob
            candidates.extend(sorted(glob.glob(os.path.join(d, "layerwise_eta*.json"))))
        except Exception:
            pass
        eta_path = next((p for p in candidates if os.path.isfile(p)), None)
        if eta_path:
            return load_eta_json(eta_path), eta_path
        if args.skip_missing_eta:
            raise FileNotFoundError(f"missing per-file eta in {d!r}")
        return (load_eta_json(args.eta_json) if args.eta_json else Eta()), args.eta_json

    # Group points by key
    by_key: dict[Any, list[tuple[float, float, float]]] = defaultdict(list)
    n_skip = 0
    any_rows = False
    for path in args.json_paths:
        try:
            eta, _eta_used = _eta_for_file(path)
        except Exception:
            # Missing eta for this measured file (skip)
            n_skip += 1
            continue
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", []) if isinstance(data, dict) else []
        if not rows:
            continue
        any_rows = True
        for r in rows:
            mp = r.get("model_preset")
            if not mp or mp not in cat["presets"]:
                n_skip += 1
                continue
            meas = float(r.get("measured_ms", 0))
            if meas <= 0:
                n_skip += 1
                continue
            B = int(r.get("batch_size", 0) or 0)
            if B <= 0:
                n_skip += 1
                continue
            kv_cli = str(r.get("kv_type", "f16"))
            try:
                kv_key = resolve_kv_quant_key(kv_cli)
            except Exception:
                kv_key = str(kv_cli)
            ctx = mid_ctx(r)
            model = dict(cat["presets"][mp])
            wbits = float(r.get("weight_bits", 16.0))

            # Neutral attention scaling for decomposition.
            total_s, events = simulate_decode_step(
                model,
                batch_size=B,
                ctx_len=ctx,
                hw=hw,
                eta=eta,
                weight_bpe=wbits / 8.0,
                norm_bpe=16.0 / 8.0,
                kv_quant_key=str(kv_key),
                kv_group_size=None,
                kv_asym=False,
                gguf_tensor_bytes=None,
                attn_impl=str(phys.get("attn_impl", "simple")),
                fa_bc=int(phys.get("fa_bc", 128)),
                attn_naive_spill=bool(phys.get("attn_naive_spill", False)),
                kv_attn_byte_mode=str(phys.get("kv_attn_byte_mode", "fp16_equiv_dequant")),
                attn_time_scale=1.0,
                attn_time_scale_inv_batch=0.0,
                attn_scale_by_batch=None,
                attn_scale_by_batch_and_kv=None,
                weight_tag=str(r.get("weight_tag", "") or "") or None,
                weight_time_scale_by_tag=phys.get("weight_time_scale_by_tag"),
            )

            attn_s = sum(e.seconds for e in events if e.family == "attn_core")
            other_s = float(total_s) - float(attn_s)
            attn_ms = float(attn_s) * 1000.0 / float(B)
            other_ms = float(other_s) * 1000.0 / float(B)

            key: Any
            if args.mode == "batch":
                key = int(B)
            else:
                key = (str(kv_key), int(B))
            by_key[key].append((attn_ms, other_ms, meas))

    if not any_rows:
        raise ValueError("No rows found in any input JSON.")

    if not by_key:
        raise ValueError("No usable rows found to fit attention scales.")

    if args.mode == "batch":
        out_map: dict[int, float] = {}
        for b in sorted(by_key.keys()):
            pts = by_key[b]
            if len(pts) < int(args.min_rows):
                out_map[int(b)] = 1.0
                continue
            out_map[int(b)] = _fit_scale(pts, min_scale=args.min_scale, max_scale=args.max_scale)
        phys["attn_scale_by_batch"] = out_map
        phys["attn_scale_by_batch_and_kv"] = phys.get("attn_scale_by_batch_and_kv", {})
    else:
        out_map2: dict[str, dict[int, float]] = defaultdict(dict)
        for (kv_key, b) in sorted(by_key.keys()):
            pts = by_key[(kv_key, b)]
            if len(pts) < int(args.min_rows):
                out_map2[str(kv_key)][int(b)] = 1.0
                continue
            out_map2[str(kv_key)][int(b)] = _fit_scale(pts, min_scale=args.min_scale, max_scale=args.max_scale)
        phys["attn_scale_by_batch_and_kv"] = dict(out_map2)
        phys["attn_scale_by_batch"] = phys.get("attn_scale_by_batch", {})

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(phys, f, indent=2)
        f.write("\n")

    print(f"Wrote {args.out}")
    print(f"mode={args.mode}  groups={len(by_key)}  skipped={n_skip}")


if __name__ == "__main__":
    main()

