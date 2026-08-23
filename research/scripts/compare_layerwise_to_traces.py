#!/usr/bin/env python3
"""Compare layerwise_roofline_sim vs aggregate roofline_ms vs measured_ms from kv_timing JSON."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from benchmark_kv_timing import roofline_ms  # noqa: E402
from layerwise_roofline_sim import (  # noqa: E402
    Eta,
    HARDWARE_PRESETS,
    calibrated_ms_per_token,
    load_calibration_json,
    load_eta_json,
    load_structure_catalog,
    resolve_kv_quant_key,
    resolve_sim_physics,
    resolve_weight_bits,
    simulate_decode_step,
)
from gguf_layerwise_weights import load_gguf_tensor_n_bytes  # noqa: E402

DEFAULT_HW = "h100-sxm"


def mid_ctx(r: dict) -> int:
    pl = int(r.get("prompt_len", 0))
    dl = int(r.get("decode_len", 0))
    return int(r.get("mid_ctx", pl + dl // 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "json_paths",
        nargs="*",
        help="kv_timing*.json files (default: all under research/results)",
    )
    ap.add_argument("--hw", default=DEFAULT_HW, help="Hardware key for both models")
    ap.add_argument(
        "--catalog",
        default=os.path.join(_SCRIPT_DIR, "model_structures.json"),
        help="model_structures.json path",
    )
    ap.add_argument(
        "--eta-json",
        default=None,
        help="Fitted η from fit_layerwise_eta.py (layerwise column uses this η)",
    )
    ap.add_argument(
        "--calibration-json",
        default=None,
        help="From fit_layerwise_calibration.py (cal column: t_floor/B + scale×lay)",
    )
    ap.add_argument(
        "--gguf",
        default=None,
        metavar="PATH",
        help="Optional GGUF path: layerwise column uses per-tensor weight bytes (same preset rows only)",
    )
    ap.add_argument(
        "--attn-impl",
        default="simple",
        choices=["simple", "flash"],
        help="Passed to simulate_decode_step (default: simple)",
    )
    ap.add_argument(
        "--fa-bc",
        type=int,
        default=128,
        metavar="N",
        help="Flash KV tile metadata (simulate_decode_step)",
    )
    ap.add_argument(
        "--attn-naive-spill",
        action="store_true",
        help="With --attn-impl simple: unfused logits/probs HBM traffic",
    )
    ap.add_argument(
        "--sim-physics-json",
        default=None,
        metavar="PATH",
        help="layerwise_roofline_sim sim physics (kv_attn_byte_mode, attn scales)",
    )
    ap.add_argument(
        "--kv-attn-byte-mode",
        choices=["fp16_equiv_dequant", "storage"],
        default=None,
        help="Override attention KV byte model",
    )
    ap.add_argument("--attn-time-scale", type=float, default=None, metavar="F")
    ap.add_argument("--attn-time-scale-inv-batch", type=float, default=None, metavar="F")
    args = ap.parse_args()

    if args.json_paths:
        paths = sorted(args.json_paths)
    else:
        paths = sorted(
            glob.glob(os.path.join(_SCRIPT_DIR, "../results/**/kv_timing*.json"), recursive=True)
        )

    if not paths:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    cat = load_structure_catalog(args.catalog)
    hw = dict(HARDWARE_PRESETS[args.hw])
    eta = load_eta_json(args.eta_json) if args.eta_json else Eta()
    calib = load_calibration_json(args.calibration_json) if args.calibration_json else None
    gguf_tb = None
    if args.gguf:
        gguf_tb = load_gguf_tensor_n_bytes(os.path.expanduser(args.gguf))

    phys = resolve_sim_physics(
        args.sim_physics_json,
        kv_attn_byte_mode=args.kv_attn_byte_mode,
        attn_time_scale=args.attn_time_scale,
        attn_time_scale_inv_batch=args.attn_time_scale_inv_batch,
    )

    rows_out: list[dict] = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", [])
        for r in rows:
            mp = r.get("model_preset")
            if not mp or mp not in cat["presets"]:
                continue
            model = dict(cat["presets"][mp])
            B = int(r["batch_size"])
            ctx = mid_ctx(r)
            meas = float(r.get("measured_ms", 0))
            if meas <= 0:
                continue
            wbits = float(r.get("weight_bits", model.get("weight_bits", 16)))
            wtag = r.get("weight_tag")
            wb, _ = resolve_weight_bits(model, str(wtag) if wtag else None, wbits)
            wbpe = wb / 8.0
            norm_bpe = 16.0 / 8.0
            kv_key = resolve_kv_quant_key(str(r.get("kv_type", "f16")))

            total_s, _ = simulate_decode_step(
                model,
                batch_size=B,
                ctx_len=ctx,
                hw=hw,
                eta=eta,
                weight_bpe=wbpe,
                norm_bpe=norm_bpe,
                kv_quant_key=kv_key,
                gguf_tensor_bytes=gguf_tb,
                attn_impl=args.attn_impl,
                fa_bc=args.fa_bc,
                attn_naive_spill=args.attn_naive_spill,
                kv_attn_byte_mode=str(phys["kv_attn_byte_mode"]),
                attn_time_scale=float(phys["attn_time_scale"]),
                attn_time_scale_inv_batch=float(phys["attn_time_scale_inv_batch"]),
                attn_scale_by_batch=phys.get("attn_scale_by_batch"),
                attn_scale_by_batch_and_kv=phys.get("attn_scale_by_batch_and_kv"),
            )
            layer_ms = total_s * 1000.0 / float(B)

            roof = roofline_ms(str(mp), args.hw, wbits, str(r["kv_type"]), ctx, batch_size=B)
            rel = os.path.relpath(path, os.path.join(_SCRIPT_DIR, ".."))
            rowd: dict = {
                "file": rel,
                "preset": mp,
                "weight_tag": wtag,
                "kv_type": r["kv_type"],
                "B": B,
                "ctx": ctx,
                "measured_ms": meas,
                "roofline_ms": roof,
                "layerwise_ms": layer_ms,
            }
            if calib is not None:
                rowd["cal_ms"] = calibrated_ms_per_token(
                    layer_ms,
                    B,
                    float(calib["t_floor_ms"]),
                    float(calib["scale"]),
                )
            rows_out.append(rowd)

    if not rows_out:
        print("No comparable rows.", file=sys.stderr)
        sys.exit(1)

    print("Comparison: cluster measured vs aggregate roofline vs layerwise simulator")
    print(
        f"hw={args.hw!r}  attn_impl={args.attn_impl!r}  "
        f"layerwise_η={'fitted ' + repr(args.eta_json) if args.eta_json else 'defaults'}"
        + (f"  cal={args.calibration_json!r}" if args.calibration_json else "")
        + (f"  gguf={args.gguf!r}" if args.gguf else "")
        + f"  sim_physics={phys!r}"
    )
    print()
    if calib is not None:
        hdr = (
            f"{'file':<42} {'preset':<12} {'kv':<6} {'B':>3}  {'meas':>8} {'roof':>8} "
            f"{'lay':>8} {'cal':>8}  {'m/r':>5} {'m/l':>5} {'m/c':>5}"
        )
    else:
        hdr = f"{'file':<42} {'preset':<12} {'kv':<6} {'B':>3}  {'meas':>8} {'roof':>8} {'lay':>8}  {'m/r':>5} {'m/l':>5}"
    print(hdr)
    print("-" * len(hdr))

    errs_roof = []
    errs_lay = []
    errs_cal: list[float] = []
    for x in rows_out:
        m = x["measured_ms"]
        roof = x["roofline_ms"]
        lay = x["layerwise_ms"]
        rr = m / roof if roof and not math.isnan(roof) and roof > 0 else float("nan")
        rl = m / lay if lay > 0 else float("nan")
        if not math.isnan(roof) and roof > 0:
            errs_roof.append(m - roof)
        errs_lay.append(m - lay)
        if calib is not None:
            c = x["cal_ms"]
            rc = m / c if c > 0 else float("nan")
            errs_cal.append(m - c)
            print(
                f"{x['file']:<42} {x['preset']:<12} {x['kv_type']:<6} {x['B']:>3}  "
                f"{m:8.3f} {roof:8.3f} {lay:8.3f} {c:8.3f}  {rr:5.2f} {rl:5.2f} {rc:5.2f}"
            )
        else:
            print(
                f"{x['file']:<42} {x['preset']:<12} {x['kv_type']:<6} {x['B']:>3}  "
                f"{m:8.3f} {roof:8.3f} {lay:8.3f}  {rr:5.2f} {rl:5.2f}"
            )

    def rmse(errs: list[float]) -> float:
        return math.sqrt(sum(e * e for e in errs) / len(errs))

    def mae(errs: list[float]) -> float:
        return sum(abs(e) for e in errs) / len(errs)

    print()
    print(f"n_rows = {len(rows_out)}")
    print(
        f"MAE  (meas - roof):   {mae(errs_roof):.4f} ms/tok   RMSE: {rmse(errs_roof):.4f}"
    )
    print(
        f"MAE  (meas - layer):  {mae(errs_lay):.4f} ms/tok   RMSE: {rmse(errs_lay):.4f}"
    )
    if calib is not None and errs_cal:
        print(
            f"MAE  (meas - cal):    {mae(errs_cal):.4f} ms/tok   RMSE: {rmse(errs_cal):.4f}"
        )
    mean_rr = sum(
        x["measured_ms"] / x["roofline_ms"]
        for x in rows_out
        if x["roofline_ms"] and not math.isnan(x["roofline_ms"]) and x["roofline_ms"] > 0
    ) / max(
        1,
        sum(
            1
            for x in rows_out
            if x["roofline_ms"] and not math.isnan(x["roofline_ms"]) and x["roofline_ms"] > 0
        ),
    )
    mean_rl = sum(x["measured_ms"] / x["layerwise_ms"] for x in rows_out) / len(rows_out)
    line = f"mean(meas/roofline) ≈ {mean_rr:.3f}   mean(meas/layerwise) ≈ {mean_rl:.3f}"
    if calib is not None:
        mean_rc = sum(x["measured_ms"] / x["cal_ms"] for x in rows_out) / len(rows_out)
        line += f"   mean(meas/cal) ≈ {mean_rc:.3f}"
    print(line)


if __name__ == "__main__":
    main()
