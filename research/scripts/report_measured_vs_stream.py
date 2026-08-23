#!/usr/bin/env python3
"""Compare measured kv_timing rows vs stream_perf_model predictions.

Without --calibration-json and without --auto-calibration (or if auto finds no JSON),
predictions use stream_perf_model.default_calib() — fixed η defaults, no fit to cluster.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import layerwise_roofline_sim as sim  # noqa: E402
import stream_perf_model as spm  # noqa: E402
from gguf_layerwise_weights import load_gguf_tensor_n_bytes  # noqa: E402


def _load_rows(path: str) -> list[dict[str, Any]]:
    data = json.load(open(path, encoding="utf-8"))
    return list(data.get("rows", []))


def _fmt(x: float, width: int = 9) -> str:
    if math.isnan(x) or math.isinf(x):
        return " " * (width - 3) + "nan"
    return f"{x:{width}.4f}"


def _ms_per_tok_to_tok_s(ms_per_tok: float) -> float:
    if ms_per_tok <= 0 or math.isnan(ms_per_tok) or math.isinf(ms_per_tok):
        return float("nan")
    return 1000.0 / float(ms_per_tok)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured-json", action="append", default=[])
    ap.add_argument("--glob-measured", default=None)
    ap.add_argument("--hw", default="h100-sxm", choices=list(sim.HARDWARE_PRESETS.keys()))
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--calibration-json", default=None, help="stream_perf_model calibration JSON")
    ap.add_argument(
        "--auto-calibration",
        action="store_true",
        help="Auto-pick stream_calib_decode_h100.json next to each measured JSON.",
    )
    ap.add_argument("--gguf-dir", action="append", default=[], help="Extra GGUF search dirs")
    ap.add_argument("--kv-attn-byte-mode", choices=["fp16_equiv_dequant", "storage"], default=None)
    ap.add_argument("--attn-impl", default="simple", choices=["simple", "flash"])
    ap.add_argument("--fa-bc", type=int, default=128)
    ap.add_argument("--attn-naive-spill", action="store_true")
    ap.add_argument("--only-batch", type=int, default=0)
    ap.add_argument("--only-preset", default=None)
    ap.add_argument("--only-kv", default=None)
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument(
        "--diag-streams",
        action="store_true",
        help="With per-row output, append roofline stream times (ms per decode step: Tc,Tw,Tkv,max,overlap,step).",
    )
    ap.add_argument(
        "--unit",
        default="ms_per_tok",
        choices=["ms_per_tok", "tok_s"],
    )
    args = ap.parse_args()

    if args.diag_streams and args.summary_only:
        print("[warn] --diag-streams ignored with --summary-only", file=sys.stderr)

    paths: list[str] = []
    for p in args.measured_json:
        paths.append(os.path.abspath(os.path.expanduser(p)))
    if args.glob_measured:
        paths.extend(sorted(glob.glob(args.glob_measured, recursive=True)))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No measured JSONs")

    cat = sim.load_structure_catalog(args.catalog)
    gguf_cache: dict[str, dict[str, int]] = {}

    def _find_gguf_for_row(r: dict[str, Any]) -> Optional[str]:
        mp = r.get("model_path")
        if not mp:
            return None
        base = os.path.basename(str(mp))
        if not base.endswith(".gguf"):
            return None
        for d in list(args.gguf_dir) + ["/home/qshao/Project/Fun/models"]:
            if not d:
                continue
            p = os.path.join(os.path.abspath(os.path.expanduser(d)), base)
            if os.path.isfile(p):
                return p
        p0 = os.path.abspath(os.path.expanduser(str(mp)))
        return p0 if os.path.isfile(p0) else None

    def _gguf_tb(r: dict[str, Any]) -> Optional[dict[str, int]]:
        gp = _find_gguf_for_row(r)
        if not gp:
            return None
        if gp not in gguf_cache:
            gguf_cache[gp] = load_gguf_tensor_n_bytes(gp)
        return gguf_cache[gp]

    def _pick_cal(measured_path: str) -> tuple[dict[str, Any], str]:
        if args.auto_calibration:
            d = os.path.dirname(os.path.abspath(measured_path))
            cand = os.path.join(d, "stream_calib_decode_h100.json")
            if os.path.isfile(cand):
                return spm.load_stream_calib(cand), cand
        if args.calibration_json:
            p = os.path.abspath(os.path.expanduser(args.calibration_json))
            return spm.load_stream_calib(p), p
        return spm.default_calib(str(args.hw)), "(default_calib — uncalibrated)"

    meas_lbl = "meas_tok/s" if args.unit == "tok_s" else "meas_ms"
    pred_lbl = "pred_tok/s" if args.unit == "tok_s" else "pred_ms"
    err_lbl = "err_tok/s" if args.unit == "tok_s" else "err_ms"
    hdr_base = (
        f"{'preset':10s} {'wtag':8s} {'kv':6s} {'B':>3s} {'mid':>5s} "
        f"{meas_lbl:>9s} {pred_lbl:>9s} {err_lbl:>9s} {'pred/meas':>9s} {'dom':>8s}"
    )
    hdr_diag = (
        f" {'Tc_ms':>7} {'Tw_ms':>7} {'Tk_ms':>7} {'mx_ms':>7} "
        f"{'wm_ol':>7} {'mk_ol':>7} {'step_ms':>8}"
    )
    hdr = hdr_base + (hdr_diag if args.diag_streams else "")

    all_errs: list[float] = []
    all_ratios: list[float] = []

    for measured_path in paths:
        cal, cal_src = _pick_cal(measured_path)
        for w in spm.stream_calib_report_warnings(
            cal,
            hw_name=str(args.hw),
            attn_impl=str(args.attn_impl),
            fa_bc=int(args.fa_bc),
            attn_naive_spill=bool(args.attn_naive_spill),
            kv_attn_byte_mode_cli=args.kv_attn_byte_mode,
        ):
            print(f"[warn] {measured_path}: {w}", file=sys.stderr)
        if args.kv_attn_byte_mode is not None:
            cal = dict(cal)
            cal["kv_attn_byte_mode"] = str(args.kv_attn_byte_mode)

        rows = _load_rows(measured_path)
        errs: list[float] = []
        ratios: list[float] = []

        if not args.summary_only:
            print("=" * 120)
            print("MEASURED vs stream_perf_model", flush=True)
            print(f"measured_json={measured_path!r}", flush=True)
            print(f"stream_calibration={cal_src!r}", flush=True)
            print("=" * 120)
            print(hdr)
            print("-" * len(hdr))

        shown = 0
        n_uniform_weights = 0
        for r in rows:
            mp = str(r.get("model_preset", "")).strip()
            if not mp or mp not in cat["presets"]:
                continue
            if args.only_preset and mp != args.only_preset:
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

            mid = spm.mid_ctx(r)
            meas = float(r.get("measured_ms", 0.0))
            if meas <= 0:
                continue

            kv_key = sim.resolve_kv_quant_key(kv_cli)
            model = dict(cat["presets"][mp])
            wtag = str(r.get("weight_tag", "") or "").strip()
            wb, _ = sim.resolve_weight_bits(model, wtag if wtag else None, float(r.get("weight_bits", 16.0)))
            tb = _gguf_tb(r)
            if tb is None:
                n_uniform_weights += 1

            feats = spm.extract_decode_features(
                model,
                batch_size=B,
                ctx_len=mid,
                hw_name=str(args.hw),
                kv_quant_key=str(kv_key),
                weight_bpe=float(wb) / 8.0,
                norm_bpe=16.0 / 8.0,
                gguf_tensor_bytes=tb,
                kv_group_size=None,
                kv_asym=False,
                attn_impl=str(args.attn_impl),
                fa_bc=int(args.fa_bc),
                attn_naive_spill=bool(args.attn_naive_spill),
                kv_attn_byte_mode=str(cal.get("kv_attn_byte_mode", "fp16_equiv_dequant")),
            )
            brk = spm.decode_stream_breakdown_s(
                feats,
                batch_size=B,
                hw_name=str(args.hw),
                kv_quant_key=str(kv_key),
                cal=cal,
            )
            pred = float(brk["ms_per_tok"])
            dom = str(brk["dominant"])

            if args.unit == "tok_s":
                meas_u = _ms_per_tok_to_tok_s(meas)
                pred_u = _ms_per_tok_to_tok_s(pred)
            else:
                meas_u = meas
                pred_u = pred

            err = pred_u - meas_u
            ratio = pred_u / meas_u if meas_u else float("nan")

            if not args.summary_only:
                line = (
                    f"{mp:10s} {wtag:8s} {kv_cli:6s} {B:3d} {mid:5d} "
                    f"{_fmt(meas_u)} {_fmt(pred_u)} {_fmt(err)} {ratio:9.3f} {dom:>8s}"
                )
                if args.diag_streams:
                    tc = float(brk["tc_s"]) * 1000.0
                    tw = float(brk["tw_s"]) * 1000.0
                    tk = float(brk["tk_s"]) * 1000.0
                    mx = float(brk["t_max_s"]) * 1000.0
                    owm = float(brk["t_alpha_wm_s"]) * 1000.0
                    omk = float(brk["t_alpha_mk_s"]) * 1000.0
                    stp = float(brk["t_step_s"]) * 1000.0
                    line += (
                        f" {tc:7.3f} {tw:7.3f} {tk:7.3f} {mx:7.3f} "
                        f"{owm:7.3f} {omk:7.3f} {stp:8.3f}"
                    )
                print(line)

            errs.append(err)
            ratios.append(ratio)
            shown += 1

        if n_uniform_weights and shown:
            print(
                f"[warn] {measured_path}: {n_uniform_weights}/{shown} comparable rows used "
                f"uniform_bpw weight bytes (GGUF tensor map not resolved)",
                file=sys.stderr,
            )

        if not errs:
            print(f"[skip] no rows: {measured_path!r}", file=sys.stderr)
            continue

        mae = sum(abs(e) for e in errs) / len(errs)
        rmse = math.sqrt(sum(e * e for e in errs) / len(errs))
        mean_ratio = sum(ratios) / len(ratios)
        all_errs.extend(errs)
        all_ratios.extend(ratios)

        base = os.path.relpath(measured_path, os.getcwd())
        print(f"[summary] {base}: n={len(errs)}  MAE={mae:.4f}  RMSE={rmse:.4f}  mean(pred/meas)={mean_ratio:.3f}")

    if all_errs:
        mae = sum(abs(e) for e in all_errs) / len(all_errs)
        rmse = math.sqrt(sum(e * e for e in all_errs) / len(all_errs))
        mean_ratio = sum(all_ratios) / len(all_ratios)
        print(f"\n[overall] files={len(paths)}  n={len(all_errs)}  MAE={mae:.4f}  RMSE={rmse:.4f}  mean(pred/meas)={mean_ratio:.3f}")
    else:
        print("No comparable rows.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
