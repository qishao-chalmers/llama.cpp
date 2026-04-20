#!/usr/bin/env python3
"""Flatten stream_calib_decode_h100.json files into one CSV for hyperparameter trends.

Joins each calibration with model_structures.json (by model_preset from the sibling
kv_timing_h100.json when present) and emits one row per profile calibration.

Typical workflow:
  python3 research/scripts/collect_stream_calib_hyperparams.py \\
    --glob-calib 'research/results/**/profile/stream_calib_decode_h100.json' \\
    --out research/results/stream_calib_hyperparams.csv

  # Quick text summary (mean/std by weight_tag for key etas)
  python3 research/scripts/collect_stream_calib_hyperparams.py --glob-calib '...' --summary
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import layerwise_roofline_sim as sim  # noqa: E402


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _profile_slug_from_calib(calib_path: str) -> str:
    segs = os.path.abspath(calib_path).replace("\\", "/").split("/")
    try:
        i = segs.index("results")
        return segs[i + 1]
    except (ValueError, IndexError):
        return os.path.basename(os.path.dirname(os.path.dirname(calib_path)))


def _kv_timing_path(calib_path: str) -> str:
    d = os.path.dirname(os.path.abspath(calib_path))
    return os.path.join(d, "kv_timing_h100.json")


def _sample_row_from_kv_timing(path: str) -> Optional[dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    data = _load_json(path)
    for r in data.get("rows", []):
        if float(r.get("measured_ms", 0) or 0) > 0:
            return r
    return None


def _get_eta_kv(m: dict[str, Any], key: str) -> str:
    ek = m.get("eta_kv_bw_by_kv") or {}
    if not isinstance(ek, dict):
        return ""
    v = ek.get(key)
    if v is None:
        return ""
    return f"{float(v):.8f}"


def collect_rows(
    calib_paths: list[str],
    catalog_path: str,
) -> tuple[list[str], list[dict[str, str]]]:
    cat = sim.load_structure_catalog(catalog_path)
    presets = cat.get("presets", {})

    fieldnames = [
        "profile_slug",
        "calib_relpath",
        "hw",
        "kv_attn_byte_mode",
        "model_preset",
        "weight_tag",
        "weight_bits",
        "family",
        "n_layers",
        "d_model",
        "ffn_dim",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "eta_weight_bw",
        "eta_compute",
        "eta_kv_fp16",
        "eta_kv_int8_ch",
        "eta_kv_int4_ch",
        "fixed_B1",
        "fixed_B4",
        "fixed_B8",
        "fixed_B16",
        "fixed_B32",
        "tail_ms",
        "alpha_wm",
        "alpha_mk",
        "fit_n_rows",
        "fit_n_params",
        "fit_rows_per_param",
        "fit_mae_ms_per_tok",
        "fit_rmse_ms_per_tok",
        "fit_allow_no_gguf",
        "fit_gguf_cache_n",
        "fit_attn_impl",
    ]

    rows_out: list[dict[str, str]] = []
    cwd = os.getcwd()

    for cp in calib_paths:
        cp_abs = os.path.abspath(cp)
        cal = _load_json(cp_abs)
        slug = _profile_slug_from_calib(cp_abs)
        rel = os.path.relpath(cp_abs, cwd)

        kv_path = _kv_timing_path(cp_abs)
        sample = _sample_row_from_kv_timing(kv_path)
        preset_name = ""
        wtag = ""
        wb = ""
        if sample:
            preset_name = str(sample.get("model_preset", "") or "").strip()
            wtag = str(sample.get("weight_tag", "") or "").strip()
            wb = str(sample.get("weight_bits", "") or "").strip()

        struct = dict(presets.get(preset_name, {})) if preset_name else {}
        family = str(struct.get("family", "") or "")
        nl = str(struct.get("n_layers", "") or "")
        dm = str(struct.get("d_model", "") or "")
        ff = str(struct.get("ffn_dim", "") or "")
        nh = str(struct.get("n_heads", "") or "")
        nkh = str(struct.get("n_kv_heads", "") or "")
        hd = str(struct.get("head_dim", "") or "")

        fo = cal.get("fixed_overhead_ms_by_batch") or {}
        if not isinstance(fo, dict):
            fo = {}

        def fb(b: int) -> str:
            v = fo.get(str(b), fo.get(b))
            if v is None:
                return ""
            return f"{float(v):.8f}"

        fit = cal.get("fit") or {}
        if not isinstance(fit, dict):
            fit = {}

        row: dict[str, str] = {
            "profile_slug": slug,
            "calib_relpath": rel,
            "hw": str(cal.get("hw", "") or ""),
            "kv_attn_byte_mode": str(cal.get("kv_attn_byte_mode", "") or ""),
            "model_preset": preset_name,
            "weight_tag": wtag,
            "weight_bits": wb,
            "family": family,
            "n_layers": nl,
            "d_model": dm,
            "ffn_dim": ff,
            "n_heads": nh,
            "n_kv_heads": nkh,
            "head_dim": hd,
            "eta_weight_bw": f"{float(cal.get('eta_weight_bw', 0)):.8f}",
            "eta_compute": f"{float(cal.get('eta_compute', 0)):.8f}",
            "eta_kv_fp16": _get_eta_kv(cal, "fp16"),
            "eta_kv_int8_ch": _get_eta_kv(cal, "int8_ch"),
            "eta_kv_int4_ch": _get_eta_kv(cal, "int4_ch"),
            "fixed_B1": fb(1),
            "fixed_B4": fb(4),
            "fixed_B8": fb(8),
            "fixed_B16": fb(16),
            "fixed_B32": fb(32),
            "tail_ms": f"{float(cal.get('tail_ms', 0)):.8f}",
            "alpha_wm": f"{float(cal.get('alpha_wm', 0)):.8f}",
            "alpha_mk": f"{float(cal.get('alpha_mk', 0)):.8f}",
            "fit_n_rows": str(int(fit.get("n_rows", 0) or 0)),
            "fit_n_params": str(int(fit.get("n_params", 0) or 0)),
            "fit_rows_per_param": str(fit.get("rows_per_param", "") or ""),
            "fit_mae_ms_per_tok": str(fit.get("mae_ms_per_tok", "") or ""),
            "fit_rmse_ms_per_tok": str(fit.get("rmse_ms_per_tok", "") or ""),
            "fit_allow_no_gguf": str(bool(fit.get("allow_no_gguf", False))),
            "fit_gguf_cache_n": str(int(fit.get("gguf_cache_n", 0) or 0)),
            "fit_attn_impl": str(fit.get("attn_impl", "") or ""),
        }
        rows_out.append(row)

    return fieldnames, rows_out


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def print_trend_ols_weight_bits(rows: list[dict[str, str]]) -> None:
    """Simple level-2 trend: eta_weight_bw ~ a + b * weight_bits (unweighted OLS)."""

    try:
        import numpy as np
    except ImportError:
        print("numpy required for --trend-ols", file=sys.stderr)
        return

    xs: list[float] = []
    ys: list[float] = []
    for r in rows:
        wb = _f(r["weight_bits"])
        ew = _f(r["eta_weight_bw"])
        if math.isfinite(wb) and math.isfinite(ew):
            xs.append(wb)
            ys.append(ew)
    if len(xs) < 3:
        print(f"\n=== OLS eta_weight_bw ~ weight_bits === insufficient rows (n={len(xs)}, need >=3)")
        return
    a = np.column_stack([np.ones(len(xs)), np.array(xs, dtype=float)])
    b = np.array(ys, dtype=float)
    coef, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    resid = b - a @ coef
    rmse = float(math.sqrt(float(np.mean(resid**2))))
    print("\n=== OLS eta_weight_bw ~ 1 + weight_bits (all profiles in table) ===")
    print(f"  intercept={float(coef[0]):.6f}  slope={float(coef[1]):.6f}  n={len(xs)}  rmse={rmse:.6f}")


def print_summary(rows: list[dict[str, str]]) -> None:
    by_tag: dict[str, list[float]] = defaultdict(list)
    by_preset: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        tag = r.get("weight_tag") or "(empty)"
        by_tag[tag].append(_f(r["eta_weight_bw"]))
        pr = r.get("model_preset") or "(empty)"
        by_preset[pr].append(_f(r["eta_compute"]))

    def stat(xs: list[float]) -> str:
        xs = [x for x in xs if not math.isnan(x)]
        if not xs:
            return "n=0"
        m = sum(xs) / len(xs)
        v = sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)
        sd = math.sqrt(v) if len(xs) > 1 else 0.0
        return f"n={len(xs)} mean={m:.4f} std={sd:.4f}"

    print("=== eta_weight_bw by weight_tag ===")
    for tag in sorted(by_tag.keys()):
        print(f"  {tag:12s}  {stat(by_tag[tag])}")
    print("\n=== eta_compute by model_preset ===")
    for pr in sorted(by_preset.keys()):
        print(f"  {pr:16s}  {stat(by_preset[pr])}")

    # KV etas across profiles (pooled)
    for col, label in (
        ("eta_kv_fp16", "fp16"),
        ("eta_kv_int8_ch", "int8_ch"),
        ("eta_kv_int4_ch", "int4_ch"),
    ):
        vals = [_f(r[col]) for r in rows if r.get(col)]
        vals = [x for x in vals if not math.isnan(x)]
        if vals:
            m = sum(vals) / len(vals)
            sd = math.sqrt(sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1))
            print(f"\n=== eta_kv {label} (pooled) === n={len(vals)} mean={m:.4f} std={sd:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-json", action="append", default=[], help="Calibration JSON (repeatable)")
    ap.add_argument("--glob-calib", default=None, help="Glob for stream_calib*.json (recursive)")
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("-o", "--out", default=None, help="Write CSV here (default: stdout)")
    ap.add_argument("--summary", action="store_true", help="Print grouped mean/std to stdout")
    ap.add_argument(
        "--trend-ols",
        action="store_true",
        help="Print unweighted OLS: eta_weight_bw ~ 1 + weight_bits (needs numpy, n>=3)",
    )
    args = ap.parse_args()

    paths: list[str] = []
    for p in args.calib_json:
        paths.append(os.path.abspath(os.path.expanduser(p)))
    if args.glob_calib:
        paths.extend(sorted(glob.glob(args.glob_calib, recursive=True)))
    paths = sorted(dict.fromkeys(paths))
    if not paths:
        print("No calibration files. Use --calib-json or --glob-calib.", file=sys.stderr)
        sys.exit(2)

    fieldnames, rows = collect_rows(paths, args.catalog)

    if args.summary:
        print_summary(rows)
    if args.trend_ols:
        print_trend_ols_weight_bits(rows)

    out_f = (
        open(os.path.abspath(os.path.expanduser(args.out)), "w", encoding="utf-8", newline="")
        if args.out
        else sys.stdout
    )
    close_out = bool(args.out)
    try:
        w = csv.DictWriter(out_f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    finally:
        if close_out:
            out_f.close()


if __name__ == "__main__":
    main()
