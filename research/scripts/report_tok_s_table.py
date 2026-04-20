#!/usr/bin/env python3
"""Emit measured vs stream-model predicted decode tok/s in Markdown or CSV.

Calibration and physics match report_measured_vs_stream.py (per-profile
stream_calib_decode_h100.json when --auto-calibration is set).
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import os
import sys
from typing import Any, Iterator, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import layerwise_roofline_sim as sim  # noqa: E402
import stream_perf_model as spm  # noqa: E402
from gguf_layerwise_weights import load_gguf_tensor_n_bytes  # noqa: E402


def _load_rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("rows", []))


def _ms_per_tok_to_tok_s(ms_per_tok: float) -> float:
    if ms_per_tok <= 0 or math.isnan(ms_per_tok) or math.isinf(ms_per_tok):
        return float("nan")
    return 1000.0 / float(ms_per_tok)


def _profile_slug(measured_path: str) -> str:
    segs = os.path.abspath(measured_path).replace("\\", "/").split("/")
    try:
        i = segs.index("results")
        return segs[i + 1]
    except (ValueError, IndexError):
        return os.path.basename(os.path.dirname(os.path.dirname(measured_path)))


def _collect_paths(measured_json: list[str], glob_measured: Optional[str]) -> list[str]:
    paths: list[str] = []
    for p in measured_json:
        paths.append(os.path.abspath(os.path.expanduser(p)))
    if glob_measured:
        paths.extend(sorted(glob.glob(glob_measured, recursive=True)))
    return sorted(dict.fromkeys(paths))


def iter_tok_s_rows(
    *,
    paths: list[str],
    catalog_path: str,
    hw_name: str,
    auto_calibration: bool,
    calibration_json: Optional[str],
    gguf_dirs: list[str],
    only_batch: int,
    only_preset: Optional[str],
    only_kv: Optional[str],
    attn_impl: str,
    fa_bc: int,
    attn_naive_spill: bool,
    kv_attn_byte_mode: Optional[str],
    file_weight_stats: Optional[dict[str, list[int]]] = None,
) -> Iterator[dict[str, Any]]:
    cat = sim.load_structure_catalog(catalog_path)
    gguf_cache: dict[str, dict[str, int]] = {}
    search_dirs = list(gguf_dirs) + ["/home/qshao/Project/Fun/models"]

    def _find_gguf_for_row(r: dict[str, Any]) -> Optional[str]:
        mp = r.get("model_path")
        if not mp:
            return None
        base = os.path.basename(str(mp))
        if not base.endswith(".gguf"):
            return None
        for d in search_dirs:
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

    def _pick_cal(measured_path: str) -> dict[str, Any]:
        if auto_calibration:
            d = os.path.dirname(os.path.abspath(measured_path))
            cand = os.path.join(d, "stream_calib_decode_h100.json")
            if os.path.isfile(cand):
                return spm.load_stream_calib(cand)
        if calibration_json:
            return spm.load_stream_calib(calibration_json)
        return spm.default_calib(str(hw_name))

    for measured_path in paths:
        cal = _pick_cal(measured_path)
        for w in spm.stream_calib_report_warnings(
            cal,
            hw_name=str(hw_name),
            attn_impl=str(attn_impl),
            fa_bc=int(fa_bc),
            attn_naive_spill=bool(attn_naive_spill),
            kv_attn_byte_mode_cli=kv_attn_byte_mode,
        ):
            print(f"[warn] {measured_path}: {w}", file=sys.stderr)
        if kv_attn_byte_mode is not None:
            cal = dict(cal)
            cal["kv_attn_byte_mode"] = str(kv_attn_byte_mode)

        prof = _profile_slug(measured_path)
        rel_json = os.path.relpath(measured_path, os.getcwd())

        for r in _load_rows(measured_path):
            mp = str(r.get("model_preset", "")).strip()
            if not mp or mp not in cat["presets"]:
                continue
            if only_preset and mp != only_preset:
                continue
            kv_cli = str(r.get("kv_type", "")).strip()
            if not kv_cli:
                continue
            if only_kv and kv_cli != only_kv:
                continue
            bsz = int(r.get("batch_size", 0) or 0)
            if bsz <= 0:
                continue
            if only_batch and bsz != int(only_batch):
                continue

            mid = spm.mid_ctx(r)
            meas_ms = float(r.get("measured_ms", 0.0))
            if meas_ms <= 0:
                continue

            kv_key = sim.resolve_kv_quant_key(kv_cli)
            model = dict(cat["presets"][mp])
            wtag = str(r.get("weight_tag", "") or "").strip()
            wb, _ = sim.resolve_weight_bits(model, wtag if wtag else None, float(r.get("weight_bits", 16.0)))
            tb = _gguf_tb(r)
            if file_weight_stats is not None:
                st = file_weight_stats.setdefault(measured_path, [0, 0])
                st[0] += 1
                if tb is None:
                    st[1] += 1

            feats = spm.extract_decode_features(
                model,
                batch_size=bsz,
                ctx_len=mid,
                hw_name=str(hw_name),
                kv_quant_key=str(kv_key),
                weight_bpe=float(wb) / 8.0,
                norm_bpe=16.0 / 8.0,
                gguf_tensor_bytes=tb,
                kv_group_size=None,
                kv_asym=False,
                attn_impl=str(attn_impl),
                fa_bc=int(fa_bc),
                attn_naive_spill=bool(attn_naive_spill),
                kv_attn_byte_mode=str(cal.get("kv_attn_byte_mode", "fp16_equiv_dequant")),
            )
            pred_ms = spm.predict_decode_ms_per_tok(
                feats,
                batch_size=bsz,
                hw_name=str(hw_name),
                kv_quant_key=str(kv_key),
                cal=cal,
            )
            dom = spm.dominant_stream(
                feats,
                batch_size=bsz,
                hw_name=str(hw_name),
                kv_quant_key=str(kv_key),
                cal=cal,
            )

            meas_ts = _ms_per_tok_to_tok_s(meas_ms)
            pred_ts = _ms_per_tok_to_tok_s(pred_ms)
            ratio = pred_ts / meas_ts if meas_ts and not math.isnan(meas_ts) else float("nan")

            yield {
                "profile": prof,
                "measured_json": rel_json,
                "preset": mp,
                "wtag": wtag,
                "kv": kv_cli,
                "B": bsz,
                "mid_ctx": mid,
                "meas_tok_s": meas_ts,
                "pred_tok_s": pred_ts,
                "pred_over_meas": ratio,
                "dominant": dom,
            }


def _write_csv(rows: list[dict[str, Any]], out: io.TextIOBase) -> None:
    fieldnames = [
        "profile",
        "preset",
        "wtag",
        "kv",
        "B",
        "mid_ctx",
        "meas_tok_s",
        "pred_tok_s",
        "pred_over_meas",
        "dominant",
        "measured_json",
    ]
    w = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        r = dict(row)
        for k in ("meas_tok_s", "pred_tok_s", "pred_over_meas"):
            v = r.get(k)
            if isinstance(v, float) and not (math.isnan(v) or math.isinf(v)):
                r[k] = round(v, 6)
        w.writerow(r)


def _write_markdown_wide(rows: list[dict[str, Any]], out: io.TextIOBase) -> None:
    by_kv: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kv.setdefault(row["kv"], []).append(row)

    batches = sorted({r["B"] for r in rows})
    profiles = sorted({r["profile"] for r in rows})

    for kv in sorted(by_kv.keys()):
        out.write(f"### KV: `{kv}`\n\n")
        subcols: list[str] = []
        for b in batches:
            subcols.append(f"B={b} meas")
            subcols.append(f"B={b} pred")
        hdr = "| profile | preset | wtag | " + " | ".join(subcols) + " |\n"
        sep = "|" + "|".join(["---"] * (3 + 2 * len(batches))) + "|\n"
        out.write(hdr)
        out.write(sep)

        lookup = {(r["profile"], r["kv"], r["B"]): r for r in by_kv[kv]}
        for prof in profiles:
            sample = next((x for x in by_kv[kv] if x["profile"] == prof), None)
            if not sample:
                continue
            cells: list[str] = []
            for b in batches:
                d = lookup.get((prof, kv, b))
                if d:
                    cells.append(f"{d['meas_tok_s']:.2f}")
                    cells.append(f"{d['pred_tok_s']:.2f}")
                else:
                    cells.extend(["", ""])
            out.write("| " + " | ".join([prof, sample["preset"], sample["wtag"]] + cells) + " |\n")
        out.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measured-json", action="append", default=[])
    ap.add_argument("--glob-measured", default=None)
    ap.add_argument("--hw", default="h100-sxm", choices=list(sim.HARDWARE_PRESETS.keys()))
    ap.add_argument("--catalog", default=os.path.join(_SCRIPT_DIR, "model_structures.json"))
    ap.add_argument("--calibration-json", default=None)
    ap.add_argument(
        "--auto-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load stream_calib_decode_h100.json beside each measured JSON (default: on).",
    )
    ap.add_argument("--gguf-dir", action="append", default=[], help="Extra GGUF search dirs")
    ap.add_argument("--kv-attn-byte-mode", choices=["fp16_equiv_dequant", "storage"], default=None)
    ap.add_argument("--attn-impl", default="simple", choices=["simple", "flash"])
    ap.add_argument("--fa-bc", type=int, default=128)
    ap.add_argument("--attn-naive-spill", action="store_true")
    ap.add_argument("--only-batch", type=int, default=0)
    ap.add_argument("--only-preset", default=None)
    ap.add_argument("--only-kv", default=None)
    ap.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    ap.add_argument(
        "--out",
        default="-",
        help="Output path, or '-' for stdout (default: -).",
    )
    args = ap.parse_args()

    paths = _collect_paths(args.measured_json, args.glob_measured)
    if not paths:
        print("No measured JSONs (use --measured-json or --glob-measured).", file=sys.stderr)
        sys.exit(2)

    file_weight_stats: dict[str, list[int]] = {}
    rows = list(
        iter_tok_s_rows(
            paths=paths,
            catalog_path=args.catalog,
            hw_name=str(args.hw),
            auto_calibration=bool(args.auto_calibration),
            calibration_json=args.calibration_json,
            gguf_dirs=list(args.gguf_dir),
            only_batch=int(args.only_batch),
            only_preset=args.only_preset,
            only_kv=args.only_kv,
            attn_impl=str(args.attn_impl),
            fa_bc=int(args.fa_bc),
            attn_naive_spill=bool(args.attn_naive_spill),
            kv_attn_byte_mode=args.kv_attn_byte_mode,
            file_weight_stats=file_weight_stats,
        )
    )
    for mp, st in sorted(file_weight_stats.items()):
        tot, uni = st[0], st[1]
        if uni and tot:
            print(
                f"[warn] {mp}: {uni}/{tot} rows used uniform_bpw weight bytes "
                f"(GGUF tensor map not resolved)",
                file=sys.stderr,
            )
    if not rows:
        print("No comparable rows.", file=sys.stderr)
        sys.exit(2)

    if args.out == "-":
        out_f: io.TextIOBase = sys.stdout
        close_out = False
    else:
        out_f = open(os.path.abspath(os.path.expanduser(args.out)), "w", encoding="utf-8", newline="")
        close_out = True

    try:
        if args.format == "csv":
            _write_csv(rows, out_f)
        else:
            out_f.write(
                "Measured vs predicted decode tok/s (stream_perf_model). "
                f"hw={args.hw} attn_impl={args.attn_impl} auto_calib={args.auto_calibration}\n\n"
            )
            _write_markdown_wide(rows, out_f)
    finally:
        if close_out:
            out_f.close()


if __name__ == "__main__":
    main()
