#!/usr/bin/env python3
"""Run roofline_layer.py against cluster kv_timing JSON rows and record calibration checks.

Uses downloaded profile results (research/results/<preset>/profile/kv_timing_h100.json) — no GGUF
required. For each row, subprocesses roofline with matching --model, --batch-size, prompt/decode,
--main-gguf-quant, --kv-quant / --calibration-kv-type, and --calibration-json pointing at that file.

Writes a JSON report (default: research/results/roofline_calibration_verify.json) with:
  measured_ms (from row), roofline_ms_this, decode_scale, scaled_tok_s, ok flag.

Example:
  python3 research/scripts/verify_roofline_calibration.py \\
      --scan-root research/results --glob '**/qwen3-8b*/profile/kv_timing_h100.json'
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
ROOFLINE = os.path.join(_SCRIPT_DIR, "roofline_layer.py")

# Native KV names from benchmark_kv_timing → roofline_layer --kv-quant
NATIVE_KV_TO_ROOFLINE = {
    "f16":  "fp16",
    "q8_0": "int8_ch",
    "q4_0": "int4_ch",
}


def _parse_roofline_output(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m = re.search(
        r"Measured ms/tok \(cluster / profile\):\s*([\d.eE+-]+)", text)
    if m:
        out["parsed_measured_ms"] = float(m.group(1))
    m = re.search(
        r"decode_scale = measured / roofline_this =\s*([\d.eE+-]+)", text)
    if m:
        out["decode_scale"] = float(m.group(1))
    m = re.search(
        r"decode_scale = measured / ref_fp16 =\s*([\d.eE+-]+)", text)
    if m:
        out["decode_scale"] = float(m.group(1))
    m = re.search(r"Scaled decode[^\n]*→\s*([\d.]+)\s*tok/s", text)
    if m:
        out["scaled_tok_s"] = float(m.group(1))
    m = re.search(r"mode=(\S+)", text)
    if m:
        out["mode"] = m.group(1)
    return out


def _run_one(
        profile_path: str,
        hw: str,
        model: str,
        batch_size: int,
        n_prompt: int,
        n_decode: int,
        weight_tag: str,
        native_kv: str,
) -> tuple[str, Dict[str, Any]]:
    kvq = NATIVE_KV_TO_ROOFLINE.get(native_kv, "fp16")
    ckv = native_kv
    # Row-level check: compare to measured_ms (mean). match_ctx would use decode_buckets.
    cmd = [
        sys.executable,
        ROOFLINE,
        "--model", model,
        "--hw", hw,
        "--batch-size", str(batch_size),
        "--n-prompt", str(n_prompt),
        "--n-decode", str(n_decode),
        "--kv-quant", kvq,
        "--main-gguf-quant", weight_tag,
        "--calibration-kv-type", ckv,
        "--calibration-bucket-policy", "mean",
        "--calibration-json", profile_path,
    ]
    p = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    merged: Dict[str, Any] = {
        "returncode": p.returncode,
        "stdout_tail": p.stdout[-4000:] if p.stdout else "",
    }
    if p.stderr:
        merged["stderr_tail"] = p.stderr[-2000:]
    merged.update(_parse_roofline_output(p.stdout or ""))
    return p.stdout or "", merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scan-root", default=os.path.join(_REPO_ROOT, "research", "results"),
        help="Root to glob for kv_timing JSON (default: research/results)")
    ap.add_argument(
        "--glob", default="**/qwen3-8b*/profile/kv_timing_h100.json",
        help="Glob under scan-root (default: qwen3-8b* profiles)")
    ap.add_argument(
        "--profiles", nargs="*", metavar="PATH",
        help="Explicit kv_timing JSON paths (skip glob if set)")
    ap.add_argument("--hw", default="h100-sxm")
    ap.add_argument(
        "-o", "--output",
        default=os.path.join(_REPO_ROOT, "research", "results", "roofline_calibration_verify.json"),
        help="Write verification report JSON")
    ap.add_argument(
        "--filter-model", default=None,
        help="Only rows where model_preset equals this (e.g. qwen3-8b)")
    args = ap.parse_args()

    paths: List[str] = []
    if args.profiles:
        paths = [os.path.expanduser(p) for p in args.profiles]
    else:
        pattern = os.path.join(args.scan_root, args.glob)
        paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        print("No profile JSON files found.", file=sys.stderr)
        sys.exit(1)

    report: Dict[str, Any] = {
        "_comment": "verify_roofline_calibration.py — roofline vs cluster kv_timing rows",
        "hw": args.hw,
        "runs": [],
    }

    for profile_path in paths:
        if not os.path.isfile(profile_path):
            print(f"skip missing: {profile_path}", file=sys.stderr)
            continue
        with open(profile_path, encoding="utf-8") as f:
            cal = json.load(f)
        rows = cal.get("rows") or []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            mp = row.get("model_preset")
            if args.filter_model and mp != args.filter_model:
                continue
            measured = float(row["measured_ms"])
            wtag = row["weight_tag"]
            kv_t = str(row.get("kv_type", "f16"))
            stdout, meta = _run_one(
                profile_path=os.path.abspath(profile_path),
                hw=args.hw,
                model=mp,
                batch_size=int(row["batch_size"]),
                n_prompt=int(row["prompt_len"]),
                n_decode=int(row["decode_len"]),
                weight_tag=wtag,
                native_kv=kv_t,
            )
            ps = meta.get("parsed_measured_ms")
            scale = meta.get("decode_scale")
            ok = (
                meta.get("returncode") == 0
                and ps is not None
                and abs(ps - measured) < 1e-3
                and scale is not None
            )
            report["runs"].append({
                "profile": os.path.relpath(profile_path, _REPO_ROOT),
                "row_index": i,
                "model_preset": mp,
                "weight_tag": wtag,
                "kv_type": kv_t,
                "batch_size": row["batch_size"],
                "prompt_len": row["prompt_len"],
                "decode_len": row["decode_len"],
                "measured_ms_row": measured,
                "parsed_measured_ms": ps,
                "decode_scale": scale,
                "scaled_tok_s": meta.get("scaled_tok_s"),
                "calibration_mode": meta.get("mode"),
                "ok": ok,
            })
            if not ok:
                report["runs"][-1]["debug_returncode"] = meta.get("returncode")

    n_ok = sum(1 for r in report["runs"] if r.get("ok"))
    report["summary"] = {
        "total": len(report["runs"]),
        "ok": n_ok,
        "failed": len(report["runs"]) - n_ok,
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
