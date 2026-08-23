#!/usr/bin/env python3
"""Sweep llama-batched-bench across npp and npl, collect CUDA PERF decode breakdown.

Outputs
  • CSV path: --output (default sweep_results.csv), columns npp/npl + SUMMARY ms/pct + wall/tok/s.
  • Text tables on stderr: «DECODE DURATION (ms)» and «DECODE PERCENTAGE (%)» — save the first
    block to a file such as research/results/decode_sweep.txt for roofline_layer --sweep-csv.

Fixed arguments passed to llama-batched-bench (not configurable here unless via trailing extras):
  -fa 1   -ngl 99
  -c C    where C = max(--min-context, (npp + ntg) * npl + 1024)
  -npp -ntg -npl from each grid point

Example with every sweep_batched_bench parameter set explicitly to its default (adjust paths):

  python3 research/scripts/sweep_batched_bench.py \\
    --model ./models/Qwen3-8B-Q8_0.gguf \\
    --bin ./build_nvtx/bin/llama-batched-bench \\
    --npp 1024,2048,4096,8192,16384 \\
    --npl 1,2,4,8,16,32 \\
    --ntg 128 \\
    --min-context 1024 \\
    --output research/results/decode_sweep.csv \\
    2> research/results/decode_sweep_bench.stderr.txt

  • Optional per-op scaling tables (fixed op order, matches cuda print):
    --print-op-detail --op-detail-file PATH --ops-long-csv PATH

Optional KV types (defaults: omit = bench default):

  python3 research/scripts/sweep_batched_bench.py -m ./models/Qwen3-8B-Q8_0.gguf \\
    --bin ./build/bin/llama-batched-bench \\
    --npp 1024,2048,4096,8192,16384 \\
    --npl 1,2,4,8,16,32 \\
    --ntg 128 \\
    --min-context 1024 \\
    --cache-type-k f16 --cache-type-v f16 \\
    -o research/results/decode_sweep.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys


def _op_sort_key(key: str):
    """Match ggml-cuda GGML_CUDA_PERF sort_key_for_op (structural order, not by time)."""
    colon = key.find(":")
    name = key if colon < 0 else key[:colon]
    ty = "" if colon < 0 else key[colon + 1 :]

    def starts(pfx: str) -> bool:
        return name.startswith(pfx)

    tier = 1000
    if "norm" in name and ty in ("RMS_NORM", "NORM"):
        tier = 10
    elif name == "Qcur" and ty == "MUL_MAT":
        tier = 20
    elif name == "Kcur" and ty == "MUL_MAT":
        tier = 21
    elif name == "Vcur" and ty == "MUL_MAT":
        tier = 22
    elif name == "Qcur" and ty == "ROPE":
        tier = 30
    elif name == "Kcur" and ty == "ROPE":
        tier = 31
    elif starts("cache_k"):
        tier = 40
    elif starts("cache_v"):
        tier = 41
    elif "fattn" in name or name == "__fattn__":
        tier = 50
    elif "result_output" in name or "attn_output" in name:
        tier = 60
    elif name == "l_out":
        tier = 70
    elif "ffn_inp" in name:
        tier = 71
    elif "ffn_gate" in name:
        tier = 80
    elif "ffn_up" in name:
        tier = 81
    elif "ffn_swiglu" in name:
        tier = 82
    elif "ffn_out" in name or "ffn_down" in name:
        tier = 83
    elif name.startswith("node_"):
        tier = 200
    elif "copy" in key:
        tier = 300
    return (tier, key)


def write_op_detail_table(dest, r: dict, npp: int, npl: int, ntg: int, model: str) -> None:
    """Fixed-order op lines (same tiers as ggml-cuda.cu) for vimdiff across npp/npl."""
    ops = r.get("ops") or {}
    if not ops:
        return
    hdr = (
        f"\n### CUDA PERF DECODE row  npp={npp}  npl={npl}  ntg={ntg}  "
        f"model={model}\n"
        f"  kernel={r.get('kernel_ms', 0):.2f} ms  wall={r.get('gpu_wall_ms', 0):.2f} ms  "
        f"calls={r.get('calls', 0)}\n"
    )
    dest.write(hdr)
    dest.write(f"  {'op':<36} {'avg_ms':>10} {'pct':>8} {'count':>8}\n")
    dest.write(f"  {'-' * 36} {'-' * 10} {'-' * 8} {'-' * 8}\n")
    for k in sorted(ops.keys(), key=_op_sort_key):
        o = ops[k]
        dest.write(f"  {k:<36} {o['ms']:>10.3f} {o['pct']:>7.1f}% {o['count']:>8}\n")
    dest.write(f"  {'-' * 36} {'-' * 10} {'-' * 8} {'-' * 8}\n")


def parse_decode_block(stderr_text):
    """Parse the CUDA PERF DECODE block from stderr."""
    pattern = r"=== CUDA PERF DECODE \((\d+) tok/step, (\d+) calls\) \[.*?(\d+\.\d+) ms kernel, (\d+\.\d+) ms wall\] ===(.*?)={40,}"
    m = re.search(pattern, stderr_text, re.DOTALL)
    if not m:
        return None

    result = {
        "tok_per_step": int(m.group(1)),
        "calls": int(m.group(2)),
        "kernel_ms": float(m.group(3)),
        "wall_ms": float(m.group(4)),
    }

    body = m.group(5)

    # parse summary categories
    for cat in ["QKV\\+O proj", "RoPE", "Attention", "FFN", "Norm", "Other"]:
        pat = rf"\[SUMMARY\]\s+{cat}\s+([\d.]+)\s+([\d.]+)%"
        cm = re.search(pat, body)
        if cm:
            key = cat.replace("\\+", "+").replace("\\", "")
            result[f"{key}_ms"] = float(cm.group(1))
            result[f"{key}_pct"] = float(cm.group(2))

    # parse per-op lines
    op_pattern = r"^\s+([\w\*\(\):_ ]+?)\s{2,}([\d.]+)\s+([\d.]+)%\s+(\d+)\s*$"
    ops = {}
    for line in body.split("\n"):
        om = re.match(op_pattern, line)
        if om:
            name = om.group(1).strip()
            if not name.startswith("["):
                ops[name] = {"ms": float(om.group(2)), "pct": float(om.group(3)), "count": int(om.group(4))}
    result["ops"] = ops

    # parse totals
    for label, key in [("Kernel sum", "kernel_sum_ms"), ("GPU wall", "gpu_wall_ms"),
                        ("Launch/sync overhead", "overhead_ms"),
                        ("tok/s \\(per seq\\)", "toks_per_seq"),
                        ("tok/s \\(total\\)", "toks_total")]:
        pat = rf"\[{label}\]\s+([\d.]+)"
        tm = re.search(pat, body)
        if tm:
            result[key] = float(tm.group(1))

    return result


def run_bench(args, npp, npl, op_detail_file=None):
    """Run llama-batched-bench with given npp and npl."""
    ntg = args.ntg
    c = max(args.min_context, (npp + ntg) * npl + 1024)

    cmd = [
        args.bin, "-m", args.model,
        "-fa", "1", "-ngl", "99",
        "-c", str(c),
        "-npp", str(npp),
        "-ntg", str(ntg),
        "-npl", str(npl),
    ]
    if args.cache_type_k:
        cmd += ["--cache-type-k", args.cache_type_k]
    if args.cache_type_v:
        cmd += ["--cache-type-v", args.cache_type_v]
    cmd += args.extra

    cmd_str = " ".join(cmd)
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  npp={npp}, npl={npl}, c={c}", file=sys.stderr)
    print(f"  {cmd_str}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT", file=sys.stderr)
        return None

    result = parse_decode_block(proc.stderr)
    if result is None:
        print(f"  WARNING: No CUDA PERF DECODE block found", file=sys.stderr)
        # check for errors
        for line in proc.stderr.split("\n"):
            if "failed" in line.lower() or "error" in line.lower():
                print(f"  {line.strip()}", file=sys.stderr)
        return None

    # parse llama-batched-bench table for S_TG
    tg_pattern = r"\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)\s*\|"
    tg_match = re.search(tg_pattern, proc.stdout + proc.stderr)
    if tg_match:
        result["bench_tg_toks"] = float(tg_match.group(1))

    result["npp"] = npp
    result["npl"] = npl
    print(f"  OK: wall={result.get('gpu_wall_ms', '?'):.2f}ms, "
          f"attn={result.get('Attention_pct', '?'):.1f}%, "
          f"ffn={result.get('FFN_pct', '?'):.1f}%, "
          f"tok/s(seq)={result.get('toks_per_seq', '?'):.1f}", file=sys.stderr)

    if op_detail_file is not None and result.get("ops"):
        write_op_detail_table(op_detail_file, result, npp, npl, args.ntg, args.model)
    if getattr(args, "print_op_detail", False) and result.get("ops"):
        write_op_detail_table(sys.stderr, result, npp, npl, args.ntg, args.model)
    return result


def main():
    p = argparse.ArgumentParser(
        description="Sweep llama-batched-bench across npp × npl; parse CUDA PERF DECODE block.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", "-m", required=True, help="Path to GGUF (llama-batched-bench -m).")
    p.add_argument("--bin", default="./build_nvtx/bin/llama-batched-bench",
                   help="llama-batched-bench binary.")
    p.add_argument("--npp", type=str, default="1024,2048,4096,8192,16384",
                   help="Comma-separated prompt lengths (-npp).")
    p.add_argument("--npl", type=str, default="1,2,4,8,16,32",
                   help="Comma-separated parallel decode lanes (-npl).")
    p.add_argument("--ntg", type=int, default=128, help="Tokens to generate per lane (-ntg).")
    p.add_argument("--cache-type-k", type=str, default=None,
                   help="Passed as --cache-type-k to the bench if set.")
    p.add_argument("--cache-type-v", type=str, default=None,
                   help="Passed as --cache-type-v to the bench if set.")
    p.add_argument("--min-context", type=int, default=1024, dest="min_context",
                   help="Floor for -c; context is max(this, (npp+ntg)*npl+1024).")
    p.add_argument("--output", "-o", type=str, default="sweep_results.csv",
                   help="CSV written with per-row SUMMARY ms/pct and wall/tok/s.")
    p.add_argument("--print-op-detail", action="store_true",
                   help="After each successful grid point, print per-op breakdown (fixed structural order) to stderr.")
    p.add_argument("--op-detail-file", type=str, default=None,
                   help="Write the same per-op tables here (truncated at start). Same order as GGML_CUDA_PERF in ggml-cuda.cu.")
    p.add_argument("--ops-long-csv", type=str, default=None,
                   help="Tall CSV: one row per (npp, npl, op) for batch/context scaling plots.")
    p.add_argument("extra", nargs="*", help="Extra tokens appended to the bench command line.")
    args = p.parse_args()

    npp_list = [int(x) for x in args.npp.split(",")]
    npl_list = [int(x) for x in args.npl.split(",")]

    op_detail_fp = None
    if args.op_detail_file:
        op_detail_fp = open(
            os.path.expanduser(args.op_detail_file), "w", encoding="utf-8")
        op_detail_fp.write(
            "# Per-op CUDA PERF decode (structural sort; vimdiff across runs).\n"
            f"# model={args.model} ntg={args.ntg}\n\n")

    ops_long_fp = None
    ops_long_w = None
    if args.ops_long_csv:
        ops_long_fp = open(os.path.expanduser(args.ops_long_csv), "w", newline="", encoding="utf-8")
        ops_long_fields = [
            "npp", "npl", "ntg", "model", "kernel_ms", "gpu_wall_ms", "calls",
            "op", "avg_ms", "pct", "count",
        ]
        ops_long_w = csv.DictWriter(ops_long_fp, fieldnames=ops_long_fields)
        ops_long_w.writeheader()

    results = []
    for npp in npp_list:
        for npl in npl_list:
            r = run_bench(args, npp, npl, op_detail_file=op_detail_fp)
            if r:
                results.append(r)
                if ops_long_w and r.get("ops"):
                    for k in sorted(r["ops"].keys(), key=_op_sort_key):
                        o = r["ops"][k]
                        ops_long_w.writerow({
                            "npp": npp,
                            "npl": npl,
                            "ntg": args.ntg,
                            "model": args.model,
                            "kernel_ms": r.get("kernel_ms", ""),
                            "gpu_wall_ms": r.get("gpu_wall_ms", ""),
                            "calls": r.get("calls", ""),
                            "op": k,
                            "avg_ms": o["ms"],
                            "pct": o["pct"],
                            "count": o["count"],
                        })

    if op_detail_fp:
        op_detail_fp.close()
        print(f"\nOp detail written to {args.op_detail_file}", file=sys.stderr)
    if ops_long_fp:
        ops_long_fp.close()
        print(f"Ops long CSV written to {args.ops_long_csv}", file=sys.stderr)

    if not results:
        print("No results collected!", file=sys.stderr)
        return

    # write CSV
    categories_csv = ["QKV+O proj", "RoPE", "Attention", "FFN", "Norm", "Other"]
    fieldnames = ["npp", "npl", "kernel_ms", "wall_ms"]
    for cat in categories_csv:
        fieldnames += [f"{cat}_ms", f"{cat}_pct"]
    fieldnames += ["overhead_ms", "toks_per_seq", "toks_total", "bench_tg_toks"]

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row)
    print(f"\nResults written to {args.output}", file=sys.stderr)

    # print two summary tables
    categories = ["QKV+O proj", "RoPE", "Attention", "FFN", "Norm", "Other"]
    cat_short  = ["QKV+O", "RoPE", "Attn", "FFN", "Norm", "Other"]

    def print_table(title, value_key, fmt, suffix=""):
        print(f"\n{'='*110}", file=sys.stderr)
        print(f"  {title}  (model: {args.model}, ntg={args.ntg})", file=sys.stderr)
        print(f"{'='*110}", file=sys.stderr)
        hdr = f"{'npp':>6} {'npl':>4}"
        for c in cat_short:
            hdr += f" {c:>8}"
        hdr += f" {'Wall':>8} {'tok/s':>8}"
        print(hdr, file=sys.stderr)
        print("-" * 110, file=sys.stderr)

        prev_npp = None
        for r in results:
            if prev_npp is not None and r["npp"] != prev_npp:
                print("-" * 110, file=sys.stderr)
            prev_npp = r["npp"]

            line = f"{r['npp']:>6} {r['npl']:>4}"
            for cat in categories:
                val = r.get(f"{cat}_{value_key}", 0)
                line += f" {fmt.format(val):>8}"
            wall = r.get('gpu_wall_ms', 0)
            line += f" {wall:>8.2f}"
            npl = r.get('npl', 1)
            per_seq = r.get('toks_per_seq', 0)
            total_toks = per_seq * npl
            line += f" {total_toks:>8.1f}"
            print(line, file=sys.stderr)

        print(f"{'='*110}\n", file=sys.stderr)

    print_table("DECODE DURATION (ms)", "ms", "{:.2f}")
    print_table("DECODE PERCENTAGE (%)", "pct", "{:.1f}")


if __name__ == "__main__":
    main()
