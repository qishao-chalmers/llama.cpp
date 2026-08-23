#!/usr/bin/env python3
"""profile_ops_sweep.py — Run llama-bench across configs, parse per-op timing.

Requires a build with -DGGML_CUDA_PERF=ON -DGGML_CUDA_GRAPHS=OFF.

Usage:
    python3 research/scripts/profile_ops_sweep.py \\
        --model models/Qwen3-8B-Q8_0.gguf \\
        --bin ./build_nvtx/bin/llama-bench \\
        --prompt-lengths 512 1024 2048 4096 8192 \\
        --gen-tokens 128 \\
        --reps 1

    # Save CSV:
    python3 research/scripts/profile_ops_sweep.py \\
        --model models/Qwen3-8B-Q8_0.gguf \\
        --out research/results/perf_ops.csv
"""

import argparse
import csv
import io
import re
import subprocess
import sys
from collections import defaultdict


def parse_perf_blocks(stderr_text: str):
    """Parse CUDA PERF output blocks from stderr.

    Returns list of dicts, each with:
      - call_num: int
      - n_ops: int
      - total_ms: float
      - ops: dict[str, {time_ms, pct, count}]
      - summary: dict[str, {time_ms, pct}]
    """
    blocks = []
    block_re = re.compile(
        r"=== CUDA PERF \[call #(\d+), (\d+) ops, ([\d.]+) ms total\] ==="
    )
    op_re = re.compile(
        r"^\s{2}(\S+)\s+([\d.]+)\s+([\d.]+)%\s+(\d+)\s*$", re.MULTILINE
    )
    summary_re = re.compile(
        r"^\s{2}\[SUMMARY\]\s+(.+?)\s+([\d.]+)\s+([\d.]+)%\s*$", re.MULTILINE
    )

    for m in block_re.finditer(stderr_text):
        call_num = int(m.group(1))
        n_ops = int(m.group(2))
        total_ms = float(m.group(3))

        start = m.end()
        next_block = block_re.search(stderr_text, start)
        end = next_block.start() if next_block else len(stderr_text)
        chunk = stderr_text[start:end]

        ops = {}
        for om in op_re.finditer(chunk):
            name = om.group(1)
            if name.startswith("---"):
                continue
            ops[name] = {
                "time_ms": float(om.group(2)),
                "pct": float(om.group(3)),
                "count": int(om.group(4)),
            }

        summary = {}
        for sm in summary_re.finditer(chunk):
            summary[sm.group(1)] = {
                "time_ms": float(sm.group(2)),
                "pct": float(sm.group(3)),
            }

        blocks.append({
            "call_num": call_num,
            "n_ops": n_ops,
            "total_ms": total_ms,
            "ops": ops,
            "summary": summary,
        })

    return blocks


def run_llama_bench(bin_path, model, prompt_len, gen_tokens, reps, extra_args):
    """Run llama-bench and return (stdout, stderr)."""
    cmd = [
        bin_path,
        "-m", model,
        "-p", str(prompt_len),
        "-n", str(gen_tokens),
        "-r", str(reps),
        "-ngl", "99",
        "-fa", "1",
    ] + extra_args

    print(f"  Running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return result.stdout, result.stderr


def extract_tps(stdout_text: str):
    """Extract tok/s from llama-bench stdout (pp and tg lines)."""
    pp_tps, tg_tps = None, None
    for line in stdout_text.strip().split("\n"):
        if not line or line.startswith("|") is False:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 8:
            continue
        try:
            tps = float(parts[-2])
        except (ValueError, IndexError):
            continue
        # pp line has n_gen=0, tg line has n_prompt=0
        if "pp" in line.lower() or (len(parts) > 4 and parts[4] == "0"):
            continue
        tg_tps = tps

    # simpler: just grab all numbers from lines containing t/s
    for line in stdout_text.strip().split("\n"):
        if "t/s" not in line and "tok/s" not in line:
            continue
    return pp_tps, tg_tps


def main():
    parser = argparse.ArgumentParser(description="Sweep llama-bench configs, parse per-op CUDA timing")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--bin", default="./build_nvtx/bin/llama-bench", help="llama-bench binary")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=None,
                        help="Explicit prompt lengths (e.g. 512 1024 2048)")
    parser.add_argument("--prompt-range", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                        help="Prompt range with power-of-2 steps (e.g. --prompt-range 512 16384 gives 512,1024,...,16384)")
    parser.add_argument("--gen-tokens", type=int, default=128, help="Tokens to generate per run")
    parser.add_argument("--reps", type=int, default=1, help="Repetitions per config")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra args for llama-bench (e.g. -ctk q8_0)")
    args = parser.parse_args()

    if args.prompt_lengths:
        prompt_lengths = args.prompt_lengths
    elif args.prompt_range:
        lo, hi = args.prompt_range
        prompt_lengths = []
        v = lo
        while v <= hi:
            prompt_lengths.append(v)
            v *= 2
    else:
        prompt_lengths = [512, 1024, 2048, 4096]

    all_results = []

    for p_len in prompt_lengths:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"  Prompt length: {p_len}, gen: {args.gen_tokens}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)

        stdout, stderr = run_llama_bench(
            args.bin, args.model, p_len, args.gen_tokens, args.reps, args.extra
        )

        blocks = parse_perf_blocks(stderr)

        if not blocks:
            print(f"  WARNING: No CUDA PERF blocks found. Is the binary built with -DGGML_CUDA_PERF=ON?",
                  file=sys.stderr)
            continue

        # Typically: first blocks are prefill, later blocks are decode
        # With -r 1: we expect a few prefill calls + many decode calls
        # After warmup skip (call_count <= 2), we get the real measurements
        for blk in blocks:
            row = {
                "prompt_len": p_len,
                "gen_tokens": args.gen_tokens,
                "call_num": blk["call_num"],
                "total_ms": blk["total_ms"],
                "n_ops": blk["n_ops"],
            }
            for sname, sval in blk["summary"].items():
                key = sname.replace(" ", "_").replace("+", "_")
                row[f"summary_{key}_ms"] = sval["time_ms"]
                row[f"summary_{key}_pct"] = sval["pct"]

            for op_name, op_val in blk["ops"].items():
                row[f"op_{op_name}_ms"] = op_val["time_ms"]
                row[f"op_{op_name}_pct"] = op_val["pct"]
                row[f"op_{op_name}_count"] = op_val["count"]

            all_results.append(row)

        # Print summary for this config
        print(f"\n  --- Summary for p={p_len} ---", file=sys.stderr)
        for blk in blocks:
            print(f"  call #{blk['call_num']}: {blk['total_ms']:.2f} ms, {blk['n_ops']} ops",
                  file=sys.stderr)
            for sname, sval in blk["summary"].items():
                print(f"    {sname:20s}: {sval['time_ms']:8.3f} ms ({sval['pct']:.1f}%)",
                      file=sys.stderr)

    # Write CSV
    if args.out and all_results:
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())
        all_keys = sorted(all_keys)

        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"\nResults saved to {args.out}", file=sys.stderr)

    # Print a compact final summary table
    if all_results:
        print("\n" + "=" * 90)
        print(f"{'p_len':>8} {'call':>6} {'total_ms':>10} {'QKV+O':>10} {'Attn':>10} "
              f"{'FFN':>10} {'Norm':>10} {'Other':>10}")
        print("-" * 90)
        for r in all_results:
            proj  = r.get("summary_QKV_O_proj_ms", r.get("summary_QKV+O_proj_ms", 0))
            attn  = r.get("summary_Attention_ms", 0)
            ffn   = r.get("summary_FFN_ms", 0)
            norm  = r.get("summary_Norm_ms", 0)
            other = r.get("summary_Other_ms", 0)
            print(f"{r['prompt_len']:>8} {r['call_num']:>6} {r['total_ms']:>10.2f} "
                  f"{proj:>10.3f} {attn:>10.3f} {ffn:>10.3f} {norm:>10.3f} {other:>10.3f}")
        print("=" * 90)


if __name__ == "__main__":
    main()
