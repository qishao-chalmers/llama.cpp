#!/usr/bin/env python3
"""show_kv_quant_toks.py — Show decode tok/s vs context length and KV quant.

Generates a table of decode tok/s for each combination of context length
and KV quantization format, using the roofline_layer.py engine.

Usage:
    python3 research/scripts/show_kv_quant_toks.py
    python3 research/scripts/show_kv_quant_toks.py --model qwen3-8b --hw a100-80g --batch-size 1
    python3 research/scripts/show_kv_quant_toks.py --batch-size 32
    python3 research/scripts/show_kv_quant_toks.py --batch-size 1 32 128
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
from roofline_layer import (
    HARDWARE_PRESETS, MODEL_PRESETS, QUANT_CONFIGS, ops_for_layer,
)


def run_table(model_name, hw_name, B, quants, ctx_list):
    hw = HARDWARE_PRESETS[hw_name]
    cp = hw["compute_tflops"] * 1e12
    bw = hw["memory_bw_gbps"] * 1e9
    m  = dict(MODEL_PRESETS[model_name])
    nl = m["n_layers"]

    kw_base = dict(model=m, compute_eff=0.70, mem_eff=0.85, attn_eff=0.85,
                   compute_peak=cp, weight_bw=bw, attn_bw=bw, act_bw=bw,
                   flash_attn=True, kv_group_size=64, padding_eff=1.0, weight_bits=16)

    print(f"\n  B={B} {model_name} / {hw_name} — Decode tok/s vs context length and KV quant")
    w = 130
    print(f"  {'═' * w}")
    hdr = f"{'ctx':>8} |"
    for q in quants:
        hdr += f" {q:>14}"
    hdr += "  |  int4/fp16  int2/fp16  KV%fp16  sys_tok/s(fp16)"
    print(f"  {hdr}")
    print(f"  {'─' * w}")

    for ctx in ctx_list:
        row = f"{ctx:>8} |"
        tps = {}
        for q in quants:
            kw = dict(kw_base, batch_size=B, kv_quant=q)
            ops = ops_for_layer("decode", B, ctx, **kw)
            t = sum(op.time_s * nl for op in ops)
            tok_s = 1.0 / t
            tps[q] = tok_s
            row += f" {tok_s:>14.1f}"

        r_int4 = tps["int4_ch"] / tps["fp16"]
        r_int2 = tps["int2_ch"] / tps["fp16"]

        kw_fp = dict(kw_base, batch_size=B, kv_quant="fp16")
        ops_fp = ops_for_layer("decode", B, ctx, **kw_fp)
        t_kv  = sum(op.time_s * nl for op in ops_fp if op.kv_dependent)
        t_all = sum(op.time_s * nl for op in ops_fp)
        kvpct = t_kv / t_all * 100 if t_all > 0 else 0
        sys_tps = tps["fp16"] * B

        row += f"  |  {r_int4:>8.2f}x  {r_int2:>8.2f}x  {kvpct:>5.0f}%  {sys_tps:>14.0f}"
        print(f"  {row}")

    print(f"  {'─' * w}")
    print(f"  Note: tok/s values are per-sequence. sys_tok/s = tok/s/seq × B (total throughput).")


def main():
    p = argparse.ArgumentParser(description="Decode tok/s vs context length and KV quant")
    p.add_argument("--model", default="qwen3-8b", choices=list(MODEL_PRESETS))
    p.add_argument("--hw",    default="a100-80g",  choices=list(HARDWARE_PRESETS))
    p.add_argument("--batch-size", type=int, nargs="+", default=[1],
                   help="Batch size(s) to sweep (default: 1)")
    p.add_argument("--ctx", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                   help="Context lengths to sweep")
    args = p.parse_args()

    quants = ["fp16", "int8_ch", "int4_ch", "int3_ch", "int3_half_1357_ch", "int2_ch"]

    for B in args.batch_size:
        run_table(args.model, args.hw, B, quants, args.ctx)
    print()


if __name__ == "__main__":
    main()
