#!/usr/bin/env python3
"""kv_cache_size.py — Show KV cache memory usage for one or more GGUFs.

Usage:
    python3 research/scripts/kv_cache_size.py models/Qwen3-8B-Q8_0.gguf
    python3 research/scripts/kv_cache_size.py models/*.gguf --ctx 4096 8192 22288
    python3 research/scripts/kv_cache_size.py models/*.gguf --quants fp16 int8 int4 int2
    python3 research/scripts/kv_cache_size.py model.gguf --batch 4
    python3 research/scripts/kv_cache_size.py model.gguf --batch 1 2 4 8
"""

import argparse
import sys
from pathlib import Path

try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

QUANT_BYTES = {
    "fp16": 2.0,
    "fp32": 4.0,
    "int8": 1.0,
    "int4": 0.5,
    "int3": 0.375,
    "int2": 0.25,
}

DEFAULT_CTX    = [512, 1024, 2048, 4096, 8192, 16384, 22288, 32768]
DEFAULT_QUANTS = ["fp16", "int8", "int4", "int2"]
DEFAULT_BATCH  = [1]


def read_arch(path: str) -> dict:
    """Extract KV-relevant architecture fields from a GGUF file.

    GGUF fields use an arch-specific prefix (e.g. 'qwen3.', 'llama.',
    'mistral.').  We match by suffix so this works for any model family.
    """
    r = gguf.GGUFReader(path, "r")
    fields = {kv.name: kv for kv in r.fields.values()}

    def find(suffix, default=None):
        for name, kv in fields.items():
            if name.endswith(suffix):
                return int(kv.parts[-1][0])
        return default

    n_layers   = find(".block_count")
    n_kv_heads = find(".attention.head_count_kv")
    head_dim   = find(".attention.key_length")

    # Fallback: head_dim = n_embd / n_heads
    if head_dim is None:
        n_embd  = find(".embedding_length")
        n_heads = find(".attention.head_count")
        if n_embd and n_heads:
            head_dim = n_embd // n_heads

    return {
        "path":      path,
        "name":      Path(path).name,
        "n_layers":  n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim":  head_dim,
    }


def kv_bytes(arch: dict, n_ctx: int, bytes_per_elem: float) -> float:
    """Total KV cache bytes (K + V, all layers)."""
    return (2 * arch["n_layers"] * arch["n_kv_heads"]
            * arch["head_dim"] * n_ctx * bytes_per_elem)


def fmt_gb(b: float) -> str:
    gb = b / 1024**3
    if gb >= 10:
        return f"{gb:6.1f} GB"
    return f"{gb:6.2f} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="GGUF file path(s)")
    ap.add_argument("--ctx", nargs="+", type=int, default=DEFAULT_CTX,
                    metavar="N", help="context lengths to show")
    ap.add_argument("--quants", nargs="+", default=DEFAULT_QUANTS,
                    choices=list(QUANT_BYTES), metavar="QUANT",
                    help=f"KV quant types: {list(QUANT_BYTES)}")
    ap.add_argument("--batch", nargs="+", type=int, default=DEFAULT_BATCH,
                    metavar="B", help="batch size(s): KV bytes × B parallel sequences "
                    "each with ctx tokens (default: 1)")
    args = ap.parse_args()

    archs = []
    for p in args.models:
        try:
            a = read_arch(p)
            if None in (a["n_layers"], a["n_kv_heads"], a["head_dim"]):
                print(f"[warn] missing arch fields in {p}, skipping", file=sys.stderr)
                continue
            archs.append(a)
        except Exception as e:
            print(f"[warn] {p}: {e}", file=sys.stderr)

    if not archs:
        sys.exit("No valid GGUF files found.")

    ctx_list    = sorted(set(args.ctx))
    quant_list  = args.quants
    batch_list  = sorted(set(args.batch))
    for bsz in batch_list:
        if bsz < 1:
            sys.exit("--batch values must be >= 1")

    for arch in archs:
        print(f"\n{'='*70}")
        print(f"  {arch['name']}")
        print(f"  n_layers={arch['n_layers']}  n_kv_heads={arch['n_kv_heads']}"
              f"  head_dim={arch['head_dim']}")
        kb_per_tok = (2 * arch["n_layers"] * arch["n_kv_heads"]
                      * arch["head_dim"] * QUANT_BYTES["fp16"]) / 1024
        print(f"  fp16 KV: {kb_per_tok:.0f} KB/token (per sequence, not × batch)")
        print(f"{'='*70}")

        col_w = 10

        if len(batch_list) > 1:
            # One header; rows cover every (batch, ctx) — avoids repeating batch lines
            # and the quant header for each batch.
            header = (f"{'batch':>8}{'ctx':>8}"
                      + "".join(f"{q:>{col_w}}" for q in quant_list))
            print(header)
            print("-" * len(header))
            for bsz in batch_list:
                for ctx in ctx_list:
                    row = f"{bsz:>8}{ctx:>8}"
                    for q in quant_list:
                        b = kv_bytes(arch, ctx, QUANT_BYTES[q]) * bsz
                        row += f"{fmt_gb(b):>{col_w}}"
                    print(row)
        else:
            bsz = batch_list[0]
            if bsz != 1:
                print(f"\n  batch={bsz}  (total KV = per-sequence table × {bsz})")
            header = f"{'ctx':>8}" + "".join(f"{q:>{col_w}}" for q in quant_list)
            print(header)
            print("-" * len(header))
            for ctx in ctx_list:
                row = f"{ctx:>8}"
                for q in quant_list:
                    b = kv_bytes(arch, ctx, QUANT_BYTES[q]) * bsz
                    row += f"{fmt_gb(b):>{col_w}}"
                print(row)

    print()


if __name__ == "__main__":
    main()
