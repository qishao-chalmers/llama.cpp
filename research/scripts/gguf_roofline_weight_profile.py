#!/usr/bin/env python3
"""Emit a weight-bpw JSON profile for research/scripts/roofline_layer.py from a GGUF file.

Maps llama.cpp-style tensor names (blk.N.*) into roofline buckets:
  attn  — attn_qkv / attn_q,k,v / attn_output weights
  mlp   — ffn_gate / ffn_up / ffn_down weights
  default — all other *.weight tensors (embeddings, output head, norms, …)

Bits-per-weight is element-weighted: (n_bytes * 8) / n_elements per bucket.
Use: python3 research/scripts/roofline_layer.py ... --weight-bpw-profile out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", help="Path to .gguf model")
    ap.add_argument("-o", "--output", metavar="PATH", help="Write JSON here (default: stdout)")
    args = ap.parse_args()

    root = _repo_root()
    sys.path.insert(0, os.path.join(root, "gguf-py"))
    from gguf import GGUFReader  # noqa: E402

    # Match convert naming in gguf-py constants (MODEL_TENSOR.*).
    attn_re = re.compile(
        r"blk\.\d+\.(attn_qkv|attn_q|attn_k|attn_v|attn_output)\.weight$"
    )
    ffn_re = re.compile(r"blk\.\d+\.(ffn_gate|ffn_up|ffn_down)\.weight$")

    reader = GGUFReader(args.gguf, "r")
    attn_b = attn_e = 0
    mlp_b = mlp_e = 0
    other_b = other_e = 0
    for t in reader.tensors:
        name = t.name
        if not name.endswith(".weight"):
            continue
        nb, ne = t.n_bytes, int(t.n_elements)
        if ne <= 0:
            continue
        if attn_re.search(name):
            attn_b += nb
            attn_e += ne
        elif ffn_re.search(name):
            mlp_b += nb
            mlp_e += ne
        else:
            other_b += nb
            other_e += ne

    def bpw(nbytes: int, nel: int) -> float:
        return (nbytes * 8.0) / nel if nel else float("nan")

    def_bpw = bpw(other_b, other_e)
    if other_e <= 0 and (attn_e + mlp_e) > 0:
        def_bpw = bpw(attn_b + mlp_b, attn_e + mlp_e)

    out: dict = {
        "attn": round(bpw(attn_b, attn_e), 6),
        "mlp": round(bpw(mlp_b, mlp_e), 6),
        "default": round(def_bpw, 6) if def_bpw == def_bpw else 16.0,
        "_meta": {
            "gguf": os.path.abspath(args.gguf),
            "attn_elements": attn_e,
            "mlp_elements": mlp_e,
            "other_elements": other_e,
        },
    }

    txt = json.dumps(out, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(txt)
            f.write("\n")
    else:
        print(txt)


if __name__ == "__main__":
    main()
