#!/usr/bin/env python3
"""Simple log: each GGUF tensor's quantization type and effective bits/weight.

For K-quants (e.g. Q4_K_M in the filename), every weight row still has a single
``tensor_type`` such as ``Q4_K`` or ``Q6_K`` per tensor. The "mixed 4 vs 6 bit"
story is *inside* a Q4_K super-block, not separate GGUF tensors — this script
shows storage type + average BPW for each tensor.

Usage:
  python3 research/scripts/gguf_log_tensor_types.py /path/to/model.gguf
  python3 research/scripts/gguf_log_tensor_types.py model.gguf --match 'blk\\..*\\.weight'
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", help="Model .gguf path")
    ap.add_argument(
        "--match",
        default=None,
        help="If set, only tensors whose name matches this regex (e.g. 'blk\\\\..*weight').",
    )
    args = ap.parse_args()

    path = Path(args.gguf).expanduser()
    if not path.is_file():
        print(f"Not a file: {path}", file=sys.stderr)
        sys.exit(2)

    rx = re.compile(args.match) if args.match else None
    reader = GGUFReader(str(path), "r")

    print(f"# {path.name}  tensors={len(reader.tensors)}")
    print(f"{'type':12}  {'bpw':>7}  n_elems  name")
    for t in reader.tensors:
        if rx and not rx.search(t.name):
            continue
        bpw = (float(t.n_bytes) * 8.0 / float(t.n_elements)) if t.n_elements else 0.0
        print(f"{t.tensor_type.name:12}  {bpw:7.3f}  {t.n_elements:8}  {t.name}")


if __name__ == "__main__":
    main()
